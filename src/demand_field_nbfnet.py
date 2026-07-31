"""CUDA-capable OD-conditioned bidirectional Neural Bellman-Ford network."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as functional
from torch.utils.checkpoint import checkpoint


NBFNET_VARIANTS = (
    "base",
    "origin_only",
    "destination_only",
    "shared_parameters",
    "undirected",
    "degree_rewired",
    "shuffled_od",
    "fixed_diffusion",
    "graphsage",
    "no_edge_features",
    "no_interactions",
    "last_layer_only",
    "no_ranking",
    "propagation_deep",
    "propagation_residual",
    "propagation_doubling",
    "propagation_residual_doubling",
)

PROPAGATION_ONLY_VARIANTS = frozenset(
    {
        "propagation_deep",
        "propagation_residual",
        "propagation_doubling",
        "propagation_residual_doubling",
    }
)
RESIDUAL_PROPAGATION_VARIANTS = frozenset(
    {"propagation_residual", "propagation_residual_doubling"}
)
DOUBLING_PROPAGATION_VARIANTS = frozenset(
    {"propagation_doubling", "propagation_residual_doubling"}
)


@dataclass(frozen=True, slots=True)
class NBFNetConfig:
    hidden_dim: int = 32
    propagation_layers: int = 6
    demand_scale: float = 1000.0
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    rank_weight: float = 0.20
    huber_delta: float = 1.0
    prototype_batch_size: int = 1
    max_epochs: int = 100
    patience: int = 20
    min_improvement: float = 0.0001
    mixed_precision: bool = True
    gradient_checkpointing: bool = True
    zero_initialize_prediction_head: bool = False
    variant: str = "base"
    randomization_seed: int = 20260730

    def validate(self) -> None:
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.propagation_layers <= 0:
            raise ValueError("propagation_layers must be positive")
        if self.demand_scale <= 0.0:
            raise ValueError("demand_scale must be positive")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("learning rate must be positive and weight decay non-negative")
        if self.rank_weight < 0.0:
            raise ValueError("rank_weight must be non-negative")
        if self.huber_delta <= 0.0:
            raise ValueError("huber_delta must be positive")
        if self.prototype_batch_size <= 0:
            raise ValueError("prototype_batch_size must be positive")
        if self.max_epochs <= 0 or self.patience <= 0:
            raise ValueError("max_epochs and patience must be positive")
        if self.variant not in NBFNET_VARIANTS:
            raise ValueError(
                f"variant must be one of {', '.join(NBFNET_VARIANTS)}"
            )


class _DirectionalLayer(nn.Module):
    def __init__(self, hidden_dim: int, edge_dim: int) -> None:
        super().__init__()
        self.message_state = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.message_edge_gate = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.update_state = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.update_aggregate = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.update_gate = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(
        self,
        state: torch.Tensor,
        sender_index: torch.Tensor,
        receiver_index: torch.Tensor,
        edge_features: torch.Tensor,
        receiver_normalizer: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, node_count, hidden_dim = state.shape
        sender = state[:, sender_index, :]
        receiver = state[:, receiver_index, :]
        repeated_edges = edge_features.unsqueeze(0).expand(batch_size, -1, -1)
        message = self.message_state(sender) * self.message_edge_gate(repeated_edges)
        gate_input = torch.cat((sender, receiver, repeated_edges), dim=-1)
        message = message * torch.sigmoid(self.gate(gate_input))

        batch_offsets = (
            torch.arange(batch_size, device=state.device, dtype=receiver_index.dtype)
            * node_count
        )
        flat_receivers = (receiver_index.unsqueeze(0) + batch_offsets[:, None]).reshape(-1)
        aggregate = message.new_zeros((batch_size * node_count, hidden_dim))
        aggregate.index_add_(0, flat_receivers, message.reshape(-1, hidden_dim))
        aggregate = aggregate.reshape(batch_size, node_count, hidden_dim)
        aggregate = aggregate / receiver_normalizer.to(dtype=aggregate.dtype).unsqueeze(
            0
        ).unsqueeze(-1)

        update_input = torch.cat((state, aggregate), dim=-1)
        candidate = functional.relu(
            self.update_state(state) + self.update_aggregate(aggregate)
        )
        update_gate = torch.sigmoid(self.update_gate(update_input))
        return update_gate * candidate + (1.0 - update_gate) * state


class _GraphSAGELayer(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.self_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.neighbor_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(
        self,
        state: torch.Tensor,
        sender_index: torch.Tensor,
        receiver_index: torch.Tensor,
        receiver_normalizer: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, node_count, hidden_dim = state.shape
        message = state[:, sender_index, :]
        batch_offsets = (
            torch.arange(batch_size, device=state.device, dtype=receiver_index.dtype)
            * node_count
        )
        flat_receivers = (receiver_index.unsqueeze(0) + batch_offsets[:, None]).reshape(-1)
        aggregate = message.new_zeros((batch_size * node_count, hidden_dim))
        aggregate.index_add_(0, flat_receivers, message.reshape(-1, hidden_dim))
        aggregate = aggregate.reshape(batch_size, node_count, hidden_dim)
        aggregate = aggregate / receiver_normalizer.to(
            dtype=aggregate.dtype
        ).unsqueeze(0).unsqueeze(-1)
        return functional.relu(
            self.self_projection(state) + self.neighbor_projection(aggregate)
        )


class _PropagationOnlyLayer(nn.Module):
    """One sparse directed propagation step with an optional residual path."""

    def __init__(self, hidden_dim: int, edge_dim: int, residual: bool) -> None:
        super().__init__()
        self.residual = residual
        self.message_state = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.message_edge_gate = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.message_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.aggregate_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
        )
        self.normalization = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        state: torch.Tensor,
        sender_index: torch.Tensor,
        receiver_index: torch.Tensor,
        edge_features: torch.Tensor,
        receiver_normalizer: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, node_count, hidden_dim = state.shape
        sender = state[:, sender_index, :]
        receiver = state[:, receiver_index, :]
        repeated_edges = edge_features.unsqueeze(0).expand(batch_size, -1, -1)
        message = self.message_state(sender) * self.message_edge_gate(repeated_edges)
        message = message * torch.sigmoid(
            self.message_gate(torch.cat((sender, receiver, repeated_edges), dim=-1))
        )
        batch_offsets = (
            torch.arange(batch_size, device=state.device, dtype=receiver_index.dtype)
            * node_count
        )
        flat_receivers = (receiver_index.unsqueeze(0) + batch_offsets[:, None]).reshape(-1)
        aggregate = message.new_zeros((batch_size * node_count, hidden_dim))
        aggregate.index_add_(0, flat_receivers, message.reshape(-1, hidden_dim))
        aggregate = aggregate.reshape(batch_size, node_count, hidden_dim)
        aggregate = aggregate / receiver_normalizer.to(dtype=aggregate.dtype).unsqueeze(
            0
        ).unsqueeze(-1)
        propagated = self.aggregate_projection(aggregate)
        if self.residual:
            propagated = propagated + state
        return self.normalization(propagated)


class BidirectionalNBFNet(nn.Module):
    """Predict candidate-region value from weighted OD demand prototypes.

    The forward pass accepts a batch of prototype-specific origin and destination
    fields. Each field is propagated on the shared directed graph, pooled over
    candidate regions, and returned as one score per prototype and region.
    """

    def __init__(
        self,
        node_feature_dim: int,
        region_feature_dim: int,
        edge_type_count: int,
        config: NBFNetConfig,
    ) -> None:
        super().__init__()
        config.validate()
        if min(node_feature_dim, region_feature_dim, edge_type_count) <= 0:
            raise ValueError("feature dimensions and edge_type_count must be positive")
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.edge_dim = edge_type_count + 1
        self.propagation_only = config.variant in PROPAGATION_ONLY_VARIANTS
        self.residual_propagation = config.variant in RESIDUAL_PROPAGATION_VARIANTS
        self.doubling_readout = config.variant in DOUBLING_PROPAGATION_VARIANTS
        self.origin_encoder = self._encoder(node_feature_dim, config.hidden_dim)
        self.destination_encoder = (
            self.origin_encoder
            if config.variant == "shared_parameters"
            else self._encoder(node_feature_dim, config.hidden_dim)
        )
        self.origin_token = nn.Parameter(torch.empty(config.hidden_dim))
        nn.init.normal_(self.origin_token, std=config.hidden_dim**-0.5)
        if config.variant == "shared_parameters":
            self.destination_token = self.origin_token
        else:
            self.destination_token = nn.Parameter(torch.empty(config.hidden_dim))
            nn.init.normal_(self.destination_token, std=config.hidden_dim**-0.5)
        if self.propagation_only:
            self.origin_layers = nn.ModuleList()
            self.destination_layers = nn.ModuleList()
            self.graphsage_layers = nn.ModuleList()
            self.origin_propagation_layers = nn.ModuleList(
                _PropagationOnlyLayer(
                    config.hidden_dim,
                    self.edge_dim,
                    self.residual_propagation,
                )
                for _ in range(config.propagation_layers)
            )
            self.destination_propagation_layers = nn.ModuleList(
                _PropagationOnlyLayer(
                    config.hidden_dim,
                    self.edge_dim,
                    self.residual_propagation,
                )
                for _ in range(config.propagation_layers)
            )
        elif config.variant == "fixed_diffusion":
            self.origin_layers = nn.ModuleList()
            self.destination_layers = nn.ModuleList()
            self.graphsage_layers = nn.ModuleList()
            self.origin_propagation_layers = nn.ModuleList()
            self.destination_propagation_layers = nn.ModuleList()
        elif config.variant == "graphsage":
            self.origin_layers = nn.ModuleList()
            self.destination_layers = nn.ModuleList()
            self.graphsage_layers = nn.ModuleList(
                _GraphSAGELayer(config.hidden_dim)
                for _ in range(config.propagation_layers)
            )
            self.origin_propagation_layers = nn.ModuleList()
            self.destination_propagation_layers = nn.ModuleList()
        else:
            self.origin_layers = nn.ModuleList(
                _DirectionalLayer(config.hidden_dim, self.edge_dim)
                for _ in range(config.propagation_layers)
            )
            self.destination_layers = (
                self.origin_layers
                if config.variant == "shared_parameters"
                else nn.ModuleList(
                    _DirectionalLayer(config.hidden_dim, self.edge_dim)
                    for _ in range(config.propagation_layers)
                )
            )
            self.graphsage_layers = nn.ModuleList()
            self.origin_propagation_layers = nn.ModuleList()
            self.destination_propagation_layers = nn.ModuleList()
        self.readout_depths = self._readout_depths(config)
        self.depth_logits = nn.Parameter(torch.zeros(len(self.readout_depths)))
        pooled_dim = config.hidden_dim * 8
        prediction_input_dim = (
            pooled_dim if self.propagation_only else pooled_dim + region_feature_dim
        )
        self.prediction_head = nn.Sequential(
            nn.Linear(prediction_input_dim, config.hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1),
        )
        if config.zero_initialize_prediction_head:
            output_layer = self.prediction_head[-1]
            assert isinstance(output_layer, nn.Linear)
            nn.init.zeros_(output_layer.weight)
            nn.init.zeros_(output_layer.bias)

    @staticmethod
    def _encoder(input_dim: int, hidden_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    @staticmethod
    def _readout_depths(config: NBFNetConfig) -> tuple[int, ...]:
        if config.variant in DOUBLING_PROPAGATION_VARIANTS:
            depths: list[int] = []
            depth = 1
            while depth <= config.propagation_layers:
                depths.append(depth)
                depth *= 2
            if depths[-1] != config.propagation_layers:
                depths.append(config.propagation_layers)
            return tuple(depths)
        if config.variant == "propagation_residual":
            return tuple(range(1, config.propagation_layers + 1))
        if config.variant == "propagation_deep":
            return (config.propagation_layers,)
        return tuple(range(config.propagation_layers + 1))

    def forward(
        self,
        node_features: torch.Tensor,
        edge_source: torch.Tensor,
        edge_target: torch.Tensor,
        edge_features: torch.Tensor,
        origin_fields: torch.Tensor,
        destination_fields: torch.Tensor,
        region_nodes: torch.Tensor,
        region_features: torch.Tensor,
        receiver_normalizer_forward: torch.Tensor,
        receiver_normalizer_reverse: torch.Tensor,
    ) -> torch.Tensor:
        if origin_fields.shape != destination_fields.shape:
            raise ValueError("origin and destination field shapes must match")
        if origin_fields.ndim != 2 or origin_fields.shape[1] != node_features.shape[0]:
            raise ValueError("prototype fields must have shape (batch, node_count)")
        origin_seed = self.origin_encoder(node_features)
        destination_seed = self.destination_encoder(node_features)
        origin_seed = origin_seed + self.origin_token.to(dtype=origin_seed.dtype)
        destination_seed = destination_seed + self.destination_token.to(
            dtype=destination_seed.dtype
        )
        origin_strength = torch.log1p(
            origin_fields * self.config.demand_scale
        ).to(dtype=origin_seed.dtype)
        destination_strength = torch.log1p(
            destination_fields * self.config.demand_scale
        ).to(dtype=destination_seed.dtype)
        origin_state = origin_seed.unsqueeze(0) * origin_strength.unsqueeze(-1)
        destination_state = destination_seed.unsqueeze(0) * destination_strength.unsqueeze(
            -1
        )
        if self.config.variant == "origin_only":
            destination_state = torch.zeros_like(destination_state)
        elif self.config.variant == "destination_only":
            origin_state = torch.zeros_like(origin_state)
        origin_states = [origin_state]
        destination_states = [destination_state]
        for layer_index in range(self.config.propagation_layers):
            if self.propagation_only:
                origin_state = self._run_propagation_layer(
                    self.origin_propagation_layers[layer_index],
                    origin_state,
                    edge_source,
                    edge_target,
                    edge_features,
                    receiver_normalizer_forward,
                )
                destination_state = self._run_propagation_layer(
                    self.destination_propagation_layers[layer_index],
                    destination_state,
                    edge_target,
                    edge_source,
                    edge_features,
                    receiver_normalizer_reverse,
                )
            elif self.config.variant == "fixed_diffusion":
                origin_state = _fixed_mean_diffusion(
                    origin_state,
                    edge_source,
                    edge_target,
                    receiver_normalizer_forward,
                )
                destination_state = _fixed_mean_diffusion(
                    destination_state,
                    edge_target,
                    edge_source,
                    receiver_normalizer_reverse,
                )
            elif self.config.variant == "graphsage":
                layer = self.graphsage_layers[layer_index]
                origin_state = layer(
                    origin_state,
                    edge_source,
                    edge_target,
                    receiver_normalizer_forward,
                )
                destination_state = layer(
                    destination_state,
                    edge_target,
                    edge_source,
                    receiver_normalizer_reverse,
                )
            else:
                origin_state = self._run_directional_layer(
                    self.origin_layers[layer_index],
                    origin_state,
                    edge_source,
                    edge_target,
                    edge_features,
                    receiver_normalizer_forward,
                )
                destination_state = self._run_directional_layer(
                    self.destination_layers[layer_index],
                    destination_state,
                    edge_target,
                    edge_source,
                    edge_features,
                    receiver_normalizer_reverse,
                )
            origin_states.append(origin_state)
            destination_states.append(destination_state)

        if self.config.variant == "last_layer_only":
            depth_weights = torch.zeros_like(self.depth_logits)
            depth_weights[-1] = 1.0
        elif self.config.variant == "propagation_deep":
            depth_weights = torch.ones_like(self.depth_logits)
        else:
            depth_weights = torch.softmax(self.depth_logits, dim=0)
        fused = origin_states[0].new_zeros(
            (origin_state.shape[0], origin_state.shape[1], self.hidden_dim * 4)
        )
        for weight, depth in zip(depth_weights, self.readout_depths):
            depth_features = self._combine_fields(
                origin_states[depth],
                destination_states[depth],
            )
            fused = fused + weight * depth_features

        pooled = self._pool_regions(fused, region_nodes)
        if self.propagation_only:
            return self.prediction_head(pooled).squeeze(-1)
        expanded_region_features = region_features.unsqueeze(0).expand(pooled.shape[0], -1, -1)
        prediction_input = torch.cat((pooled, expanded_region_features), dim=-1)
        return self.prediction_head(prediction_input).squeeze(-1)

    def _combine_fields(
        self,
        origin: torch.Tensor,
        destination: torch.Tensor,
    ) -> torch.Tensor:
        zeros = torch.zeros_like(origin)
        if self.config.variant == "origin_only":
            return torch.cat((origin, zeros, zeros, zeros), dim=-1)
        if self.config.variant == "destination_only":
            return torch.cat((zeros, destination, zeros, zeros), dim=-1)
        if self.config.variant == "no_interactions":
            return torch.cat((origin, destination, zeros, zeros), dim=-1)
        return torch.cat(
            (
                origin,
                destination,
                origin * destination,
                torch.abs(origin - destination),
            ),
            dim=-1,
        )

    def _run_directional_layer(
        self,
        layer: _DirectionalLayer,
        state: torch.Tensor,
        sender_index: torch.Tensor,
        receiver_index: torch.Tensor,
        edge_features: torch.Tensor,
        receiver_normalizer: torch.Tensor,
    ) -> torch.Tensor:
        if self.config.gradient_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(
                layer,
                state,
                sender_index,
                receiver_index,
                edge_features,
                receiver_normalizer,
                use_reentrant=False,
            )
        return layer(
            state,
            sender_index,
            receiver_index,
            edge_features,
            receiver_normalizer,
        )

    def _run_propagation_layer(
        self,
        layer: _PropagationOnlyLayer,
        state: torch.Tensor,
        sender_index: torch.Tensor,
        receiver_index: torch.Tensor,
        edge_features: torch.Tensor,
        receiver_normalizer: torch.Tensor,
    ) -> torch.Tensor:
        if self.config.gradient_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(
                layer,
                state,
                sender_index,
                receiver_index,
                edge_features,
                receiver_normalizer,
                use_reentrant=False,
            )
        return layer(
            state,
            sender_index,
            receiver_index,
            edge_features,
            receiver_normalizer,
        )

    @staticmethod
    def _pool_regions(node_states: torch.Tensor, region_nodes: torch.Tensor) -> torch.Tensor:
        values = node_states[:, region_nodes, :]
        return torch.cat((values.mean(dim=2), values.amax(dim=2)), dim=-1)


def _fixed_mean_diffusion(
    state: torch.Tensor,
    sender_index: torch.Tensor,
    receiver_index: torch.Tensor,
    receiver_normalizer: torch.Tensor,
) -> torch.Tensor:
    batch_size, node_count, hidden_dim = state.shape
    message = state[:, sender_index, :]
    batch_offsets = (
        torch.arange(batch_size, device=state.device, dtype=receiver_index.dtype)
        * node_count
    )
    flat_receivers = (receiver_index.unsqueeze(0) + batch_offsets[:, None]).reshape(-1)
    aggregate = message.new_zeros((batch_size * node_count, hidden_dim))
    aggregate.index_add_(0, flat_receivers, message.reshape(-1, hidden_dim))
    aggregate = aggregate.reshape(batch_size, node_count, hidden_dim)
    aggregate = aggregate / receiver_normalizer.to(
        dtype=aggregate.dtype
    ).unsqueeze(0).unsqueeze(-1)
    return 0.5 * (state + aggregate)


def build_edge_features(
    edge_length: torch.Tensor,
    edge_type: torch.Tensor,
    edge_type_count: int,
) -> torch.Tensor:
    if edge_length.ndim != 1 or edge_type.shape != edge_length.shape:
        raise ValueError("edge_length and edge_type must be equal one-dimensional tensors")
    if torch.any(edge_type < 0) or torch.any(edge_type >= edge_type_count):
        raise ValueError("edge_type contains an index outside edge_type_count")
    return torch.cat(
        (
            edge_length.unsqueeze(-1),
            functional.one_hot(edge_type.long(), num_classes=edge_type_count).to(
                dtype=edge_length.dtype
            ),
        ),
        dim=-1,
    )


def build_receiver_normalizers(
    receiver_index: torch.Tensor,
    node_count: int,
) -> torch.Tensor:
    if receiver_index.ndim != 1:
        raise ValueError("receiver_index must be one-dimensional")
    degree = torch.zeros(node_count, device=receiver_index.device, dtype=torch.float32)
    degree.index_add_(0, receiver_index, torch.ones_like(receiver_index, dtype=torch.float32))
    return degree.clamp_min(1.0)


def iter_slices(size: int, batch_size: int) -> Iterable[slice]:
    if size < 0 or batch_size <= 0:
        raise ValueError("size must be non-negative and batch_size must be positive")
    for start in range(0, size, batch_size):
        yield slice(start, min(start + batch_size, size))
