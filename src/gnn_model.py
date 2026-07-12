"""用于预测压缩区域种子价值的轻量 GraphSAGE 模型。"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as functional


class MeanGraphSageLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.self_projection = nn.Linear(input_dim, output_dim)
        self.neighbor_projection = nn.Linear(input_dim, output_dim, bias=False)

    def forward(
        self,
        features: torch.Tensor,
        edge_source: torch.Tensor,
        edge_target: torch.Tensor,
        target_degree: torch.Tensor,
    ) -> torch.Tensor:
        aggregated = torch.zeros_like(features)
        aggregated.index_add_(0, edge_target, features[edge_source])
        aggregated = aggregated / target_degree.clamp_min(1.0).unsqueeze(1)
        return self.self_projection(features) + self.neighbor_projection(aggregated)


class SeedValueGraphSage(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.first_layer = MeanGraphSageLayer(input_dim, hidden_dim)
        self.second_layer = MeanGraphSageLayer(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 1)
        self.dropout = dropout

    def forward(
        self,
        features: torch.Tensor,
        edge_source: torch.Tensor,
        edge_target: torch.Tensor,
        target_degree: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.first_layer(features, edge_source, edge_target, target_degree)
        hidden = functional.relu(hidden)
        hidden = functional.dropout(hidden, self.dropout, self.training)
        hidden = self.second_layer(hidden, edge_source, edge_target, target_degree)
        hidden = functional.relu(hidden)
        hidden = functional.dropout(hidden, self.dropout, self.training)
        return torch.sigmoid(self.output_layer(hidden).squeeze(1))


class SeedValueMlp(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.first_layer = nn.Linear(input_dim, hidden_dim)
        self.second_layer = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 1)
        self.dropout = dropout

    def forward(
        self,
        features: torch.Tensor,
        edge_source: torch.Tensor,
        edge_target: torch.Tensor,
        target_degree: torch.Tensor,
    ) -> torch.Tensor:
        del edge_source, edge_target, target_degree
        hidden = functional.relu(self.first_layer(features))
        hidden = functional.dropout(hidden, self.dropout, self.training)
        hidden = functional.relu(self.second_layer(hidden))
        hidden = functional.dropout(hidden, self.dropout, self.training)
        return torch.sigmoid(self.output_layer(hidden).squeeze(1))
