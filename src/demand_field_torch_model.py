"""使用 CUDA 训练的第二版无传播区域 MLP。"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as functional

from src.demand_field_model import MLPConfig, regression_metrics


TORCH_MLP_MODEL_SCHEMA = "aic.gnn_v2.region_mlp.torch_cuda.v1"


def require_cuda(device_name: str = "cuda") -> torch.device:
    device = torch.device(device_name)
    if device.type != "cuda":
        raise ValueError("the formal MLP baseline requires a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but torch.cuda.is_available() is false")
    torch.empty(1, device=device)
    return device


def cuda_environment(device: torch.device) -> dict[str, object]:
    index = device.index if device.index is not None else torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    return {
        "framework": "pytorch",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device_type": "cuda",
        "device_index": index,
        "device_name": properties.name,
        "device_total_memory_bytes": properties.total_memory,
    }


class _RegionMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        dimensions = (input_dim, *hidden_dims, 1)
        for layer_index, (input_width, output_width) in enumerate(
            zip(dimensions, dimensions[1:])
        ):
            linear = nn.Linear(input_width, output_width)
            nn.init.kaiming_normal_(linear.weight, nonlinearity="relu")
            nn.init.zeros_(linear.bias)
            layers.append(linear)
            if layer_index < len(dimensions) - 2:
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


class TorchCudaMLPRegressor:
    def __init__(
        self,
        input_dim: int,
        config: MLPConfig,
        seed: int,
        device: torch.device,
    ) -> None:
        config.validate()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if device.type != "cuda":
            raise ValueError("TorchCudaMLPRegressor requires CUDA")
        self.input_dim = input_dim
        self.config = config
        self.seed = seed
        self.device = device
        self.x_mean = np.zeros(input_dim, dtype=np.float32)
        self.x_scale = np.ones(input_dim, dtype=np.float32)
        self.y_mean = 0.0
        self.y_scale = 1.0
        self.best_epoch = 0
        self.training_seconds = 0.0
        self.peak_allocated_bytes = 0
        self.peak_reserved_bytes = 0
        self._set_seed()
        self.model = _RegionMLP(input_dim, config.hidden_dims).to(device)

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_validation: np.ndarray,
        y_validation: np.ndarray,
    ) -> list[dict[str, float]]:
        if x_train.ndim != 2 or x_train.shape[1] != self.input_dim:
            raise ValueError("training feature shape does not match model input")
        if not len(x_train) or not len(x_validation):
            raise ValueError("training and validation sets must be non-empty")
        self._fit_scalers(x_train, y_train)
        train_x = self._tensor(self._scale_x(x_train))
        train_y = self._tensor(self._scale_y(y_train))
        validation_x = self._tensor(self._scale_x(x_validation))
        validation_y = self._tensor(self._scale_y(y_validation))
        optimizer = torch.optim.Adam(
            [
                {
                    "params": [
                        parameter
                        for name, parameter in self.model.named_parameters()
                        if name.endswith("weight")
                    ],
                    "weight_decay": self.config.weight_decay,
                },
                {
                    "params": [
                        parameter
                        for name, parameter in self.model.named_parameters()
                        if name.endswith("bias")
                    ],
                    "weight_decay": 0.0,
                },
            ],
            lr=self.config.learning_rate,
        )
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.seed)
        best_validation_loss = float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        epochs_without_improvement = 0
        history: list[dict[str, float]] = []
        torch.cuda.reset_peak_memory_stats(self.device)
        started = time.perf_counter()

        for epoch in range(1, self.config.max_epochs + 1):
            self.model.train()
            order = torch.randperm(len(train_x), generator=generator, device=self.device)
            batch_losses: list[float] = []
            for start in range(0, len(order), self.config.batch_size):
                batch_indices = order[start : start + self.config.batch_size]
                batch_x = train_x[batch_indices]
                batch_y = train_y[batch_indices]
                optimizer.zero_grad(set_to_none=True)
                prediction = self.model(batch_x)
                loss = functional.huber_loss(
                    prediction,
                    batch_y,
                    reduction="mean",
                    delta=self.config.huber_delta,
                )
                loss = loss + self.config.rank_weight * _sampled_pairwise_loss(
                    prediction,
                    batch_y,
                    generator,
                )
                loss.backward()
                optimizer.step()
                batch_losses.append(float(loss.detach().item()))

            self.model.eval()
            with torch.no_grad():
                validation_prediction = self.model(validation_x)
                validation_loss = _evaluation_loss(
                    validation_prediction,
                    validation_y,
                    self.config,
                )
                train_prediction = self.model(train_x)
                validation_unscaled = self._unscale_y(
                    validation_prediction.detach().cpu().numpy()
                )
                history.append(
                    {
                        "epoch": float(epoch),
                        "train_batch_loss": float(np.mean(batch_losses)),
                        "train_huber": float(
                            functional.huber_loss(
                                train_prediction,
                                train_y,
                                reduction="mean",
                                delta=self.config.huber_delta,
                            ).item()
                        ),
                        "validation_loss": validation_loss,
                        "validation_spearman": regression_metrics(
                            validation_unscaled,
                            y_validation.astype(np.float64, copy=False),
                        )["spearman"],
                    }
                )
            if validation_loss < best_validation_loss - self.config.min_improvement:
                best_validation_loss = validation_loss
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in self.model.state_dict().items()
                }
                self.best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if epochs_without_improvement >= self.config.patience:
                break

        torch.cuda.synchronize(self.device)
        self.training_seconds = time.perf_counter() - started
        self.peak_allocated_bytes = torch.cuda.max_memory_allocated(self.device)
        self.peak_reserved_bytes = torch.cuda.max_memory_reserved(self.device)
        if best_state is None:
            raise RuntimeError("MLP training did not produce a finite validation checkpoint")
        self.model.load_state_dict(best_state)
        self.model.to(self.device)
        return history

    def predict(self, features: np.ndarray) -> np.ndarray:
        if features.ndim != 2 or features.shape[1] != self.input_dim:
            raise ValueError("prediction feature shape does not match model input")
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(self._tensor(self._scale_x(features)))
        return self._unscale_y(prediction.cpu().numpy()).astype(np.float64)

    def save(self, weights_path: Path, manifest_path: Path, metadata: dict) -> dict:
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict": {
                name: value.detach().cpu() for name, value in self.model.state_dict().items()
            },
            "x_mean": torch.from_numpy(self.x_mean),
            "x_scale": torch.from_numpy(self.x_scale),
            "y_mean": self.y_mean,
            "y_scale": self.y_scale,
        }
        torch.save(payload, weights_path)
        manifest = {
            "schema": TORCH_MLP_MODEL_SCHEMA,
            "input_dim": self.input_dim,
            "seed": self.seed,
            "best_epoch": self.best_epoch,
            "config": {
                **asdict(self.config),
                "hidden_dims": list(self.config.hidden_dims),
            },
            "weights_file": str(weights_path.resolve()),
            "weights_sha256": _sha256(weights_path),
            "execution": {
                **cuda_environment(self.device),
                "training_seconds": self.training_seconds,
                "peak_allocated_bytes": self.peak_allocated_bytes,
                "peak_reserved_bytes": self.peak_reserved_bytes,
            },
            **metadata,
        }
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return manifest

    @classmethod
    def load(
        cls,
        weights_path: Path,
        manifest_path: Path,
        device: torch.device,
    ) -> "TorchCudaMLPRegressor":
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != TORCH_MLP_MODEL_SCHEMA:
            raise ValueError(f"unsupported model schema: {manifest.get('schema')}")
        if manifest.get("weights_sha256") != _sha256(weights_path):
            raise ValueError("MLP weights digest mismatch")
        raw_config = dict(manifest["config"])
        raw_config["hidden_dims"] = tuple(raw_config["hidden_dims"])
        model = cls(
            int(manifest["input_dim"]),
            MLPConfig(**raw_config),
            int(manifest["seed"]),
            device,
        )
        payload = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.model.load_state_dict(payload["state_dict"])
        model.model.to(device)
        model.x_mean = payload["x_mean"].numpy()
        model.x_scale = payload["x_scale"].numpy()
        model.y_mean = float(payload["y_mean"])
        model.y_scale = float(payload["y_scale"])
        model.best_epoch = int(manifest["best_epoch"])
        return model

    def _set_seed(self) -> None:
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

    def _fit_scalers(self, features: np.ndarray, labels: np.ndarray) -> None:
        self.x_mean = features.mean(axis=0, dtype=np.float64).astype(np.float32)
        scale = features.std(axis=0, dtype=np.float64).astype(np.float32)
        self.x_scale = np.where(scale > 1e-8, scale, 1.0).astype(np.float32)
        self.y_mean = float(np.mean(labels, dtype=np.float64))
        label_scale = float(np.std(labels, dtype=np.float64))
        self.y_scale = label_scale if label_scale > 1e-8 else 1.0

    def _scale_x(self, features: np.ndarray) -> np.ndarray:
        return (
            features.astype(np.float32, copy=False) - self.x_mean
        ) / self.x_scale

    def _scale_y(self, labels: np.ndarray) -> np.ndarray:
        return (
            labels.astype(np.float32, copy=False) - self.y_mean
        ) / self.y_scale

    def _unscale_y(self, labels: np.ndarray) -> np.ndarray:
        return labels.astype(np.float64, copy=False) * self.y_scale + self.y_mean

    def _tensor(self, values: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(values, dtype=torch.float32, device=self.device)


def _sampled_pairwise_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    pair_count = max(1, len(prediction))
    left = torch.randint(
        len(prediction), (pair_count,), generator=generator, device=prediction.device
    )
    right = torch.randint(
        len(prediction), (pair_count,), generator=generator, device=prediction.device
    )
    valid = target[left] != target[right]
    left = left[valid]
    right = right[valid]
    if not len(left):
        return prediction.sum() * 0.0
    sign = torch.sign(target[left] - target[right])
    margin = sign * (prediction[left] - prediction[right])
    return functional.softplus(-margin).mean()


def _evaluation_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    config: MLPConfig,
) -> float:
    huber = functional.huber_loss(
        prediction,
        target,
        reduction="mean",
        delta=config.huber_delta,
    )
    left, right = torch.triu_indices(
        len(target), len(target), offset=1, device=prediction.device
    )
    valid = target[left] != target[right]
    left = left[valid]
    right = right[valid]
    if len(left):
        sign = torch.sign(target[left] - target[right])
        margin = sign * (prediction[left] - prediction[right])
        rank = functional.softplus(-margin).mean()
    else:
        rank = prediction.sum() * 0.0
    return float((huber + config.rank_weight * rank).item())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
