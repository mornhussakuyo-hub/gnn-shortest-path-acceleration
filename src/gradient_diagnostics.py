"""Numerically stable gradient diagnostics for deep demand propagation models."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


def finite_tensor_statistics(
    tensor: torch.Tensor,
    *,
    include_fp64_l2: bool = True,
) -> dict[str, object]:
    values = tensor.detach()
    finite_mask = torch.isfinite(values)
    finite_values = torch.where(finite_mask, values, torch.zeros_like(values))
    finite_abs = finite_values.abs()
    finite_count = int(finite_mask.sum().item())
    element_count = values.numel()
    maximum_absolute_value = (
        float(finite_abs.max().item()) if element_count else 0.0
    )
    statistics: dict[str, object] = {
        "dtype": str(values.dtype).removeprefix("torch."),
        "shape": list(values.shape),
        "element_count": element_count,
        "finite_count": finite_count,
        "nonfinite_count": element_count - finite_count,
        "nan_count": int(torch.isnan(values).sum().item()),
        "positive_inf_count": int(torch.isposinf(values).sum().item()),
        "negative_inf_count": int(torch.isneginf(values).sum().item()),
        "maximum_absolute_finite_value": maximum_absolute_value,
        "all_finite": finite_count == element_count,
    }
    if include_fp64_l2:
        squared_norm = finite_values.double().square().sum()
        statistics["finite_l2_norm_fp64"] = float(torch.sqrt(squared_norm).item())
    return statistics


@dataclass(slots=True)
class _DeviceTensorAggregate:
    call_count: int
    element_count: int
    finite_count: torch.Tensor
    nan_count: torch.Tensor
    positive_inf_count: torch.Tensor
    negative_inf_count: torch.Tensor
    maximum_absolute_finite_value: torch.Tensor


class DeviceTensorAccumulator:
    """Aggregate hook tensors on-device and synchronize only when finalized."""

    def __init__(self) -> None:
        self._aggregates: dict[str, _DeviceTensorAggregate] = {}

    def add(self, name: str, tensor: torch.Tensor) -> None:
        values = tensor.detach()
        finite_mask = torch.isfinite(values)
        maximum_absolute_value = torch.where(
            finite_mask,
            values.abs(),
            torch.zeros((), device=values.device, dtype=values.dtype),
        ).max()
        finite_count = finite_mask.sum(dtype=torch.int64)
        nan_count = torch.isnan(values).sum(dtype=torch.int64)
        positive_inf_count = torch.isposinf(values).sum(dtype=torch.int64)
        negative_inf_count = torch.isneginf(values).sum(dtype=torch.int64)
        aggregate = self._aggregates.get(name)
        if aggregate is None:
            self._aggregates[name] = _DeviceTensorAggregate(
                call_count=1,
                element_count=values.numel(),
                finite_count=finite_count,
                nan_count=nan_count,
                positive_inf_count=positive_inf_count,
                negative_inf_count=negative_inf_count,
                maximum_absolute_finite_value=maximum_absolute_value,
            )
            return
        aggregate.call_count += 1
        aggregate.element_count += values.numel()
        aggregate.finite_count = aggregate.finite_count + finite_count
        aggregate.nan_count = aggregate.nan_count + nan_count
        aggregate.positive_inf_count = (
            aggregate.positive_inf_count + positive_inf_count
        )
        aggregate.negative_inf_count = (
            aggregate.negative_inf_count + negative_inf_count
        )
        aggregate.maximum_absolute_finite_value = torch.maximum(
            aggregate.maximum_absolute_finite_value,
            maximum_absolute_value,
        )

    def finalize(self) -> dict[str, dict[str, object]]:
        result: dict[str, dict[str, object]] = {}
        for name, aggregate in sorted(self._aggregates.items()):
            finite_count = int(aggregate.finite_count.item())
            nonfinite_count = aggregate.element_count - finite_count
            result[name] = {
                "call_count": aggregate.call_count,
                "element_count": aggregate.element_count,
                "finite_count": finite_count,
                "nonfinite_count": nonfinite_count,
                "nan_count": int(aggregate.nan_count.item()),
                "positive_inf_count": int(aggregate.positive_inf_count.item()),
                "negative_inf_count": int(aggregate.negative_inf_count.item()),
                "maximum_absolute_finite_value": float(
                    aggregate.maximum_absolute_finite_value.item()
                ),
                "all_finite": nonfinite_count == 0,
            }
        return result


def parameter_gradient_report(
    model: torch.nn.Module,
) -> dict[str, object]:
    parameter_rows: list[dict[str, object]] = []
    first_parameter = next(model.parameters())
    finite_squared_norm_fp64 = torch.zeros(
        (), device=first_parameter.device, dtype=torch.float64
    )
    raw_squared_norm_fp32 = torch.zeros(
        (), device=first_parameter.device, dtype=torch.float32
    )
    total_element_count = 0
    total_nonfinite_count = 0
    maximum_absolute_value = 0.0
    missing_gradient_names: list[str] = []
    nonfinite_gradient_names: list[str] = []

    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        if gradient is None:
            missing_gradient_names.append(name)
            parameter_rows.append(
                {
                    "name": name,
                    "requires_grad": parameter.requires_grad,
                    "gradient_missing": True,
                    "shape": list(parameter.shape),
                    "element_count": parameter.numel(),
                }
            )
            continue
        statistics = finite_tensor_statistics(gradient, include_fp64_l2=True)
        statistics.update(
            {
                "name": name,
                "requires_grad": parameter.requires_grad,
                "gradient_missing": False,
            }
        )
        parameter_rows.append(statistics)
        total_element_count += gradient.numel()
        nonfinite_count = int(statistics["nonfinite_count"])
        total_nonfinite_count += nonfinite_count
        if nonfinite_count:
            nonfinite_gradient_names.append(name)
        maximum_absolute_value = max(
            maximum_absolute_value,
            float(statistics["maximum_absolute_finite_value"]),
        )
        finite_l2 = float(statistics["finite_l2_norm_fp64"])
        finite_squared_norm_fp64 += finite_l2 * finite_l2
        raw_squared_norm_fp32 += gradient.detach().float().square().sum()

    finite_l2_norm_fp64 = float(torch.sqrt(finite_squared_norm_fp64).item())
    raw_l2_norm_fp32 = float(torch.sqrt(raw_squared_norm_fp32).item())
    return {
        "summary": {
            "parameter_tensor_count": len(parameter_rows),
            "gradient_tensor_count": len(parameter_rows) - len(missing_gradient_names),
            "gradient_element_count": total_element_count,
            "nonfinite_gradient_element_count": total_nonfinite_count,
            "all_gradient_elements_finite": total_nonfinite_count == 0,
            "maximum_absolute_finite_gradient": maximum_absolute_value,
            "finite_l2_norm_fp64": finite_l2_norm_fp64,
            "raw_l2_norm_fp32": raw_l2_norm_fp32,
            "raw_fp32_norm_is_finite": math.isfinite(raw_l2_norm_fp32),
            "missing_gradient_names": missing_gradient_names,
            "nonfinite_gradient_names": nonfinite_gradient_names,
        },
        "parameters": parameter_rows,
    }
