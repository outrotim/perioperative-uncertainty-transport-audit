"""Privacy-preserving reference implementation for the accompanying manuscript.

This module exposes the model and evaluation formulas without data-loading
logic, institutional field names, fitted preprocessing values, or model
weights. It is intended for method inspection and reuse, not clinical use.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn


class MTLHeteroscedasticMLP(nn.Module):
    """Shared-trunk regression with task-specific mean/log-variance heads."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Iterable[int] = (256, 128),
        dropout: float = 0.15,
        n_tasks: int = 2,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = int(input_dim)
        for hidden in hidden_dims:
            layers.extend(
                [
                    nn.Linear(width, int(hidden)),
                    nn.ReLU(),
                    nn.Dropout(p=float(dropout)),
                ]
            )
            width = int(hidden)
        self.trunk = nn.Sequential(*layers)
        self.heads = nn.ModuleList([nn.Linear(width, 2) for _ in range(int(n_tasks))])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk(x)
        outputs = [head(hidden) for head in self.heads]
        means = torch.cat([output[:, 0:1] for output in outputs], dim=1)
        log_variances = torch.cat([output[:, 1:2] for output in outputs], dim=1)
        return means, torch.clamp(log_variances, min=-12.0, max=8.0)


def gaussian_nll(
    targets: torch.Tensor,
    means: torch.Tensor,
    log_variances: torch.Tensor,
) -> torch.Tensor:
    """Mean factorized Gaussian negative log-likelihood across rows and tasks."""
    inverse_variances = torch.exp(-log_variances)
    losses = 0.5 * (
        inverse_variances * (targets - means) ** 2 + log_variances
    )
    return losses.mean()


@torch.no_grad()
def deterministic_prediction(
    model: nn.Module,
    features: np.ndarray,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic task means and standard deviations."""
    model = model.to(device)
    model.eval()
    tensor = torch.as_tensor(features, dtype=torch.float32, device=device)
    means, log_variances = model(tensor)
    standard_deviations = torch.sqrt(torch.exp(log_variances))
    return (
        means.cpu().numpy().astype(np.float32),
        standard_deviations.cpu().numpy().astype(np.float32),
    )


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample split-conformal quantile using the higher order statistic."""
    clean = np.asarray(scores, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return float("nan")
    quantile_level = np.ceil((clean.size + 1) * (1 - float(alpha))) / clean.size
    quantile_level = float(np.clip(quantile_level, 0, 1))
    ordered = np.sort(clean)
    index = int(np.ceil(quantile_level * clean.size) - 1)
    return float(ordered[np.clip(index, 0, clean.size - 1)])


def normalized_split_conformal(
    calibration_targets: np.ndarray,
    calibration_means: np.ndarray,
    calibration_sigmas: np.ndarray,
    test_means: np.ndarray,
    test_sigmas: np.ndarray,
    alpha: float = 0.1,
    sigma_floor: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct per-task normalized split-conformal prediction intervals."""
    y_cal = np.asarray(calibration_targets, dtype=float)
    mu_cal = np.asarray(calibration_means, dtype=float)
    sigma_cal = np.maximum(np.asarray(calibration_sigmas, dtype=float), sigma_floor)
    mu_test = np.asarray(test_means, dtype=float)
    sigma_test = np.maximum(np.asarray(test_sigmas, dtype=float), sigma_floor)
    if not (y_cal.shape == mu_cal.shape == sigma_cal.shape):
        raise ValueError("Calibration arrays must have identical shapes.")
    if mu_test.shape != sigma_test.shape:
        raise ValueError("Test mean and sigma arrays must have identical shapes.")

    scores = np.abs(y_cal - mu_cal) / sigma_cal
    q_hats = np.asarray(
        [conformal_quantile(scores[:, task], alpha) for task in range(scores.shape[1])]
    )
    lower = mu_test - sigma_test * q_hats
    upper = mu_test + sigma_test * q_hats
    return lower, upper, q_hats


def regression_metrics(targets: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    """Return MAE, RMSE, and R-squared."""
    y = np.asarray(targets, dtype=float)
    pred = np.asarray(predictions, dtype=float)
    errors = pred - y
    residual_sum = float(np.square(errors).sum())
    total_sum = float(np.square(y - y.mean()).sum())
    return {
        "mae": float(np.abs(errors).mean()),
        "rmse": float(np.sqrt(np.square(errors).mean())),
        "r2": float(1 - residual_sum / total_sum) if total_sum > 0 else float("nan"),
    }


def interval_metrics(
    targets: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> dict[str, float]:
    """Return empirical coverage and interval-width summaries."""
    y = np.asarray(targets, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    widths = hi - lo
    return {
        "coverage": float(np.mean((y >= lo) & (y <= hi))),
        "mean_width": float(widths.mean()),
        "median_width": float(np.median(widths)),
    }


def uncertainty_error_spearman(
    targets: np.ndarray,
    predictions: np.ndarray,
    uncertainty: np.ndarray,
) -> float:
    """Spearman correlation between predicted uncertainty and absolute error."""
    errors = np.abs(np.asarray(targets, dtype=float) - np.asarray(predictions, dtype=float))
    score = np.asarray(uncertainty, dtype=float)
    score_ranks = pd.Series(score).rank(method="average").to_numpy(float)
    error_ranks = pd.Series(errors).rank(method="average").to_numpy(float)
    return float(np.corrcoef(score_ranks, error_ranks)[0, 1])


def risk_retention_curve(
    targets: np.ndarray,
    predictions: np.ndarray,
    uncertainty: np.ndarray,
    retention_fractions: np.ndarray,
) -> np.ndarray:
    """MAE(retained least-uncertain rows) minus MAE(all rows)."""
    errors = np.abs(np.asarray(targets, dtype=float) - np.asarray(predictions, dtype=float))
    score = np.asarray(uncertainty, dtype=float)
    fractions = np.asarray(retention_fractions, dtype=float)
    if not np.all((fractions > 0) & (fractions <= 1)):
        raise ValueError("Retention fractions must lie in (0, 1].")
    order = np.argsort(score, kind="mergesort")
    baseline = float(errors.mean())
    changes = []
    for fraction in fractions:
        keep_count = max(1, int(np.floor(fraction * errors.size)))
        changes.append(float(errors[order[:keep_count]].mean() - baseline))
    return np.asarray(changes)
