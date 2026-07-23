# Perioperative uncertainty transport audit

This repository accompanies the manuscript **“When Nominal Coverage Does Not Ensure Useful Triage: A Temporal and Hospital Information-System Shift Audit of Uncertainty-Aware Postoperative Pain Prediction.”**

It provides a minimal, privacy-preserving reference implementation and aggregate data sufficient to inspect the reported model and uncertainty formulas and to redraw the three main figures. It is not a release of patient-level data or a deployable clinical prediction model.

## Repository contents

- `reference_methods.py` — multi-task heteroscedastic neural-network architecture, Gaussian negative log-likelihood, normalized split-conformal intervals, performance metrics, and risk-retention calculations.
- `model_specification.json` — frozen public model, preprocessing, split, calibration, and evaluation specifications without local paths or internal field names.
- `main_figure_data.json` — non-identifiable aggregate values underlying Figures 1–3.
- `plot_main_figures.py` — standalone script that redraws Figures 1–3 from `main_figure_data.json`.
- `requirements.txt` — tested software versions.
- `LICENSE.md` — dual-license terms.

## Data availability

The aggregate, non-identifiable values underlying the main figures are provided in `main_figure_data.json` under the Creative Commons Attribution 4.0 International license.

Individual-level data are not included. They were derived from institutional perioperative information systems and postoperative assessment records and cannot be publicly released because of privacy, ethics, and institutional-governance restrictions. The repository contains no hospitalization identifiers, row-level outcomes, row-level predictions, split identifiers, clinical free text, fitted preprocessing values, or model checkpoints.

## Reproducing the main figures

Python 3.12 was used for the public reference package.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python plot_main_figures.py
```

The script writes PDF and PNG files to a local `figures/` directory. No network access or clinical data are required.

## Method reference

The reported model used a shared multi-task multilayer perceptron with task-specific heads for resting and activity pain. Each head returned a conditional mean and log variance. The main intervals used normalized split conformal calibration:

```text
score_i = |y_i - mu_i| / sigma_i
interval(x) = mu(x) +/- q_hat * sigma(x)
```

Uncertainty-guided deferral ranked records by deterministic predicted standard deviation. At each retention fraction, the least-uncertain records were retained. The plotted quantity was:

```text
delta_MAE(retention) = MAE(retained records) - MAE(all test records)
```

Negative values indicate lower error after deferral; positive values indicate higher error.

## Important caveats

- This was a retrospective, single-centre internal temporal transport evaluation, not an external or prospective validation.
- The audit did not have a fixed real-time prediction trigger. Some formula-derived predictors could overlap early postoperative outcome assessment.
- Temporal R-squared was negative for both main outcomes. The fitted model is not supported for deployment.
- Near-nominal marginal coverage did not establish useful case-level uncertainty ranking or deferral.
- Any future operational model would require redevelopment or updating, prespecified prediction timing, multisite external validation, and prospective workflow evaluation.

## Licenses

Python source code and dependency specifications are released under the MIT License. Aggregate figure data and the non-code model specification are released under CC BY 4.0. See `LICENSE.md`.

## Citation

A complete citation will be added after publication. Until then, please cite the accompanying manuscript by title and repository URL.

