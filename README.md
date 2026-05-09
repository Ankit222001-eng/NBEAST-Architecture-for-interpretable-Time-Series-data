# N-BEATS vs LSTM Benchmark

A two-part deep learning study benchmarking **N-BEATS** (Neural Basis Expansion Analysis for Time Series) against a vanilla **LSTM** baseline, covering both univariate and multivariate forecasting scenarios. Hyperparameters are tuned end-to-end using **Bayesian Optimization (Optuna TPE)**.

---

## Overview

| Part | Notebook | Dataset | Task |
|------|----------|---------|------|
| 1 | `NBeats_LSTM_Benchmark_part1.ipynb` | Air Passengers (monthly) | Univariate forecasting |
| 2 | `Multivariate_Part2_.ipynb` | Beijing Air Quality (hourly) | Multivariate PM2.5 forecasting |

---

## Models

### N-BEATS Generic
A purely data-driven doubly residual stacking architecture. Each block learns a backcast (to remove what it explains from the input) and a forecast, with learnable basis expansion matrices — no domain assumptions.

### N-BEATS Interpretable
Replaces generic basis functions with structured ones — **polynomial (trend)** and **Fourier (seasonality)** bases — making the forecast decomposable into interpretable components out of the box.

### LSTM Baseline
A standard multi-layer LSTM with dropout and a fully connected output head, used as the benchmark to compare against the N-BEATS variants.

---

## Part 1 — Univariate (Air Passengers)

- **Dataset:** Classic Box & Jenkins airline passengers dataset (144 monthly observations)
- **Horizon:** 12 months
- **Input window:** 36 months (3× horizon)
- **Split:** 80% train / 10% val / 10% test
- **Loss:** sMAPE
- **Metrics:** sMAPE, MAE, RMSE, MAPE
- **Tuning:** Optuna TPE sampler, 30 trials per model

**Hyperparameter search space (N-BEATS):**

| Parameter | Range |
|-----------|-------|
| `n_stacks` | 5–40 |
| `n_blocks` | 1–3 |
| `n_layers` | 2–6 |
| `hidden_size` | {128, 256, 512} |
| `lr` | 1e-4 – 1e-2 (log) |
| `weight_decay` | 1e-6 – 1e-3 (log) |

**Hyperparameter search space (LSTM):**

| Parameter | Range |
|-----------|-------|
| `hidden_size` | {64, 128, 256} |
| `num_layers` | 1–3 |
| `dropout` | 0.0–0.4 |
| `lr` | 1e-4 – 1e-2 (log) |
| `weight_decay` | 1e-6 – 1e-3 (log) |

---

## Part 2 — Multivariate (Beijing PM2.5)

- **Dataset:** Beijing Multi-Site Air Quality (`PRSA_Data_Aotizhongxin_20130301-20170228.csv`) via KaggleHub
- **Features:** PM2.5, PM10, SO2, NO2, CO, O3, TEMP, PRES, DEWP, WSPM (10 features)
- **Target:** PM2.5 concentration
- **Horizon:** 8 hours ahead
- **Input window:** 96 hours (4 days)
- **Split:** 70% train / 15% val / 15% test
- **Preprocessing:** Forward/backward fill for NaNs, FFT denoising on PM2.5 (top 10% frequency components retained), z-score normalization per feature
- **Loss:** MSE
- **Metrics:** MAE, RMSE, MAPE, sMAPE
- **Tuning:** Optuna TPE sampler, 20 trials per model

The N-BEATS Interpretable model additionally outputs separated **trend** and **seasonality** components, enabling decomposition plots of the PM2.5 signal.

---

## Project Structure

```
├── NBeats_LSTM_Benchmark_part1.ipynb   # Part 1: univariate Air Passengers
├── Multivariate_Part2_.ipynb           # Part 2: multivariate Beijing PM2.5
└── README.md
```

**Generated outputs (saved during notebook runs):**

```
fft_denoising.png       # PM2.5 original vs FFT-denoised signal
mv_benchmark.png        # Training curves + forecast vs actual comparison
pm25_decomposition.png  # N-BEATS Interpretable trend/seasonality/residual decomposition
```

---

## Requirements

```bash
pip install torch optuna ucimlrepo kagglehub matplotlib pandas numpy
```

> GPU is auto-detected (`cuda` if available, otherwise `cpu`).

---

## How to Run

**Part 1 (Univariate):**
```bash
jupyter notebook NBeats_LSTM_Benchmark_part1.ipynb
```
Run all cells in order. Optuna will run 30 tuning trials per model, then train final models and print the benchmark table.

**Part 2 (Multivariate):**
```bash
jupyter notebook Multivariate_Part2_.ipynb
```
Requires a Kaggle account and `kagglehub` configured for dataset download. Run all cells in order. Optuna will run 20 tuning trials per model.

---

## Architecture Details

### N-BEATS Block

Each block takes the current residual input, passes it through a fully connected stack (ReLU activations), then projects to `theta_b` (backcast coefficients) and `theta_f` (forecast coefficients). These are expanded through a basis function to produce the backcast and forecast signals.

```
Input → FC Stack → θ_b → Basis → Backcast
                 → θ_f → Basis → Forecast
```

### Residual Stacking

Blocks are chained: each block's backcast is subtracted from its input before passing to the next block, allowing each subsequent block to focus on unexplained variance.

### Basis Functions

| Type | Backcast | Forecast |
|------|----------|----------|
| **Generic** | Learned linear projection | Learned linear projection |
| **Trend** | Polynomial (degree 2) | Polynomial (degree 2) |
| **Seasonality** | Fourier (cos + sin) | Fourier (cos + sin) |

---

## References

- Oreshkin et al. (2020) — [N-BEATS: Neural basis expansion analysis for interpretable time series forecasting](https://arxiv.org/abs/1905.10437)
- Akiba et al. (2019) — [Optuna: A Next-generation Hyperparameter Optimization Framework](https://arxiv.org/abs/1907.10902)
- Air Passengers dataset — Box & Jenkins (1976)
- Beijing Multi-Site Air Quality dataset — UCI / Kaggle
