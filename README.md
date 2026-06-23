# Advanced Portfolio Autoencoders: True Holdout Research

This repository contains a research notebook for testing whether an **Autoencoder → Black-Litterman** pipeline can generate portfolio views that are stable enough to use in equity portfolio construction.

The project combines classical quantitative finance, machine learning, covariance denoising, and stress testing. The goal is not to produce a deployable trading strategy, but to evaluate whether the modeling pipeline survives out-of-sample and synthetic regime tests.

## Project Summary

Most portfolio backtests look better in notebooks than they do in live markets.

This project tries to avoid that by focusing on robustness rather than just performance. It extends a traditional equity portfolio construction notebook into a full research pipeline using:

* Autoencoder latent-factor modeling
* Black-Litterman portfolio optimization
* Ledoit-Wolf shrinkage
* Random Matrix Theory covariance denoising
* PCA factor modeling
* Synthetic stress testing
* Bootstrap confidence intervals
* True holdout evaluation

The main conclusion:

**The model is promising, but not ready to deploy.**

There is some evidence of signal, but not enough robustness to trust the strategy with real capital.

## Research Question

Can autoencoder-derived latent factors be converted into Black-Litterman views that improve portfolio performance versus a classical Ledoit-Wolf baseline?

More specifically:

* Do autoencoder factors contain useful cross-sectional information?
* Are the resulting Black-Litterman views stable across regimes?
* Does the model outperform a simpler baseline out-of-sample?
* Is the improvement statistically reliable?
* Does the model remain stable under synthetic stress tests?

## Asset Universe

The notebook uses 18 US large-cap equities across several sectors and themes:

* AAPL
* MSFT
* NVDA
* AMZN
* GOOGL
* META
* AVGO
* AMD
* COST
* JPM
* UNH
* LLY
* V
* MA
* CRWD
* PANW
* TSLA
* NEE

Benchmarks:

* SPY
* QQQ

The full complete panel begins in 2019 because some assets, such as CRWD, do not have earlier full public trading history.

## Important Caveat: Universe Selection Bias

The 18-stock universe was selected in 2026 with knowledge of which companies had performed well historically.

This introduces a major source of hindsight bias.

Even if the walk-forward training procedure avoids looking ahead inside the pipeline, the universe itself is still pre-selected using future information.

Because of that, any apparent alpha should be interpreted carefully.

This notebook is a research artifact, not an allocation engine.

## Methodology

The research pipeline includes the following components.

### 1. Data Collection

Daily and monthly adjusted price data is collected using `yfinance`.

The notebook computes:

* Daily returns
* Monthly returns
* Benchmark returns
* Full-panel return matrices
* Normalized price paths

### 2. Classical Portfolio Models

Several traditional portfolio construction methods are implemented:

* Minimum variance
* Mean-variance optimization
* Max Sharpe optimization
* Risk parity
* Hierarchical Risk Parity
* CVaR optimization
* Robust mean-variance optimization
* Black-Litterman optimization

All portfolios are long-only and fully invested, with position caps.

### 3. Covariance Estimation

The notebook compares multiple covariance estimators:

* Sample covariance
* Ledoit-Wolf shrinkage
* PCA factor covariance
* Autoencoder factor covariance
* Denoising autoencoder covariance
* Random Matrix Theory denoised covariance

The purpose is to test whether more advanced risk models produce more stable optimization inputs.

### 4. Autoencoder Modeling

The notebook trains three neural network models:

* Basic autoencoder
* Denoising autoencoder
* Variational autoencoder

Each model compresses the cross-section of asset returns into a lower-dimensional latent representation.

The autoencoder is not used directly as a trading model. Instead, its latent factors are used to study:

* Hidden factor structure
* Asset clustering
* Stress regimes
* Reconstruction error
* Synthetic return scenarios

### 5. PCA Benchmark

PCA is used as a serious baseline.

This is important because an autoencoder with only 18 assets may not have enough cross-sectional data to justify its added complexity.

In the notebook, PCA performs very competitively and in some tests outperforms the autoencoder on reconstruction error.

### 6. Autoencoder → Black-Litterman Views

The core experimental idea is to convert autoencoder latent information into Black-Litterman views.

The pipeline attempts to use latent factor structure to generate systematic views, which are then fed into Black-Litterman allocation.

The goal is to test whether neural latent factors can improve portfolio construction in a disciplined and mechanically defined way.

### 7. Random Matrix Theory Cleaning

Random Matrix Theory is used to denoise the empirical correlation matrix.

The idea is to separate likely signal eigenvalues from noisy eigenvalues, then construct a cleaner covariance matrix for optimization.

This helps reduce false precision in portfolio weights.

### 8. Stress Testing

The notebook uses synthetic alternate histories to test whether the AE → Black-Litterman view engine is stable across different market regimes.

This is one of the most important parts of the project.

A model that performs well in one historical sample but breaks under plausible synthetic regimes is not robust enough to deploy.

## Key Findings

The model shows some promising signs:

* Positive median Sharpe lift versus the Ledoit-Wolf baseline
* 80% probability of outperforming in the tested setup
* Useful latent-space diagnostics
* Clear regime and stress sensitivity signals

However, the evidence is not strong enough:

* Bootstrap confidence intervals cross zero
* Portfolio weights are unstable under most stress regimes
* The view engine is regime-sensitive
* The asset universe is too small for deep learning to clearly dominate PCA
* Universe pre-selection creates unavoidable hindsight bias

## Main Conclusion

The model is **not ready to deploy**.

There may be signal, but the distribution of outcomes is too wide to trust with capital.

The project is best understood as a research pipeline and diagnostic framework, not a production trading system.

## Lessons Learned

### Autoencoders Need Scale

With only 18 assets, autoencoders do not have enough cross-sectional depth to clearly justify their complexity.

For this type of model, hundreds of assets would likely be more appropriate.

### PCA Is a Strong Baseline

At this universe size, PCA remains a serious competitor.

Any neural model needs to beat PCA not only in-sample, but also out-of-sample and under stress.

### Universe Selection Matters

Walk-forward training can reduce look-ahead bias inside the model pipeline, but it cannot fix hindsight bias in the original asset selection.

A proper production-grade study should use a full index universe with no pre-selection.

### Stress Testing Is Essential

Synthetic stress testing is useful because it reveals where a model breaks.

It does not just show performance.

It shows fragility.

## Repository Structure

```text
.
├── advanced_portfolio_autoencoders_true_holdout.ipynb
├── README.md
└── requirements.txt
```

Suggested optional structure:

```text
.
├── notebooks/
│   └── advanced_portfolio_autoencoders_true_holdout.ipynb
├── src/
│   ├── data.py
│   ├── models.py
│   ├── optimization.py
│   ├── risk_models.py
│   └── evaluation.py
├── outputs/
│   ├── figures/
│   └── tables/
├── requirements.txt
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/ae-black-litterman-portfolio-research.git
cd ae-black-litterman-portfolio-research
```

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If running in Google Colab, install the missing packages inside the notebook:

```python
!pip install -q yfinance cvxpy
```

## Main Dependencies

The notebook uses:

```text
numpy
pandas
matplotlib
scipy
scikit-learn
torch
cvxpy
yfinance
```

## How to Run

Open the notebook:

```bash
jupyter notebook advanced_portfolio_autoencoders_true_holdout.ipynb
```

Then run all cells from top to bottom.

The notebook downloads market data, trains the models, runs portfolio optimization tests, and generates diagnostic charts.

## Results Are Not Financial Advice

This repository is for educational and research purposes only.

It is not financial advice, investment advice, or a recommendation to buy or sell any security.

The results are affected by:

* Universe selection bias
* Limited asset count
* Transaction cost assumptions
* Model instability
* Regime sensitivity
* Limited out-of-sample evidence

Use this project as a research framework, not as a trading system.

## Next Steps

Future improvements should include:

* Full-index universe instead of 18 selected stocks
* No pre-selection based on known winners
* Larger cross-sectional dataset
* Cleaner train-validation-test split
* Longer true holdout period
* More realistic transaction cost modeling
* Comparison against additional factor models
* Better regime-aware view calibration
* Robustness tests across global equity universes

## Final Takeaway

The AE → Black-Litterman pipeline is an interesting research direction.

But the current evidence is insufficient.

The model shows potential, but not enough stability.

For now:

**Promising research. Not a deployable strategy.**
