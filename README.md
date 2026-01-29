# Deep BSDE Solver for Option Pricing

## Overview
This project implements a **Deep Learning-based BSDE Solver** (based on Han et al., 2018) to solve high-dimensional partial differential equations (PDEs) for option pricing and dynamic hedging.

The primary goal is to overcome the "Curse of Dimensionality" in traditional finance models while ensuring mathematical robustness.

## Key Features
- **Deep BSDE Solver**: Uses PyTorch to approximate the unknown gradient term ($\nabla u$, i.e., Delta) directly via neural networks.
- **Malliavin Calculus Validation**: Implements a standard Monte Carlo simulation with Malliavin weights as a benchmark to validate the AI model's accuracy.
- **Delta Smile Visualization**: Successfully captures the non-linear hedging strategy, especially in Deep OTM (Out-of-The-Money) regions where traditional approximations often fail.

## Results
The model was trained for **3000 epochs**.
- **Convergence**: The loss function shows stable convergence, and the predicted initial price $Y_0$ matches the theoretical benchmark within < 0.5% relative error.
- **Hedging Strategy**: The learned Delta path demonstrates accurate sensitivity to Gamma risk near the strike price ($t=0$) and smooth convergence in OTM scenarios.
