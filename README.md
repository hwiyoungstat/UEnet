# U-ENet: Mean-Anchored Elastic Net Regression

**U-ENet** is a mean-anchored elastic net regression method designed to reduce systematic regression-to-the-mean (RTM) bias in prediction. This repository provides a **base R implementation** using the **alternating direction method of multipliers (ADMM)**.

The implementation imposes mean-anchoring constraints on observations whose response is below or above the overall sample mean, while retaining elastic net regularization.

## Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Quick start](#quick-start)
- [Function arguments](#function-arguments)
- [Returned values](#returned-values)
- [Optimization formulation](#optimization-formulation)
- [Implementation notes](#implementation-notes)
- [Source code](#source-code)

## Overview

The implementation combines:

- **Elastic net regularization:** an L1/L2 penalty on predictor coefficients.
- **Mean anchoring:** equality constraints that match the average fitted and observed response values in the lower- and upper-response groups.
- **ADMM optimization:** coefficient updates via Cholesky factorization, soft thresholding, and adaptive penalty parameter `rho`.
- **Unpenalized intercept:** the **first column of `X` must be an intercept column** (typically all ones).

The model returns fitted coefficients, in-sample predictions, and basic optimization diagnostics.

## Requirements

- R (base R only; no additional packages required).
- A numeric predictor matrix `X` with an intercept in **column 1**.
- A numeric response vector `Y` of length `nrow(X)`.

> **Important:** Include the intercept yourself. The function does **not** add one automatically. Both the groups `Y < mean(Y)` and `Y > mean(Y)` must contain observations, and the inputs should not contain missing or non-finite values. The function currently does not validate these conditions.

## Quick start

Place [`UEnet.R`](UEnet.R) in the same directory as your script and run:

```r
source("UEnet.R")

# 1. Generate reproducible example data
set.seed(123)
n <- 200
x1 <- rnorm(n)
x2 <- rnorm(n)
Y <- 50 + 5 * x1 - 3 * x2 + rnorm(n, sd = 4)

# The intercept MUST be the first column.
X <- cbind(intercept = 1, x1 = x1, x2 = x2)

# 2. Fit U-ENet
fit <- UEnet(X, Y, alpha = 1/3, lambda = 0.5)

# 3. Estimated coefficients
setNames(fit$betahat, colnames(X))

# 4. Fitted values and convergence diagnostics
head(fit$predict)
fit$iteration
c(primal = fit$r.norm, dual = fit$s.norm)
```

### Check the mean-anchoring constraints

The group means of the fitted values should be close to those of the observed responses **when the ADMM algorithm converges**.

```r
lower <- Y < mean(Y)
upper <- Y > mean(Y)

data.frame(
  group = c("Below response mean", "Above response mean"),
  observed_mean = c(mean(Y[lower]), mean(Y[upper])),
  fitted_mean = c(mean(fit$predict[lower]), mean(fit$predict[upper]))
)
```

### Predict on new data

The function returns regression coefficients. Multiply a new design matrix by `fit$betahat` to obtain predictions:

```r
X_new <- cbind(
  intercept = 1,
  x1 = c(-1, 0, 1),
  x2 = c(0.5, -0.2, 1.2)
)

pred_new <- as.vector(X_new %*% fit$betahat)
pred_new
```

**Use the same predictor order and preprocessing for new data as for training data.** The function currently does not provide a separate `predict()` method.

## Function arguments

```r
UEnet(X, Y, alpha = 1/3, lambda = 0.5)
```

| Argument | Description | Default |
| --- | --- | --- |
| `X` | Numeric `n × p` design matrix; the first column is the unpenalized intercept. | Required |
| `Y` | Numeric response vector of length `n`. | Required |
| `alpha` | Elastic net mixing parameter: `0` for ridge-type penalty and `1` for lasso-type penalty. | `1/3` |
| `lambda` | Nonnegative overall regularization strength. | `0.5` |

For the usual elastic net interpretation, select `0 <= alpha <= 1` and `lambda >= 0`. These ranges are not enforced by the current function.

## Returned values

`UEnet()` returns a list:

| Component | Description |
| --- | --- |
| `betahat` | Estimated regression coefficients used for prediction. |
| `gammahat` | Auxiliary ADMM coefficients after soft thresholding. |
| `predict` | In-sample fitted values, `X %*% betahat`. |
| `iteration` | Number of ADMM iterations performed. |
| `r.norm` | Final primal residual norm. |
| `s.norm` | Final dual residual norm. |
| `beta.gamma.diff` | Maximum absolute difference between `betahat` and `gammahat`. |

## Optimization formulation

Let \(L = \{i : y_i < \bar y\}\) and \(U = \{i : y_i > \bar y\}\), with sizes \(n_L\) and \(n_U\). Write \(X_L\) and \(X_U\) for the associated rows of \(X\), and \(\bar y_L, \bar y_U\) for the corresponding response means.

The implemented optimization problem is

$$
\begin{aligned}
\min_{\beta,\gamma}\quad &
\frac{1}{2n}\lVert Y-X\beta\rVert_2^2
+ \lambda\left(
\alpha\lVert\gamma_{-1}\rVert_1
+\frac{1-\alpha}{2}\lVert\gamma_{-1}\rVert_2^2
\right) \\
\text{subject to}\quad
&\beta=\gamma, \\
&\frac{1}{n_L}\mathbf{1}^{\top}X_L\beta=\bar{y}_L, \\
&\frac{1}{n_U}\mathbf{1}^{\top}X_U\beta=\bar{y}_U.
\end{aligned}
$$

Here, \(\gamma_{-1}\) excludes the first (intercept) coefficient from regularization. The two mean constraints anchor the lower- and upper-response groups separately.

Observations exactly equal to the overall mean response remain part of the regression fit but do not belong to either anchoring group.

## Implementation notes

- **Algorithm:** ADMM with a soft-thresholding update and Cholesky-based linear solves.
- **Adaptive ADMM penalty:** `rho` starts at 1 and is adjusted every 10 iterations based on the relative primal and dual residuals.
- **Stopping criterion:** both residual norms must fall below `1e-4`; the maximum number of iterations is `10,000`.
- **Preprocessing:** standardizing non-intercept predictors can be useful when their scales differ substantially; apply the same transformations to new data.
- **Convergence:** inspect `iteration`, `r.norm`, and `s.norm`. Hitting `10,000` iterations does not guarantee convergence.
- **Scope:** mean anchoring is imposed on the *training data* groups. It does not guarantee calibration, unbiasedness, or prediction accuracy on external data.

## Source code

The standalone implementation is in **[`UEnet.R`](UEnet.R)**. For convenience, the full source is also included below.

<details>
<summary>Show the complete R implementation</summary>

```r
# U-ENet: Mean-Anchored Elastic Net Regression
# Base R implementation using ADMM.
# IMPORTANT: X must include an intercept as its first column.

soft <- function(x, t) {
  temp <- (x - t) * ((x - t) > 0) -
    (-x - t) * ((-x - t) > 0)
  return(temp)
}

UEnet <- function(X, Y, alpha = 1/3, lambda = 0.5) {

  ite <- 10000  # Maximum iterations (early stopping enabled)

  p <- ncol(X)
  n <- nrow(X)
  mean.Y <- mean(Y)
  L <- which(Y < mean.Y)
  U <- which(Y > mean.Y)

  n.L <- length(L)
  n.U <- length(U)

  X.L <- X[L, , drop = FALSE]
  X.U <- X[U, , drop = FALSE]

  Y.L <- Y[L]
  Y.U <- Y[U]

  mean.Y.L <- mean(Y.L)
  mean.Y.U <- mean(Y.U)

  # Equality constraints: beta = gamma, plus two mean anchors.
  A <- rbind(
    diag(p),
    matrix(1 / n.L, nrow = 1, ncol = n.L) %*% X.L,
    matrix(1 / n.U, nrow = 1, ncol = n.U) %*% X.U
  )
  B <- -rbind(diag(p), rep(0, p), rep(0, p))
  C <- rbind(matrix(0, nrow = p), mean.Y.L, mean.Y.U)

  XtX <- crossprod(X) / n
  XtY <- crossprod(X, Y) / n
  AtA <- crossprod(A)

  # ADMM settings
  rho <- 1
  mu <- 10
  tau <- 2
  eps <- 1e-4

  beta.old <- rep(1, p)
  gamma.old <- beta.old
  omega.old <- rep(0, p + 2)

  # Cholesky factor used in the beta update.
  R <- chol(XtX + rho * AtA)

  for (i in 1:ite) {

    # Beta update
    Bgamma_minus_C_plus_omega <- c(
      -gamma.old + omega.old[1:p],
      -mean.Y.L + omega.old[p + 1],
      -mean.Y.U + omega.old[p + 2]
    )

    rhs <- XtY - rho * crossprod(A, Bgamma_minus_C_plus_omega)
    z <- forwardsolve(t(R), rhs)
    beta.new <- as.vector(backsolve(R, z))

    # Gamma update: the first coefficient (intercept) is unpenalized.
    v <- beta.new + omega.old[1:p]
    gamma.new <- numeric(p)
    gamma.new[1] <- v[1]
    gamma.new[-1] <- 1 / (lambda * (1 - alpha) + rho) *
      soft(rho * v[-1], lambda * alpha)

    # Primal and dual residuals
    r <- A %*% beta.new + B %*% gamma.new - C
    s <- -rho * (gamma.new - gamma.old)

    # Euclidean residual norms (works for both matrices and vectors).
    r.norm <- sqrt(sum(r^2))
    s.norm <- sqrt(sum(s^2))

    omega.new <- omega.old + r

    # Adaptive rho; update the factorization when rho changes.
    if (i %% 10 == 0) {
      if (r.norm > mu * s.norm) {
        rho <- tau * rho
        omega.new <- omega.new / tau
        R <- chol(XtX + rho * AtA)
      } else if (s.norm > mu * r.norm) {
        rho <- rho / tau
        omega.new <- omega.new * tau
        R <- chol(XtX + rho * AtA)
      }
    }

    beta.old <- beta.new
    gamma.old <- gamma.new
    omega.old <- omega.new

    if (r.norm < eps && s.norm < eps) {
      break
    }
  }

  predicted <- X %*% beta.new

  return(list(
    betahat = beta.new,
    gammahat = gamma.new,
    predict = predicted,
    iteration = i,
    r.norm = r.norm,
    s.norm = s.norm,
    beta.gamma.diff = max(abs(beta.new - gamma.new))
  ))
}
```

