# `benchmark_utils` is a module in which you can define code to reuse in
# the benchmark objective, datasets, and solvers. The folder should have the
# name `benchmark_utils`, and code defined inside will be importable using
# the usual import syntax.
import numpy as np


def gradient_ols(X, y, beta):
    return X.T @ (X @ beta - y)


def value_ols(X, y, beta):
    return 0.5 * np.linalg.vector_norm(y - X @ beta) ** 2
