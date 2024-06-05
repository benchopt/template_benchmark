# `benchmark_utils` is a module in which you can define code to reuse in
# the benchmark objective, datasets, and solvers. The folder should have the
# name `benchmark_utils`, and code defined inside will be importable using
# the usual import syntax. To import external packages in this file, use the
# safe_import_context as follows:

"""
from benchopt.utils import safe_import_context

with safe_import_context() as import_ctx:
    from sklearn.linear_model import Lasso
"""

def gradient_ols(X, y, beta):
    return X.T @ (X @ beta - y)
