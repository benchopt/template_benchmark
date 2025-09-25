from benchopt import BaseObjective

import numpy as np
from benchmark_utils import value_ols, gradient_ols


# The benchmark objective must be named `Objective` and
# inherit from `BaseObjective` for `benchopt` to work properly.
class Objective(BaseObjective):

    # Name to select the objective in the CLI and to display the results.
    name = "Ordinary Least Squares"

    # URL of the main repo for this benchmark.
    url = "https://github.com/#ORG/#BENCHMARK_NAME"

    # List of packages needed to run the benchmark.
    # They are installed with conda; to use pip, use 'pip:packagename'. To
    # install from a specific conda channel, use 'channelname:packagename'.
    # Packages that are not necessary to the whole benchmark but only to some
    # solvers or datasets should be declared in Dataset or Solver (see
    # simulated.py and python-gd.py).
    # Example syntax: requirements = ['numpy', 'pip::jax', 'pytorch::pytorch']
    requirements = ["numpy"]

    # Minimal version of benchopt required to run this benchmark.
    # Bump it up if the benchmark depends on a new feature of benchopt.
    min_benchopt_version = "1.7"

    def set_data(self, X, y):
        # The keyword arguments of this function are the keys of the dictionary
        # returned by `Dataset.get_data`. This defines the benchmark's
        # API to pass data. This is customizable for each benchmark.
        self.X, self.y = X, y

    def evaluate_result(self, beta):
        # The keyword arguments of this function are the keys of the
        # dictionary returned by `Solver.get_result`. This defines the
        # benchmark's API to pass solvers' result. This is customizable for
        # each benchmark.

        # Here we can compute any metric to evaluate the quality of the
        # solution. We compute the value of the objective function and the
        # norm of the gradient.
        grad = gradient_ols(self.X, self.y, beta)
        value = value_ols(self.X, self.y, beta)

        # This method can return many metrics in a dictionary. One of these
        # metrics needs to be `value` for convergence detection purposes.
        return dict(
            value=value,
            grad_norm=np.linalg.vector_norm(grad),
        )

    def get_one_result(self):
        # Return one solution. The return value should be an object compatible
        # with `self.evaluate_result`. This is mainly for testing purposes.
        return dict(beta=np.zeros(self.X.shape[1]))

    def get_objective(self):
        # Define the information to pass to each solver to run the benchmark.
        # The output of this function are the keyword arguments
        # for `Solver.set_objective`. This defines the
        # benchmark's API for passing the objective to the solver.
        # It is customizable for each benchmark.
        return dict(
            X=self.X,
            y=self.y,
        )
