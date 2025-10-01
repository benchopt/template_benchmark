from benchopt import BaseSolver

import numpy as np

# Reusable function can be imported from the benchmark_utils module, which is
# dynamically installed when running the benchmark.
from benchmark_utils import gradient_ols


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(BaseSolver):

    # Name to select the solver in the CLI and to display the results.
    name = 'GD'

    # List of packages needed to run the solver. See the corresponding
    # section in objective.py. This is an optional attribute.
    requirements = []

    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    parameters = {
        'learning_rate': [0.1, 0.5],
    }

    # Evaluation strategy for the performance curve.
    # It describe when how and when the solver will be evaluated.
    # You can also use `iteration`, `tolerance` or `run_once`, as described in
    # https://benchopt.github.io/performance_curves.html
    # For optimization solvers, we recommend to use 'callback' which can be
    # used regularly to log the progress of the solver and implement a stopping
    # criterion.
    # For machine learning methods, `run_once` is usually more adapted, to
    # evaluate method only at the end of the training phase.
    sampling_strategy = 'callback'

    def set_objective(self, X, y):
        # Define the information received by each solver from the objective.
        # The arguments of this function are the results of the
        # `Objective.get_objective`. This defines the benchmark's API for
        # passing the objective to the solver.
        # It is customizable for each benchmark.
        self.X, self.y = X, y

    def run(self, callback):
        # This is the function that is called to run the method.
        # When using ``sampling_strategy='callback'``, the function is provided
        # with a ``callback`` function that must be called regularly to
        # log the progress of the solver. The callback function returns
        # ``True`` until the solver should stop.
        # See https://benchopt.github.io/guide/auto_stop.html for more details.

        self.beta = np.zeros(self.X.shape[1])
        L = np.linalg.norm(self.X, ord=2)**2
        while callback():
            self.beta -= (self.learning_rate / L) * gradient_ols(
                self.X, self.y, self.beta
            )

    def get_result(self):
        # Return the result of the method.
        #
        # The outputs of this function is a dictionary which defines the
        # keyword arguments for `Objective.evaluate_result`
        # This defines the benchmark's API for solvers' results.
        # it is customizable for each benchmark.
        return dict(beta=self.beta)
