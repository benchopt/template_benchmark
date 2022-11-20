import numpy as np


from benchopt import BaseSolver


class Solver(BaseSolver):
    """Gradient descent solver, optionally accelerated."""
    name = 'GD'

    # any parameter defined here is accessible as a class attribute
    parameters = {'scale_step': [1, 2]}

    def set_objective(self, X, y):
        # The arguments of this function are the results of the
        # `get_objective` method of the objective.
        # They are customizable.
        self.X, self.y = X, y

    def run(self, n_iter):
        # This is the function that is called to evaluate the solver.
        # It runs the algorithm for a given a number of iterations `n_iter`.

        L = np.linalg.norm(self.X, ord=2) ** 2
        alpha = self.scale_step / L
        w = np.zeros(self.X.shape[1])
        for _ in range(n_iter):
            w -= alpha * self.X.T.dot(self.X.dot(w) - self.y)

        self.w = w

    def get_result(self):
        # The outputs of this function are the arguments of the
        # `compute` method of the objective.
        # They are customizable.
        return self.w
