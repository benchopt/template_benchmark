from benchopt import BaseSolver, safe_import_context

# Protect the import with `safe_import_ctx`. This allows:
# - To skip import to fasten autocompletion for instance.
# - To get requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import numpy as np

    # import your reusable functions here
    from benchmark_utils import gradient_ols


# The benchmark solvers must be name `Solver` and
# derive from `BaseSolver` for `benchopt` to work properly.
class Solver(BaseSolver):

    # Name to select the solver in the CLI and to display the results.
    name = 'GD'

    # List of parameters for the sovler. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    parameters = {
        'scale_step': [1, 1.99],
    }

    def set_objective(self, X, y):
        # Define the information received by each solver from the objective.
        # The arguments of this function are the results of the
        # `Objective.get_objective`. This defines the benchmark's API for
        # passing the objective to the solver.
        # This is customizable for each benchmark.
        self.X, self.y = X, y

    def run(self, n_iter):
        # This is the function that is called to evaluate the solver.
        # It runs the algorithm for a given a number of iterations `n_iter`.

        L = np.linalg.norm(self.X, ord=2) ** 2
        alpha = self.scale_step / L
        w = np.zeros(self.X.shape[1])
        for _ in range(n_iter):
            w -= alpha * gradient_ols(self.X, self.y, w)

        self.w = w

    def get_result(self):
        # Return the result from one optimization run.
        # The outputs of this function are the arguments of the
        # `compute` method of the objective. This defines the benchmark's
        # API for solvers' results. This is customizable for each benchmark.
        return self.w
