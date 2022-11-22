from benchopt import BaseObjective, safe_import_context

# Protect the import with `safe_import_ctx`. This allows:
# - To skip import to fasten autocompletion for instance.
# - To get requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import numpy as np

    # import your reusable functions here
    from benchmark_utils.utils import reusable_function


# The benchmark objective must be name `Objective` and
# derive from `BaseObjective` for `benchopt` to work properly.
class Objective(BaseObjective):

    # Name to select the objective in the CLI and to display the results.
    name = "Ordinary Least Squares"

    # List of parameters for the objective. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    # This means the OLS objective will have a parameter `self.whiten_y`.
    parameters = {
        'whiten_y': [False, True],
    }

    # Minimal version of benchopt required to run this benchmark.
    # Bump it up if the benchmark depends on a new feature of benchopt.
    min_benchopt_version = "1.3"

    def set_data(self, X, y):
        # The keyword arguments of this function are the keys of the `data`
        # dict returned by `Dataset.get_data`. This defines the benchmark's
        # API to pass data. This is customizable for each benchmark.
        self.X, self.y = X, y

        # `set_data` can be used to preprocess the data. For instance,
        # if `whiten_y` is True, remove the mean of `y`.
        if self.whiten_y:
            y -= y.mean(axis=0)

    def compute(self, beta):
        # The arguments of this function are the outputs of the
        # `Sovler.get_result`. This defines the benchmark's API to pass
        # solvers' result. This is customizable for each benchmark.
        diff = self.y - self.X.dot(beta)

        # This method can return many metrics in a dictionary. One of these
        # metrics needs to be `value` for convergence detection purposes.
        return dict(
            value=.5 * diff.dot(diff),
        )

    def get_one_solution(self):
        # Return one solution. This should be an object compatible
        # with `self.compute`. This is mainly for testing purposes.
        return np.zeros(self.X.shape[1])

    def get_objective(self):
        # Define the information to pass to each solver to run the benchmark.
        # The output of this function are the keyword arguments
        # for the `set_objective` method of the solver. This defines the
        # benchmark's API for passing the objective to the solver.
        # This is customizable for each benchmark.
        return dict(
            X=self.X,
            y=self.y,
        )
