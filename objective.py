from benchopt import BaseObjective, safe_import_context

# Protect import to allow manipulating objective without importing library
# Useful for autocompletion and install commands
with safe_import_context() as import_ctx:
    import numpy as np


class Objective(BaseObjective):
    name = "Ordinary Least Squares"

    # All parameters 'p' defined here are available as 'self.p'
    parameters = {
        'fit_intercept': [False],
    }

    def get_one_solution(self):
        # Return one solution. This should be compatible with 'self.compute'.
        return np.zeros(self.X.shape[1])

    def set_data(self, X, y):
        self.X, self.y = X, y

    def compute(self, beta):
        diff = self.y - self.X.dot(beta)
        return .5 * diff.dot(diff)

    def to_dict(self):
        return dict(X=self.X, y=self.y, fit_intercept=self.fit_intercept)
