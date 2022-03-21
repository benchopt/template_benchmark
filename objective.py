from benchopt import BaseObjective


class Objective(BaseObjective):
    name = "Ordinary Least Squares"

    parameters = {
        'fit_intercept': [False],
    }

    def __init__(self, fit_intercept=False):
        self.fit_intercept = fit_intercept

    def set_data(self, X, y):
        # The keyword arguments of this function are the keys of the `data`
        # dict in the `get_data` function of the dataset.
        # They are customizable.
        self.X, self.y = X, y

    def compute(self, beta):
        # The arguments of this function are the outputs of the
        # `get_result` method of the solver.
        # They are customizable.
        diff = self.y - self.X.dot(beta)
        return .5 * diff.dot(diff)

    def to_dict(self):
        # The output of this function are the keyword arguments
        # for the `set_objective` method of the solver.
        # They are customizable.
        return dict(X=self.X, y=self.y, fit_intercept=self.fit_intercept)
