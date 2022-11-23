# define reusable functions across the benchmark folder
# to be imported in datafits, objective, and solvers


def gradient_ols(X, y, w):
    return X.T.dot(X.dot(w) - y)
