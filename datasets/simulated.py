from benchopt import BaseDataset, safe_import_context


# Protect the import with `safe_import_ctx`. This allows:
# - To skip import to fasten autocompletion for instance.
# - To get requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import numpy as np

    # import your reusable functions here
    from benchmark_utils.utils import reusable_function  # noqa


# All datasets must be name `Dataset` and derive from the `BaseDataset`
class Dataset(BaseDataset):

    # Name to select the dataset in the CLI and to display the results.
    name = "Simulated"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    parameters = {
        'n_samples, n_features': [
            (1000, 500),
            (5000, 200),
        ],
        'random_state': [27],
    }

    def get_data(self):
        # The return arguments of this function are passed as keyword arguments
        # to `Objective.set_data`. This defines the benchmark's
        # API to pass data. This is customizable for each benchmark.

        # Generate data using `numpy`.
        rng = np.random.RandomState(self.random_state)
        X = rng.randn(self.n_samples, self.n_features)
        y = rng.randn(self.n_samples)

        # `data` defines the keyword arguments for `Objective.set_data`.
        return dict(X=X, y=y)
