from benchopt import BaseDataset

import numpy as np


# All datasets must be named `Dataset` and inherit from `BaseDataset`
class Dataset(BaseDataset):

    # Name to select the dataset in the CLI and to display the results.
    name = "Simulated"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    # Any parameters 'param' defined here is available as `self.param`.
    parameters = {
        'n_samples, n_features': [
            (1000, 500),
            (5000, 200),
        ],
    }

    # List of packages needed to run the dataset. See the corresponding
    # section in objective.py. This is an optional attribute.
    requirements = []

    def get_data(self):
        # The return arguments of this function are passed as keyword arguments
        # to `Objective.set_data`. This defines the benchmark's
        # API to pass data. It is customizable for each benchmark.

        # Get a random seed to generate the data. The seed is generated from
        # the `get_seed` method, which ensures that the same seed is used
        # across different runs of the benchmark, and different solvers.
        # The use of `use_repetition=True` ensures that the seed changes across
        # different repetitions of the benchmark, which can be useful to
        # generate different data for each repetition.
        seed = self.get_seed(use_repetition=True)

        # Generate pseudorandom data using `numpy`.
        rng = np.random.RandomState(seed)
        X = rng.randn(self.n_samples, self.n_features)
        y = rng.randn(self.n_samples)

        # The dictionary defines the keyword arguments for `Objective.set_data`
        return dict(X=X, y=y)
