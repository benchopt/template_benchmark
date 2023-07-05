from datasets.simulated import Dataset
from objective import Objective
from solvers.python_gd import Solver

# In general the test requires having installed the dependencies before
# For the original template having benchopt installed is sufficient
def test_pipeline():
    # We add manually attributes of dataset, objective, solver for the test

    dataset = Dataset()
    dataset.n_samples = 1000
    dataset.n_features = 500
    dataset.random_state = 27
    dataset_params = dataset.get_data()
    objective = Objective()
    objective.whiten_y = True
    objective.set_data(**dataset_params)
    objective_params = objective.get_objective()

    solver = Solver()
    solver.scale_step = 1.
    solver.set_objective(**objective_params)
    solver.run(n_iter=10)
    results = solver.get_result()

    metrics = objective.compute(results)
    assert 'value' in metrics.keys()
    print('Pipeline functional')

if __name__ == '__main__':
    test_pipeline()