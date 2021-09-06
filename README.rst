Template for BenchOpt Benchmark repositories
=============================================
|Build Template|

This repo should be used with the following steps:

1. Hit the `Use this template` button,
2. Create the repo,
3. Clone it on your computer,
4. Run ``python clean_template.py`` script that will replace the necessary
   part in the ``README.rst`` with your repo and org name.
5. Remove this section and describe

My BenchOpt Benchmark
=====================
|Build Status| |Python 3.6+|

BenchOpt is a package to simplify and make more transparent and
reproducible the comparisons of optimization algorithms.
This benchmark is dedicated to solver of **describe your problem**:

.. math::

    \min_{w} f(X, w)

where n (or n_samples) stands for the number of samples, p (or n_features) stands for the number of features and

.. math::

 X = [x_1^\top, \dots, x_n^\top]^\top \in \mathbb{R}^{n \times p}

Install
--------

This benchmark can be run using the following commands:

.. code-block::

   $ pip install -U benchopt
   $ git clone https://github.com/#ORG/#BENCHMARK_NAME
   $ benchopt run #BENCHMARK_NAME

Apart from the problem, options can be passed to `benchopt run`, to restrict the benchmarks to some solvers or datasets, e.g.:

.. code-block::

	$ benchopt run #BENCHMARK_NAME -s solver1 -d dataset2 --max-runs 10 --n-repetitions 10


Use `benchopt run -h` for more details about these options, or visit https://benchopt.github.io/api.html.

.. |Build Template| image:: https://github.com/benchopt/template_benchmark/workflows/Tests/badge.svg
   :target: https://github.com/benchopt/template_benchmark/actions
.. |Build Status| image:: https://github.com/#ORG/#BENCHMARK_NAME/workflows/Tests/badge.svg
   :target: https://github.com/#ORG/#BENCHMARK_NAME/actions
.. |Python 3.6+| image:: https://img.shields.io/badge/python-3.6%2B-blue
   :target: https://www.python.org/downloads/release/python-360/
