.. _parmest_estimability_section:

Estimability Analysis
=====================

After estimating the model parameters with their associated uncertainty, as demonstrated in the
:ref:`driversection` and :ref:`parmest_covariance_section` Sections, estimability analysis is required to identify
parameters that cannot be reliably estimated from the available data due to limitations in the mathematical
model structure. If such parameters exist, the model may need to be reformulated, replaced with an
alternative structure, or augmented with additional prior information. In parmest, estimability analysis can
be performed using eigen-decomposition of the parameter covariance matrix, profile likelihood methods, or
multistart optimization.

Eigen-decomposition
-------------------

The estimability of model parameters can be analyzed through eigen-decomposition of the covariance matrix
obtained from parameter estimation. This covariance matrix quantifies parameter uncertainty and captures both
parameter variances and correlations. Eigen-decomposition of this matrix identifies principal directions
in parameter space along which uncertainty is largest or smallest. These directions provide insight into
parameter identifiability and reveal combinations of parameters that are either structurally identifiable or
non-identifiable based on the underlying model formulation.

.. note::

   Detailed descriptions and example code for this method will be added in a future update.

Profile Likelihood
------------------

Profile likelihood analysis evaluates parameter estimability by systematically varying one parameter while
re-optimizing the remaining parameters to maintain consistency with the model and observed data. This approach
is closely related to likelihood ratio–based uncertainty quantification and provides a robust characterization
of practical identifiability through the shape of the likelihood surface. In addition, it can reveal structural
non-identifiability when flat or unbounded profiles indicate parameter combinations that are not uniquely
determined by the model formulation, particularly in nonlinear systems.

.. note::

   Detailed descriptions and example code for this method will be added in a future update.

Multistart Optimization
-----------------------

Multistart optimization assesses parameter estimability by exploring a range of initial guesses. Because
parameter estimation problems are often nonlinear and may exhibit multiple local minima, different initializations
can lead to different parameter estimates. By solving the estimation problem from multiple starting points, one can
evaluate the robustness of the solution and identify potential issues related to non-convexity or non-identifiability.
Consistent convergence to a unique solution across initializations suggests that the parameters are structurally
identifiable within the model formulation, whereas sensitivity to initialization indicates potential estimability
issues.

In parmest, multistart optimization is performed with
:class:`~pyomo.contrib.parmest.parmest.Estimator.theta_est_multistart`, which solves the parameter estimation
problem from several starting values and returns the best start that terminates optimally. The starting values are
sampled between the parameter bounds, so every fitted parameter must have finite lower and upper bounds. The
``multistart_sampling_method`` argument selects ``"uniform_random"`` (the default), ``"latin_hypercube"``, or
``"sobol_sampling"`` (which works best when ``n_starts`` is a power of 2), and ``seed`` makes the starting values
reproducible. Starting values can also be given directly with ``user_supplied_vals``. Using the data and
:class:`~pyomo.contrib.parmest.experiment.Experiment` class from the :ref:`driversection` Section:

.. testsetup:: *
    :skipif: not __import__('pyomo.contrib.parmest.parmest').contrib.parmest.parmest.parmest_available

    # Data
    import pandas as pd
    data = pd.DataFrame(
        data=[[1, 8.3], [2, 10.3], [3, 19.0],
              [4, 16.0], [5, 15.6], [7, 19.8]],
        columns=['hour', 'y'],
    )

    # Create the Experiment class
    from pyomo.contrib.parmest.examples.rooney_biegler.rooney_biegler import RooneyBieglerExperiment

    exp_list = []
    for i in range(data.shape[0]):
        exp_list.append(RooneyBieglerExperiment(data.loc[i, :]))

.. doctest::
    :skipif: not __import__('pyomo.contrib.parmest.parmest').contrib.parmest.parmest.parmest_available

    >>> import pyomo.contrib.parmest.parmest as parmest
    >>> pest = parmest.Estimator(exp_list, obj_function="SSE")
    >>> results, best_theta, best_obj = pest.theta_est_multistart(
    ...     n_starts=8, multistart_sampling_method="latin_hypercube", seed=42
    ... )

``results`` is a Pandas DataFrame with one row per start: the starting and converged (``converged_<name>``)
parameter values, the objective value, the solver termination condition, and the solve time. Comparing the
converged values and objectives of the optimal starts shows whether the estimate depends on the initialization.
The solution of the best start is kept in the :class:`~pyomo.contrib.parmest.parmest.Estimator` object, so
methods such as :class:`~pyomo.contrib.parmest.parmest.Estimator.cov_est` can be called afterwards.
