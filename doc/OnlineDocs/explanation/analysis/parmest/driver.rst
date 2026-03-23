.. _driversection:

Parmest Quick Start Guide 
=========================

This quick start guide shows how to use parmest to estimate model parameters
from experimental data as well as compute their uncertainty. The model and data used in this
guide were taken from [RB01]_.

The mathematical model of interest is:

.. math::
   y_i(\theta_1, \theta_2, t_i) = \theta_1 \left(1 - e^{-\theta_2 t_i} \right),
    \quad \forall \quad i \, \in \, {1, \ldots, n}

Where :math:`y` is the observation of the measured variable, :math:`t` is the time, :math:`\theta_1`
is the asymptote, and :math:`\theta_2` is the rate constant.

The experimental data is given in the table below:

.. list-table:: Data
   :header-rows: 1
   :widths: 30 30

   * - hour
     - y
   * - 1
     - 8.3
   * - 2
     - 10.3
   * - 3
     - 19.0
   * - 4
     - 16.0
   * - 5
     - 15.6
   * - 7
     - 19.8

To use parmest to estimate :math:`\theta_1` and :math:`\theta_2` from the data, we provide the following
detailed steps:

.. testsetup:: *

    import pandas as pd
    import pyomo.contrib.parmest.parmest as parmest

    # Data
    data = pd.DataFrame(
        data=[[1, 8.3], [2, 10.3], [3, 19.0], [4, 16.0], [5, 15.6], [7, 19.8]],
        columns=["hour", "y"]
    )

    # Create an experiment list
    from pyomo.contrib.parmest.examples.rooney_biegler.rooney_biegler import RooneyBieglerExperiment

    exp_list = []
    for i in range(data.shape[0]):
        exp_list.append(RooneyBieglerExperiment(data.loc[i, :]))

Step 0: Import parmest and Pandas
---------------------------------

Before solving the parameter estimation problem, the following code must be executed to import the
required packages for parameter estimation in parmest:

.. doctest::
    :skipif: not __import__('pyomo.contrib.parmest.parmest').contrib.parmest.parmest.parmest_available

    >>> import pyomo.contrib.parmest.parmest as parmest
    >>> import pandas as pd

.. _ExperimentClass:

Step 1: Create an Experiment Class
----------------------------------

Parmest requires that the user create an :class:`~pyomo.contrib.parmest.experiment.Experiment` class that
builds an annotated Pyomo model denoting experiment outputs, unknown parameters, and measurement errors using
Pyomo `Suffix` components.

* ``m.experiment_outputs`` maps the experiment output (or measurement) terms in the model
  (Pyomo `Param`, `Var`, or `Expression`) to their associated data values (float, int).
* ``m.unknown_parameters`` maps the model parameters to estimate (Pyomo `Param` or `Var`)
  to their component unique identifier (Pyomo `ComponentUID`) which is used to identify equivalent
  parameters across multiple experiments.
  Within parmest, any parameters that are to be estimated are converted to unfixed variables.
  Variables that are to be estimated are also unfixed.
* ``m.measurement_error`` maps the measurement error (float, int) of the experiment output, or measurement
  (Pyomo `Param`, `Var`, or `Expression`) defined in the model.

The experiment class has one required method:

* :class:`~pyomo.contrib.parmest.experiment.Experiment.get_labeled_model` which returns the labeled Pyomo model.

An example :class:`~pyomo.contrib.parmest.experiment.Experiment` class is shown below.

.. literalinclude:: /../../pyomo/contrib/parmest/examples/rooney_biegler/rooney_biegler.py
   :language: python
   :pyobject: RooneyBieglerExperiment
   :caption: RooneyBieglerExperiment class from the parmest example

Step 2: Load the Data and Create a List of Experiments
------------------------------------------------------

Load the experimental data into Python and create an instance of your
:class:`~pyomo.contrib.parmest.experiment.Experiment` class for each set of experimental data.
In this example, each measurement of `y` is treated as a separate experiment.

.. doctest::
    :skipif: not __import__('pyomo.contrib.parmest.parmest').contrib.parmest.parmest.parmest_available

    >>> data = pd.DataFrame(data=[[1, 8.3], [2, 10.3], [3, 19.0], [4, 16.0], [5, 15.6], [7, 19.8]],
    ...                     columns=["hour", "y"])
    >>> exp_list = []
    >>> for i in range(data.shape[0]):
    ...     exp_list.append(RooneyBieglerExperiment(data.loc[i, :]))

.. _EstimatorObj:

Step 3: Create the Estimator Object
-----------------------------------

To use parmest, the user creates an :class:`~pyomo.contrib.parmest.parmest.Estimator` object which includes
the following methods:

.. autosummary::
   :nosignatures:

   ~pyomo.contrib.parmest.parmest.Estimator.theta_est
   ~pyomo.contrib.parmest.parmest.Estimator.cov_est
   ~pyomo.contrib.parmest.parmest.Estimator.theta_est_bootstrap
   ~pyomo.contrib.parmest.parmest.Estimator.theta_est_leaveNout
   ~pyomo.contrib.parmest.parmest.Estimator.objective_at_theta
   ~pyomo.contrib.parmest.parmest.Estimator.confidence_region_test
   ~pyomo.contrib.parmest.parmest.Estimator.likelihood_ratio_test
   ~pyomo.contrib.parmest.parmest.Estimator.leaveNout_bootstrap_test

Additional functions are available in parmest to plot
results and fit distributions to theta values.

.. autosummary::
   :nosignatures:

   ~pyomo.contrib.parmest.graphics.pairwise_plot
   ~pyomo.contrib.parmest.graphics.grouped_boxplot
   ~pyomo.contrib.parmest.graphics.grouped_violinplot
   ~pyomo.contrib.parmest.graphics.fit_rect_dist
   ~pyomo.contrib.parmest.graphics.fit_mvn_dist
   ~pyomo.contrib.parmest.graphics.fit_kde_dist
    
A :class:`~pyomo.contrib.parmest.parmest.Estimator` object can be
created using the following code. A description of the arguments are
listed below.

.. doctest::
    :skipif: not __import__('pyomo.contrib.parmest.parmest').contrib.parmest.parmest.parmest_available

    >>> pest = parmest.Estimator(exp_list, obj_function="SSE")

Alternatively, the weighted sum of squared errors objective can be used.

.. doctest::
    :skipif: not __import__('pyomo.contrib.parmest.parmest').contrib.parmest.parmest.parmest_available

    >>> pest = parmest.Estimator(exp_list, obj_function="SSE_weighted")

Optionally, solver options can be supplied, e.g.,

.. doctest::
    :skipif: not __import__('pyomo.contrib.parmest.parmest').contrib.parmest.parmest.parmest_available

    >>> solver_options = {"max_iter": 6000}
    >>> pest = parmest.Estimator(exp_list, obj_function="SSE", solver_options=solver_options)

Objective function
^^^^^^^^^^^^^^^^^^

The ``obj_function`` keyword argument is used to specify the objective function to use for parameter
estimation if the user has not implemented their own custom objective function.
Parmest includes two built-in objective functions ("SSE" and "SSE_weighted") to compute
the sum of squared errors between the ``m.experiment_outputs`` model values and
data values. If the user wants to use an objective that is different from the built-in
options, a custom objective function can be specified in the user's model, however,
covariance matrix estimation (see :ref:`covariancesection` Section) is not supported
for custom objective functions.

When declaring a custom objective function, parmest assumes the model has the structure of 
a two-stage stochastic programming problem so the objective function should be implemented
using Pyomo Expressions for the first stage cost (named "FirstStageCost") and the second stage
cost (named "SecondStageCost"). For parameter estimation problems the first stage cost is usually 
set to zero and the second stage cost is usually defined as the deviation between the model and 
the observations.

Step 4: Estimate the Parameters
-------------------------------

After creating the :class:`~pyomo.contrib.parmest.parmest.Estimator` object with the desired objective function,
solve the parameter estimation problem by calling :class:`~pyomo.contrib.parmest.parmest.Estimator.theta_est`,
e.g.,

.. doctest::
    :skipif: not __import__('pyomo.contrib.parmest.parmest').contrib.parmest.parmest.parmest_available

    >>> pest = parmest.Estimator(exp_list, obj_function="SSE")
    >>> obj_val, theta_val = pest.theta_est()

Suggested Initialization Procedure for Parameter Estimation Problems
--------------------------------------------------------------------

To check the quality of initial guess values provided for the fitted parameters, we suggest solving a 
square instance of the problem prior to solving the parameter estimation problem using the following steps:

1. Create :class:`~pyomo.contrib.parmest.parmest.Estimator` object. To initialize the parameter 
estimation solve from the square problem solution, set optional argument ``solver_options = {bound_push: 1e-8}``.

2. Call :class:`~pyomo.contrib.parmest.parmest.Estimator.objective_at_theta` with optional 
argument ``(initialize_parmest_model=True)``. Different initial guess values for the fitted 
parameters can be provided using optional argument `theta_values` (**Pandas Dataframe**)

More Examples Beyond this Quick Guide
-------------------------------------

More detailed examples, such as parameter estimation of reaction kinetics are provided in the
:ref:`examplesection` Section.
