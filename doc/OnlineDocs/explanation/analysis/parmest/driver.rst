.. _driversection:

Parameter Estimation 
====================

Parameter Estimation using parmest requires a Pyomo model, experimental
data which defines multiple scenarios, and parameters
(thetas) to estimate.  parmest uses Pyomo [PyomoBookIII]_ and (optionally)
mpi-sppy [KMM+23]_ to solve a
two-stage stochastic programming problem, where the experimental data is
used to create a scenario tree.  The objective function needs to be
written with the Pyomo Expression for first stage cost
(named "FirstStageCost") set to zero and the Pyomo Expression for second
stage cost (named "SecondStageCost") defined as the deviation between
the model and the observations (typically defined as the sum of squared
deviation between model values and observed values).

If the Pyomo model is not formatted as a two-stage stochastic
programming problem in this format, the user can choose either the
built-in "SSE" or "SSE_weighted" objective functions, or supply a custom
objective function to use as the second stage cost. The Pyomo model will then be
modified within parmest to match the required specifications.
The stochastic programming callback function is also defined within parmest.
The callback function returns a populated and initialized model for each scenario.

Quick Guide
===========
We use a simple model to provide a quick guide on how to use parmest to estimate model parameters
from experimental data as well as compute their uncertainty. The model and data used in this
guide is taken from: Rooney, W. C.; Biegler, L. T. Design for Model Parameter Uncertainty Using
Nonlinear Confidence Regions. AIChE J. 2001, 47 (8), 1794–1804.

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

Step 0: Import Pyomo, parmest, Experiment Class, and Pandas
-----------------------------------------------------------

Before solving the parameter estimation problem, the following code must be executed to import the
required packages for parameter estimation in parmest:

.. doctest::
    :skipif: not __import__('pyomo.contrib.parmest.parmest').contrib.parmest.parmest.parmest_available

    >>> import pyomo.environ as pyo
    >>> import pyomo.contrib.parmest.parmest as parmest
    >>> from pyomo.contrib.parmest.experiment import Experiment
    >>> import pandas as pd

.. _ExperimentClass:

Step 1: Create the Experiment Class for the Model
-------------------------------------------------

parmest requires that the user create an :class:`~pyomo.contrib.parmest.experiment.Experiment` class that
builds an annotated Pyomo model denoting experiment outputs, unknown parameters, and measurement errors using Pyomo `Suffix` components.

* ``m.experiment_outputs`` maps the experiment output, or measurement, terms in the model (Pyomo `Param`, `Var`, or `Expression`) to their associated data values (float, int).
* ``m.unknown_parameters`` maps the model parameters to estimate (Pyomo `Param` or `Var`)
  to their component unique identifier (Pyomo `ComponentUID`) which is used to identify equivalent parameters across multiple experiments.
  Within parmest, any parameters that are to be estimated are converted to unfixed variables.
  Variables that are to be estimated are also unfixed.

The experiment class has one required method:

* :class:`~pyomo.contrib.parmest.experiment.Experiment.get_labeled_model` which returns the labeled Pyomo model.
  Note that the model does not have to be specifically written as a
  two-stage stochastic programming problem for parmest.
  That is, parmest can modify the
  objective, see the :ref:`EstimatorObj` Section below.

This step shows how to create the :class:`~pyomo.contrib.parmest.experiment.Experiment` class using the
mathematical model outlined in the introduction section of this Quick Start.

.. doctest::

    >>> class RooneyBieglerExperiment(Experiment):
    ...     def __init__(self, data):
    ...         self.data = data
    ...         self.model = None
    ...
    ...     def create_model(self):
    ...         # the model expects a dataframe
    ...         data_df = self.data.to_frame().transpose()
    ...
    ...         # create the pyomo model
    ...         m = self.model = pyo.ConcreteModel()
    ...
    ...         # add asymptote and rate constant to the model
    ...         m.asymptote = pyo.Var(initialize=15)
    ...         m.rate_constant = pyo.Var(initialize=0.5)
    ...
    ...         # add the measured variable, y, to the model
    ...         m.y = pyo.Var(data_df.hour, within=pyo.PositiveReals, initialize=5)
    ...
    ...         # add the mathematical equation for predicting y
    ...         def response_rule(m, h):
    ...             return m.y[h] == m.asymptote * (1 - pyo.exp(-m.rate_constant * h))
    ...
    ...         m.response_function = pyo.Constraint(data_df.hour, rule=response_rule)
    ...
    ...         return m
    ...
    ...     def label_model(self):
    ...
    ...         m = self.model
    ...
    ...         # label the experiment outputs
    ...         m.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
    ...         m.experiment_outputs.update([(m.y[self.data['hour']], self.data['y'])])
    ...
    ...         # label the unknown parameters in the model
    ...         m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
    ...         m.unknown_parameters.update(
    ...             (k, pyo.ComponentUID(k)) for k in [m.asymptote, m.rate_constant]
    ...         )
    ...
    ...         # add the measurement error assumed to be constant and 0.1
    ...         m.measurement_error = pyo.Suffix(direction=pyo.Suffix.LOCAL)
    ...         m.measurement_error.update([(m.y[self.data['hour']], 0.1)])
    ...
    ...     def finalize_model(self):
    ...
    ...         m = self.model
    ...         pass
    ...
    ...     def get_labeled_model(self):
    ...         self.create_model()
    ...         self.label_model()
    ...         self.finalize_model()
    ...
    ...         return self.model


Step 2: Load the Data and Create a List Experiments
-----------------------------------------------------------------------

After creating an :class:`~pyomo.contrib.parmest.experiment.Experiment` class instance for the model, a list of the
model's :class:`~pyomo.contrib.parmest.experiment.Experiment` class for all the experimental data points should be
created.

.. doctest::

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

The second argument is an optional argument if the objective function has already been included in the
Pyomo model, which defines the optimization objective function to use in parameter estimation. However, if the
objective function has not been included in the Pyomo model, like the one in the :ref:`ExperimentClass` Section
above, the user is required to supply the second argument.

If ``obj_function`` is not specified, the Pyomo model is used "as is" and
should be defined with "FirstStageCost" and "SecondStageCost"
expressions that are used to build an objective for the two-stage 
stochastic programming problem.

If the Pyomo model is not written as a two-stage stochastic programming problem in
this format, the user can select the "SSE" or "SSE_weighted" built-in objective
functions. If the user wants to use an objective that is different from the built-in
options, a custom objective function can be defined for parameter estimation. However,
covariance matrix estimation (see :ref:`covariancesection` Section) is not supported for custom objective functions. The objective function (built-in or custom) has a single argument,
which is the model from a single experiment. The objective function returns a Pyomo expression
which is used to define "SecondStageCost". The objective function can be used to customize data
points and weights that are used in parameter estimation.

Parmest includes two built-in objective functions ("SSE" and "SSE_weighted") to compute
the sum of squared errors between the ``m.experiment_outputs`` model values and
data values.

Step 3: Estimate the Parameters
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
