.. _driversection:

Parameter Estimation 
==================================

Parameter Estimation using parmest requires a Pyomo model, experimental
data which defines multiple scenarios, and a list of parameter names
(thetas) to estimate.  parmest uses Pyomo [PyomoBookII]_ and (optionally) 
mpi-sppy [mpisppy]_ to solve a
two-stage stochastic programming problem, where the experimental data is
used to create a scenario tree.  The objective function needs to be
written with the Pyomo Expression for first stage cost
(named "FirstStageCost") set to zero and the Pyomo Expression for second
stage cost (named "SecondStageCost") defined as the deviation between
the model and the observations (typically defined as the sum of squared
deviation between model values and observed values).

If the Pyomo model is not formatted as a two-stage stochastic
programming problem in this format, the user can supply a custom
function to use as the second stage cost and the Pyomo model will be
modified within parmest to match the required specifications.
The stochastic programming callback function is also defined within parmest.  The callback
function returns a populated and initialized model for each scenario.

To use parmest, the user creates a :class:`~pyomo.contrib.parmest.parmest.Estimator` object 
which includes the following methods:

.. autosummary::
   :nosignatures:

   ~pyomo.contrib.parmest.parmest.Estimator.theta_est
   ~pyomo.contrib.parmest.parmest.Estimator.theta_est_bootstrap
   ~pyomo.contrib.parmest.parmest.Estimator.theta_est_leaveNout
   ~pyomo.contrib.parmest.parmest.Estimator.objective_at_theta
   ~pyomo.contrib.parmest.parmest.Estimator.confidence_region_test
   ~pyomo.contrib.parmest.parmest.Estimator.likelihood_ratio_test
   ~pyomo.contrib.parmest.parmest.Estimator.leaveNout_bootstrap_test

Additional functions are available in parmest to group data, plot
results, and fit distributions to theta values.

.. autosummary::
   :nosignatures:

   ~pyomo.contrib.parmest.parmest.group_data
   ~pyomo.contrib.parmest.graphics.pairwise_plot
   ~pyomo.contrib.parmest.graphics.grouped_boxplot
   ~pyomo.contrib.parmest.graphics.grouped_violinplot
   ~pyomo.contrib.parmest.graphics.fit_rect_dist
   ~pyomo.contrib.parmest.graphics.fit_mvn_dist
   ~pyomo.contrib.parmest.graphics.fit_kde_dist
    
A :class:`~pyomo.contrib.parmest.parmest.Estimator` object can be
created using the following code. A description of each argument is
listed below.  Examples are provided in the :ref:`examplesection`
Section.

.. testsetup:: *
    :skipif: not __import__('pyomo.contrib.parmest.parmest').contrib.parmest.parmest.parmest_available

    import pandas as pd
    from pyomo.contrib.parmest.examples.rooney_biegler.rooney_biegler import rooney_biegler_model as model_function
    data = pd.DataFrame(data=[[1,8.3],[2,10.3],[3,19.0],
                              [4,16.0],[5,15.6],[6,19.8]],
                        columns=['hour', 'y'])
    theta_names = ['asymptote', 'rate_constant']
    def objective_function(model, data):
        expr = sum((data.y[i] - model.response_function[data.hour[i]])**2 for i in data.index)
        return expr

.. doctest::
    :skipif: not __import__('pyomo.contrib.parmest.parmest').contrib.parmest.parmest.parmest_available

    >>> import pyomo.contrib.parmest.parmest as parmest
    >>> pest = parmest.Estimator(model_function, data, theta_names, objective_function)

Optionally, solver options can be supplied, e.g.,

.. doctest::
    :skipif: not __import__('pyomo.contrib.parmest.parmest').contrib.parmest.parmest.parmest_available

    >>> solver_options = {"max_iter": 6000}
    >>> pest = parmest.Estimator(model_function, data, theta_names, objective_function, solver_options)
        


Model function
--------------

The first argument is a function which uses data for a single scenario
to return a populated and initialized Pyomo model for that scenario.

Parameters that the user would like to estimate can be defined as
**mutable parameters (Pyomo `Param`) or variables (Pyomo `Var`)**.  
Within parmest, any parameters that are to be estimated are converted to unfixed variables. 
Variables that are to be estimated are also unfixed.

The model does not have to be specifically written as a 
two-stage stochastic programming problem for parmest. 
That is, parmest can modify the
objective, see :ref:`ObjFunction` below.

Data
----

The second argument is the data which will be used to populate the Pyomo
model.  Supported data formats include:

* **Pandas Dataframe** where each row is a separate scenario and column
  names refer to observed quantities.  Pandas DataFrames are easily
  stored and read in from csv, excel, or databases, or created directly
  in Python.
* **List of Pandas Dataframe** where each entry in the list is a separate scenario. 
  Dataframes store observed quantities, referenced by index and column.
* **List of dictionaries** where each entry in the list is a separate
  scenario and the keys (or nested keys) refer to observed quantities.
  Dictionaries are often preferred over DataFrames when using static and
  time series data.  Dictionaries are easily stored and read in from
  json or yaml files, or created directly in Python.
* **List of json file names** where each entry in the list contains a
  json file name for a separate scenario.  This format is recommended
  when using large datasets in parallel computing.

The data must be compatible with the model function that returns a
populated and initialized Pyomo model for a single scenario.  Data can
include multiple entries per variable (time series and/or duplicate
sensors).  This information can be included in custom objective
functions, see :ref:`ObjFunction` below.

Theta names
-----------

The third argument is a list of parameters or variable names that the user wants to
estimate.  The list contains strings with `Param` and/or `Var` names from the Pyomo
model.

.. _ObjFunction:

Objective function
------------------

The fourth argument is an optional argument which defines the
optimization objective function to use in parameter estimation.  

If no objective function is specified, the Pyomo model is used "as is" and
should be defined with "FirstStageCost" and "SecondStageCost"
expressions that are used to build an objective for the two-stage 
stochastic programming problem.  

If the Pyomo model is not written as a two-stage stochastic programming problem in
this format, and/or if the user wants to use an objective that is
different than the original model, a custom objective function can be
defined for parameter estimation.  The objective function arguments
include `model` and `data` and the objective function returns a Pyomo
expression which is used to define "SecondStageCost".  The objective
function can be used to customize data points and weights that are used
in parameter estimation.

Suggested initialization procedure for parameter estimation problems
--------------------------------------------------------------------

To check the quality of initial guess values provided for the fitted parameters, we suggest solving a 
square instance of the problem prior to solving the parameter estimation problem using the following steps:

1. Create :class:`~pyomo.contrib.parmest.parmest.Estimator` object. To initialize the parameter estimation solve from the square problem solution, set optional argument ``solver_options = {bound_push: 1e-8}``.

2. Call :class:`~pyomo.contrib.parmest.parmest.Estimator.objective_at_theta` with optional argument ``(initialize_parmest_model=True)``. Different initial guess values for the fitted parameters can be provided using optional argument `theta_values` (**Pandas Dataframe**)

3. Solve parameter estimation problem by calling :class:`~pyomo.contrib.parmest.parmest.Estimator.theta_est`
