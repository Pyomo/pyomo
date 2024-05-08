.. _driversection:

Parameter Estimation 
==================================

Parameter Estimation using parmest requires a Pyomo model, experimental
data which defines multiple scenarios, and parameters
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
created using the following code. A description of each argument is
listed below.  Examples are provided in the :ref:`examplesection`
Section.

.. testsetup:: *
    :skipif: not __import__('pyomo.contrib.parmest.parmest').contrib.parmest.parmest.parmest_available

    # Data
    import pandas as pd
    data = pd.DataFrame(
        data=[[1, 8.3], [2, 10.3], [3, 19.0], 
              [4, 16.0], [5, 15.6], [7, 19.8]],
        columns=['hour', 'y'],
    )

    # Sum of squared error function
    def SSE(model):
        expr = (
            model.experiment_outputs[model.y]
            - model.response_function[model.experiment_outputs[model.hour]]
        ) ** 2
        return expr

    # Create an experiment list
    from pyomo.contrib.parmest.examples.rooney_biegler.rooney_biegler import RooneyBieglerExperiment
    exp_list = []
    for i in range(data.shape[0]):
        exp_list.append(RooneyBieglerExperiment(data.loc[i, :]))

.. doctest::
    :skipif: not __import__('pyomo.contrib.parmest.parmest').contrib.parmest.parmest.parmest_available

    >>> import pyomo.contrib.parmest.parmest as parmest
    >>> pest = parmest.Estimator(exp_list, obj_function=SSE)

Optionally, solver options can be supplied, e.g.,

.. doctest::
    :skipif: not __import__('pyomo.contrib.parmest.parmest').contrib.parmest.parmest.parmest_available

    >>> solver_options = {"max_iter": 6000}
    >>> pest = parmest.Estimator(exp_list, obj_function=SSE, solver_options=solver_options)


List of experiment objects
--------------------------

The first argument is a list of experiment objects which is used to
create one labeled model for each expeirment. 
The template :class:`~pyomo.contrib.parmest.experiment.Experiment` 
can be used to generate a list of experiment objects.

A labeled Pyomo model ``m`` has the following additional suffixes (Pyomo `Suffix`):

* ``m.experiment_outputs`` which defines experiment output (Pyomo `Param`, `Var`, or `Expression`)
  and their associated data values (float, int).
* ``m.unknown_parameters`` which defines the mutable parameters or variables (Pyomo `Param` or `Var`)
  to estimate along with their component unique identifier (Pyomo `ComponentUID`). 
  Within parmest, any parameters that are to be estimated are converted to unfixed variables. 
  Variables that are to be estimated are also unfixed.

The experiment class has one required method:

* :class:`~pyomo.contrib.parmest.experiment.Experiment.get_labeled_model` which returns the labeled Pyomo model.
  Note that the model does not have to be specifically written as a 
  two-stage stochastic programming problem for parmest. 
  That is, parmest can modify the
  objective, see :ref:`ObjFunction` below.
 
Parmest comes with several :ref:`examplesection` that illustrates how to set up the list of experiment objects.
The examples commonly include additional :class:`~pyomo.contrib.parmest.experiment.Experiment` class methods to
create the model, finalize the model, and label the model.  The user can customize methods to suit their needs.

.. _ObjFunction:

Objective function
------------------

The second argument is an optional argument which defines the
optimization objective function to use in parameter estimation.  

If no objective function is specified, the Pyomo model is used "as is" and
should be defined with "FirstStageCost" and "SecondStageCost"
expressions that are used to build an objective for the two-stage 
stochastic programming problem.  

If the Pyomo model is not written as a two-stage stochastic programming problem in
this format, and/or if the user wants to use an objective that is
different than the original model, a custom objective function can be
defined for parameter estimation.  The objective function has a single argument, 
which is the model from a single experiment.
The objective function returns a Pyomo
expression which is used to define "SecondStageCost".  The objective
function can be used to customize data points and weights that are used
in parameter estimation.

Parmest includes one built in objective function to compute the sum of squared errors ("SSE") between the 
``m.experiment_outputs`` model values and data values.

Suggested initialization procedure for parameter estimation problems
--------------------------------------------------------------------

To check the quality of initial guess values provided for the fitted parameters, we suggest solving a 
square instance of the problem prior to solving the parameter estimation problem using the following steps:

1. Create :class:`~pyomo.contrib.parmest.parmest.Estimator` object. To initialize the parameter 
estimation solve from the square problem solution, set optional argument ``solver_options = {bound_push: 1e-8}``.

2. Call :class:`~pyomo.contrib.parmest.parmest.Estimator.objective_at_theta` with optional 
argument ``(initialize_parmest_model=True)``. Different initial guess values for the fitted 
parameters can be provided using optional argument `theta_values` (**Pandas Dataframe**)

3. Solve parameter estimation problem by calling :class:`~pyomo.contrib.parmest.parmest.Estimator.theta_est`
