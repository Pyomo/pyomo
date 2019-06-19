.. _driversection:

Parameter Estimation using parmest
=======================================

Parameter Estimation using parmest requires a Pyomo model, experimental data which defines 
multiple scenarios, and a list of thetas to estimate. 
parmest uses PySP [PyomoBookII]_ to solve a two-stage stochastic programming 
problem, where the experimental data is used to create a scenario tree.
The objective function needs to be written in PySP form with the 
Pyomo Expression for first stage cost (named "FirstStateCost") set to zero and the 
Pyomo Expression for second stage cost (named "SecondStageCost") defined as the 
deviation between model and the observations (typically defined as
the sum of squared deviation between
model values and observed values).

If the Pyomo model is not formatted as a two-stage stochastic programming 
problem in this format, the user can supply a custom function to use as the second stage cost
and the Pyomo model will be modified within parmest to match the specifications required by PySP.
The PySP callback function is also defined within parmest.
The callback function returns a populated 
and initialized model for each scenario.

To use parmest, the user creates a :class:`~pyomo.contrib.parmest.parmest.Estimator` object 
and uses its methods for:

* Parameter estimation, :class:`~pyomo.contrib.parmest.parmest.Estimator.theta_est`
* Bootstrap resampling for parameter estimation, :class:`~pyomo.contrib.parmest.parmest.Estimator.theta_est_bootstrap`
* Compute the objective at theta values, :class:`~pyomo.contrib.parmest.parmest.Estimator.objective_at_theta`
* Compute likelihood ratio, :class:`~pyomo.contrib.parmest.parmest.Estimator.likelihood_ratio_test`

A :class:`~pyomo.contrib.parmest.parmest.Estimator` object can be created using 
the following code. A description of each argument is listed below.  Examples are provided in the :ref:`examplesection` Section.

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
 

Model function
----------------
The first argument is a function which uses data for a single scenario to return a 
populated and initialized Pyomo model for that scenario.
Parameters that the user would like to estimate must be defined as variables (Pyomo `Var`).
The variables can be fixed (parmest unfixes variables that will be estimated). 
The model does not have to be specifically written for parmest. That is, parmest can modify the objective for pySP, see :ref:`ObjFunction` below.

Data
-----------------------

The second argument is the data which will be used to populate the Pyomo model.  
Supported data formats include:

* **Pandas Dataframe** where each row is a separate scenario and column names refer to observed quantities. 
  Pandas DataFrames are easily stored and read in from csv, excel, or databases, or created directly in Python.
* **List of dictionaries** where each entry in the list is a separate scenario and the keys (or nested keys) 
  refer to observed quantities.  
  Dictionaries are often preferred over DataFrames when using static and time series data.  
  Dictionaries are easily stored and read in from json or yaml files, or created directly in Python.
* **List of json file names** where each entry in the list contains a json file name for a separate scenario.
  This format is recommended when using large datasets in parallel computing.

The data must be compatible with the model function that returns a populated and initialized Pyomo model for a 
single scenario.
Data can include multiple entries per variable (time series and/or duplicate sensors).  
This information can be included in custom objective functions, see :ref:`ObjFunction` below.

Theta names
-----------------------

The third argument is a list of variable names that the user wants to estimate. 
The list contains strings with `Var` names from the Pyomo model.

.. _ObjFunction:

Objective function 
-----------------------------

The forth argument is an optional argument which defines the optimization objective function to use in 
parameter estimation.
If no objective function is specified, the Pyomo model is used 
"as is" and should be defined with a "FirstStateCost" and 
"SecondStageCost" expression that are used to build an objective 
for PySP.
If the Pyomo model is not written as a two stage stochastic programming problem in this format, 
and/or if the user wants to use an objective that is different than the original model, 
a custom objective function can be defined for parameter estimation.
The objective function arguments include `model` and `data` and the objective function returns 
a Pyomo expression which are used to define "SecondStageCost".
The objective function can be used to customize data points and weights that are used in parameter estimation.
