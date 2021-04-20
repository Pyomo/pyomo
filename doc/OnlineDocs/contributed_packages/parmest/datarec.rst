.. _datarecsection:

Data Reconciliation 
====================

The method :class:`~pyomo.contrib.parmest.parmest.Estimator.theta_est`
can optionally return model values.  This feature can be used to return
reconciled data using a user specified objective. In this case, the list
of variable names the user wants to estimate (theta_names) is set to an
empty list and the objective function is defined to minimize
measurement to model error. Note that the model used for data
reconciliation may differ from the model used for parameter estimation.

The following example illustrates the use of parmest for data
reconciliation.  The functions
:class:`~pyomo.contrib.parmest.graphics.grouped_boxplot` or
:class:`~pyomo.contrib.parmest.graphics.grouped_violinplot` can be used
to visually compare the original and reconciled data.

Here's a stylized code snippet showing how box plots might be created:

.. doctest::
    :skipif: True

    >>> import pyomo.contrib.parmest.parmest as parmest
    >>> pest = parmest.Estimator(model_function, data, [], objective_function)
    >>> obj, theta, data_rec = pest.theta_est(return_values=['A', 'B']) 
    >>> parmest.graphics.grouped_boxplot(data, data_rec)

Returned Values
^^^^^^^^^^^^^^^

Here's a full program that can be run to see returned values (in this case it
is the response function that is defined in the model file):

.. doctest::
   :skipif: not ipopt_available or not parmest_available
   
   >>> import pandas as pd   
   >>> import pyomo.contrib.parmest.parmest as parmest
   >>> from pyomo.contrib.parmest.examples.rooney_biegler.rooney_biegler import rooney_biegler_model

   >>> theta_names = ['asymptote', 'rate_constant']

   >>> data = pd.DataFrame(data=[[1,8.3],[2,10.3],[3,19.0],
   ...                           [4,16.0],[5,15.6],[7,19.8]],
   ...                     columns=['hour', 'y'])

   >>> def SSE(model, data):  
   ...     expr = sum((data.y[i]\
   ...                 - model.response_function[data.hour[i]])**2 for i in data.index)
   ...     return expr

   >>> pest = parmest.Estimator(rooney_biegler_model, data, theta_names, SSE,
   ...                          solver_options=None)
   >>> obj, theta, var_values = pest.theta_est(return_values=['response_function'])
   >>> #print(var_values)
   
