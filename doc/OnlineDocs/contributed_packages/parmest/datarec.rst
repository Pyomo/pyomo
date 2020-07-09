.. _datarecsection:

Data Reconciliation 
=================================

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

.. doctest::
    :skipif: True

    >>> import pyomo.contrib.parmest.parmest as parmest
    >>> pest = parmest.Estimator(model_function, data, [], objective_function)
    >>> obj, theta, data_rec = pest.theta_est(return_values=['A', 'B']) 
    >>> parmest.grouped_boxplot(data, data_rec)
