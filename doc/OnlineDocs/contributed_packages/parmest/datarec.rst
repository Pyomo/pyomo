.. _datarecsection:

Data Reconciliation 
====================

The optional argument ``return_values`` in :class:`~pyomo.contrib.parmest.parmest.Estimator.theta_est`
can be used for data reconciliation or to return model values based on the specified objective. 

For data reconciliation, the ``m.unknown_parameters`` is empty
and the objective function is defined to minimize
measurement to model error. Note that the model used for data
reconciliation may differ from the model used for parameter estimation.

The functions
:class:`~pyomo.contrib.parmest.graphics.grouped_boxplot` or
:class:`~pyomo.contrib.parmest.graphics.grouped_violinplot` can be used
to visually compare the original and reconciled data.

The following example from the reactor design subdirectory returns reconciled values for experiment outputs 
(`ca`, `cb`, `cc`, and `cd`) and then uses those values in 
parameter estimation (`k1`, `k2`, and `k3`).

.. literalinclude:: ../../../../pyomo/contrib/parmest/examples/reactor_design/datarec_example.py
   :language: python
   
The following example returns model values from a Pyomo Expression. 

.. doctest::
   :skipif: not ipopt_available or not parmest_available
   
   >>> import pandas as pd   
   >>> import pyomo.contrib.parmest.parmest as parmest
   >>> from pyomo.contrib.parmest.examples.rooney_biegler.rooney_biegler import RooneyBieglerExperiment

   >>> # Generate data
   >>> data = pd.DataFrame(data=[[1,8.3],[2,10.3],[3,19.0],
   ...                           [4,16.0],[5,15.6],[7,19.8]],
   ...                     columns=['hour', 'y'])

   >>> # Create an experiment list
   >>> exp_list = []
   >>> for i in range(data.shape[0]):
   ...     exp_list.append(RooneyBieglerExperiment(data.loc[i, :]))

   >>> # Define objective
   >>> def SSE(model):
   ...     expr = (model.experiment_outputs[model.y]
   ...             - model.response_function[model.experiment_outputs[model.hour]]
   ...            ) ** 2
   ...     return expr

   >>> pest = parmest.Estimator(exp_list, obj_function=SSE, solver_options=None)
   >>> obj, theta, var_values = pest.theta_est(return_values=['response_function'])
   >>> #print(var_values)
