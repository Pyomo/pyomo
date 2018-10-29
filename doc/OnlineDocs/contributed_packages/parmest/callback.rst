.. _callbacksection:

Callback Function
==================

The modeller must provide a function that instantiates
the model for each experiment, which is more-or-less the same function
that needs to be supplied to PySP if the modeler chooses
to use PySP directly for parameter estimation and eschew the
use of parmest. There are two options for the function signature:

* parmest Signature
* PySP Signature

pamest Signature
------------------

The parmest signature is recommenced, unless you are planning to use PySP directly
The parmest signature is shown below:

.. doctest::
   
   >>> def instance_creation_callback(experiment_number = None, cb_data = None):
   ...    pass
.. doctest::
   :hide:

    ...    pass

Any function name is allowed; the name is supplied to the parmest constructor.
Use of data in the `cb_data` object is optional, but it must appear in the
callback function definition. To supply data via the `cb_data` argument,
an object with the data should be supplied to the parmest constructor. 
If no data is supplied, the `cb_data` argument will have the value `None`.


PySP Signature
----------------------

This signature matches that expected by PySP if it is used directly. It
is recommended only when using PySP directly for parameter estimation. 
The parmest signature is shown below:

.. doctest::
   
   >>> def pysp_instance_creation_callback(scenario_tree_model, scenario_name, node_names):
    ...    pass
   
.. doctest::
   :hide:

    ...     pass
	
Any function name is allowed; the name is supplied to the parmest constructor.

The experiments are assumed to be numbered and the experiment numbers
will be placed by parmest as the last part of the `scenario_name`
that is passed into the callback. It is good practice to 

.. doctest::
   
   >>> import re

and then start the callback function with a line like

.. doctest::
   :hide:

    >>> scenario_name = "Scenario1"

.. doctest::

   >>> i = int(re.compile(r'(\d+)$').search(scenario_name).group(1))

which will place the experiment number in the Python variable `i`.

.. _objective:

Objective Requirements
-------------------------
The modeller must provide objective Expressions that are the same as those 
that need to be used if the modeler chooses to use PySP directly for parameter 
estimation and eschew the use of parmest. These Expressions are created using 
the Pyomo `Expression` component and are part of the model that is often in the 
same file as the callback function.

The first stage cost expression should simply be zero and should be called FirstStageCost. 
Although any name could be used for the second stage expression, the path of least 
resistance is to call it SecondStageCost. This expression just computes the sum of the 
squared deviation of observations from model computed Var values.

.. _cb_data:

Data from Experiments
---------------------------------

The writer of the callback can obtain scenario data in any manner that
is appropriate. For many applications, we anticipate that it will come
from a database of some sort; however, for test applications, the
experiment data will often be assigned to an object in Python variables in the
global namespace of the py file with the callback function. In other
applications, the callback may rely on data passed to the 
:class:`~pyomo.contrib.parmest.parmest.ParmEstimator` object.

The numbering given to the experiments must be used by the 
:class:`~pyomo.contrib.parmest.parmest.ParmEstimator` object; 
see :ref:`NumbersList` for more information.

