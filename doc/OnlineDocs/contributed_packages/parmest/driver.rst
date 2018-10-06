.. _driversection:

Parameter Estimation using parmest
=======================================

To use parmest, the user writes a Python code
that creates a :class:`~pyomo.contrib.parmest.parmest.ParmEstimator` object and uses its methods for parameter
estimation, confidence region estimation, and scenario creation. Once
a parameter estimator object is created, one typically wants to call member
function such as :class:`~pyomo.contrib.parmest.parmest.ParmEstimator.theta_est`, 
:class:`~pyomo.contrib.parmest.parmest.ParmEstimator.bootstrap` and 
:class:`~pyomo.contrib.parmest.parmest.ParmEstimator.likelihood_ratio`. Examples
are provided in the :ref:`examplesection` Section.

A :class:`~pyomo.contrib.parmest.parmest.ParmEstimator` object can be created using 
the following code. A description of each argument is listed below.

.. testsetup:: *

    theta_list = ['asymptote', 'rate_constant']
    num_samples = 6
    exp_numbers_list = range(self.num_samples)
    model_file = \
            "pyomo.contrib.parmest.examples.rooney_biegler.rooney_biegler"
    callback = "instance_creation_callback"
    cb_data = pd.DataFrame(data=[[1,8.3],[2,10.3],[3,19.0],
                                   [4,16.0],[5,15.6],[6,19.8]],
                                   columns=['hour', 'y'])


..doctest::

    >>> import pyomo.contrib.parmest.parmest as parmest
    >>> pest = parmest.ParmEstimator(model_file, callback,
    ...                              cost_expression, exp_numbers_list,
    ...                              theta_list, cb_data=cb_data)
 
.. _CallbackSpec:

Callback
-----------------------

The first two arguments (`model_file` and `callback`) specify the model and location of the callback
function. For more information about what the callback function 
does, see :ref:`callbacksection`.
The second argument can be either a string (a Python `str`)
giving the function name, or the function name itself (not a Python
`str`, but the name of the function, which has the Python type
`func`). If the function is given as a string, then the first
argument gives the module either as a string or as a module name (not
a string).  If the second argument gives the callback function as a
function (not as a string), then the first argument is ignored.
For an example of the callback function given as a function name (not a string), see :ref:`AllInOne`.

Cost Expression
-----------------------

The third argument (`cost_expression`) of the constructor call is a string with the name of the `Expression` in the
Pyomo model that has the least squares objective. This will often be given as "SecondStageCost"
in the model. See :ref:`objective` for more information.

.. _NumbersList:

Experiment Numbers
-----------------------

The fourth argument (`exp_numbers_list`) is a
list of experiment numbers in the form of a Python list. These numbers
can be any set of unique, non-negative numbers. See :ref:`cb_data` for
information about how these numbers can be used in the scenario creation
callback.

Theta
-----------------------

The fifth argument (`theta_list`) is a list of strings with `Var` names
from the Pyomo model that are the parameters to be estimated.

Callback Data
-----------------------

The sixth argument (`cb_data`) is an optional argument which supplies data to the callback.
See :ref:`cb_data` for more information. 
