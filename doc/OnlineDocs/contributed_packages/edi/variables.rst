Variables
=========

Overview
--------
Variables are the mathematical representation of individual decisions being considered by the optimizer.  In engineering design, variables are often representations of geometric parameters and operating conditions.  

Using the EDI package, variables can be defined as both scalar (``pyomo.core.base.var.ScalarVar``) and vector/matrix/tensor (``pyomo.core.base.var.IndexedVar``), and can exist in many mathematical spaces (All Real, Integers, etc).

The Variable constructor is a very thin wrapper, and so experienced pyomo users will not see any significant differences from base pyomo.  


Construction
------------

Variables are constructed by 1) creating an instance of a new variable in a EDI Formulation and 2) passing out this newly constructed variable to be used in objective and constraint construction.  

.. py:function:: f.Variable(name, guess, units, description='', size=None, bounds=None, domain=None)

    Declares a variable in a pyomo.edi.formulation

   :param name: The name of the variable for the purposes of tracking in the formulation.  Commonly, this will be the same as the variable name in local namespace.
   :type  name: str
   :param guess: The initial guess of the variable.  For scalar variables, this should be a valid float or int for the specified domain.  For vector variables, this will most often also be a single float or int, but a dictionary of index-value pairs is also accepted as in accordance with base pyomo.  Numpy arrays will be supported in a future release (see `this issue <https://github.com/codykarcher/pyomo/issues/1>`_)
   :type  guess: float or int or dict
   :param units: The units of the variable.  Every entry in a vector variable must have the same units.  Entries of '', ' ', '-', 'None', and 'dimensionless' all become units.dimensionless
   :type  units: str or pyomo.core.base.units_container._PyomoUnit
   :param description: A description of the variable
   :type  description: str
   :param size: The size (or shape) of the variable.  Entries of 0, 1, and None all correspond to scalar variables.  Other integers correspond to vector variables.  Matrix and tensor variable are declared using lists of ints, ex: [10,10].  Matrix and tensor variables with a dimension of 1 (ie, [10,10,1]) will be rejected as the extra dimension holds no meaningful value.  
   :type  size: int or list
   :param bounds: The bounds on the variable.  A list or tuple of two elements [lower_bound, upper_bound] where the two bounds are assumed to be either ints or floats.  WARNING:  User is currently responsible for ensuring the units are correct (see `this issue <https://github.com/codykarcher/pyomo/issues/8>`__)
   :type  bounds: list or tuple
   :param domain: The domain of the variable (ex: Reals, Integers, etc).  Default of None constructs a variable in Reals.
   :type  domain: pyomo set

   :return: The variable that was declared in the formulation
   :rtype: pyomo.core.base.var.ScalarVar or pyomo.core.base.var.IndexedVar


Relation to Pyomo Var
---------------------

The fields: name, domain, and bounds are directly passed to the pyomo ``Var`` constructor, with some minor checking.  The guess field is passed to initalize.  The description field is passed to the doc field in the pyomo ``Var``.  Units are passed directly with an additional check.

Non-scalar variables are constructed using pyomo ``Sets``.  Sets are constructed to be integer sets that fill the entire interval from lower bound to upper bound, ie a vector variable of length 5 would create a pyomo ``Set`` with valid indices [0,1,2,3,4] with no skips.  In this way, non-scalar constatants are slightly less flexible than general non-scalar pyomo ``Params``.


Examples
--------


A standard declaration statement
++++++++++++++++++++++++++++++++

.. code-block:: python

   from pyomo.contrib.edi import Formulation
   f = Formulation()
   x = f.Variable(name = 'x', guess = 1.0, units = 'm' , description = 'The x variable')


Shortest possible declaration
+++++++++++++++++++++++++++++

.. code-block:: python

   from pyomo.contrib.edi import Formulation
   f = Formulation()
   x = f.Variable('x', 1.0, 'm' )


A variable with bounds
++++++++++++++++++++++

.. code-block:: python

   from pyomo.contrib.edi import Formulation
   f = Formulation()
   x = f.Variable( name = 'x', 
                   guess = 1.0, 
                   units = 'm' , 
                   description = 'The x variable', 
                   bounds = [-10,10] )


An integer variable
+++++++++++++++++++

.. code-block:: python

   from pyomo.contrib.edi import Formulation
   from pyomo.environ import Integers
   f = Formulation()
   x = f.Variable( name = 'x', 
                   guess = 1.0, 
                   units = 'm' , 
                   description = 'The x variable', 
                   domain = Integers )


An alternative units definition
+++++++++++++++++++++++++++++++

.. code-block:: python

   from pyomo.environ import units
   from pyomo.contrib.edi import Formulation
   f = Formulation()
   x = f.Variable( name = 'x', 
                   guess = 1.0, 
                   units = units.m , 
                   description = 'The x variable' )


A vector variable
+++++++++++++++++

.. code-block:: python

   from pyomo.environ import units
   from pyomo.contrib.edi import Formulation
   f = Formulation()
   x = f.Variable( name = 'x', 
                   guess = 1.0, 
                   units = 'm' , 
                   description = 'The x variable', 
                   size = 5 )


A matrix/tensor variable
++++++++++++++++++++++++

.. code-block:: python

   from pyomo.environ import units
   from pyomo.contrib.edi import Formulation
   f = Formulation()
   x = f.Variable( name = 'x', 
                   guess = 1.0, 
                   units = 'm' , 
                   description = 'The x variable', 
                   size = [10,2] )


More complicated units definition
+++++++++++++++++++++++++++++++++

.. code-block:: python

   from pyomo.environ import units
   from pyomo.contrib.edi import Formulation
   f = Formulation()
   x = f.Variable( name = 'x', 
                   guess = 1.0, 
                   units = 'kg*m/s**2' , 
                   description = 'The x variable' )


Tips
----

* Declare variables in alphabetical order.  Trust me.  It's a pain at first, but it saves a huge amount of time down the road, especially for large models.
* Designate a section in your file for variable declarations, as is done in the :doc:`introductory example <./quickstart>`
* Align all of your variable declarations in a pretty, grid like fashion.  Again, a pain at first, but a big time saver
* Use the keyword names during variable declarations.  Takes extra space, but is a massive boost to readability and intrepretability
* Declare one variable on one single line with no breaks, no matter what style guides tell you.  Again, this is a significant boost to readability
* Do not skimp out on the description field, it is extremely helpful





