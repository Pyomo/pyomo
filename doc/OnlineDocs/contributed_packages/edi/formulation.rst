Formulation
===========

The core object in EDI is called a *Formulation*.  For experienced pyomo users, a Formulation inherits from a pyomo *ConcreteModel*, and can therefore be treated exactly as a typical *ConcreteModel* with a few additional features.  

Each modeling element (ex: Variable, Constant, Objective, and Constraint) has a constructor that is used to create the corresponding element in the *Formulation* instance.  In addition, there are a number of helper functions that collect and return model elements or perform supporting actions.

Construction
------------
A *Formulation* is constructed as follows:

.. code-block:: python
  :linenos:

  from pyomo.contrib.edi import Formulation
  f = Formulation()

Standard practice is to construct a formulation to namespace variable ``f``, but any valid python name can be used.  Standard pyomo practice would be to construct this to ``model`` or ``m``.


Declaring Variables
-------------------

See the :doc:`Variables <./variables>` Documentation

Variables are declared using the ``f.Variable()`` function.  This function creates an instance of ``pyomo.environ.Var`` and adds it to the ``edi.contrib.Formulation``.  The function returns an instance of a ``pyomo.environ.Var`` that can be used in later construction.

.. code-block:: python
  :linenos:

  from pyomo.contrib.edi import Formulation
  f = Formulation()
  x = f.Variable(name = 'x', guess = 1.0, units = '')



Declaring Constants
-------------------

See the :doc:`Constants <./constants>` Documentation

Constants (referred to in base pyomo as parameters or ``Params``) are declared using the ``f.Constant()`` function.  This function creates an instance of a ``pyomo.environ.Param`` and adds it to the ``edi.contrib.Formulation``.  This function also returns an instance of ``pyomo.environ.Param`` that can be used in later construction.

.. code-block:: python
  :linenos:

  from pyomo.contrib.edi import Formulation
  f = Formulation()
  c = f.Constant(name = 'c', value = 1.0, units = '')


Declaring Objectives
--------------------

See the :doc:`Objectives <./objectives>` Documentation

Objectives are declared using the ``f.Objective()`` function.  This function creates an instance of a ``pyomo.environ.Objective`` and adds it to the ``edi.contrib.Formulation``.  Multiple objectives can be declared, but interpretation of multiple objectives will depend on the solver.  The returned values of the ``f.Variable()`` and ``f.Constant()`` declarations can be used to construct the objective.  Black-box (ie, Runtime) objectives are not supported at this time, but are planned in a future update.

.. code-block:: python
  :linenos:

  from pyomo.contrib.edi import Formulation
  f = Formulation()
  x = f.Variable(name = 'x', guess = 1.0, units = '')
  y = f.Variable(name = 'y', guess = 1.0, units = '')
  c = f.Constant(name = 'c', value = 1.0, units = '')
  f.Objective( c*x + y )


By default, objectives are minimized, but can be switched to a maximize using the ``sense`` keyword from pyomo:

.. code-block:: python
  :linenos:

  from pyomo.contrib.edi import Formulation
  from pyomo.environ import maximize, minimize
  f = Formulation()
  x = f.Variable(name = 'x', guess = 1.0, units = '')
  y = f.Variable(name = 'y', guess = 1.0, units = '')
  c = f.Constant(name = 'c', value = 1.0, units = '')
  f.Objective( c*x + y , sense = maximize)


Note:  Future version will allow a string to be passed into ``sense`` (see `this issue <https://github.com/codykarcher/pyomo/issues/7>`_)



Declaring Constraints
---------------------

See the :doc:`Constraints <./constraints>` Documentation

Constraints can be declared in two ways.  First is using the standard ``f.Constraint()`` constructor.  This function creates an instance of ``pyomo.environ.Constraint`` and adds it to the ``edi.contrib.Formulation``.  The operators ``<=``, ``>=``, and ``==`` are used as constraint constructors.

.. code-block:: python
  :linenos:

  from pyomo.contrib.edi import Formulation
  f = Formulation()
  x = f.Variable(name = 'x', guess = 1.0, units = '')
  y = f.Variable(name = 'y', guess = 1.0, units = '')
  c = f.Constant(name = 'c', value = 1.0, units = '')
  f.Objective( c*x + y )
  f.Constraint( x**2 + y**2 <= 1.0 )
  f.Constraint( x >= 0)
  f.Constraint( y <= 0)

Constraints can also be declared using the ``f.ConstraintList()`` function.  This function takes in a list of constraints and allows for multiple constraints to be declared in one go:

.. code-block:: python
  :linenos:

  from pyomo.contrib.edi import Formulation
  f = Formulation()
  x = f.Variable(name = 'x', guess = 1.0, units = '')
  y = f.Variable(name = 'y', guess = 1.0, units = '')
  c = f.Constant(name = 'c', value = 1.0, units = '')
  f.Objective( c*x + y )
  f.ConstraintList( 
      [
          x**2 + y**2 <= 1.0 ,
          x >= 0,
          y <= 0,
      ]
  )

The constraint list can also be declared a priori and passed in to the ``f.ConstraintList()`` function, which may be beneficial for complex models:

.. code-block:: python
  :linenos:

  from pyomo.contrib.edi import Formulation
  f = Formulation()
  x = f.Variable(name = 'x', guess = 1.0, units = '')
  y = f.Variable(name = 'y', guess = 1.0, units = '')
  c = f.Constant(name = 'c', value = 1.0, units = '')
  f.Objective( c*x + y )

  constraintList = [
      x**2 + y**2 <= 1.0 ,
      x >= 0,
      y <= 0,
  ]

  f.ConstraintList( constraintList )

We recommend that most users should be using the ``f.ConstraintList()`` function, with the ``f.Constraint()`` function being reserved for under-the-hood usage.


Declaring Black-Box Constraints
-------------------------------

See the :doc:`Black-Box Constraints <./blackboxconstraints>` Documentation


Support Functions
-----------------

.. |br| raw:: html

   <br />


``f.get_variables()``  |br|
Returns a list variables that have been defined in the formulation in declaration order.  Will only return variables defined via EDI.

``f.get_constants()``  |br|
Returns a list of constants that have been defined in the formulation in declaration order.  Will only return constants/parameters defined via EDI.

``f.get_objectives()``  |br|
Returns a list of objectives that have been defined in the formulation in declaration order.  Will only return objectives defined via EDI.

``f.get_constraints()``  |br|
Returns a list of constraints that have been defined in the formulation in declaration order.  This command returns a list that includes both explicit and runtime (black-box) constraints, but only constraints that have been defined via EDI.

``f.get_explicitConstraints()``  |br|
Returns a list of *explicit* constraints that have been defined in the formulation in declaration order.  This command returns a list that includes *only* the explicit constraints and *not* the runtime (black-box) constraints.  Only includes constraints that have been defined via EDI.

``f.get_runtimeConstraints()``  |br|
Returns a list of *runtime* (ie. black-box) constraints that have been defined in the formulation in declaration order.  This command returns a list that includes *only* the runtime constraints and *not* the explicit constraints.  Only includes constraints that have been defined via EDI.

``f.check_units()``  |br|
Checks the units of each objective and constraint for consistency.  Will only check objectives and constraints defined via EDI.












