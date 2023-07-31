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


Declaring Constants
-------------------

See the :doc:`Constants <./constants>` Documentation



Declaring Objectives
--------------------

See the :doc:`Objectives <./objectives>` Documentation


Declaring Constraints
---------------------

See the :doc:`Constraints <./constraints>` Documentation



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












