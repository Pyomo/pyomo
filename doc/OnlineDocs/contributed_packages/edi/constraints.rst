Constraints
===========


Overview
--------

Constraints are the mathematical representation of rules that are imposed on your decisions/variables when minimizing or maximizing.  In engineering design, constraints are often imposed by physics or operational limits.

The Constraint constructor is a very thin wrapper on pyomo ``Constraint``, and so experienced pyomo users will not see any significant differences from base pyomo.  


Construction
------------

Constraints are constructed by creating an instance of a new constraint in a EDI Formulation 

.. py:function:: f.Constraint(expr)

    Declares a constraint in a pyomo.edi.formulation

   :param expr: The expression representing the constraint
   :type  expr: pyomo expression

   :return: None
   :rtype: None


However, the expected use case is the ``f.ConstraintList()`` function:

.. py:function:: f.ConstraintList(conList)

    Declares new constraints in a pyomo.edi.formulation from a list of inputs

   :param conList: The list of constraints to be generated.  Entries will be pyomo expressions, or lists/tuples/dicts that are used to create RuntimeConstraints (see :doc:`here <./blackboxconstraints>`)
   :type  conList: list

   :return: None
   :rtype: None


Relation to Pyomo Constraint
----------------------------

The EDI constraint constructor is essentially a direct pass through to base pyomo.  Constraints will be added to the ``pyomo.ConcreteModel`` in increasing order with key ``constraint_###`` where the the index of the objective appears after the underscore.  First constraint is labeled as ``constraint_1``, and constraint names are never padded with zeros.  RuntimeConstraints also contribute to this counter.


Examples
--------

A standard declaration statement
++++++++++++++++++++++++++++++++

.. code-block:: python

    from pyomo.environ import units
    from pyomo.contrib.edi import Formulation
    f = Formulation()
    x = f.Variable(name = 'x', guess = 1.0, units = 'm'  , description = 'The x variable')
    y = f.Variable(name = 'y', guess = 1.0, units = 'm'  , description = 'The y variable')
    c = f.Constant(name = 'c', value = 1.0, units = ''   , description = 'A constant c' )
    f.Objective( c*x + y )
    f.ConstraintList(
        [
            x**2 + y**2 <= 1.0*units.m**2 ,
            x <= 0.75*units.m,
            x >= y ,
        ]
    )


With the core constructor
+++++++++++++++++++++++++

.. code-block:: python

    from pyomo.environ import units
    from pyomo.contrib.edi import Formulation
    f = Formulation()
    x = f.Variable(name = 'x', guess = 1.0, units = 'm'  , description = 'The x variable')
    y = f.Variable(name = 'y', guess = 1.0, units = 'm'  , description = 'The y variable')
    c = f.Constant(name = 'c', value = 1.0, units = ''   , description = 'A constant c' )
    f.Objective(  c*x + y )
    f.Constraint( x**2 + y**2 <= 1.0*units.m**2 )
    f.Constraint( x <= 0.75*units.m )
    f.Constraint( x >= y )


Using indexed variables and constants
+++++++++++++++++++++++++++++++++++++

.. code-block:: python

    from pyomo.contrib.edi import Formulation
    f = Formulation()
    x = f.Variable( name = 'x', 
                    guess = 1.0, 
                    units = 'm' , 
                    description = 'The x variable',
                    bounds = [0,100] ,
                    size = 3    )
    y = f.Variable( name = 'y', 
                    guess = 1.0, 
                    units = 'm' , 
                    description = 'The y variable',
                    size = [2,2])
    c = f.Constant( name = 'c', 
                    value = 1.0, 
                    units = ''  , 
                    description = 'A constant c'  ,
                    size = 3    )
    f.Objective( c[0]*x[0] + c[1]*x[1] + c[2]*x[2] 
                 + y[0,0]**4
                 + y[0,1]**4
                 + y[1,0]**4
                 + y[1,1]**4 ) # Default is minimize
    f.ConstraintList(
        [
            x[0]**2 + x[1]**2 + x[2]**2 <= 1.0*units.m,
            y[0,0] >= 1.0*units.m,
            y[0,1] >= 1.0*units.m,
            y[1,0] >= 1.0*units.m,
            y[1,1] >= 1.0*units.m,
            x[0] >= y[0,0] ,
        ]
    )


Tips
----

* For the typical user, constraints should always be constructed using the ``f.ConstraintList()`` function to produce a cleaner input file that is easier to modify


Known Issues
------------

* Indexed variables must be broken up using either indices or a pyomo rule (see `this issue <https://github.com/codykarcher/pyomo/issues/3>`__)
* Units that are inconsistent, but not the same (ie, meters and feet) will flag as invalid when checking units (see `this issue <https://github.com/codykarcher/pyomo/issues/6>`__)
