Objectives
==========

Overview
--------

Objectives are the mathematical representation of what you wish to minimize or maximize.  In engineering design, objectives are often minimizing cost, material, or time, or alternatively maximizing profit or utility.

The Objective constructor is a very thin wrapper on pyomo ``Objective``, and so experienced pyomo users will not see any significant differences from base pyomo.  


Construction
------------

Objectives are constructed by creating an instance of a new objective in a EDI Formulation 

.. py:function:: f.Objective(expr, sense=minimize)

    Declares an objective in a pyomo.edi.formulation

   :param expr: The expression to be optimized
   :type  expr: pyomo expression
   :param sense: The sense in which the objective should be optimized, either minimized or maximized.  Can import ``minimize`` and ``maximize`` from ``pyomo.environ``, but minimize corresponds to an integer of 1 and maximize to an integer of -1.
   :type  sense: int

   :return: None
   :rtype: None


Relation to Pyomo Objective
---------------------------

The EDI objective constructor is essentially a direct pass through to base pyomo.  Objectives will be added to the ``pyomo.ConcreteModel`` in increasing order with key ``objective_###`` where the the index of the objective appears after the underscore.  First objective is labeled as ``objective_1``, and objective names are never padded with zeros.


Examples
--------

A standard declaration statement
++++++++++++++++++++++++++++++++

.. code-block:: python

    from pyomo.contrib.edi import Formulation
    f = Formulation()
    x = f.Variable(name = 'x', guess = 1.0, units = 'm'  , description = 'The x variable')
    y = f.Variable(name = 'y', guess = 1.0, units = 'm'  , description = 'The y variable')
    c = f.Constant(name = 'c', value = 1.0, units = ''   , description = 'A constant c' )
    f.Objective( c*x + y )  # Default is minimize


With a non-linear objective
+++++++++++++++++++++++++++

.. code-block:: python

    from pyomo.contrib.edi import Formulation
    f = Formulation()
    x = f.Variable(name = 'x', guess = 1.0, units = 'm'  , description = 'The x variable')
    y = f.Variable(name = 'y', guess = 1.0, units = 'm'  , description = 'The y variable')
    c = f.Constant(name = 'c', value = 1.0, units = ''   , description = 'A constant c' )
    f.Objective( c*x**4 + y**4 ) # Default is minimize


Explicitly minimize
+++++++++++++++++++

.. code-block:: python

    from pyomo.contrib.edi import Formulation
    from pyomo.environ import minimize, maximize
    f = Formulation()
    x = f.Variable(name = 'x', guess = 1.0, units = 'm'  , description = 'The x variable')
    y = f.Variable(name = 'y', guess = 1.0, units = 'm'  , description = 'The y variable')
    c = f.Constant(name = 'c', value = 1.0, units = ''   , description = 'A constant c' )
    f.Objective( c*x**4 + y**4 , sense = minimize)


Explicitly minimize using integer
+++++++++++++++++++++++++++++++++

.. code-block:: python

    from pyomo.contrib.edi import Formulation
    from pyomo.environ import minimize, maximize
    f = Formulation()
    x = f.Variable(name = 'x', guess = 1.0, units = 'm'  , description = 'The x variable')
    y = f.Variable(name = 'y', guess = 1.0, units = 'm'  , description = 'The y variable')
    c = f.Constant(name = 'c', value = 1.0, units = ''   , description = 'A constant c' )
    f.Objective( c*x**4 + y**4 , sense = 1 ) # 1 corresponds to minimize


Maximizing
++++++++++

.. code-block:: python

    from pyomo.contrib.edi import Formulation
    from pyomo.environ import minimize, maximize
    f = Formulation()
    x = f.Variable(name = 'x', guess = 1.0, units = 'm'  , description = 'The x variable')
    y = f.Variable(name = 'y', guess = 1.0, units = 'm'  , description = 'The y variable')
    c = f.Constant(name = 'c', value = 1.0, units = ''   , description = 'A constant c' )
    f.Objective( -c*x**4 - y**4 , sense = maximize)


Maximizing using integer
++++++++++++++++++++++++

.. code-block:: python

    from pyomo.contrib.edi import Formulation
    from pyomo.environ import minimize, maximize
    f = Formulation()
    x = f.Variable(name = 'x', guess = 1.0, units = 'm'  , description = 'The x variable')
    y = f.Variable(name = 'y', guess = 1.0, units = 'm'  , description = 'The y variable')
    c = f.Constant(name = 'c', value = 1.0, units = ''   , description = 'A constant c' )
    f.Objective( -c*x**4 - y**4 , sense = -1 ) # -1 corresponds to maximize


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



Tips
----

* Objectives are a pretty natural place to break your file.  Put at least one blank line above and below the objective constructor and use good sectioning to create a whitespace easily identifiable when scrolling quickly 



