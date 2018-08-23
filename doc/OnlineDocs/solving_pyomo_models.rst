Solving Pyomo Models
====================

.. doctest::
   :hide:

   >>> import pyomo.environ as pyo
   >>> from pyomo.opt import SolverFactory
   >>> m = pyo.AbstractModel()
   >>> m.n = pyo.Param(default=4)
   >>> m.x = pyo.Var(pyo.RangeSet(m.n), within=pyo.Binary)
   >>> def o_rule(m):
   >>>    return pyo.summation(m.x)
   >>> m.o = pyo.Objective(rule=o_rule)
   >>> m.c = pyo.ConstraintList()
   >>> model = m.create_instance()


Solving ConcreteModels
----------------------
If you have a ConcreteModel, add these lines at the bottom of your Python script to solve it

.. doctest::

   >>> opt = pyo.SolverFactory('glpk')
   >>> opt.solve(model)

Solving AbstractModels
----------------------
If you have an AbstractModel, you must create a concrete instance of your model before solving it using the same lines as above:

.. doctest::
   :hide:

   >>> model = m

.. doctest::

   >>> instance = model.create_instance()
   >>> opt = pyo.SolverFactory('glpk')
   >>> opt.solve(instance)

Supported Solvers
-----------------
Pyomo supports any solvers that read `.lp` or `.nl` files and also includes special interfaces to a few others.

