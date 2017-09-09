Persistent Solvers
==================

The purpose of the persistent solver interfaces is to efficiently
notify the solver of incremental changes to a Pyomo model. The
persistent solver interfaces create and store model instances from the
Python API for the corresponding solver. For example, the
:py:class:`GurobiPersistent` class maintaints a pointer to a gurobipy
Model object. Thus, we can make small changes to the model and notify
the solver rather than recreating the entire model using the solver
Python API (or rewriting an entire model file - e.g., an lp file)
every time the model is solved.

.. warning:: Users are responsible for notifying persistent solver
   interfaces when changes to a model are made!


Using Persistent Solvers
------------------------

.. doctest::
   Create a model
   >>> import pyomo.environ as pe
   >>> m = pe.ConcreteModel()
   >>> m.x = pe.Var()
   >>> m.y = pe.Var()
   >>> m.obj = pe.Objective(expr=m.x**2 + m.y**2)
   >>> m.c = pe.Constraint(expr=m.y >= -2*m.x + 5)

   Create a persistent solver
   >>> opt = pe.SolverFactory('gurobi_persistent')

   This returns an instance of :py:class:`GurobiPersistent`. Now we
   need to tell the solver about our model.

   >>> opt.set_instance(m)

   This will create a gurobipy Model object and include the
   appropriate variables and constraints. We can now solve the model.

   >>> results = opt.solve()

   We can also add or remove variables, constraints, blocks, and
   objectives. For example,

   >>> m.c2 = pe.Constraint(expr=m.y >= m.x)
   >>> opt.add_constraint(m.c2)

   This tells the solver to add one new constraint but otherwise leave
   the model unchanged. We can now resolve the model.

   >>> results = opt.solve()
