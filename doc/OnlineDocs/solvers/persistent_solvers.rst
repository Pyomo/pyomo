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

The first step in using a persistent solver is to create a Pyomo model
as usual.

>>> import pyomo.environ as pe
>>> m = pe.ConcreteModel()
>>> m.x = pe.Var()
>>> m.y = pe.Var()
>>> m.obj = pe.Objective(expr=m.x**2 + m.y**2)
>>> m.c = pe.Constraint(expr=m.y >= -2*m.x + 5)

You can create an instance of a persistent solver through the SolverFactory.

>>> opt = pe.SolverFactory('gurobi_persistent')

This returns an instance of :py:class:`GurobiPersistent`. Now we need
to tell the solver about our model.

>>> opt.set_instance(m)

This will create a gurobipy Model object and include the appropriate
variables and constraints. We can now solve the model.

>>> results = opt.solve()

We can also add or remove variables, constraints, blocks, and
objectives. For example,

>>> m.c2 = pe.Constraint(expr=m.y >= m.x)
>>> opt.add_constraint(m.c2)

This tells the solver to add one new constraint but otherwise leave
the model unchanged. We can now resolve the model.

>>> results = opt.solve()

To remove a component, simply call the corresponding remove method.

>>> opt.remove_constraint(m.c2)
>>> del m.c2
>>> results = opt.solve()

If a pyomo component is replaced with another component with the same
name, the first component must be removed from the solver. Otherwise,
the solver will have multiple components. For example, the following
code will run without error, but the solver will have an extra
constraint. The solver will have both y >= -2*x + 5 and y <= x, which
is not what was intended!

>>> m = pe.ConcreteModel()
>>> m.x = pe.Var()
>>> m.y = pe.Var()
>>> m.c = pe.Constraint(expr=m.y >= -2*m.x + 5)
>>> opt = pe.SolverFactory('gurobi_persistent')
>>> opt.set_instance(m)
>>> # WRONG:
>>> del m.c
>>> m.c = pe.Constraint(expr=m.y <= m.x)
>>> opt.add_constraint(m.c)

The correct way to do this is:

>>> m = pe.ConcreteModel()
>>> m.x = pe.Var()
>>> m.y = pe.Var()
>>> m.c = pe.Constraint(expr=m.y >= -2*m.x + 5)
>>> opt = pe.SolverFactory('gurobi_persistent')
>>> opt.set_instance(m)
>>> # Correct:
>>> opt.remove_constraint(m.c)
>>> del m.c
>>> m.c = pe.Constraint(expr=m.y <= m.x)
>>> opt.add_constraint(m.c)

.. warning:: Components removed from a pyomo model must be removed
             from the solver instance by the user.

Additionally, unexpected behavior may result if a component is
modified before being removed.

>>> m = pe.ConcreteModel()
>>> m.b = pe.Block()
>>> m.b.x = pe.Var()
>>> m.b.y = pe.Var()
>>> m.b.c = pe.Constraint(expr=m.b.y >= -2*m.b.x + 5)
>>> opt = pe.SolverFactory('gurobi_persistent')
>>> opt.set_instance(m)
>>> m.b.c2 = pe.Constraint(expr=m.b.y <= m.b.x)
>>> # ERROR: The constraint referenced by m.b.c2 does not
>>> # exist in the solver model.
>>> # opt.remove_block(m.b) 

In most cases, the only way to modify a component is to remove it from
the solver instance, modify it with Pyomo, and then add it back to the
solver instance. The only exception is with variables. Variables may
be modified and then updated with with solver:

>>> m = pe.ConcreteModel()
>>> m.x = pe.Var()
>>> m.y = pe.Var()
>>> m.obj = pe.Objective(expr=m.x**2 + m.y**2)
>>> m.c = pe.Constraint(expr=m.y >= -2*m.x + 5)
>>> opt = pe.SolverFactory('gurobi_persistent')
>>> opt.set_instance(m)
>>> m.x.setlb(1.0)
>>> opt.update_var(m.x)
