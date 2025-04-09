z3 SMT Sat Solver Interface
===========================

The z3 Satisfiability Solver interface can convert pyomo variables and expressions for
use with the z3 Satisfiability Solver

Installation
------------
z3 is required for use of the Sat Solver can be installed via the command

.. code::

    pip install z3-solver

Using z3 Sat Solver
-------------------
To use the sat solver define your pyomo model as usual:

.. doctest::

  Required import
  >>> import pyomo.environ as pyo
  >>> from pyomo.contrib.satsolver.satsolver import SMTSatSolver

  Create a simple model
  >>> m = pyo.ConcreteModel()
  >>> m.x = pyo.Var()
  >>> m.y = pyo.Var()
  >>> m.obj = pyo.Objective(expr=m.x**2 + m.y**2)
  >>> m.c = pyo.Constraint(expr=m.y >= -2*m.x + 5)

  Invoke the sat solver using optional argument model to automatically process
  pyomo model
  >>> is_feasible = SMTSatSolver(model = m).check()# doctest: +SKIP
