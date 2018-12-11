Solving Pyomo Models
====================

.. doctest::
   :hide:

   >>> import pyomo.environ as pyo

   >>> m = pyo.AbstractModel()
   >>> m.n = pyo.Param(default=4)
   >>> m.x = pyo.Var(pyo.RangeSet(m.n), within=pyo.Binary)
   >>> def o_rule(m):
   ...    return pyo.summation(m.x)
   >>> m.o = pyo.Objective(rule=o_rule)

   >>> model = m.create_instance()
   >>> model.c = pyo.Constraint(expr=model.x[2]+model.x[3]>=1)


Solving ConcreteModels
----------------------

If you have a ConcreteModel, add these lines at the bottom of your
Python script to solve it

.. doctest::

   >>> opt = pyo.SolverFactory('glpk')
   >>> opt.solve(model) # doctest: +SKIP

Solving AbstractModels
----------------------

If you have an AbstractModel, you must create a concrete instance of
your model before solving it using the same lines as above:

.. doctest::
   :hide:

   >>> model = m

.. doctest::

   >>> instance = model.create_instance()
   >>> opt = pyo.SolverFactory('glpk')
   >>> opt.solve(instance) # doctest: +SKIP

``pyomo solve`` Command
-----------------------

To solve a ConcreteModel contained in the file ``my_model.py`` using the
``pyomo`` command and the solver GLPK, use the following line in a
terminal window::

   pyomo solve my_model.py --solver='glpk'

To solve an AbstractModel contained in the file ``my_model.py`` with data
in the file ``my_data.dat`` using the ``pyomo`` command and the solver GLPK,
use the following line in a terminal window::

   pyomo solve my_model.py my_data.dat --solver='glpk'

Supported Solvers
-----------------

Pyomo supports a wide variety of solvers.  Pyomo has specialized
interfaces to some solvers (for example, BARON, CBC, CPLEX, and Gurobi).
It also has generic interfaces that support calling any solver that can
read AMPL "``.nl``" and write "``.sol``" files and the ability to
generate GAMS-format models and retrieve the results.  You can get the
current list of supported solvers using the ``pyomo`` command::

   pyomo help --solvers
