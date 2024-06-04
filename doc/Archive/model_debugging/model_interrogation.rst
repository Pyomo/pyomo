Interrogating Pyomo Models
==========================

.. doctest::
   :hide:

   >>> import pyomo.environ as pyo
   >>> from pyomo.opt import SolverFactory
   >>> model = pyo.ConcreteModel()
   >>> model.n = pyo.Param(default=4)
   >>> model.x = pyo.Var(pyo.RangeSet(model.n), within=pyo.Binary)
   >>> def o_rule(model):
   ...    return pyo.summation(model.x)
   >>> model.o = pyo.Objective(rule=o_rule)
   >>> model.c = pyo.Constraint(expr=model.x[2] + model.x[3] >= 1)
   >>> r = SolverFactory('glpk').solve(model)

Show solver output by adding the `tee=True` option when calling the
`solve` function

.. doctest::

   >>> SolverFactory('glpk').solve(model, tee=True) # doctest: +SKIP
   
You can use the `pprint` function to display the model or individual
model components

.. doctest::

   >>> model.pprint() # doctest: +SKIP
   >>> model.x.pprint() # doctest: +SKIP

