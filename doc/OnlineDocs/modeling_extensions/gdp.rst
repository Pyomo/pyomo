.. _gdpchapt:

Generalized Disjunctive Programming
===================================

See :ref:`disjunctstart` for an introduction to the Pyomo disjunctive programming
extension. A disjunction is a set of collections of variables, parameters, and 
constraints that are linked by an exclusive OR.

The following models all work and are equivalent:

.. doctest::
   
   Option 1: maximal verbosity, abstract-like

   >>> from pyomo.environ import *
   >>> from pyomo.gdp import *
   >>> model = ConcreteModel()

   >>> model.x = Var()
   >>> model.y = Var()

   >>> # Two conditions
   >>> def _d(disjunct, flag):
   ...    model = disjunct.model()
   ...    if flag:
   ...       # x == 0
   ...       disjunct.c = Constraint(expr=model.x == 0)
   ...    else:
   ...       # y == 0
   ...       disjunct.c = Constraint(expr=model.y == 0)
   >>> model.d = Disjunct([0,1], rule=_d)
 
   >>> # Define the disjunction
   >>> def _c(model):
   ...    return [model.d[0], model.d[1]]
   >>> model.c = Disjunction(rule=_c) 

   Option 2: Maximal verbosity, concrete-like:

   >>> from pyomo.environ import *
   >>> from pyomo.gdp import *
   >>> model = ConcreteModel()
 
   >>> model.x = Var()
   >>> model.y = Var()
 
   >>> model.fix_x = Disjunct()
   >>> model.fix_x.c = Constraint(expr=model.x == 0)
 
   >>> model.fix_y = Disjunct()
   >>> model.fix_y.c = Constraint(expr=model.y == 0)
 
   >>> model.c = Disjunction(expr=[model.fix_x, model.fix_y])
 
   Option 3: Implicit disjuncts (disjunction rule returns a list of expressions or a 
   list of lists of expressions)
 
   >>> from pyomo.environ import *
   >>> from pyomo.gdp import *
   >>> model = ConcreteModel()
 
   >>> model.x = Var()
   >>> model.y = Var()

   >>> model.c = Disjunction(expr=[model.x == 0, model.y == 0])


