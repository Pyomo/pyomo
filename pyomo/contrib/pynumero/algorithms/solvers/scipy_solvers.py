from pyomo.core.base.objective import Objective
from pyomo.common.modeling import unique_component_name
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP

import scipy as sp


class ScipyRootSolver(object):

    def solve(model):

        active_objs = list(model.component_data_objects(Objective, active=True))
        if len(active_objs) == 0:
            obj_name = unique_component_name(model, "_obj")
            obj = pyo.Objective(expr=0.0)
            model.add_component(name, obj)

        nlp = PyomoNLP(model)

        if len(active_objs) == 0:
            model.del_component(obj_name)
        
        # Call to solve(nlp)

        # Transfer values back to Pyomo model


class ScipySquareNlpSolver(object):

    def __init__(self, nlp):
        self._nlp = nlp
        # TODO: Make sure nlp only has equality constraints

    def solve(self):

        sp.optimize.fsolve(
            self.evaluate_function,
            x0,
            fprime=self.evaluate_jacobian,
        )

    def evaluate_function(self, x0):
        self._nlp.set_primals(x0)
        return self._nlp.evaluate_constraints_eq()

    def evaluate_jacobian(self, x0):
        self._nlp.set_primals(x0)
        return self._nlp.evaluate_jacobian()
