#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

from pyomo.core import ConcreteModel, Param, Var, Expression, Objective, Constraint, Piecewise
from pyomo.solvers.tests.models.base import _BaseTestModel, register_model


@register_model
class LP_piecewise(_BaseTestModel):
    """
    A continuous linear model
    """

    description = "LP_piecewise"
    capabilities = set(['linear'])

    def __init__(self):
        _BaseTestModel.__init__(self)
        self.add_results(self.description+".json")

    def _generate_model(self):
        self.model = ConcreteModel()
        model = self.model
        model._name = self.description

        model.x = Var()
        model.y = Var()

        model.obj = Objective(expr=model.y)
        model.p = Piecewise(model.y, model.x,
                            pw_pts=[-1,0,1],
                            f_rule=[1,0.5,1],
                            pw_repn='SOS2',
                            pw_constr_type='LB',
                            unbounded_domain_var=True)

    def warmstart_model(self):
        assert self.model is not None
        model = self.model
        model.x = None
        model.y = 1.0


@register_model
class LP_piecewise_nosuffixes(LP_piecewise):

    description = "LP_piecewise_nosuffixes"
    test_pickling = False

    def __init__(self):
        LP_piecewise.__init__(self)
        self.disable_suffix_tests = True
        self.add_results("LP_piecewise.json")

