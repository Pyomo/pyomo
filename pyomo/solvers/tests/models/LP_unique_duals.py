#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.core import ConcreteModel, Param, Var, Objective, Constraint, RangeSet, NonNegativeReals, Suffix, sum_product
from pyomo.solvers.tests.models.base import _BaseTestModel, register_model

def c_rule(model, j):
    return 5 if j<5 else 9.0/2

def b_rule(model, i):
    if i == 4:
        i = 5
    elif i == 5:
        i = 4
    return 5 if i<5 else 7.0/2

def A_rule(model, i, j):
    if i == 4:
        i = 5
    elif i == 5:
        i = 4
    return 2 if i==j else 1

def primalcon_rule(model, i):
    return sum(model.A[i,j]*model.x[j] for j in model.N) >= model.b[i]


@register_model
class LP_unique_duals(_BaseTestModel):
    """
    A LP with unique dual values
    """

    description = "LP_unique_duals"
    capabilities = set(['linear'])

    def __init__(self):
        _BaseTestModel.__init__(self)
        self.add_results(self.description+".json")

    def _generate_model(self):
        self.model = None
        self.model = ConcreteModel()
        model = self.model
        model._name = self.description

        n = 7
        m = 7
        model.N = RangeSet(1,n)
        model.M = RangeSet(1,m)

        model.c = Param(model.N, rule=c_rule)

        model.b = Param(model.M, rule=b_rule)

        model.A = Param(model.M, model.N, rule=A_rule)

        model.x = Var(model.N, within=NonNegativeReals)
        model.y = Var(model.M, within=NonNegativeReals)

        model.cost = Objective(expr=sum_product(model.c, model.x))

        model.primalcon = Constraint(model.M, rule=primalcon_rule)

        #model.dual = Suffix(direction=Suffix.IMPORT)
        #model.rc = Suffix(direction=Suffix.IMPORT)
        model.slack = Suffix(direction=Suffix.IMPORT)
        model.urc = Suffix(direction=Suffix.IMPORT)
        model.lrc = Suffix(direction=Suffix.IMPORT)

    def warmstart_model(self):
        assert self.model is not None
        model = self.model
        for i in model.x:
            model.x[i] = None
        for i in model.y:
            model.y[i] = None

