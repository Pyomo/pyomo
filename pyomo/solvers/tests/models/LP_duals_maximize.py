#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

from pyomo.core import ConcreteModel, Param, Var, Expression, Objective, Constraint, RangeSet, maximize, ConstraintList
from pyomo.solvers.tests.models.base import _BaseTestModel, register_model


@register_model
class LP_duals_maximize(_BaseTestModel):
    """
    A continuous linear model designed to test every form of
    constraint when collecting duals for a maximization
    objective
    """

    description = "LP_duals_maximize"
    level = ('nightly', 'expensive')
    capabilities = set(['linear'])

    def __init__(self):
        _BaseTestModel.__init__(self)
        self.add_results(self.description+".json")

    def _generate_model(self):
        self.model = ConcreteModel()
        model = self.model
        model._name = self.description

        model.neg1 = Param(initialize=-1.0, mutable=True)
        model.pos1 = Param(initialize=1.0, mutable=True)

        model.s = RangeSet(1,12)
        model.x = Var(model.s)
        model.x[1].setlb(model.neg1)
        model.x[1].setub(model.pos1)
        model.x[2].setlb(model.neg1)
        model.x[2].setub(model.pos1)
        model.obj = Objective(expr=sum(model.x[i]*((-1)**(i))
                                       for i in model.x.index_set()),
                              sense=maximize)
        model.c = ConstraintList()
        # to make the variable used in the constraint match the name
        model.c.add(Constraint.Skip)
        model.c.add(Constraint.Skip)
        model.c.add(model.x[3]>=-1.)
        model.c.add(model.x[4]<=1.)
        model.c.add(model.x[5]==-1.)
        model.c.add(model.x[6]==-1.)
        model.c.add(model.x[7]==1.)
        model.c.add(model.x[8]==1.)
        model.c.add((model.neg1,model.x[9],model.neg1))
        model.c.add((-1.,model.x[10],-1.))
        model.c.add((1.,model.x[11],1.))
        model.c.add((1.,model.x[12],1.))

        model.c_inactive = ConstraintList()
        # to make the variable used in the constraint match the name
        model.c_inactive.add(Constraint.Skip)
        model.c_inactive.add(Constraint.Skip)
        model.c_inactive.add(model.x[3]>=-2.)
        model.c_inactive.add(model.x[4]<=2.)

    def warmstart_model(self):
        assert self.model is not None
        model = self.model
        for i in model.s:
            model.x[i] = None

