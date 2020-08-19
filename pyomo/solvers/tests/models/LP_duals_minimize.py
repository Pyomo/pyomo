#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyomo.kernel as pmo
from pyomo.core import ConcreteModel, Var, Objective, Constraint, RangeSet, ConstraintList
from pyomo.solvers.tests.models.base import _BaseTestModel, register_model

@register_model
class LP_duals_minimize(_BaseTestModel):
    """
    A continuous linear model designed to test every form of
    constraint when collecting duals for a minimization
    objective
    """

    description = "LP_duals_minimize"
    level = ('nightly', 'expensive')
    capabilities = set(['linear'])
    size = (12, 12, None)

    def __init__(self):
        _BaseTestModel.__init__(self)
        self.add_results(self.description+".json")

    def _generate_model(self):
        self.model = None
        self.model = ConcreteModel()
        model = self.model
        model._name = self.description

        model.s = RangeSet(1,12)
        model.x = Var(model.s)
        model.x[1].setlb(-1)
        model.x[1].setub(1)
        model.x[2].setlb(-1)
        model.x[2].setub(1)
        model.obj = Objective(expr=sum(model.x[i]*((-1)**(i+1))
                                       for i in model.x.index_set()))
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
        model.c.add((-1.,model.x[9],-1.))
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
            model.x[i].value = None

@register_model
class LP_duals_minimize_kernel(LP_duals_minimize):

    def _generate_model(self):
        self.model = None
        self.model = pmo.block()
        model = self.model
        model._name = self.description

        model.s = list(range(1,13))
        model.x = pmo.variable_dict(
            (i, pmo.variable()) for i in model.s)
        model.x[1].lb = -1
        model.x[1].ub = 1
        model.x[2].lb = -1
        model.x[2].ub = 1
        model.obj = pmo.objective(sum(model.x[i]*((-1)**(i+1))
                                      for i in model.s))
        model.c = pmo.constraint_dict()
        # to make the variable used in the constraint match the name
        model.c[3] = pmo.constraint(model.x[3]>=-1.)
        model.c[4] = pmo.constraint(model.x[4]<=1.)
        model.c[5] = pmo.constraint(model.x[5]==-1.)
        model.c[6] = pmo.constraint(model.x[6]==-1.)
        model.c[7] = pmo.constraint(model.x[7]==1.)
        model.c[8] = pmo.constraint(model.x[8]==1.)
        model.c[9] = pmo.constraint((-1.,model.x[9],-1.))
        model.c[10] = pmo.constraint((-1.,model.x[10],-1.))
        model.c[11] = pmo.constraint((1.,model.x[11],1.))
        model.c[12] = pmo.constraint((1.,model.x[12],1.))

        model.c_inactive = pmo.constraint_dict()
        # to make the variable used in the constraint match the name
        model.c_inactive[3] = pmo.constraint(model.x[3]>=-2.)
        model.c_inactive[4] = pmo.constraint(model.x[4]<=2.)
