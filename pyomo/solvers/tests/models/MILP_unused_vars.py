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
from pyomo.core import ConcreteModel, Param, Var, Expression, Objective, Constraint, ConstraintList, Set, Integers, IntegerInterval, summation, Block
from pyomo.solvers.tests.models.base import _BaseTestModel, register_model

@register_model
class MILP_unused_vars(_BaseTestModel):
    """
    A continuous linear model where some vars aren't used
    and some used vars start out with the stale flag as True
    """

    description = "MILP_unused_vars"
    capabilities = set(['linear', 'integer'])

    def __init__(self):
        _BaseTestModel.__init__(self)
        self.disable_suffix_tests = True
        self.add_results(self.description+".json")

    def _generate_model(self):
        self.model = ConcreteModel()
        model = self.model
        model._name = self.description

        model.s = Set(initialize=[1,2])

        model.x_unused = Var(within=Integers)
        model.x_unused.stale = False

        model.x_unused_initialy_stale = Var(within=Integers)
        model.x_unused_initialy_stale.stale = True

        model.X_unused = Var(model.s, within=Integers)
        model.X_unused_initialy_stale = Var(model.s, within=Integers)
        for i in model.s:
            model.X_unused[i].stale = False
            model.X_unused_initialy_stale[i].stale = True

        model.x = Var(within=IntegerInterval(bounds=(None,None)))
        model.x.stale = False

        model.x_initialy_stale = Var(within=Integers)
        model.x_initialy_stale.stale = True

        model.X = Var(model.s, within=Integers)
        model.X_initialy_stale = Var(model.s, within=Integers)
        for i in model.s:
            model.X[i].stale = False
            model.X_initialy_stale[i].stale = True

        model.obj = Objective(expr= model.x + \
                                    model.x_initialy_stale + \
                                    summation(model.X) + \
                                    summation(model.X_initialy_stale))

        model.c = ConstraintList()
        model.c.add( model.x          >= 1 )
        model.c.add( model.x_initialy_stale    >= 1 )
        model.c.add( model.X[1]       >= 0 )
        model.c.add( model.X[2]       >= 1 )
        model.c.add( model.X_initialy_stale[1] >= 0 )
        model.c.add( model.X_initialy_stale[2] >= 1 )

        # Test that stale flags do not get updated
        # on inactive blocks (where "inactive blocks" mean blocks
        # that do NOT follow a path of all active parent blocks
        # up to the top-level model)
        flat_model = model.clone()
        model.b = Block()
        model.B = Block(model.s)
        model.b.b = flat_model.clone()
        model.B[1].b = flat_model.clone()
        model.B[2].b = flat_model.clone()

        model.b.deactivate()
        model.B.deactivate()
        model.b.b.activate()
        model.B[1].b.activate()
        model.B[2].b.deactivate()
        assert model.b.active is False
        assert model.B[1].active is False
        assert model.B[1].active is False
        assert model.b.b.active is True
        assert model.B[1].b.active is True
        assert model.B[2].b.active is False

    def warmstart_model(self):
        assert self.model is not None
        model = self.model
        model.x_unused.value = -1
        model.x_unused_initialy_stale.value = -1
        for i in model.s:
            model.X_unused[i].value = -1
            model.X_unused_initialy_stale[i].value = -1

        model.x.value = -1
        model.x_initialy_stale.value = -1
        for i in model.s:
            model.X[i].value = -1
            model.X_initialy_stale[i].value = -1

@register_model
class MILP_unused_vars_kernel(MILP_unused_vars):

    def _generate_model(self):
        self.model = pmo.block()
        model = self.model
        model._name = self.description

        model.s = [1,2]

        model.x_unused = pmo.variable(domain=pmo.Integers)
        model.x_unused.stale = False

        model.x_unused_initialy_stale = pmo.variable(domain=pmo.Integers)
        model.x_unused_initialy_stale.stale = True

        model.X_unused = pmo.create_variable_dict(keys=model.s,
                                                  domain=pmo.Integers)
        model.X_unused_initialy_stale = \
            pmo.create_variable_dict(keys=model.s,
                                     domain=pmo.Integers)
        for i in model.s:
            model.X_unused[i].stale = False
            model.X_unused_initialy_stale[i].stale = True

        model.x = pmo.variable(domain=pmo.IntegerInterval(bounds=(None,None)))
        model.x.stale = False

        model.x_initialy_stale = pmo.variable(domain=pmo.Integers)
        model.x_initialy_stale.stale = True

        model.X = pmo.create_variable_dict(keys=model.s,
                                           domain=pmo.Integers)
        model.X_initialy_stale = pmo.create_variable_dict(keys=model.s,
                                                          domain=pmo.Integers)
        for i in model.s:
            model.X[i].stale = False
            model.X_initialy_stale[i].stale = True

        model.obj = pmo.objective(model.x + \
                                  model.x_initialy_stale + \
                                  sum(model.X.values()) + \
                                  sum(model.X_initialy_stale.values()))

        model.c = pmo.constraint_dict()
        model.c[1] = pmo.constraint(model.x          >= 1)
        model.c[2] = pmo.constraint(model.x_initialy_stale    >= 1)
        model.c[3] = pmo.constraint(model.X[1]       >= 0)
        model.c[4] = pmo.constraint(model.X[2]       >= 1)
        model.c[5] = pmo.constraint(model.X_initialy_stale[1] >= 0)
        model.c[6] = pmo.constraint(model.X_initialy_stale[2] >= 1)

        # Test that stale flags do not get updated
        # on inactive blocks (where "inactive blocks" mean blocks
        # that do NOT follow a path of all active parent blocks
        # up to the top-level model)
        flat_model = model.clone()
        model.b = pmo.block()
        model.B = pmo.block_dict()
        model.b.b = flat_model.clone()
        model.B[1] = pmo.tiny_block()
        model.B[1].b = flat_model.clone()
        model.B[2] = pmo.tiny_block()
        model.B[2].b = flat_model.clone()

        model.b.deactivate()
        model.B.deactivate()
        model.b.b.activate()
        model.B[1].b.activate()
        model.B[2].b.deactivate()
        assert model.b.active is False
        assert model.B[1].active is False
        assert model.B[1].active is False
        assert model.b.b.active is True
        assert model.B[1].b.active is True
        assert model.B[2].b.active is False
