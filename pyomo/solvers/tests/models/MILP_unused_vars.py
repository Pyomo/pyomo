#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.kernel import block, variable, objective, constraint, variable_dict, constraint_dict, block_dict, IntegerSet
from pyomo.core import ConcreteModel, Var, Objective, ConstraintList, Set, Integers, RangeSet, sum_product, Block
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

        model.x = Var(within=RangeSet(None,None))
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
                                    sum_product(model.X) + \
                                    sum_product(model.X_initialy_stale))

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
        self.model =  block()
        model = self.model
        model._name = self.description

        model.s = [1,2]

        model.x_unused =  variable(domain= IntegerSet)
        model.x_unused.stale = False

        model.x_unused_initialy_stale =  variable(domain= IntegerSet)
        model.x_unused_initialy_stale.stale = True

        model.X_unused =  variable_dict(
            (i,  variable(domain= IntegerSet)) for i in model.s)
        model.X_unused_initialy_stale =  variable_dict(
            (i,  variable(domain= IntegerSet)) for i in model.s)
        for i in model.s:
            model.X_unused[i].stale = False
            model.X_unused_initialy_stale[i].stale = True

        model.x =  variable(domain=RangeSet(None,None))
        model.x.stale = False

        model.x_initialy_stale =  variable(domain= IntegerSet)
        model.x_initialy_stale.stale = True

        model.X =  variable_dict(
            (i,  variable(domain= IntegerSet)) for i in model.s)
        model.X_initialy_stale =  variable_dict(
            (i,  variable(domain= IntegerSet)) for i in model.s)
        for i in model.s:
            model.X[i].stale = False
            model.X_initialy_stale[i].stale = True

        model.obj =  objective(model.x + \
                                  model.x_initialy_stale + \
                                  sum(model.X.values()) + \
                                  sum(model.X_initialy_stale.values()))

        model.c =  constraint_dict()
        model.c[1] =  constraint(model.x          >= 1)
        model.c[2] =  constraint(model.x_initialy_stale    >= 1)
        model.c[3] =  constraint(model.X[1]       >= 0)
        model.c[4] =  constraint(model.X[2]       >= 1)
        model.c[5] =  constraint(model.X_initialy_stale[1] >= 0)
        model.c[6] =  constraint(model.X_initialy_stale[2] >= 1)

        # Test that stale flags do not get updated
        # on inactive blocks (where "inactive blocks" mean blocks
        # that do NOT follow a path of all active parent blocks
        # up to the top-level model)
        flat_model = model.clone()
        model.b =  block()
        model.B =  block_dict()
        model.b.b = flat_model.clone()
        model.B[1] =  block()
        model.B[1].b = flat_model.clone()
        model.B[2] =  block()
        model.B[2].b = flat_model.clone()

        model.b.deactivate()
        model.B.deactivate(shallow=False)
        model.b.b.activate()
        model.B[1].b.activate()
        model.B[2].b.deactivate()
        assert model.b.active is False
        assert model.B[1].active is False
        assert model.B[1].active is False
        assert model.b.b.active is True
        assert model.B[1].b.active is True
        assert model.B[2].b.active is False
