#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

from pyomo.core import ConcreteModel, Param, Var, Expression, Objective, Constraint, RangeSet, ConstraintList
from pyomo.solvers.tests.models.base import _BaseTestModel, register_model
from pyomo.repn.beta.matrix import compile_block_linear_constraints


@register_model
class LP_compiled(_BaseTestModel):
    """
    A continuous linear model that is compiled into a
    MatrixConstraint object.
    """

    description = "LP_compiled"
    capabilities = set(['linear'])
    test_pickling = False

    def __init__(self):
        _BaseTestModel.__init__(self)
        self.add_results(self.description+".json")
        self.disable_suffix_tests = True

    def _generate_model(self):
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
        cdata = model.c.add((0, 1, 3))
        assert cdata.lower == 0
        assert cdata.upper == 3
        assert cdata.body() == 1
        assert not cdata.equality
        cdata = model.c.add((0, 2, 3))
        assert cdata.lower == 0
        assert cdata.upper == 3
        assert cdata.body() == 2
        assert not cdata.equality
        cdata = model.c.add((0, 1, None))
        assert cdata.lower == 0
        assert cdata.upper is None
        assert cdata.body() == 1
        assert not cdata.equality
        cdata = model.c.add((None, 0, 1))
        assert cdata.lower is None
        assert cdata.upper == 1
        assert cdata.body() == 0
        assert not cdata.equality
        cdata = model.c.add((1,1))
        assert cdata.lower == 1
        assert cdata.upper == 1
        assert cdata.body() == 1
        assert cdata.equality

        model.fixed_var = Var()
        model.fixed_var.fix(1.0)
        cdata = model.c.add((0, 1+model.fixed_var, 3))
        cdata = model.c.add((0, 2 + model.fixed_var, 3))
        cdata = model.c.add((0, model.fixed_var, None))
        cdata = model.c.add((None, model.fixed_var, 1))
        cdata = model.c.add((model.fixed_var, 1))

        model.c_inactive = ConstraintList()
        # to make the variable used in the constraint match the name
        model.c_inactive.add(Constraint.Skip)
        model.c_inactive.add(Constraint.Skip)
        model.c_inactive.add(model.x[3]>=-2.)
        model.c_inactive.add(model.x[4]<=2.)

        compile_block_linear_constraints(model, 'Amatrix')

    def warmstart_model(self):
        assert self.model is not None
        model = self.model
        for i in model.s:
            model.x[i] = None

