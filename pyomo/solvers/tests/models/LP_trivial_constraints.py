#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

from pyomo.core import ConcreteModel, Param, Var, Expression, Objective, Constraint, RealInterval, ConstraintList
from pyomo.solvers.tests.models.base import _BaseTestModel, register_model


@register_model
class LP_trivial_constraints(_BaseTestModel):
    """
    A continuous linear model with trivial constraints
    """

    description = "LP_trivial_constraints"
    capabilities = set(['linear'])
    test_pickling = False

    def __init__(self):
        _BaseTestModel.__init__(self)
        self.add_results(self.description+".json")

    def _generate_model(self):
        self.model = None
        self.model = ConcreteModel()
        model = self.model
        model._name = self.description

        model.x = Var(domain=RealInterval(bounds=(float('-inf'), None)))
        model.y = Var(bounds=(None, float('inf')))
        model.obj = Objective(expr=model.x - model.y)
        model.c = ConstraintList()
        model.c.add(model.x >= -2)
        model.c.add(model.y <= 3)
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

    def warmstart_model(self):
        assert self.model is not None
        pass

    def post_solve_test_validation(self, tester, results):
        if tester is None:
            symbol_map = results._smap
            assert not symbol_map is None
            for i in self.model.c:
                assert id(self.model.c[i]) in symbol_map.byObject
        else:
            symbol_map = results._smap
            tester.assertNotEqual(symbol_map, None)
            for i in self.model.c:
                tester.assertTrue(id(self.model.c[i]) in symbol_map.byObject)

