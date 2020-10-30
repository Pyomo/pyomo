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
from pyomo.core import ConcreteModel, Var, Objective, Constraint, RealInterval, ConstraintList
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
        model.d = Constraint(
            rule=lambda m: (float('-inf'), m.x, float('inf')))
        assert not model.d.equality

    def warmstart_model(self):
        assert self.model is not None
        pass

    def post_solve_test_validation(self, tester, results):
        symbol_map = results._smap
        assert not symbol_map is None
        if tester is None:
            for i in self.model.c:
                assert id(self.model.c[i]) in symbol_map.byObject
            assert id(self.model.d[i]) not in symbol_map.byObject
        else:
            for i in self.model.c:
                tester.assertTrue(id(self.model.c[i]) in symbol_map.byObject)
            tester.assertTrue(id(self.model.d) not in symbol_map.byObject)

@register_model
class LP_trivial_constraints_kernel(LP_trivial_constraints):

    def _generate_model(self):
        self.model = None
        self.model = pmo.block()
        model = self.model
        model._name = self.description

        model.x = pmo.variable(domain=RealInterval(bounds=(float('-inf'), None)))
        model.y = pmo.variable(ub=float('inf'))
        model.obj = pmo.objective(model.x - model.y)
        model.c = pmo.constraint_dict()
        model.c[1] = pmo.constraint(model.x >= -2)
        model.c[2] = pmo.constraint(model.y <= 3)
        cdata = model.c[3] = pmo.constraint((0, 1, 3))
        assert cdata.lb == 0
        assert cdata.ub == 3
        assert cdata.body() == 1
        assert not cdata.equality
        cdata = model.c[4] = pmo.constraint((0, 2, 3))
        assert cdata.lb == 0
        assert cdata.ub == 3
        assert cdata.body() == 2
        assert not cdata.equality
        cdata = model.c[5] = pmo.constraint((0, 1, None))
        assert cdata.lb == 0
        assert cdata.ub is None
        assert cdata.body() == 1
        assert not cdata.equality
        cdata = model.c[6] = pmo.constraint((None, 0, 1))
        assert cdata.lb is None
        assert cdata.ub == 1
        assert cdata.body() == 0
        assert not cdata.equality
        cdata = model.c[7] = pmo.constraint((1,1))
        assert cdata.lb == 1
        assert cdata.ub == 1
        assert cdata.body() == 1
        assert cdata.equality

    def post_solve_test_validation(self, tester, results):
        symbol_map = results.Solution(0).symbol_map
        assert not symbol_map is None
        if tester is None:
            for i in self.model.c:
                assert id(self.model.c[i]) in symbol_map.byObject
        else:
            tester.assertNotEqual(symbol_map, None)
            for i in self.model.c:
                tester.assertTrue(id(self.model.c[i]) in symbol_map.byObject)
