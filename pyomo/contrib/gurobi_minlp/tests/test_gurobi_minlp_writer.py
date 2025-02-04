#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2024
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.common.dependencies import attempt_import
from pyomo.environ import (
    Binary,
    ConcreteModel,
    Constraint,
    Integers,
    log,
    NonNegativeIntegers,
    NonNegativeReals,
    NonPositiveIntegers,
    NonPositiveReals,
    Objective,
    Reals,
    Var,
)
from pyomo.opt import WriterFactory
from pyomo.contrib.gurobi_minlp.repn.gurobi_direct_minlp import (
    GurobiMINLPVisitor
)
from pyomo.contrib.gurobi_minlp.tests.test_gurobi_minlp_walker import (
    CommonTest
)

## DEBUG
from pytest import set_trace

gurobipy, gurobipy_available = attempt_import('gurobipy', minimum_version='12.0.0')

def make_model():
    m = ConcreteModel()
    m.x1 = Var(domain=NonNegativeReals, bounds=(0, 10))
    m.x2 = Var(domain=Reals, bounds=(-3, 4))
    m.x3 = Var(domain=NonPositiveReals, bounds=(-13, 0))
    m.y1 = Var(domain=Integers, bounds=(4, 14))
    m.y2 = Var(domain=NonNegativeIntegers, bounds=(5, 16))
    m.y3 = Var(domain=NonPositiveIntegers, bounds=(-13, 0))
    m.z1 = Var(domain=Binary)

    m.c1 = Constraint(expr=2 ** m.x2 >= m.x3)
    m.c2 = Constraint(expr=m.y1 ** 2 <= 7)
    m.c3 = Constraint(expr=m.y2 + m.y3 + 5 * m.z1 >= 17)

    m.obj = Objective(expr=log(m.x1))

    return m

class TestGurobiMINLPWriter(CommonTest):
    def test_small_model(self):
        grb_model = gurobipy.Model()
        visitor = GurobiMINLPVisitor(grb_model, symbolic_solver_labels=True)

        m = make_model()

        grb_model, var_map = WriterFactory('gurobi_minlp').write(m)

        self.assertEqual(len(var_map), 7)
        x1 = var_map[id(m.x1)]
        x2 = var_map[id(m.x2)]
        x3 = var_map[id(m.x3)]
        y1 = var_map[id(m.y1)]
        y2 = var_map[id(m.y2)]
        y3 = var_map[id(m.y3)]
        z1 = var_map[id(m.z1)]

        lin_constrs = grb_model.getConstrs()
        # 
        self.assertEqual(len(lin_constrs), 3)
        quad_constrs = grb_model.getQConstrs()
        self.assertEqual(len(quad_constrs), 1)
        nonlinear_constrs = grb_model.getGenConstrs()
        self.assertEqual(nonlinear_constrs, 2)

        set_trace()
        
        grb_model.optimize()

        # TODO: assert something! :P

# ESJ: Note: It appears they don't allow x1 ** x2...?
