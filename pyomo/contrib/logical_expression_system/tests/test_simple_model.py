from pyomo.contrib.logical_expression_system.nodes import \
    (NotNode, LeafNode, OrNode, AndNode, ImplicationNode)
from pyomo.contrib.logical_expression_system.util import \
    (bring_to_conjunctive_normal_form, CNF_to_linear_constraints)
from pyomo.environ import \
    (ConcreteModel, Var, Objective, Constraint, Set, RangeSet,
     NonNegativeReals, Binary, SolverFactory, TransformationFactory)
from pyomo.gdp import (Disjunct)
import pyutilib.th as unittest

mip_solver = 'cbc'
solver_available = SolverFactory(mip_solver).available()


@unittest.skipUnless(solver_available, "%s solver is not available." % mip_solver)
class TestSimpleModels(unittest.TestCase):
    def test_CNF_constraints(self):
        m = ConcreteModel()

        m.y_idx = RangeSet(6)
        m.y = Var(m.y_idx, domain=Binary)

        n1 = LeafNode(m.y[1])
        n2 = LeafNode(m.y[2])
        n3 = LeafNode(m.y[3])
        n4 = LeafNode(m.y[4])
        n5 = LeafNode(m.y[5])
        n6 = LeafNode(m.y[6])

        N1 = NotNode(n1)
        N2 = OrNode([N1, n2])
        N3 = OrNode([n4, n5, n6])
        N4 = AndNode([N2, n3, N3])

        bring_to_conjunctive_normal_form(N4)
        CNF_to_linear_constraints(m, N4)
        m.obj = Objective(
            expr=(10 * m.y[3] + 10 * m.y[2] + 100 * m.y[1]
                  + 1 * m.y[5] + 100 * (m.y[4] + m.y[6])))

        m_trafo = TransformationFactory('gdp.chull').create_using(m)
        SolverFactory(mip_solver).solve(m_trafo, tee=False)
        self.assertTrue(
            m_trafo.y[1].value == 0.0 and m_trafo.y[2].value == 0.0
            and m_trafo.y[3].value == 1.0 and m_trafo.y[4].value == 0.0
            and m_trafo.y[5].value == 1.0 and m_trafo.y[6].value == 0.0)

    def test_simple_CNF_disjuncts(self):
        m = ConcreteModel()
        m.y_idx = Set(initialize=[1, 2])
        m.d_idx = Set(initialize=[1, 2])
        m.y = Var(m.y_idx, domain=NonNegativeReals, bounds=(0, 1))
        m.disjuncts = Disjunct(m.y_idx, m.d_idx)
        m.disjuncts[1, 1].c = Constraint(expr=m.y[1] == 0)
        m.disjuncts[1, 2].c = Constraint(expr=m.y[1] == 1)
        m.disjuncts[2, 1].c = Constraint(expr=m.y[2] == 0)
        m.disjuncts[2, 2].c = Constraint(expr=m.y[2] == 1)

        @m.Disjunction(m.y_idx)
        def disjunction(m, i):
            return [m.disjuncts[i, 1], m.disjuncts[i, 2]]

        m.obj = Objective(expr=100 * m.y[1] + 1 * m.y[2])

        l1 = LeafNode(m.disjuncts[1, 1].indicator_var)
        l2 = LeafNode(m.disjuncts[2, 2].indicator_var)
        n1 = ImplicationNode(l1, l2)
        bring_to_conjunctive_normal_form(n1)
        CNF_to_linear_constraints(m, n1)

        m_trafo = TransformationFactory('gdp.bigm').create_using(m, bigM=10)
        SolverFactory(mip_solver).solve(m_trafo)
        self.assertTrue(abs(m_trafo.y[1].value - 0 < 1e-8))
        self.assertTrue(abs(m_trafo.y[2].value - 1 < 1e-8))
