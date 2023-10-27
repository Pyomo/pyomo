import coramin
import unittest
import pyomo.environ as pe
from pyomo.common.collections import ComponentSet


class TestIterators(unittest.TestCase):
    def setUp(self):
        m = pe.ConcreteModel()
        m.x = pe.Var(bounds=(0.5, 1.5))
        m.y = pe.Var()
        m.c1 = pe.Constraint(expr=m.y == m.x)
        m.r1 = coramin.relaxations.PWUnivariateRelaxation()
        m.r1.set_input(x=m.x,
                       aux_var=m.y,
                       shape=coramin.utils.FunctionShape.CONCAVE,
                       f_x_expr=pe.log(m.x))
        m.r1.add_partition_point(value=1)
        m.r1.rebuild()
        m.b1 = pe.Block()
        m.b1.x = pe.Var(bounds=(0.5, 1.5))
        m.b1.y = pe.Var()
        m.b1.c1 = pe.Constraint(expr=m.b1.y == m.b1.x)
        m.b1.r1 = coramin.relaxations.PWUnivariateRelaxation()
        m.b1.r1.set_input(x=m.b1.x,
                          aux_var=m.b1.y,
                          shape=coramin.utils.FunctionShape.CONCAVE,
                          f_x_expr=pe.log(m.b1.x))
        m.b1.r1.add_partition_point(value=1)
        m.b1.r1.rebuild()
        m.b1.b1 = pe.Block()

        self.m = m

    def test_relaxation_data_objects(self):
        m = self.m
        rels = list(coramin.relaxations.relaxation_data_objects(m, descend_into=True, active=True))
        self.assertEqual(len(rels), 2)
        self.assertIn(m.r1, rels)
        self.assertIn(m.b1.r1, rels)

        m.r1.deactivate()
        rels = list(coramin.relaxations.relaxation_data_objects(m, descend_into=True, active=True))
        self.assertEqual(len(rels), 1)
        self.assertNotIn(m.r1, rels)
        self.assertIn(m.b1.r1, rels)

        rels = list(coramin.relaxations.relaxation_data_objects(m, descend_into=True, active=None))
        self.assertEqual(len(rels), 2)
        self.assertIn(m.r1, rels)
        self.assertIn(m.b1.r1, rels)

        m.r1.activate()
        rels = list(coramin.relaxations.relaxation_data_objects(m, descend_into=False))
        self.assertEqual(len(rels), 1)
        self.assertIn(m.r1, rels)
        self.assertNotIn(m.b1.r1, rels)

    def test_nonrelaxation_component_data_objects(self):
        m = self.m
        all_vars = list(m.component_data_objects(pe.Var, descend_into=True))
        non_relaxation_vars = list(coramin.relaxations.nonrelaxation_component_data_objects(m,
                                                                                            ctype=pe.Var,
                                                                                            descend_into=True))
        self.assertEqual(len(non_relaxation_vars), 4)
        self.assertGreater(len(all_vars), 4)

        all_vars = list(m.component_data_objects(pe.Var, descend_into=False))
        non_relaxation_vars = list(coramin.relaxations.nonrelaxation_component_data_objects(m,
                                                                                            ctype=pe.Var,
                                                                                            descend_into=False))
        self.assertEqual(len(non_relaxation_vars), 2)
        self.assertEqual(len(all_vars), 2)

        all_blocks = list(m.component_data_objects(pe.Block, descend_into=True))
        non_relaxation_blocks = list(coramin.relaxations.nonrelaxation_component_data_objects(m,
                                                                                              ctype=pe.Block,
                                                                                              descend_into=True))
        self.assertEqual(len(non_relaxation_blocks), 2)
        self.assertEqual(len(all_blocks), 8)

        all_blocks = list(m.component_data_objects(pe.Block, descend_into=False))
        non_relaxation_blocks = list(coramin.relaxations.nonrelaxation_component_data_objects(m,
                                                                                              ctype=pe.Block,
                                                                                              descend_into=False))
        self.assertEqual(len(non_relaxation_blocks), 1)
        self.assertEqual(len(all_blocks), 2)
