from coramin.relaxations.copy_relaxation import copy_relaxation_with_local_data
import unittest
import pyomo.environ as pe
import coramin
from pyomo.common.collections import ComponentSet
from pyomo.core.expr.sympy_tools import sympyify_expression


class TestCopyRelWithLocalData(unittest.TestCase):
    def test_quadratic(self):
        m = pe.ConcreteModel()
        m.x = pe.Var(bounds=(-1, 1))
        m.aux = pe.Var()
        m.rel = coramin.relaxations.PWXSquaredRelaxation()
        m.rel.build(x=m.x,
                    aux_var=m.aux,
                    pw_repn='LOG',
                    relaxation_side=coramin.utils.RelaxationSide.OVER,
                    use_linear_relaxation=False)
        m2 = pe.ConcreteModel()
        m2.x = pe.Var(bounds=(-1, 1))
        m2.aux = pe.Var()
        new_rel = copy_relaxation_with_local_data(m.rel, {id(m.x): m2.x, id(m.aux): m2.aux})
        self.assertIn(m2.x, ComponentSet(new_rel.get_rhs_vars()))
        self.assertEqual(len(new_rel.get_rhs_vars()), 1)
        self.assertEqual(m.rel.use_linear_relaxation, new_rel.use_linear_relaxation)
        self.assertEqual(m.rel.relaxation_side, new_rel.relaxation_side)
        self.assertIs(m2.aux, new_rel.get_aux_var())
        self.assertEqual(m.rel._pw_repn, new_rel._pw_repn)
        self.assertEqual(m.rel._function_shape, new_rel._function_shape)
        self.assertIsInstance(new_rel, coramin.relaxations.PWXSquaredRelaxationData)

    def test_arctan(self):
        m = pe.ConcreteModel()
        m.x = pe.Var(bounds=(-1, 1))
        m.aux = pe.Var()
        m.rel = coramin.relaxations.PWArctanRelaxation()
        m.rel.build(x=m.x,
                    aux_var=m.aux,
                    pw_repn='LOG',
                    relaxation_side=coramin.utils.RelaxationSide.OVER,
                    use_linear_relaxation=True)
        m2 = pe.ConcreteModel()
        m2.x = pe.Var(bounds=(-1, 1))
        m2.aux = pe.Var()
        new_rel = copy_relaxation_with_local_data(m.rel, {id(m.x): m2.x, id(m.aux): m2.aux})
        self.assertIn(m2.x, ComponentSet(new_rel.get_rhs_vars()))
        self.assertEqual(len(new_rel.get_rhs_vars()), 1)
        self.assertEqual(m.rel.use_linear_relaxation, new_rel.use_linear_relaxation)
        self.assertEqual(m.rel.relaxation_side, new_rel.relaxation_side)
        self.assertIs(m2.aux, new_rel.get_aux_var())
        self.assertEqual(m.rel._pw_repn, new_rel._pw_repn)
        self.assertEqual(m.rel._function_shape, new_rel._function_shape)
        self.assertIsInstance(new_rel, coramin.relaxations.PWArctanRelaxationData)

    def test_sin(self):
        m = pe.ConcreteModel()
        m.x = pe.Var(bounds=(-1, 1))
        m.aux = pe.Var()
        m.rel = coramin.relaxations.PWSinRelaxation()
        m.rel.build(x=m.x,
                    aux_var=m.aux,
                    pw_repn='LOG',
                    relaxation_side=coramin.utils.RelaxationSide.OVER,
                    use_linear_relaxation=True)
        m2 = pe.ConcreteModel()
        m2.x = pe.Var(bounds=(-1, 1))
        m2.aux = pe.Var()
        new_rel = copy_relaxation_with_local_data(m.rel, {id(m.x): m2.x, id(m.aux): m2.aux})
        self.assertIn(m2.x, ComponentSet(new_rel.get_rhs_vars()))
        self.assertEqual(len(new_rel.get_rhs_vars()), 1)
        self.assertEqual(m.rel.use_linear_relaxation, new_rel.use_linear_relaxation)
        self.assertEqual(m.rel.relaxation_side, new_rel.relaxation_side)
        self.assertIs(m2.aux, new_rel.get_aux_var())
        self.assertEqual(m.rel._pw_repn, new_rel._pw_repn)
        self.assertEqual(m.rel._function_shape, new_rel._function_shape)
        self.assertIsInstance(new_rel, coramin.relaxations.PWSinRelaxationData)

    def test_cos(self):
        m = pe.ConcreteModel()
        m.x = pe.Var(bounds=(-1, 1))
        m.aux = pe.Var()
        m.rel = coramin.relaxations.PWCosRelaxation()
        m.rel.build(x=m.x,
                    aux_var=m.aux,
                    pw_repn='LOG',
                    relaxation_side=coramin.utils.RelaxationSide.UNDER,
                    use_linear_relaxation=True)
        m2 = pe.ConcreteModel()
        m2.x = pe.Var(bounds=(-1, 1))
        m2.aux = pe.Var()
        new_rel = copy_relaxation_with_local_data(m.rel, {id(m.x): m2.x, id(m.aux): m2.aux})
        self.assertIn(m2.x, ComponentSet(new_rel.get_rhs_vars()))
        self.assertEqual(len(new_rel.get_rhs_vars()), 1)
        self.assertEqual(m.rel.use_linear_relaxation, new_rel.use_linear_relaxation)
        self.assertEqual(m.rel.relaxation_side, new_rel.relaxation_side)
        self.assertIs(m2.aux, new_rel.get_aux_var())
        self.assertEqual(m.rel._pw_repn, new_rel._pw_repn)
        self.assertEqual(m.rel._function_shape, new_rel._function_shape)
        self.assertIsInstance(new_rel, coramin.relaxations.PWCosRelaxationData)

    def test_exp(self):
        m = pe.ConcreteModel()
        m.x = pe.Var(bounds=(-1, 1))
        m.aux = pe.Var()
        m.rel = coramin.relaxations.PWUnivariateRelaxation()
        m.rel.build(x=m.x,
                    aux_var=m.aux,
                    shape=coramin.utils.FunctionShape.CONVEX,
                    f_x_expr=pe.exp(m.x),
                    pw_repn='LOG',
                    relaxation_side=coramin.utils.RelaxationSide.UNDER,
                    use_linear_relaxation=True)
        m2 = pe.ConcreteModel()
        m2.x = pe.Var(bounds=(-1, 1))
        m2.aux = pe.Var()
        new_rel = copy_relaxation_with_local_data(m.rel, {id(m.x): m2.x, id(m.aux): m2.aux})
        self.assertIn(m2.x, ComponentSet(new_rel.get_rhs_vars()))
        self.assertEqual(len(new_rel.get_rhs_vars()), 1)
        self.assertEqual(m.rel.use_linear_relaxation, new_rel.use_linear_relaxation)
        self.assertEqual(m.rel.relaxation_side, new_rel.relaxation_side)
        self.assertIs(m2.aux, new_rel.get_aux_var())
        self.assertEqual(m.rel._pw_repn, new_rel._pw_repn)
        self.assertEqual(m.rel._function_shape, new_rel._function_shape)
        self.assertIsInstance(new_rel, coramin.relaxations.PWUnivariateRelaxationData)

    def test_log(self):
        m = pe.ConcreteModel()
        m.x = pe.Var(bounds=(0.5, 1.5))
        m.aux = pe.Var()
        m.rel = coramin.relaxations.PWUnivariateRelaxation()
        m.rel.build(x=m.x,
                    aux_var=m.aux,
                    shape=coramin.utils.FunctionShape.CONCAVE,
                    f_x_expr=pe.log(m.x),
                    pw_repn='LOG',
                    relaxation_side=coramin.utils.RelaxationSide.UNDER,
                    use_linear_relaxation=True)
        m2 = pe.ConcreteModel()
        m2.x = pe.Var(bounds=(0.5, 1.5))
        m2.aux = pe.Var()
        new_rel = copy_relaxation_with_local_data(m.rel, {id(m.x): m2.x, id(m.aux): m2.aux})
        self.assertIn(m2.x, ComponentSet(new_rel.get_rhs_vars()))
        self.assertEqual(len(new_rel.get_rhs_vars()), 1)
        self.assertEqual(m.rel.use_linear_relaxation, new_rel.use_linear_relaxation)
        self.assertEqual(m.rel.relaxation_side, new_rel.relaxation_side)
        self.assertIs(m2.aux, new_rel.get_aux_var())
        self.assertEqual(m.rel._pw_repn, new_rel._pw_repn)
        self.assertEqual(m.rel._function_shape, new_rel._function_shape)
        self.assertIsInstance(new_rel, coramin.relaxations.PWUnivariateRelaxationData)

    def test_multivariate(self):
        m = pe.ConcreteModel()
        m.x = pe.Var(bounds=(-1, 1))
        m.y = pe.Var(bounds=(-1, 1))
        m.aux = pe.Var()
        m.rel = coramin.relaxations.MultivariateRelaxation()
        m.rel.build(aux_var=m.aux,
                    shape=coramin.utils.FunctionShape.CONVEX,
                    f_x_expr=(m.x**2 + m.y**2),
                    use_linear_relaxation=True)
        m2 = pe.ConcreteModel()
        m2.x = pe.Var(bounds=(-1, 1))
        m2.y = pe.Var(bounds=(-1, 1))
        m2.aux = pe.Var()
        new_rel = copy_relaxation_with_local_data(m.rel, {id(m.x): m2.x,
                                                    id(m.y): m2.y,
                                                    id(m.aux): m2.aux})
        self.assertEqual(m.rel.use_linear_relaxation, new_rel.use_linear_relaxation)
        self.assertEqual(m.rel.relaxation_side, new_rel.relaxation_side)
        rhs_vars = ComponentSet(new_rel.get_rhs_vars())
        self.assertIn(m2.x, rhs_vars)
        self.assertIn(m2.y, rhs_vars)
        self.assertEqual(len(rhs_vars), 2)
        self.assertIs(m2.aux, new_rel.get_aux_var())
        self.assertEqual(m.rel._function_shape, new_rel._function_shape)
        self.assertIsInstance(new_rel, coramin.relaxations.MultivariateRelaxationData)
        self.assertEqual(sympyify_expression(m2.x**2 + m2.y**2 - new_rel.get_rhs_expr())[1], 0)

    def test_multivariate2(self):
        m = pe.ConcreteModel()
        m.x = pe.Var(bounds=(-1, 1))
        m.y = pe.Var(bounds=(-1, 1))
        m.aux = pe.Var()
        m.rel = coramin.relaxations.MultivariateRelaxation()
        m.rel.build(aux_var=m.aux,
                    shape=coramin.utils.FunctionShape.CONCAVE,
                    f_x_expr=(-m.x**2 - m.y**2),
                    use_linear_relaxation=True)
        m2 = pe.ConcreteModel()
        m2.x = pe.Var(bounds=(-1, 1))
        m2.y = pe.Var(bounds=(-1, 1))
        m2.aux = pe.Var()
        new_rel = copy_relaxation_with_local_data(m.rel, {id(m.x): m2.x,
                                                    id(m.y): m2.y,
                                                    id(m.aux): m2.aux})
        self.assertEqual(m.rel.use_linear_relaxation, new_rel.use_linear_relaxation)
        self.assertEqual(m.rel.relaxation_side, new_rel.relaxation_side)
        rhs_vars = ComponentSet(new_rel.get_rhs_vars())
        self.assertIn(m2.x, rhs_vars)
        self.assertIn(m2.y, rhs_vars)
        self.assertEqual(len(rhs_vars), 2)
        self.assertIs(m2.aux, new_rel.get_aux_var())
        self.assertEqual(m.rel._function_shape, new_rel._function_shape)
        self.assertIsInstance(new_rel, coramin.relaxations.MultivariateRelaxationData)
        self.assertEqual(sympyify_expression(-m2.x**2 - m2.y**2 - new_rel.get_rhs_expr())[1], 0)
