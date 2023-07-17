#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyomo.common.unittest as unittest
import pyomo.environ as pyo

try:
    import coptpy

    coptpy_available = True
except:
    coptpy_available = False


class TestCoptPersistent(unittest.TestCase):
    @unittest.skipIf(not coptpy_available, "coptpy is not available")
    def test_basics(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(-10, 10))
        m.y = pyo.Var()
        m.obj = pyo.Objective(expr=m.x**2 + m.y**2)
        m.c1 = pyo.Constraint(expr=m.y >= 2 * m.x + 1)

        opt = pyo.SolverFactory('copt_persistent')
        opt.set_instance(m)

        self.assertEqual(opt.get_model_attr('Cols'), 2)
        self.assertEqual(opt.get_model_attr('Rows'), 1)
        self.assertEqual(opt.get_model_attr('QConstrs'), 0)
        self.assertEqual(opt.get_var_attr(m.x, 'LB'), -10)
        self.assertEqual(opt.get_var_attr(m.x, 'UB'), 10)

        res = opt.solve()
        self.assertAlmostEqual(m.x.value, -0.4)
        self.assertAlmostEqual(m.y.value, 0.2)
        opt.load_duals()
        self.assertAlmostEqual(m.dual[m.c1], -0.4)
        del m.dual

        m.c2 = pyo.Constraint(expr=m.y >= -m.x + 1)
        opt.add_constraint(m.c2)
        self.assertEqual(opt.get_model_attr('Cols'), 2)
        self.assertEqual(opt.get_model_attr('Rows'), 2)
        self.assertEqual(opt.get_model_attr('QConstrs'), 0)

        res = opt.solve(save_results=False, load_solutions=False)
        self.assertAlmostEqual(m.x.value, -0.4)
        self.assertAlmostEqual(m.y.value, 0.2)
        opt.load_vars()
        self.assertAlmostEqual(m.x.value, 0)
        self.assertAlmostEqual(m.y.value, 1)

        opt.remove_constraint(m.c2)
        m.del_component(m.c2)
        self.assertEqual(opt.get_model_attr('Cols'), 2)
        self.assertEqual(opt.get_model_attr('Rows'), 1)
        self.assertEqual(opt.get_model_attr('QConstrs'), 0)

        self.assertEqual(opt.get_copt_param_info('FeasTol')[2], 1e-6)
        res = opt.solve(options={'FeasTol': '1e-7'})
        self.assertEqual(opt.get_copt_param_info('FeasTol')[2], 1e-7)
        self.assertAlmostEqual(m.x.value, -0.4)
        self.assertAlmostEqual(m.y.value, 0.2)

        m.x.setlb(-5)
        m.x.setub(5)
        opt.update_var(m.x)
        self.assertEqual(opt.get_var_attr(m.x, 'LB'), -5)
        self.assertEqual(opt.get_var_attr(m.x, 'UB'), 5)

        m.x.fix(0)
        opt.update_var(m.x)
        self.assertEqual(opt.get_var_attr(m.x, 'LB'), 0)
        self.assertEqual(opt.get_var_attr(m.x, 'UB'), 0)

        m.x.unfix()
        opt.update_var(m.x)
        self.assertEqual(opt.get_var_attr(m.x, 'LB'), -5)
        self.assertEqual(opt.get_var_attr(m.x, 'UB'), 5)

        m.c2 = pyo.Constraint(expr=m.y >= m.x**2)
        opt.add_constraint(m.c2)
        self.assertEqual(opt.get_model_attr('Cols'), 2)
        self.assertEqual(opt.get_model_attr('Rows'), 1)
        self.assertEqual(opt.get_model_attr('QConstrs'), 1)

        opt.remove_constraint(m.c2)
        m.del_component(m.c2)
        self.assertEqual(opt.get_model_attr('Cols'), 2)
        self.assertEqual(opt.get_model_attr('Rows'), 1)
        self.assertEqual(opt.get_model_attr('QConstrs'), 0)

        m.z = pyo.Var()
        opt.add_var(m.z)
        self.assertEqual(opt.get_model_attr('Cols'), 3)
        opt.remove_var(m.z)
        del m.z
        self.assertEqual(opt.get_model_attr('Cols'), 2)

    @unittest.skipIf(not coptpy_available, "coptpy is not available")
    def test_update1(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.z = pyo.Var()
        m.obj = pyo.Objective(expr=m.z)
        m.c1 = pyo.Constraint(expr=m.z >= m.x**2 + m.y**2)

        opt = pyo.SolverFactory('copt_persistent')
        opt.set_instance(m)
        self.assertEqual(opt._solver_model.getAttr('QConstrs'), 0)

        opt.remove_constraint(m.c1)
        self.assertEqual(opt._solver_model.getAttr('QConstrs'), 0)

        opt.add_constraint(m.c1)
        self.assertEqual(opt._solver_model.getAttr('QConstrs'), 1)

    @unittest.skipIf(not coptpy_available, "coptpy is not available")
    def test_update2(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.z = pyo.Var()
        m.obj = pyo.Objective(expr=m.z)
        m.c2 = pyo.Constraint(expr=m.x + m.y == 1)

        opt = pyo.SolverFactory('copt_persistent')
        opt.set_instance(m)
        self.assertEqual(opt._solver_model.getAttr('QConstrs'), 0)

        opt.remove_constraint(m.c2)
        self.assertEqual(opt._solver_model.getAttr('QConstrs'), 0)

        opt.add_constraint(m.c2)
        self.assertEqual(opt._solver_model.getAttr('QConstrs'), 1)

    @unittest.skipIf(not coptpy_available, "coptpy is not available")
    def test_update3(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.z = pyo.Var()
        m.obj = pyo.Objective(expr=m.z)
        m.c1 = pyo.Constraint(expr=m.z >= m.x**2 + m.y**2)

        opt = pyo.SolverFactory('copt_persistent')
        opt.set_instance(m)
        self.assertEqual(opt._solver_model.getAttr('QConstrs'), 1)
        m.c2 = pyo.Constraint(expr=m.y >= m.x**2)
        opt.add_constraint(m.c2)
        self.assertEqual(opt._solver_model.getAttr('QConstrs'), 2)
        opt.remove_constraint(m.c2)
        self.assertEqual(opt._solver_model.getAttr('QConstrs'), 1)

    @unittest.skipIf(not coptpy_available, "coptpy is not available")
    def test_update4(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.z = pyo.Var()
        m.obj = pyo.Objective(expr=m.z)
        m.c1 = pyo.Constraint(expr=m.z >= m.x + m.y)

        opt = pyo.SolverFactory('copt_persistent')
        opt.set_instance(m)
        self.assertEqual(opt._solver_model.getAttr('QConstrs'), 1)
        m.c2 = pyo.Constraint(expr=m.y >= m.x)
        opt.add_constraint(m.c2)
        self.assertEqual(opt._solver_model.getAttr('QConstrs'), 1)
        opt.remove_constraint(m.c2)
        self.assertEqual(opt._solver_model.getAttr('QConstrs'), 1)

    @unittest.skipIf(not coptpy_available, "coptpy is not available")
    def test_update5(self):
        m = pyo.ConcreteModel()
        m.a = pyo.Set(initialize=[1, 2, 3], ordered=True)
        m.x = pyo.Var(m.a, within=pyo.Binary)
        m.y = pyo.Var(within=pyo.Binary)
        m.obj = pyo.Objective(expr=m.y)
        m.c1 = pyo.SOSConstraint(var=m.x, sos=1)

        opt = pyo.SolverFactory('copt_persistent')
        opt.set_instance(m)
        self.assertEqual(opt._solver_model.getAttr('Soss'), 0)

        opt.remove_sos_constraint(m.c1)
        self.assertEqual(opt._solver_model.getAttr('Soss'), 0)

        opt.add_sos_constraint(m.c1)
        self.assertEqual(opt._solver_model.getAttr('Soss'), 1)

    @unittest.skipIf(not coptpy_available, "coptpy is not available")
    def test_update6(self):
        m = pyo.ConcreteModel()
        m.a = pyo.Set(initialize=[1, 2, 3], ordered=True)
        m.x = pyo.Var(m.a, within=pyo.Binary)
        m.y = pyo.Var(within=pyo.Binary)
        m.obj = pyo.Objective(expr=m.y)
        m.c1 = pyo.SOSConstraint(var=m.x, sos=1)

        opt = pyo.SolverFactory('copt_persistent')
        opt.set_instance(m)
        self.assertEqual(opt._solver_model.getAttr('Soss'), 1)
        m.c2 = pyo.SOSConstraint(var=m.x, sos=2)
        opt.add_sos_constraint(m.c2)
        self.assertEqual(opt._solver_model.getAttr('Soss'), 2)
        opt.remove_sos_constraint(m.c2)
        self.assertEqual(opt._solver_model.getAttr('Soss'), 1)

    @unittest.skipIf(not coptpy_available, "coptpy is not available")
    def test_update7(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var()

        opt = pyo.SolverFactory('copt_persistent')
        opt.set_instance(m)
        self.assertEqual(opt._solver_model.getAttr('Cols'), 0)

        opt.remove_var(m.x)
        self.assertEqual(opt._solver_model.getAttr('Cols'), 1)

        opt.add_var(m.x)
        self.assertEqual(opt._solver_model.getAttr('Cols'), 2)

        opt.remove_var(m.x)
        opt.add_var(m.x)
        opt.remove_var(m.x)
        self.assertEqual(opt._solver_model.getAttr('Cols'), 1)

    @unittest.skipIf(not coptpy_available, "coptpy is not available")
    def test_quadratic_constraint_attr(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.c = pyo.Constraint(expr=m.y >= m.x**2)

        opt = pyo.SolverFactory('copt_persistent')
        opt.set_instance(m)
        self.assertEqual(opt.get_quadratic_constraint_attr(m.c, 'rhs'), 0)

    @unittest.skipIf(not coptpy_available, "coptpy is not available")
    def test_add_column(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(within=pyo.NonNegativeReals)
        m.c = pyo.Constraint(expr=(0, m.x, 1))
        m.obj = pyo.Objective(expr=-m.x)

        opt = pyo.SolverFactory('copt_persistent')
        opt.set_instance(m)
        opt.solve()
        self.assertAlmostEqual(m.x.value, 1)

        m.y = pyo.Var(within=pyo.NonNegativeReals)

        opt.add_column(m, m.y, -3, [m.c], [2])
        opt.solve()

        self.assertAlmostEqual(m.x.value, 0)
        self.assertAlmostEqual(m.y.value, 0.5)

    @unittest.skipIf(not coptpy_available, "coptpy is not available")
    def test_add_column_exceptions(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.c = pyo.Constraint(expr=(0, m.x, 1))
        m.ci = pyo.Constraint([1, 2], rule=lambda m, i: (0, m.x, i + 1))
        m.cd = pyo.Constraint(expr=(0, -m.x, 1))
        m.cd.deactivate()
        m.obj = pyo.Objective(expr=-m.x)

        opt = pyo.SolverFactory('copt_persistent')

        # set_instance not called
        self.assertRaises(RuntimeError, opt.add_column, m, m.x, 0, [m.c], [1])

        opt.set_instance(m)

        m2 = pyo.ConcreteModel()
        m2.y = pyo.Var()
        m2.c = pyo.Constraint(expr=(0, m.x, 1))

        # different model than attached to opt
        self.assertRaises(RuntimeError, opt.add_column, m2, m2.y, 0, [], [])
        # pyomo var attached to different model
        self.assertRaises(RuntimeError, opt.add_column, m, m2.y, 0, [], [])

        z = pyo.Var()
        # pyomo var floating
        self.assertRaises(RuntimeError, opt.add_column, m, z, -2, [m.c, z], [1])

        m.y = pyo.Var()
        # len(coefficients) == len(constraints)
        self.assertRaises(RuntimeError, opt.add_column, m, m.y, -2, [m.c], [1, 2])
        self.assertRaises(RuntimeError, opt.add_column, m, m.y, -2, [m.c, z], [1])

        # add indexed constraint
        self.assertRaises(AttributeError, opt.add_column, m, m.y, -2, [m.ci], [1])
        # add something not a _ConstraintData
        self.assertRaises(AttributeError, opt.add_column, m, m.y, -2, [m.x], [1])

        # constraint not on solver model
        self.assertRaises(KeyError, opt.add_column, m, m.y, -2, [m2.c], [1])

        # inactive constraint
        self.assertRaises(KeyError, opt.add_column, m, m.y, -2, [m.cd], [1])

        opt.add_var(m.y)
        # var already in solver model
        self.assertRaises(RuntimeError, opt.add_column, m, m.y, -2, [m.c], [1])
