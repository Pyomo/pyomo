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
#

import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.opt.base.solvers import UnknownSolver
from pyomo.core.plugins.transform.scaling import ScaleModel


class TestScaleModelTransformation(unittest.TestCase):
    def test_linear_scaling(self):
        model = pyo.ConcreteModel()
        model.x = pyo.Var([1, 2, 3], bounds=(-10, 10), initialize=5.0)
        model.z = pyo.Var(bounds=(10, 20))
        model.obj = pyo.Objective(expr=model.z + model.x[1])

        # test scaling of duals as well
        model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
        model.rc = pyo.Suffix(direction=pyo.Suffix.IMPORT)

        def con_rule(m, i):
            if i == 1:
                return m.x[1] + 2 * m.x[2] + 1 * m.x[3] == 4.0
            if i == 2:
                return m.x[1] + 2 * m.x[2] + 2 * m.x[3] == 5.0
            if i == 3:
                return m.x[1] + 3.0 * m.x[2] + 1 * m.x[3] == 5.0

        model.con = pyo.Constraint([1, 2, 3], rule=con_rule)
        model.zcon = pyo.Constraint(expr=model.z >= model.x[2])

        x_scale = 0.5
        obj_scale = 2.0
        z_scale = -10.0
        con_scale1 = 0.5
        con_scale2 = 2.0
        con_scale3 = -5.0
        zcon_scale = -3.0

        unscaled_model = model.clone()
        unscaled_model.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
        unscaled_model.scaling_factor[unscaled_model.obj] = obj_scale
        unscaled_model.scaling_factor[unscaled_model.x] = x_scale
        unscaled_model.scaling_factor[unscaled_model.z] = z_scale
        unscaled_model.scaling_factor[unscaled_model.con[1]] = con_scale1
        unscaled_model.scaling_factor[unscaled_model.con[2]] = con_scale2
        unscaled_model.scaling_factor[unscaled_model.con[3]] = con_scale3
        unscaled_model.scaling_factor[unscaled_model.zcon] = zcon_scale

        scaled_model = pyo.TransformationFactory('core.scale_model').create_using(
            unscaled_model
        )

        # print('*** unscaled ***')
        # unscaled_model.pprint()
        # print('*** scaled ***')
        # scaled_model.pprint()

        glpk_solver = pyo.SolverFactory('glpk')
        if isinstance(glpk_solver, UnknownSolver) or (not glpk_solver.available()):
            raise unittest.SkipTest("glpk solver not available")

        glpk_solver.solve(unscaled_model)
        glpk_solver.solve(scaled_model)

        # check vars
        self.assertAlmostEqual(
            pyo.value(unscaled_model.x[1]),
            pyo.value(scaled_model.scaled_x[1]) / x_scale,
            4,
        )
        self.assertAlmostEqual(
            pyo.value(unscaled_model.x[2]),
            pyo.value(scaled_model.scaled_x[2]) / x_scale,
            4,
        )
        self.assertAlmostEqual(
            pyo.value(unscaled_model.x[3]),
            pyo.value(scaled_model.scaled_x[3]) / x_scale,
            4,
        )
        self.assertAlmostEqual(
            pyo.value(unscaled_model.z), pyo.value(scaled_model.scaled_z) / z_scale, 4
        )
        # check var lb
        self.assertAlmostEqual(
            pyo.value(unscaled_model.x[1].lb),
            pyo.value(scaled_model.scaled_x[1].lb) / x_scale,
            4,
        )
        self.assertAlmostEqual(
            pyo.value(unscaled_model.x[2].lb),
            pyo.value(scaled_model.scaled_x[2].lb) / x_scale,
            4,
        )
        self.assertAlmostEqual(
            pyo.value(unscaled_model.x[3].lb),
            pyo.value(scaled_model.scaled_x[3].lb) / x_scale,
            4,
        )
        # note: z_scale is negative, therefore, the inequality directions swap
        self.assertAlmostEqual(
            pyo.value(unscaled_model.z.lb),
            pyo.value(scaled_model.scaled_z.ub) / z_scale,
            4,
        )
        # check var ub
        self.assertAlmostEqual(
            pyo.value(unscaled_model.x[1].ub),
            pyo.value(scaled_model.scaled_x[1].ub) / x_scale,
            4,
        )
        self.assertAlmostEqual(
            pyo.value(unscaled_model.x[2].ub),
            pyo.value(scaled_model.scaled_x[2].ub) / x_scale,
            4,
        )
        self.assertAlmostEqual(
            pyo.value(unscaled_model.x[3].ub),
            pyo.value(scaled_model.scaled_x[3].ub) / x_scale,
            4,
        )
        # note: z_scale is negative, therefore, the inequality directions swap
        self.assertAlmostEqual(
            pyo.value(unscaled_model.z.ub),
            pyo.value(scaled_model.scaled_z.lb) / z_scale,
            4,
        )
        # check var multipliers (rc)
        self.assertAlmostEqual(
            pyo.value(unscaled_model.rc[unscaled_model.x[1]]),
            pyo.value(scaled_model.rc[scaled_model.scaled_x[1]]) * x_scale / obj_scale,
            4,
        )
        self.assertAlmostEqual(
            pyo.value(unscaled_model.rc[unscaled_model.x[2]]),
            pyo.value(scaled_model.rc[scaled_model.scaled_x[2]]) * x_scale / obj_scale,
            4,
        )
        self.assertAlmostEqual(
            pyo.value(unscaled_model.rc[unscaled_model.x[3]]),
            pyo.value(scaled_model.rc[scaled_model.scaled_x[3]]) * x_scale / obj_scale,
            4,
        )
        self.assertAlmostEqual(
            pyo.value(unscaled_model.rc[unscaled_model.z]),
            pyo.value(scaled_model.rc[scaled_model.scaled_z]) * z_scale / obj_scale,
            4,
        )
        # check constraint multipliers
        self.assertAlmostEqual(
            pyo.value(unscaled_model.dual[unscaled_model.con[1]]),
            pyo.value(scaled_model.dual[scaled_model.scaled_con[1]])
            * con_scale1
            / obj_scale,
            4,
        )
        self.assertAlmostEqual(
            pyo.value(unscaled_model.dual[unscaled_model.con[2]]),
            pyo.value(scaled_model.dual[scaled_model.scaled_con[2]])
            * con_scale2
            / obj_scale,
            4,
        )
        self.assertAlmostEqual(
            pyo.value(unscaled_model.dual[unscaled_model.con[3]]),
            pyo.value(scaled_model.dual[scaled_model.scaled_con[3]])
            * con_scale3
            / obj_scale,
            4,
        )

        # put the solution from the scaled back into the original
        pyo.TransformationFactory('core.scale_model').propagate_solution(
            scaled_model, model
        )

        # compare var values and rc with the unscaled soln
        for vm in model.component_objects(ctype=pyo.Var, descend_into=True):
            cuid = pyo.ComponentUID(vm)
            vum = cuid.find_component_on(unscaled_model)
            self.assertEqual((vm in model.rc), (vum in unscaled_model.rc))
            if vm in model.rc:
                self.assertAlmostEqual(
                    pyo.value(model.rc[vm]), pyo.value(unscaled_model.rc[vum]), 4
                )
            for k in vm:
                vmk = vm[k]
                vumk = vum[k]
                self.assertAlmostEqual(pyo.value(vmk), pyo.value(vumk), 4)
                self.assertEqual((vmk in model.rc), (vumk in unscaled_model.rc))
                if vmk in model.rc:
                    self.assertAlmostEqual(
                        pyo.value(model.rc[vmk]), pyo.value(unscaled_model.rc[vumk]), 4
                    )

        # compare constraint duals and value
        for model_con in model.component_objects(
            ctype=pyo.Constraint, descend_into=True
        ):
            cuid = pyo.ComponentUID(model_con)
            unscaled_model_con = cuid.find_component_on(unscaled_model)
            self.assertEqual(
                (model_con in model.rc), (unscaled_model_con in unscaled_model.rc)
            )
            if model_con in model.dual:
                self.assertAlmostEqual(
                    pyo.value(model.dual[model_con]),
                    pyo.value(unscaled_model.dual[unscaled_model_con]),
                    4,
                )
            for k in model_con:
                mk = model_con[k]
                umk = unscaled_model_con[k]
                self.assertEqual((mk in model.dual), (umk in unscaled_model.dual))
                if mk in model.dual:
                    self.assertAlmostEqual(
                        pyo.value(model.dual[mk]),
                        pyo.value(unscaled_model.dual[umk]),
                        4,
                    )

    def test_scaling_without_rename(self):
        m = pyo.ConcreteModel()
        m.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
        m.v1 = pyo.Var(initialize=10)
        m.v2 = pyo.Var(initialize=20)
        m.v3 = pyo.Var(initialize=30)

        def c1_rule(m):
            return m.v1 == 1e6

        m.c1 = pyo.Constraint(rule=c1_rule)

        def c2_rule(m):
            return m.v2 == 1e-4

        m.c2 = pyo.Constraint(rule=c2_rule)

        m.scaling_factor[m.v1] = 1.0
        m.scaling_factor[m.v2] = 0.5
        m.scaling_factor[m.v3] = 0.25
        m.scaling_factor[m.c1] = 1e-5
        m.scaling_factor[m.c2] = 1e5

        values = {}
        values[id(m.v1)] = (m.v1.value, m.scaling_factor[m.v1])
        values[id(m.v2)] = (m.v2.value, m.scaling_factor[m.v2])
        values[id(m.v3)] = (m.v3.value, m.scaling_factor[m.v3])
        values[id(m.c1)] = (pyo.value(m.c1.body), m.scaling_factor[m.c1])
        values[id(m.c2)] = (pyo.value(m.c2.body), m.scaling_factor[m.c2])

        m.c2_ref = pyo.Reference(m.c2)
        m.v3_ref = pyo.Reference(m.v3)

        scale = pyo.TransformationFactory('core.scale_model')
        scale.apply_to(m, rename=False)

        self.assertTrue(hasattr(m, 'v1'))
        self.assertTrue(hasattr(m, 'v2'))
        self.assertTrue(hasattr(m, 'c1'))
        self.assertTrue(hasattr(m, 'c2'))

        orig_val, factor = values[id(m.v1)]
        self.assertAlmostEqual(m.v1.value, orig_val * factor)

        orig_val, factor = values[id(m.v2)]
        self.assertAlmostEqual(m.v2.value, orig_val * factor)

        orig_val, factor = values[id(m.c1)]
        self.assertAlmostEqual(pyo.value(m.c1.body), orig_val * factor)

        orig_val, factor = values[id(m.c2)]
        self.assertAlmostEqual(pyo.value(m.c2.body), orig_val * factor)

        orig_val, factor = values[id(m.v3)]
        self.assertAlmostEqual(m.v3_ref[None].value, orig_val * factor)
        # Note that because the model was not renamed,
        # v3_ref is still intact.

        lhs = m.c2.body
        monom_factor = lhs.arg(0)
        scale_factor = m.scaling_factor[m.c2] / m.scaling_factor[m.v2]
        self.assertAlmostEqual(monom_factor, scale_factor)

    def test_scaling_hierarchical(self):
        m = pyo.ConcreteModel()
        m.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
        m.v1 = pyo.Var(initialize=10)
        m.v2 = pyo.Var(initialize=20)
        m.v3 = pyo.Var(initialize=30)

        def c1_rule(m):
            return m.v1 == 1e6

        m.c1 = pyo.Constraint(rule=c1_rule)

        def c2_rule(m):
            return m.v2 == 1e-4

        m.c2 = pyo.Constraint(rule=c2_rule)

        m.scaling_factor[m.v1] = 1.0
        m.scaling_factor[m.v2] = 0.5
        m.scaling_factor[m.v3] = 0.25
        m.scaling_factor[m.c1] = 1e-5
        m.scaling_factor[m.c2] = 1e5

        m.b = pyo.Block()
        m.b.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
        m.b.v4 = pyo.Var(initialize=10)
        m.b.v5 = pyo.Var(initialize=20)
        m.b.v6 = pyo.Var(initialize=30)

        def c3_rule(m):
            return m.v4 == 1e6

        m.b.c3 = pyo.Constraint(rule=c3_rule)

        def c4_rule(m):
            return m.v5 == 1e-4

        m.b.c4 = pyo.Constraint(rule=c4_rule)

        m.b.scaling_factor[m.b.v4] = 1.0
        m.b.scaling_factor[m.b.v5] = 0.5
        m.b.scaling_factor[m.b.v6] = 0.25
        m.b.scaling_factor[m.b.c3] = 1e-5
        m.b.scaling_factor[m.b.c4] = 1e5

        values = {}
        values[id(m.v1)] = (m.v1.value, m.scaling_factor[m.v1])
        values[id(m.v2)] = (m.v2.value, m.scaling_factor[m.v2])
        values[id(m.v3)] = (m.v3.value, m.scaling_factor[m.v3])
        values[id(m.c1)] = (pyo.value(m.c1.body), m.scaling_factor[m.c1])
        values[id(m.c2)] = (pyo.value(m.c2.body), m.scaling_factor[m.c2])
        values[id(m.b.v4)] = (m.b.v4.value, m.b.scaling_factor[m.b.v4])
        values[id(m.b.v5)] = (m.b.v5.value, m.b.scaling_factor[m.b.v5])
        values[id(m.b.v6)] = (m.b.v6.value, m.b.scaling_factor[m.b.v6])
        values[id(m.b.c3)] = (pyo.value(m.b.c3.body), m.b.scaling_factor[m.b.c3])
        values[id(m.b.c4)] = (pyo.value(m.b.c4.body), m.b.scaling_factor[m.b.c4])

        m.c2_ref = pyo.Reference(m.c2)
        m.v3_ref = pyo.Reference(m.v3)

        m.b.c4_ref = pyo.Reference(m.b.c4)
        m.b.v6_ref = pyo.Reference(m.b.v6)

        scale = pyo.TransformationFactory('core.scale_model')
        scale.apply_to(m, rename=False)

        self.assertTrue(hasattr(m, 'v1'))
        self.assertTrue(hasattr(m, 'v2'))
        self.assertTrue(hasattr(m, 'c1'))
        self.assertTrue(hasattr(m, 'c2'))
        self.assertTrue(hasattr(m.b, 'v4'))
        self.assertTrue(hasattr(m.b, 'v5'))
        self.assertTrue(hasattr(m.b, 'c3'))
        self.assertTrue(hasattr(m.b, 'c4'))

        orig_val, factor = values[id(m.v1)]
        self.assertAlmostEqual(m.v1.value, orig_val * factor)

        orig_val, factor = values[id(m.v2)]
        self.assertAlmostEqual(m.v2.value, orig_val * factor)

        orig_val, factor = values[id(m.c1)]
        self.assertAlmostEqual(pyo.value(m.c1.body), orig_val * factor)

        orig_val, factor = values[id(m.c2)]
        self.assertAlmostEqual(pyo.value(m.c2.body), orig_val * factor)

        orig_val, factor = values[id(m.v3)]
        self.assertAlmostEqual(m.v3_ref[None].value, orig_val * factor)
        # Note that because the model was not renamed,
        # v3_ref is still intact.

        orig_val, factor = values[id(m.b.v4)]
        self.assertAlmostEqual(m.b.v4.value, orig_val * factor)

        orig_val, factor = values[id(m.b.v5)]
        self.assertAlmostEqual(m.b.v5.value, orig_val * factor)

        orig_val, factor = values[id(m.b.c3)]
        self.assertAlmostEqual(pyo.value(m.b.c3.body), orig_val * factor)

        orig_val, factor = values[id(m.b.c4)]
        self.assertAlmostEqual(pyo.value(m.b.c4.body), orig_val * factor)

        orig_val, factor = values[id(m.b.v6)]
        self.assertAlmostEqual(m.b.v6_ref[None].value, orig_val * factor)
        # Note that because the model was not renamed,
        # v6_ref is still intact.

        lhs = m.c2.body
        monom_factor = lhs.arg(0)
        scale_factor = m.scaling_factor[m.c2] / m.scaling_factor[m.v2]
        self.assertAlmostEqual(monom_factor, scale_factor)

        lhs = m.b.c4.body
        monom_factor = lhs.arg(0)
        scale_factor = m.b.scaling_factor[m.b.c4] / m.b.scaling_factor[m.b.v5]
        self.assertAlmostEqual(monom_factor, scale_factor)

    def test_scaling_no_solve(self):
        model = pyo.ConcreteModel()
        model.x = pyo.Var([1, 2, 3], bounds=(-10, 10), initialize=5.0)
        model.z = pyo.Var(bounds=(10, 20), initialize=15)

        def con_rule(m, i):
            if i == 1:
                return m.x[1] + 2 * m.x[2] + 1 * m.x[3] == 8.0
            if i == 2:
                return m.x[1] + 2 * m.x[2] + 2 * m.x[3] == 11.0
            if i == 3:
                return m.x[1] + 3.0 * m.x[2] + 1 * m.x[3] == 10.0

        model.con = pyo.Constraint([1, 2, 3], rule=con_rule)
        model.zcon = pyo.Constraint(expr=model.z >= model.x[2])

        model.x_ref = pyo.Reference(model.x)

        x_scale = 0.5
        obj_scale = 2.0
        z_scale = -10.0
        con_scale1 = 0.5
        con_scale2 = 2.0
        con_scale3 = -5.0
        zcon_scale = -3.0

        model.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
        model.scaling_factor[model.x] = x_scale
        model.scaling_factor[model.z] = z_scale
        model.scaling_factor[model.con[1]] = con_scale1
        model.scaling_factor[model.con[2]] = con_scale2
        model.scaling_factor[model.con[3]] = con_scale3
        model.scaling_factor[model.zcon] = zcon_scale

        # Set scaling factors for References too, but these should be ignored by the transformation
        model.scaling_factor[model.x_ref] = x_scale * 2

        scaled_model = pyo.TransformationFactory('core.scale_model').create_using(model)

        # check vars
        self.assertAlmostEqual(
            pyo.value(model.x[1]), pyo.value(scaled_model.scaled_x[1]) / x_scale, 4
        )
        self.assertAlmostEqual(
            pyo.value(model.x[2]), pyo.value(scaled_model.scaled_x[2]) / x_scale, 4
        )
        self.assertAlmostEqual(
            pyo.value(model.x[3]), pyo.value(scaled_model.scaled_x[3]) / x_scale, 4
        )
        self.assertAlmostEqual(
            pyo.value(model.z), pyo.value(scaled_model.scaled_z) / z_scale, 4
        )
        # check var lb
        self.assertAlmostEqual(
            pyo.value(model.x[1].lb),
            pyo.value(scaled_model.scaled_x[1].lb) / x_scale,
            4,
        )
        self.assertAlmostEqual(
            pyo.value(model.x[2].lb),
            pyo.value(scaled_model.scaled_x[2].lb) / x_scale,
            4,
        )
        self.assertAlmostEqual(
            pyo.value(model.x[3].lb),
            pyo.value(scaled_model.scaled_x[3].lb) / x_scale,
            4,
        )
        # note: z_scale is negative, therefore, the inequality directions swap
        self.assertAlmostEqual(
            pyo.value(model.z.lb), pyo.value(scaled_model.scaled_z.ub) / z_scale, 4
        )
        # check var ub
        self.assertAlmostEqual(
            pyo.value(model.x[1].ub),
            pyo.value(scaled_model.scaled_x[1].ub) / x_scale,
            4,
        )
        self.assertAlmostEqual(
            pyo.value(model.x[2].ub),
            pyo.value(scaled_model.scaled_x[2].ub) / x_scale,
            4,
        )
        self.assertAlmostEqual(
            pyo.value(model.x[3].ub),
            pyo.value(scaled_model.scaled_x[3].ub) / x_scale,
            4,
        )
        # note: z_scale is negative, therefore, the inequality directions swap
        self.assertAlmostEqual(
            pyo.value(model.z.ub), pyo.value(scaled_model.scaled_z.lb) / z_scale, 4
        )

        # check references to vars
        self.assertAlmostEqual(
            pyo.value(model.x[1]), pyo.value(scaled_model.scaled_x_ref[1]) / x_scale, 4
        )
        self.assertAlmostEqual(
            pyo.value(model.x[2]), pyo.value(scaled_model.scaled_x_ref[2]) / x_scale, 4
        )
        self.assertAlmostEqual(
            pyo.value(model.x[3]), pyo.value(scaled_model.scaled_x_ref[3]) / x_scale, 4
        )
        # check var lb
        self.assertAlmostEqual(
            pyo.value(model.x[1].lb),
            pyo.value(scaled_model.scaled_x_ref[1].lb) / x_scale,
            4,
        )
        self.assertAlmostEqual(
            pyo.value(model.x[2].lb),
            pyo.value(scaled_model.scaled_x_ref[2].lb) / x_scale,
            4,
        )
        self.assertAlmostEqual(
            pyo.value(model.x[3].lb),
            pyo.value(scaled_model.scaled_x_ref[3].lb) / x_scale,
            4,
        )
        # note: z_scale is negative, therefore, the inequality directions swap
        self.assertAlmostEqual(
            pyo.value(model.z.lb), pyo.value(scaled_model.scaled_z.ub) / z_scale, 4
        )
        # check var ub
        self.assertAlmostEqual(
            pyo.value(model.x[1].ub),
            pyo.value(scaled_model.scaled_x_ref[1].ub) / x_scale,
            4,
        )
        self.assertAlmostEqual(
            pyo.value(model.x[2].ub),
            pyo.value(scaled_model.scaled_x_ref[2].ub) / x_scale,
            4,
        )
        self.assertAlmostEqual(
            pyo.value(model.x[3].ub),
            pyo.value(scaled_model.scaled_x_ref[3].ub) / x_scale,
            4,
        )

        # check constraints
        self.assertAlmostEqual(
            pyo.value(model.con[1]),
            pyo.value(scaled_model.scaled_con[1]) / con_scale1,
            4,
        )
        self.assertAlmostEqual(
            pyo.value(model.con[2]),
            pyo.value(scaled_model.scaled_con[2]) / con_scale2,
            4,
        )
        self.assertAlmostEqual(
            pyo.value(model.con[3]),
            pyo.value(scaled_model.scaled_con[3]) / con_scale3,
            4,
        )
        self.assertAlmostEqual(
            pyo.value(model.zcon), pyo.value(scaled_model.scaled_zcon) / zcon_scale, 4
        )

        # Set values on scaled model and check that they map back to original
        scaled_model.scaled_x[1].set_value(1 * x_scale)
        scaled_model.scaled_x[2].set_value(2 * x_scale)
        scaled_model.scaled_x[3].set_value(3 * x_scale)
        scaled_model.scaled_z.set_value(10 * z_scale)

        # put the solution from the scaled back into the original
        pyo.TransformationFactory('core.scale_model').propagate_solution(
            scaled_model, model
        )

        # Check var values
        self.assertAlmostEqual(pyo.value(model.x[1]), 1, 4)
        self.assertAlmostEqual(pyo.value(model.x[2]), 2, 4)
        self.assertAlmostEqual(pyo.value(model.x[3]), 3, 4)
        self.assertAlmostEqual(pyo.value(model.z), 10, 4)

        # Check reference values
        self.assertAlmostEqual(pyo.value(model.x_ref[1]), 1, 4)
        self.assertAlmostEqual(pyo.value(model.x_ref[2]), 2, 4)
        self.assertAlmostEqual(pyo.value(model.x_ref[3]), 3, 4)

        # check constraints
        self.assertAlmostEqual(pyo.value(model.con[1]), 8, 4)
        self.assertAlmostEqual(pyo.value(model.con[2]), 11, 4)
        self.assertAlmostEqual(pyo.value(model.con[3]), 10, 4)
        self.assertAlmostEqual(pyo.value(model.zcon), -8, 4)

    def test_get_float_scaling_factor_top_level(self):
        m = pyo.ConcreteModel()
        m.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)

        m.b1 = pyo.Block()
        m.b1.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)

        m.b1.b2 = pyo.Block()
        m.b1.b2.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)

        m.v1 = pyo.Var(initialize=10)
        m.b1.v2 = pyo.Var(initialize=20)
        m.b1.b2.v3 = pyo.Var(initialize=30)

        m.scaling_factor[m.v1] = 0.1
        m.scaling_factor[m.b1.v2] = 0.2

        # SF should be 0.1 from top level
        sf = ScaleModel()._get_float_scaling_factor(m.v1)
        assert sf == float(0.1)
        # SF should be 0.1 from top level, lower level ignored
        sf = ScaleModel()._get_float_scaling_factor(m.b1.v2)
        assert sf == float(0.2)
        # No SF, should return 1
        sf = ScaleModel()._get_float_scaling_factor(m.b1.b2.v3)
        assert sf == 1.0

    def test_get_float_scaling_factor_local_level(self):
        m = pyo.ConcreteModel()
        m.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)

        m.b1 = pyo.Block()
        m.b1.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)

        m.b1.b2 = pyo.Block()
        m.b1.b2.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)

        m.v1 = pyo.Var(initialize=10)
        m.b1.v2 = pyo.Var(initialize=20)
        m.b1.b2.v3 = pyo.Var(initialize=30)

        m.scaling_factor[m.v1] = 0.1
        m.b1.scaling_factor[m.b1.v2] = 0.2
        m.b1.b2.scaling_factor[m.b1.b2.v3] = 0.3

        # Add an intermediate scaling factor - this should take priority
        m.b1.scaling_factor[m.b1.b2.v3] = 0.4

        # Should get SF from local levels
        sf = ScaleModel()._get_float_scaling_factor(m.v1)
        assert sf == float(0.1)
        sf = ScaleModel()._get_float_scaling_factor(m.b1.v2)
        assert sf == float(0.2)
        sf = ScaleModel()._get_float_scaling_factor(m.b1.b2.v3)
        assert sf == float(0.4)

    def test_get_float_scaling_factor_intermediate_level(self):
        m = pyo.ConcreteModel()
        m.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)

        m.b1 = pyo.Block()
        m.b1.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)

        m.b1.b2 = pyo.Block()
        # No suffix at b2 level - this should not cause an issue

        m.b1.b2.b3 = pyo.Block()
        m.b1.b2.b3.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)

        m.v1 = pyo.Var(initialize=10)
        m.b1.b2.b3.v2 = pyo.Var(initialize=20)
        m.b1.b2.b3.v3 = pyo.Var(initialize=30)

        # Scale v1 at lowest level - this should not get picked up
        m.b1.b2.b3.scaling_factor[m.v1] = 0.1

        m.b1.scaling_factor[m.b1.b2.b3.v2] = 0.2
        m.b1.scaling_factor[m.b1.b2.b3.v3] = 0.3

        m.b1.b2.b3.scaling_factor[m.b1.b2.b3.v3] = 0.4

        # v1 should be unscaled as SF set below variable level
        sf = ScaleModel()._get_float_scaling_factor(m.v1)
        assert sf == 1.0
        # v2 should get SF from b1 level
        sf = ScaleModel()._get_float_scaling_factor(m.b1.b2.b3.v2)
        assert sf == float(0.2)
        # v2 should get SF from highest level, ignoring b3 level
        sf = ScaleModel()._get_float_scaling_factor(m.b1.b2.b3.v3)
        assert sf == float(0.3)


if __name__ == "__main__":
    unittest.main()
