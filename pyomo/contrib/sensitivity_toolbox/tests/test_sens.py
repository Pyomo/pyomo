#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

# ____________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and
# Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
# rights in this software.
# This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________

"""
Unit Tests for interfacing with sIPOPT and k_aug
"""

from io import StringIO
import logging
import pyomo.common.unittest as unittest

from pyomo.environ import ConcreteModel, Param, Var, Block, Suffix, value
from pyomo.opt import SolverFactory
from pyomo.dae import ContinuousSet
from pyomo.common.dependencies import scipy_available
from pyomo.common.log import LoggingIntercept
from pyomo.common.collections import ComponentMap
from pyomo.core.expr import identify_variables
from pyomo.contrib.sensitivity_toolbox.sens import sipopt, kaug, sensitivity_calculation
import pyomo.contrib.sensitivity_toolbox.examples.parameter as param_ex
import pyomo.contrib.sensitivity_toolbox.examples.parameter_kaug as param_kaug_ex
import pyomo.contrib.sensitivity_toolbox.examples.feedbackController as fc
import pyomo.contrib.sensitivity_toolbox.examples.rangeInequality as ri
import pyomo.contrib.sensitivity_toolbox.examples.HIV_Transmission as hiv

opt = SolverFactory('ipopt_sens', solver_io='nl')
opt_kaug = SolverFactory('k_aug', solver_io='nl')
opt_dotsens = SolverFactory('dot_sens', solver_io='nl')


class FunctionDeprecationTest(unittest.TestCase):
    @unittest.skipIf(not opt.available(False), "ipopt_sens is not available")
    def test_sipopt_deprecated(self):
        m = param_ex.create_model()
        m.perturbed_eta1 = Param(initialize=4.0)
        m.perturbed_eta2 = Param(initialize=1.0)

        output = StringIO()
        with LoggingIntercept(
            output, 'pyomo.contrib.sensitivity_toolbox', logging.WARNING
        ):
            sipopt(
                m,
                [m.eta1, m.eta1],
                [m.perturbed_eta1, m.perturbed_eta2],
                cloneModel=False,
            )
        self.assertIn(
            "DEPRECATED: The sipopt function has been deprecated. Use the "
            "sensitivity_calculation() function with method='sipopt' to access",
            output.getvalue().replace('\n', ' '),
        )

    @unittest.skipIf(not opt.available(False), "ipopt_sens is not available")
    def test_sipopt_equivalent(self):
        m1 = param_ex.create_model()
        m1.perturbed_eta1 = Param(initialize=4.0)
        m1.perturbed_eta2 = Param(initialize=1.0)

        m2 = param_ex.create_model()
        m2.perturbed_eta1 = Param(initialize=4.0)
        m2.perturbed_eta2 = Param(initialize=1.0)

        m11 = sipopt(
            m1,
            [m1.eta1, m1.eta2],
            [m1.perturbed_eta1, m1.perturbed_eta2],
            cloneModel=True,
        )
        m22 = sensitivity_calculation(
            'sipopt',
            m2,
            [m2.eta1, m2.eta2],
            [m2.perturbed_eta1, m2.perturbed_eta2],
            cloneModel=True,
        )
        out1 = StringIO()
        out2 = StringIO()
        m11._SENSITIVITY_TOOLBOX_DATA.constList.pprint(ostream=out1)
        m22._SENSITIVITY_TOOLBOX_DATA.constList.pprint(ostream=out2)
        self.assertMultiLineEqual(out1.getvalue(), out2.getvalue())

    @unittest.skipIf(not opt_kaug.available(False), "k_aug is not available")
    @unittest.skipIf(not opt_dotsens.available(False), "dot_sens is not available")
    def test_kaug_deprecated(self):
        m = param_ex.create_model()
        m.perturbed_eta1 = Param(initialize=4.0)
        m.perturbed_eta2 = Param(initialize=1.0)

        output = StringIO()
        with LoggingIntercept(
            output, 'pyomo.contrib.sensitivity_toolbox', logging.WARNING
        ):
            kaug(
                m,
                [m.eta1, m.eta1],
                [m.perturbed_eta1, m.perturbed_eta2],
                cloneModel=False,
            )
        self.assertIn(
            "DEPRECATED: The kaug function has been deprecated. Use the "
            "sensitivity_calculation() function with method='k_aug'",
            output.getvalue().replace('\n', ' '),
        )

    @unittest.skipIf(not opt_kaug.available(False), "k_aug is not available")
    @unittest.skipIf(not opt_dotsens.available(False), "dot_sens is not available")
    def test_kaug_equivalent(self):
        m1 = param_ex.create_model()
        m1.perturbed_eta1 = Param(initialize=4.0)
        m1.perturbed_eta2 = Param(initialize=1.0)

        m2 = param_ex.create_model()
        m2.perturbed_eta1 = Param(initialize=4.0)
        m2.perturbed_eta2 = Param(initialize=1.0)

        m11 = kaug(
            m1,
            [m1.eta1, m1.eta2],
            [m1.perturbed_eta1, m1.perturbed_eta2],
            cloneModel=True,
        )
        m22 = sensitivity_calculation(
            'k_aug',
            m2,
            [m2.eta1, m2.eta2],
            [m2.perturbed_eta1, m2.perturbed_eta2],
            cloneModel=True,
        )
        out1 = StringIO()
        out2 = StringIO()
        m11._SENSITIVITY_TOOLBOX_DATA.constList.pprint(ostream=out1)
        m22._SENSITIVITY_TOOLBOX_DATA.constList.pprint(ostream=out2)
        self.assertMultiLineEqual(out1.getvalue(), out2.getvalue())


class TestSensitivityToolbox(unittest.TestCase):
    @unittest.skipIf(not opt.available(False), "ipopt_sens is not available")
    def test_bad_arg(self):
        m = ConcreteModel()
        m.t = ContinuousSet(bounds=(0, 1))

        m.a = Param(initialize=1, mutable=True)
        m.b = Param(initialize=2, mutable=True)
        m.c = Param(initialize=3, mutable=False)

        m.x = Var(m.t)

        list_one = [m.a, m.b]
        list_two = [m.a, m.b, m.c]
        list_three = [m.a, m.x]
        list_four = [m.a, m.c]

        # verify ValueError thrown when param and perturb list are different
        # lengths
        msg = "Length of paramList argument does not equal length of perturbList"
        with self.assertRaisesRegex(ValueError, msg):
            Result = sensitivity_calculation('sipopt', m, list_one, list_two)

        # verify ValueError thrown when param list has an unmutable param
        msg = "Parameters within paramList must be mutable"
        with self.assertRaisesRegex(ValueError, msg):
            Result = sensitivity_calculation('sipopt', m, list_four, list_one)

        # verify ValueError thrown when param list has an unfixed var.
        msg = "Specified \"parameter\" variables must be fixed"
        with self.assertRaisesRegex(ValueError, msg) as context:
            Result = sensitivity_calculation('sipopt', m, list_three, list_one)

    # test feedbackController Solution when the model gets cloned
    @unittest.skipIf(not scipy_available, "scipy is required for this test")
    @unittest.skipIf(not opt.available(False), "ipopt_sens is not available")
    def test_clonedModel_soln(self):
        m_orig = fc.create_model()
        fc.initialize_model(m_orig, 100)

        m_orig.perturbed_a = Param(initialize=-0.25)
        m_orig.perturbed_H = Param(initialize=0.55)

        m_sipopt = sensitivity_calculation(
            'sipopt',
            m_orig,
            [m_orig.a, m_orig.H],
            [m_orig.perturbed_a, m_orig.perturbed_H],
            cloneModel=True,
        )

        # verify cloned model has _SENSITIVITY_TOOLBOX_DATA block
        # and original model is untouched
        self.assertFalse(m_sipopt == m_orig)

        self.assertTrue(
            hasattr(m_sipopt, '_SENSITIVITY_TOOLBOX_DATA')
            and m_sipopt._SENSITIVITY_TOOLBOX_DATA.ctype is Block
        )

        self.assertFalse(hasattr(m_orig, '_SENSITIVITY_TOOLBOX_DATA'))
        self.assertFalse(hasattr(m_orig, 'b'))

        # verify variable declaration
        self.assertTrue(
            hasattr(m_sipopt._SENSITIVITY_TOOLBOX_DATA, 'a')
            and m_sipopt._SENSITIVITY_TOOLBOX_DATA.a.ctype is Var
        )
        self.assertTrue(
            hasattr(m_sipopt._SENSITIVITY_TOOLBOX_DATA, 'H')
            and m_sipopt._SENSITIVITY_TOOLBOX_DATA.H.ctype is Var
        )

        # verify suffixes
        self.assertTrue(
            hasattr(m_sipopt, 'sens_state_0')
            and m_sipopt.sens_state_0.ctype is Suffix
            and m_sipopt.sens_state_0[m_sipopt._SENSITIVITY_TOOLBOX_DATA.H] == 2
            and m_sipopt.sens_state_0[m_sipopt._SENSITIVITY_TOOLBOX_DATA.a] == 1
        )

        self.assertTrue(
            hasattr(m_sipopt, 'sens_state_1')
            and m_sipopt.sens_state_1.ctype is Suffix
            and m_sipopt.sens_state_1[m_sipopt._SENSITIVITY_TOOLBOX_DATA.H] == 2
            and m_sipopt.sens_state_1[m_sipopt._SENSITIVITY_TOOLBOX_DATA.a] == 1
        )

        self.assertTrue(
            hasattr(m_sipopt, 'sens_state_value_1')
            and m_sipopt.sens_state_value_1.ctype is Suffix
            and m_sipopt.sens_state_value_1[m_sipopt._SENSITIVITY_TOOLBOX_DATA.H]
            == 0.55
            and m_sipopt.sens_state_value_1[m_sipopt._SENSITIVITY_TOOLBOX_DATA.a]
            == -0.25
        )

        self.assertTrue(
            hasattr(m_sipopt, 'sens_init_constr')
            and m_sipopt.sens_init_constr.ctype is Suffix
            and m_sipopt.sens_init_constr[
                m_sipopt._SENSITIVITY_TOOLBOX_DATA.paramConst[1]
            ]
            == 1
            and m_sipopt.sens_init_constr[
                m_sipopt._SENSITIVITY_TOOLBOX_DATA.paramConst[2]
            ]
            == 2
        )

        self.assertTrue(
            hasattr(m_sipopt, 'sens_sol_state_1')
            and m_sipopt.sens_sol_state_1.ctype is Suffix
        )
        self.assertAlmostEqual(
            m_sipopt.sens_sol_state_1[m_sipopt.F[15]], -0.00102016765, 8
        )

        # These tests require way too much precision for something that
        # just needs to enforce that bounds are not active...
        self.assertTrue(
            hasattr(m_sipopt, 'sens_sol_state_1_z_L')
            and m_sipopt.sens_sol_state_1_z_L.ctype is Suffix
        )
        self.assertAlmostEqual(
            m_sipopt.sens_sol_state_1_z_L[m_sipopt.u[15]], -2.181712e-09, 13
        )

        self.assertTrue(
            hasattr(m_sipopt, 'sens_sol_state_1_z_U')
            and m_sipopt.sens_sol_state_1_z_U.ctype is Suffix
        )
        self.assertAlmostEqual(
            m_sipopt.sens_sol_state_1_z_U[m_sipopt.u[15]], 6.580899e-09, 13
        )

        # verify deactivated constraints for cloned model
        self.assertFalse(
            m_sipopt.FDiffCon[0].active
            and m_sipopt.FDiffCon[7.5].active
            and m_sipopt.FDiffCon[15].active
        )

        self.assertFalse(
            m_sipopt.x_dot[0].active
            and m_sipopt.x_dot[7.5].active
            and m_sipopt.x_dot[15].active
        )

        # verify constraints on original model are still active
        self.assertTrue(
            m_orig.FDiffCon[0].active
            and m_orig.FDiffCon[7.5].active
            and m_orig.FDiffCon[15].active
        )

        self.assertTrue(
            m_orig.x_dot[0].active
            and m_orig.x_dot[7.5].active
            and m_orig.x_dot[15].active
        )

        # verify solution
        # NOTE: This is the solution to the original problem,
        # not the result of any sensitivity update.
        self.assertAlmostEqual(value(m_sipopt.J), 0.0048956783, 8)

    @unittest.skipIf(not scipy_available, "scipy is required for this test")
    @unittest.skipIf(not opt.available(False), "ipopt_sens is not available")
    def test_noClone_soln(self):
        m_orig = fc.create_model()
        fc.initialize_model(m_orig, 100)

        m_orig.perturbed_a = Param(initialize=-0.25)
        m_orig.perturbed_H = Param(initialize=0.55)

        m_sipopt = sensitivity_calculation(
            'sipopt',
            m_orig,
            [m_orig.a, m_orig.H],
            [m_orig.perturbed_a, m_orig.perturbed_H],
            cloneModel=False,
        )

        self.assertTrue(m_sipopt == m_orig)

        # test _SENSITIVITY_TOOLBOX_DATA block exists
        self.assertTrue(
            hasattr(m_orig, '_SENSITIVITY_TOOLBOX_DATA')
            and m_orig._SENSITIVITY_TOOLBOX_DATA.ctype is Block
        )

        # test variable declaration
        self.assertTrue(
            hasattr(m_sipopt._SENSITIVITY_TOOLBOX_DATA, 'a')
            and m_sipopt._SENSITIVITY_TOOLBOX_DATA.a.ctype is Var
        )
        self.assertTrue(
            hasattr(m_sipopt._SENSITIVITY_TOOLBOX_DATA, 'H')
            and m_sipopt._SENSITIVITY_TOOLBOX_DATA.H.ctype is Var
        )

        # test for suffixes
        self.assertTrue(
            hasattr(m_sipopt, 'sens_state_0')
            and m_sipopt.sens_state_0.ctype is Suffix
            and m_sipopt.sens_state_0[m_sipopt._SENSITIVITY_TOOLBOX_DATA.H] == 2
            and m_sipopt.sens_state_0[m_sipopt._SENSITIVITY_TOOLBOX_DATA.a] == 1
        )

        self.assertTrue(
            hasattr(m_sipopt, 'sens_state_1')
            and m_sipopt.sens_state_1.ctype is Suffix
            and m_sipopt.sens_state_1[m_sipopt._SENSITIVITY_TOOLBOX_DATA.H] == 2
            and m_sipopt.sens_state_1[m_sipopt._SENSITIVITY_TOOLBOX_DATA.a] == 1
        )

        self.assertTrue(
            hasattr(m_sipopt, 'sens_state_value_1')
            and m_sipopt.sens_state_value_1.ctype is Suffix
            and m_sipopt.sens_state_value_1[m_sipopt._SENSITIVITY_TOOLBOX_DATA.H]
            == 0.55
            and m_sipopt.sens_state_value_1[m_sipopt._SENSITIVITY_TOOLBOX_DATA.a]
            == -0.25
        )

        self.assertTrue(
            hasattr(m_sipopt, 'sens_init_constr')
            and m_sipopt.sens_init_constr.ctype is Suffix
            and m_sipopt.sens_init_constr[
                m_sipopt._SENSITIVITY_TOOLBOX_DATA.paramConst[1]
            ]
            == 1
            and m_sipopt.sens_init_constr[
                m_sipopt._SENSITIVITY_TOOLBOX_DATA.paramConst[2]
            ]
            == 2
        )

        self.assertTrue(
            hasattr(m_sipopt, 'sens_sol_state_1')
            and m_sipopt.sens_sol_state_1.ctype is Suffix
        )
        self.assertAlmostEqual(
            m_sipopt.sens_sol_state_1[m_sipopt.F[15]], -0.00102016765, 8
        )

        self.assertTrue(
            hasattr(m_sipopt, 'sens_sol_state_1_z_L')
            and m_sipopt.sens_sol_state_1_z_L.ctype is Suffix
        )
        self.assertAlmostEqual(
            m_sipopt.sens_sol_state_1_z_L[m_sipopt.u[15]], -2.181712e-09, 13
        )

        self.assertTrue(
            hasattr(m_sipopt, 'sens_sol_state_1_z_U')
            and m_sipopt.sens_sol_state_1_z_U.ctype is Suffix
        )
        self.assertAlmostEqual(
            m_sipopt.sens_sol_state_1_z_U[m_sipopt.u[15]], 6.580899e-09, 13
        )

        # verify deactivated constraints on model
        self.assertFalse(
            m_sipopt.FDiffCon[0].active
            and m_sipopt.FDiffCon[7.5].active
            and m_sipopt.FDiffCon[15].active
        )

        self.assertFalse(
            m_sipopt.x_dot[0].active
            and m_sipopt.x_dot[7.5].active
            and m_sipopt.x_dot[15].active
        )

        # test model solution
        # NOTE:
        # ipopt_sens does not alter the values in the model,
        # so all this test is doing is making sure that the
        # objective value doesn't change. This test does nothing to
        # check values of the perturbed solution.
        self.assertAlmostEqual(value(m_sipopt.J), 0.0048956783, 8)

    # test indexed param mapping to var and perturbed values
    @unittest.skipIf(not scipy_available, "scipy is required for this test")
    @unittest.skipIf(not opt.available(False), "ipopt_sens is not available")
    def test_indexedParamsMapping(self):
        m = hiv.create_model()
        hiv.initialize_model(m, 10, 5, 1)

        m.epsDelta = Param(initialize=0.75001)

        q_del = {}
        q_del[(0, 0)] = 1.001
        q_del[(0, 1)] = 1.002
        q_del[(1, 0)] = 1.003
        q_del[(1, 1)] = 1.004
        q_del[(2, 0)] = 0.83001
        q_del[(2, 1)] = 0.83002
        q_del[(3, 0)] = 0.42001
        q_del[(4, 0)] = 0.17001
        m.qqDelta = Param(m.ij, initialize=q_del)

        m.aaDelta = Param(initialize=0.0001001)

        m_sipopt = sensitivity_calculation(
            'sipopt', m, [m.eps, m.qq, m.aa], [m.epsDelta, m.qqDelta, m.aaDelta]
        )

        # Make sure Param constraints have the correct form, i.e.
        # 0 <= _SENSITIVITY_TOOLBOX_DATA.PARAM_NAME - PARAM_NAME <= 0
        self.assertEqual(m_sipopt._SENSITIVITY_TOOLBOX_DATA.paramConst[1].lower, 0.0)
        self.assertEqual(m_sipopt._SENSITIVITY_TOOLBOX_DATA.paramConst[1].upper, 0.0)
        self.assertEqual(
            m_sipopt._SENSITIVITY_TOOLBOX_DATA.paramConst[1].body.to_string(),
            '_SENSITIVITY_TOOLBOX_DATA.eps - eps',
        )
        self.assertEqual(m_sipopt._SENSITIVITY_TOOLBOX_DATA.paramConst[6].lower, 0.0)
        self.assertEqual(m_sipopt._SENSITIVITY_TOOLBOX_DATA.paramConst[6].upper, 0.0)
        self.assertEqual(
            m_sipopt._SENSITIVITY_TOOLBOX_DATA.paramConst[6].body.to_string(),
            '_SENSITIVITY_TOOLBOX_DATA.qq[2,0] - qq[2,0]',
        )
        self.assertEqual(m_sipopt._SENSITIVITY_TOOLBOX_DATA.paramConst[10].lower, 0.0)
        self.assertEqual(m_sipopt._SENSITIVITY_TOOLBOX_DATA.paramConst[10].upper, 0.0)
        self.assertEqual(
            m_sipopt._SENSITIVITY_TOOLBOX_DATA.paramConst[10].body.to_string(),
            '_SENSITIVITY_TOOLBOX_DATA.aa - aa',
        )

    # test Constraint substitution
    @unittest.skipIf(not opt.available(False), "ipopt_sens is not available")
    def test_constraintSub(self):
        m = ri.create_model()

        m.pert_a = Param(initialize=0.01)
        m.pert_b = Param(initialize=1.01)

        m_sipopt = sensitivity_calculation(
            'sipopt', m, [m.a, m.b], [m.pert_a, m.pert_b]
        )

        # verify substitutions in equality constraint
        self.assertTrue(
            m_sipopt.C_equal.lower.ctype is Param
            and m_sipopt.C_equal.upper.ctype is Param
        )
        self.assertFalse(m_sipopt.C_equal.active)

        self.assertTrue(
            m_sipopt._SENSITIVITY_TOOLBOX_DATA.constList[3].lower == 0.0
            and m_sipopt._SENSITIVITY_TOOLBOX_DATA.constList[3].upper == 0.0
            and len(
                list(
                    identify_variables(
                        m_sipopt._SENSITIVITY_TOOLBOX_DATA.constList[3].body
                    )
                )
            )
            == 2
        )

        # verify substitutions in one-sided bounded constraint
        self.assertTrue(
            m_sipopt.C_singleBnd.lower is None
            and m_sipopt.C_singleBnd.upper.ctype is Param
        )
        self.assertFalse(m_sipopt.C_singleBnd.active)

        self.assertTrue(
            m_sipopt._SENSITIVITY_TOOLBOX_DATA.constList[4].lower is None
            and m_sipopt._SENSITIVITY_TOOLBOX_DATA.constList[4].upper == 0.0
            and len(
                list(
                    identify_variables(
                        m_sipopt._SENSITIVITY_TOOLBOX_DATA.constList[4].body
                    )
                )
            )
            == 2
        )

        # verify substitutions in ranged inequality constraint
        self.assertTrue(
            m_sipopt.C_rangedIn.lower.ctype is Param
            and m_sipopt.C_rangedIn.upper.ctype is Param
        )
        self.assertFalse(m_sipopt.C_rangedIn.active)

        self.assertTrue(
            m_sipopt._SENSITIVITY_TOOLBOX_DATA.constList[1].lower is None
            and m_sipopt._SENSITIVITY_TOOLBOX_DATA.constList[1].upper == 0.0
            and len(
                list(
                    identify_variables(
                        m_sipopt._SENSITIVITY_TOOLBOX_DATA.constList[1].body
                    )
                )
            )
            == 2
        )

        self.assertTrue(
            m_sipopt._SENSITIVITY_TOOLBOX_DATA.constList[2].lower is None
            and m_sipopt._SENSITIVITY_TOOLBOX_DATA.constList[2].upper == 0.0
            and len(
                list(
                    identify_variables(
                        m_sipopt._SENSITIVITY_TOOLBOX_DATA.constList[2].body
                    )
                )
            )
            == 2
        )

    # Test example `parameter.py`
    @unittest.skipIf(not opt.available(False), "ipopt_sens is not available")
    def test_parameter_example(self):
        d = param_ex.run_example()

        d_correct = {
            'eta1': 4.5,
            'eta2': 1.0,
            'x1_init': 0.15,
            'x2_init': 0.15,
            'x3_init': 0.0,
            'cost_sln': 0.5,
            'x1_sln': 0.5,
            'x2_sln': 0.5,
            'x3_sln': 0.0,
            'eta1_pert': 4.0,
            'eta2_pert': 1.0,
            'x1_pert': 0.3333333,
            'x2_pert': 0.6666667,
            'x3_pert': 0.0,
            'cost_pert': 0.55555556,
        }

        for k in d_correct.keys():
            # Check each element of the 'correct' dictionary against the returned
            # dictionary to 3 decimal places
            self.assertAlmostEqual(d[k], d_correct[k], 3)

    # Test kaug
    # Perform the same tests as for sipopt
    # test feedbackController Solution when the model gets cloned
    @unittest.skipIf(not scipy_available, "scipy is required for this test")
    @unittest.skipIf(not opt_kaug.available(False), "k_aug is not available")
    @unittest.skipIf(not opt_dotsens.available(False), "dot_sens is not available")
    def test_kaug_clonedModel_soln_kaug(self):
        m_orig = fc.create_model()
        fc.initialize_model(m_orig, 100)

        m_orig.perturbed_a = Param(initialize=-0.25)
        m_orig.perturbed_H = Param(initialize=0.55)

        m_kaug = sensitivity_calculation(
            'k_aug',
            m_orig,
            [m_orig.a, m_orig.H],
            [m_orig.perturbed_a, m_orig.perturbed_H],
            cloneModel=True,
        )

        ptb_map = ComponentMap()
        ptb_map[m_kaug.a] = value(-(m_orig.perturbed_a - m_orig.a))
        ptb_map[m_kaug.H] = value(-(m_orig.perturbed_H - m_orig.H))

        # verify cloned model has _SENSITIVITY_TOOLBOX_DATA block
        # and original model is untouched
        self.assertIsNot(m_kaug, m_orig)

        self.assertTrue(
            hasattr(m_kaug, '_SENSITIVITY_TOOLBOX_DATA')
            and m_kaug._SENSITIVITY_TOOLBOX_DATA.ctype is Block
        )

        self.assertFalse(hasattr(m_orig, '_SENSITIVITY_TOOLBOX_DATA'))
        self.assertFalse(hasattr(m_orig, 'b'))

        # verify variable declaration
        self.assertTrue(
            hasattr(m_kaug._SENSITIVITY_TOOLBOX_DATA, 'a')
            and m_kaug._SENSITIVITY_TOOLBOX_DATA.a.ctype is Var
        )
        self.assertTrue(
            hasattr(m_kaug._SENSITIVITY_TOOLBOX_DATA, 'H')
            and m_kaug._SENSITIVITY_TOOLBOX_DATA.H.ctype is Var
        )

        # verify suffixes
        self.assertTrue(
            hasattr(m_kaug, 'sens_state_0')
            and m_kaug.sens_state_0.ctype is Suffix
            and m_kaug.sens_state_0[m_kaug._SENSITIVITY_TOOLBOX_DATA.H] == 2
            and m_kaug.sens_state_0[m_kaug._SENSITIVITY_TOOLBOX_DATA.a] == 1
        )
        self.assertTrue(
            hasattr(m_kaug, 'sens_state_1')
            and m_kaug.sens_state_1.ctype is Suffix
            and m_kaug.sens_state_1[m_kaug._SENSITIVITY_TOOLBOX_DATA.H] == 2
            and m_kaug.sens_state_1[m_kaug._SENSITIVITY_TOOLBOX_DATA.a] == 1
        )
        self.assertTrue(
            hasattr(m_kaug, 'sens_state_value_1')
            and m_kaug.sens_state_value_1.ctype is Suffix
            and m_kaug.sens_state_value_1[m_kaug._SENSITIVITY_TOOLBOX_DATA.H] == 0.55
            and m_kaug.sens_state_value_1[m_kaug._SENSITIVITY_TOOLBOX_DATA.a] == -0.25
        )
        self.assertTrue(
            hasattr(m_kaug, 'sens_init_constr')
            and m_kaug.sens_init_constr.ctype is Suffix
            and m_kaug.sens_init_constr[m_kaug._SENSITIVITY_TOOLBOX_DATA.paramConst[1]]
            == 1
            and m_kaug.sens_init_constr[m_kaug._SENSITIVITY_TOOLBOX_DATA.paramConst[2]]
            == 2
        )
        self.assertTrue(hasattr(m_kaug, 'DeltaP'))
        self.assertTrue(m_kaug.DeltaP.ctype is Suffix)
        self.assertEqual(
            m_kaug.DeltaP[m_kaug._SENSITIVITY_TOOLBOX_DATA.paramConst[1]],
            ptb_map[m_kaug.a],
        )
        self.assertEqual(
            m_kaug.DeltaP[m_kaug._SENSITIVITY_TOOLBOX_DATA.paramConst[2]],
            ptb_map[m_kaug.H],
        )
        self.assertTrue(
            hasattr(m_kaug, 'dcdp')
            and m_kaug.dcdp.ctype is Suffix
            and m_kaug.dcdp[m_kaug._SENSITIVITY_TOOLBOX_DATA.paramConst[1]] == 1
            and m_kaug.dcdp[m_kaug._SENSITIVITY_TOOLBOX_DATA.paramConst[2]] == 2
        )
        self.assertTrue(
            hasattr(m_kaug, 'sens_sol_state_1')
            and m_kaug.sens_sol_state_1.ctype is Suffix
        )

        self.assertTrue(
            hasattr(m_kaug, 'ipopt_zL_in') and m_kaug.ipopt_zL_in.ctype is Suffix
        )
        self.assertAlmostEqual(
            m_kaug.ipopt_zL_in[m_kaug.u[15]], 7.162686166847096e-09, 13
        )

        self.assertTrue(
            hasattr(m_kaug, 'ipopt_zU_in') and m_kaug.ipopt_zU_in.ctype is Suffix
        )
        self.assertAlmostEqual(
            m_kaug.ipopt_zU_in[m_kaug.u[15]], -1.2439730261288605e-08, 13
        )
        # verify deactivated constraints for cloned model
        self.assertFalse(
            m_kaug.FDiffCon[0].active
            and m_kaug.FDiffCon[7.5].active
            and m_kaug.FDiffCon[15].active
        )

        self.assertFalse(
            m_kaug.x_dot[0].active
            and m_kaug.x_dot[7.5].active
            and m_kaug.x_dot[15].active
        )

        # verify constraints on original model are still active
        self.assertTrue(
            m_orig.FDiffCon[0].active
            and m_orig.FDiffCon[7.5].active
            and m_orig.FDiffCon[15].active
        )

        self.assertTrue(
            m_orig.x_dot[0].active
            and m_orig.x_dot[7.5].active
            and m_orig.x_dot[15].active
        )

        # verify solution
        # This is the only test that verifies the solution. Here we
        # verify the objective function value, which is a weak test.
        self.assertAlmostEqual(value(m_kaug.J), 0.002633263921107476, 8)
        # The original objective function value is 0.0048.
        # The answer from an attempt to reproduce this calculation with
        # PyNumero, with no inertia correction, seems to be 0.00047.
        # "Real solution" with the full nonlinear problem is 0.00138.
        # 0.00263 is the value we get after sensitivity update with k_aug
        # using MA57 and k_aug's default regularization strategy.

    @unittest.skipIf(not scipy_available, "scipy is required for this test")
    @unittest.skipIf(not opt_kaug.available(False), "k_aug is not available")
    @unittest.skipIf(not opt_dotsens.available(False), "dot_sens is not available")
    def test_noClone_soln_kaug(self):
        m_orig = fc.create_model()
        fc.initialize_model(m_orig, 100)

        m_orig.perturbed_a = Param(initialize=-0.25)
        m_orig.perturbed_H = Param(initialize=0.55)

        m_kaug = sensitivity_calculation(
            'k_aug',
            m_orig,
            [m_orig.a, m_orig.H],
            [m_orig.perturbed_a, m_orig.perturbed_H],
            cloneModel=False,
        )

        ptb_map = ComponentMap()
        ptb_map[m_kaug.a] = value(-(m_kaug.perturbed_a - m_kaug.a))
        ptb_map[m_kaug.H] = value(-(m_kaug.perturbed_H - m_kaug.H))

        self.assertTrue(m_kaug == m_orig)

        # verify suffixes
        self.assertTrue(
            hasattr(m_kaug, 'sens_state_0')
            and m_kaug.sens_state_0.ctype is Suffix
            and m_kaug.sens_state_0[m_kaug._SENSITIVITY_TOOLBOX_DATA.H] == 2
            and m_kaug.sens_state_0[m_kaug._SENSITIVITY_TOOLBOX_DATA.a] == 1
        )
        self.assertTrue(
            hasattr(m_kaug, 'sens_state_1')
            and m_kaug.sens_state_1.ctype is Suffix
            and m_kaug.sens_state_1[m_kaug._SENSITIVITY_TOOLBOX_DATA.H] == 2
            and m_kaug.sens_state_1[m_kaug._SENSITIVITY_TOOLBOX_DATA.a] == 1
        )
        self.assertTrue(
            hasattr(m_kaug, 'sens_state_value_1')
            and m_kaug.sens_state_value_1.ctype is Suffix
            and m_kaug.sens_state_value_1[m_kaug._SENSITIVITY_TOOLBOX_DATA.H] == 0.55
            and m_kaug.sens_state_value_1[m_kaug._SENSITIVITY_TOOLBOX_DATA.a] == -0.25
        )
        self.assertTrue(
            hasattr(m_kaug, 'sens_init_constr')
            and m_kaug.sens_init_constr.ctype is Suffix
            and m_kaug.sens_init_constr[m_kaug._SENSITIVITY_TOOLBOX_DATA.paramConst[1]]
            == 1
            and m_kaug.sens_init_constr[m_kaug._SENSITIVITY_TOOLBOX_DATA.paramConst[2]]
            == 2
        )
        self.assertTrue(hasattr(m_kaug, 'DeltaP'))
        self.assertIs(m_kaug.DeltaP.ctype, Suffix)
        self.assertEqual(
            m_kaug.DeltaP[m_kaug._SENSITIVITY_TOOLBOX_DATA.paramConst[1]],
            ptb_map[m_kaug.a],
        )
        self.assertEqual(
            m_kaug.DeltaP[m_kaug._SENSITIVITY_TOOLBOX_DATA.paramConst[2]],
            ptb_map[m_kaug.H],
        )
        self.assertTrue(
            hasattr(m_kaug, 'dcdp')
            and m_kaug.dcdp.ctype is Suffix
            and m_kaug.dcdp[m_kaug._SENSITIVITY_TOOLBOX_DATA.paramConst[1]] == 1
            and m_kaug.dcdp[m_kaug._SENSITIVITY_TOOLBOX_DATA.paramConst[2]] == 2
        )
        self.assertTrue(
            hasattr(m_kaug, 'sens_sol_state_1')
            and m_kaug.sens_sol_state_1.ctype is Suffix
        )

        self.assertTrue(
            hasattr(m_kaug, 'ipopt_zL_in') and m_kaug.ipopt_zL_in.ctype is Suffix
        )
        self.assertAlmostEqual(
            m_kaug.ipopt_zL_in[m_kaug.u[15]], 7.162686166847096e-09, 13
        )

        self.assertTrue(
            hasattr(m_kaug, 'ipopt_zU_in') and m_kaug.ipopt_zU_in.ctype is Suffix
        )
        self.assertAlmostEqual(
            m_kaug.ipopt_zU_in[m_kaug.u[15]], -1.2439730261288605e-08, 13
        )
        # verify deactivated constraints for cloned model
        self.assertFalse(
            m_kaug.FDiffCon[0].active
            and m_kaug.FDiffCon[7.5].active
            and m_kaug.FDiffCon[15].active
        )

        self.assertFalse(
            m_kaug.x_dot[0].active
            and m_kaug.x_dot[7.5].active
            and m_kaug.x_dot[15].active
        )

        # verify solution
        # This is the only test that verifies the solution. Here we
        # verify the objective function value, which is a weak test.
        self.assertAlmostEqual(value(m_kaug.J), 0.002633263921107476, 8)
        # The original objective function value is 0.0048.
        # The answer from an attempt to reproduce this calculation with
        # PyNumero, with no inertia correction, seems to be 0.00047.
        # "Real solution" with the full nonlinear problem is 0.00138.
        # 0.00263 is the value we get after sensitivity update with k_aug
        # using MA57 and k_aug's default regularization strategy.

    # test indexed param mapping to var and perturbed values
    @unittest.skipIf(not scipy_available, "scipy is required for this test")
    @unittest.skipIf(not opt_kaug.available(False), "k_aug is not available")
    @unittest.skipIf(not opt_dotsens.available(False), "dot_sens is not available")
    def test_indexedParamsMapping_kaug(self):
        m = hiv.create_model()
        hiv.initialize_model(m, 10, 5, 1)

        m.epsDelta = Param(initialize=0.75001)

        q_del = {}
        q_del[(0, 0)] = 1.001
        q_del[(0, 1)] = 1.002
        q_del[(1, 0)] = 1.003
        q_del[(1, 1)] = 1.004
        q_del[(2, 0)] = 0.83001
        q_del[(2, 1)] = 0.83002
        q_del[(3, 0)] = 0.42001
        q_del[(4, 0)] = 0.17001
        m.qqDelta = Param(m.ij, initialize=q_del)

        m.aaDelta = Param(initialize=0.0001001)

        m_kaug = sensitivity_calculation(
            'k_aug', m, [m.eps, m.qq, m.aa], [m.epsDelta, m.qqDelta, m.aaDelta]
        )

        # Make sure Param constraints have the correct form, i.e.
        # 0 <= _SENSITIVITY_TOOLBOX_DATA.PARAM_NAME - PARAM_NAME <= 0
        self.assertEqual(m_kaug._SENSITIVITY_TOOLBOX_DATA.paramConst[1].lower, 0.0)
        self.assertEqual(m_kaug._SENSITIVITY_TOOLBOX_DATA.paramConst[1].upper, 0.0)
        self.assertEqual(
            m_kaug._SENSITIVITY_TOOLBOX_DATA.paramConst[1].body.to_string(),
            '_SENSITIVITY_TOOLBOX_DATA.eps - eps',
        )
        self.assertEqual(m_kaug._SENSITIVITY_TOOLBOX_DATA.paramConst[6].lower, 0.0)
        self.assertEqual(m_kaug._SENSITIVITY_TOOLBOX_DATA.paramConst[6].upper, 0.0)
        self.assertEqual(
            m_kaug._SENSITIVITY_TOOLBOX_DATA.paramConst[6].body.to_string(),
            '_SENSITIVITY_TOOLBOX_DATA.qq[2,0] - qq[2,0]',
        )
        self.assertEqual(m_kaug._SENSITIVITY_TOOLBOX_DATA.paramConst[10].lower, 0.0)
        self.assertEqual(m_kaug._SENSITIVITY_TOOLBOX_DATA.paramConst[10].upper, 0.0)
        self.assertEqual(
            m_kaug._SENSITIVITY_TOOLBOX_DATA.paramConst[10].body.to_string(),
            '_SENSITIVITY_TOOLBOX_DATA.aa - aa',
        )

    # Test example `parameter_kaug.py`
    @unittest.skipIf(not opt_kaug.available(False), "k_aug is not available")
    @unittest.skipIf(not opt_dotsens.available(False), "dot_sens is not available")
    def test_parameter_example_kaug(self):
        d = param_kaug_ex.run_example()

        d_correct = {
            'eta1': 4.5,
            'eta2': 1.0,
            'x1_init': 0.15,
            'x2_init': 0.15,
            'x3_init': 0.0,
            'eta1_pert': 4.0,
            'eta2_pert': 1.0,
            'x1_pert': 0.3333333,
            'x2_pert': 0.6666667,
            'x3_pert': 0.0,
            'cost_pert': 0.55555556,
        }

        for k in d_correct.keys():
            # Check each element of the 'correct' dictionary against the returned
            # dictionary to 3 decimal places
            self.assertAlmostEqual(d[k], d_correct[k], 3)


if __name__ == "__main__":
    unittest.main()
