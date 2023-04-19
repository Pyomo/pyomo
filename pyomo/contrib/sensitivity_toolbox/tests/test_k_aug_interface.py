# ____________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and
# Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
# rights in this software.
# This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________

"""
"""
import os
import pyomo.common.unittest as unittest
from io import StringIO
import logging

import pyomo.environ as pyo
from pyomo.common.dependencies import (
    numpy as np,
    numpy_available,
    pandas as pd,
    pandas_available,
)
from pyomo.contrib.sensitivity_toolbox.sens import SensitivityInterface
from pyomo.contrib.sensitivity_toolbox.k_aug import K_augInterface

opt_ipopt = pyo.SolverFactory('ipopt', solver_io='nl', validate=False)
opt_k_aug = pyo.SolverFactory('k_aug', solver_io='nl', validate=False)
opt_dot_sens = pyo.SolverFactory('dot_sens', solver_io='nl', validate=False)


def simple_model_1():
    m = pyo.ConcreteModel()
    m.v1 = pyo.Var(initialize=10.0)
    m.v2 = pyo.Var(initialize=10.0)

    m.p = pyo.Param(mutable=True, initialize=1.0)

    m.eq_con = pyo.Constraint(expr=m.v1 * m.v2 - m.p == 0)

    m.obj = pyo.Objective(expr=m.v1**2 + m.v2**2, sense=pyo.minimize)

    return m


class TestK_augInterface(unittest.TestCase):
    @unittest.skipIf(not opt_k_aug.available(), "k_aug is not available")
    def test_clear_dir_k_aug(self):
        m = simple_model_1()
        sens = SensitivityInterface(m, clone_model=False)
        k_aug = K_augInterface()

        opt_ipopt.solve(m, tee=True)
        m.ptb = pyo.Param(mutable=True, initialize=1.5)

        cwd = os.getcwd()
        dir_contents = os.listdir(cwd)

        sens_param = [m.p]
        sens.setup_sensitivity(sens_param)

        k_aug.k_aug(m, tee=True)

        # We are back in our working directory
        self.assertEqual(cwd, os.getcwd())

        # The contents of this directory have not changed
        self.assertEqual(dir_contents, os.listdir(cwd))

        # In particular, the following files do not exist
        self.assertFalse(os.path.exists("dsdp_in_.in"))
        self.assertFalse(os.path.exists("conorder.txt"))
        self.assertFalse(os.path.exists("timings_k_aug_dsdp.txt"))

        # But they have been transferred to our k_aug interface's data
        # dict as strings.
        self.assertIsInstance(k_aug.data["dsdp_in_.in"], str)
        self.assertIsInstance(k_aug.data["conorder.txt"], str)
        self.assertIsInstance(k_aug.data["timings_k_aug_dsdp.txt"], str)

    @unittest.skipIf(not opt_k_aug.available(), "k_aug is not available")
    @unittest.skipIf(not opt_dot_sens.available(), "dot_sens is not available")
    def test_clear_dir_dot_sens(self):
        m = simple_model_1()
        sens = SensitivityInterface(m, clone_model=False)
        k_aug = K_augInterface()
        opt_ipopt.solve(m, tee=True)
        m.ptb = pyo.Param(mutable=True, initialize=1.5)

        cwd = os.getcwd()
        dir_contents = os.listdir(cwd)

        sens_param = [m.p]
        sens.setup_sensitivity(sens_param)

        # Call k_aug
        k_aug.k_aug(m, tee=True)
        self.assertIsInstance(k_aug.data["dsdp_in_.in"], str)

        sens.perturb_parameters([m.ptb])

        # Call dot_sens. In the process, we re-write dsdp_in_.in
        k_aug.dot_sens(m, tee=True)

        # Make sure we get the values we expect. This problem is easy enough
        # to solve by hand:
        # x = [1, 1, -2] = [v1, v2, dual]
        # Sensitivity system:
        # | 2 -2  1 |
        # |-2  2  1 | dx/dp = -[dL/dxdp, dc/dp]^T = -[0, 0, -1]^T
        # | 1  1  0 |
        # => dx/dp = [0.5, 0.5, 0]^T
        # Here, dp = [0.5]
        # => dx = [0.25, 0.25, 0]^T
        # => x_new = [1.25, 1.25, -2]
        self.assertAlmostEqual(m.v1.value, 1.25, 7)
        self.assertAlmostEqual(m.v2.value, 1.25, 7)

        # We are back in our working directory
        self.assertEqual(cwd, os.getcwd())

        # The contents of this directory have not changed
        self.assertEqual(dir_contents, os.listdir(cwd))
        self.assertFalse(os.path.exists("dsdp_in_.in"))
        self.assertFalse(os.path.exists("delta_p.out"))
        self.assertFalse(os.path.exists("dot_out.out"))
        self.assertFalse(os.path.exists("timings_dot_driver_dsdp.txt"))

        # And we have saved strings of the file contents.
        self.assertIsInstance(k_aug.data["dsdp_in_.in"], str)
        self.assertIsInstance(k_aug.data["delta_p.out"], str)
        self.assertIsInstance(k_aug.data["dot_out.out"], str)
        self.assertIsInstance(k_aug.data["timings_dot_driver_dsdp.txt"], str)


if __name__ == "__main__":
    unittest.main()
