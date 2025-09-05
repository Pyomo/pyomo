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

import platform
import math
import pyomo.environ as pyo
import pyomo.common.unittest as unittest
from pyomo.common.fileutils import find_library

flib = find_library("aslfunctions")
is_pypy = platform.python_implementation().lower().startswith("pypy")


@unittest.skipIf(not flib, 'Could not find the "aslfunctions" library')
@unittest.skipIf(is_pypy, "Cannot evaluate external functions under pypy")
class TestAMPLExternalFunction(unittest.TestCase):

    def test_eval_sqnsqr_function_fgh(self):
        m = pyo.ConcreteModel()
        m.tf = pyo.ExternalFunction(library=flib, function="sgnsqr")

        f, g, h = m.tf.evaluate_fgh((2,))
        self.assertEqual(f, 4)
        self.assertEqual(g, [4])
        self.assertEqual(h, [2])

        f, g, h = m.tf.evaluate_fgh((-2,))
        self.assertAlmostEqual(f, -4)
        self.assertStructuredAlmostEqual(g, [4])
        self.assertStructuredAlmostEqual(h, [-2])

    def test_eval_sqnsqr_c4_function_fgh(self):
        m = pyo.ConcreteModel()
        m.tf = pyo.ExternalFunction(library=flib, function="sgnsqr_c4")

        f, g, h = m.tf.evaluate_fgh((2,))
        self.assertEqual(f, 4)
        self.assertEqual(g, [4])
        self.assertEqual(h, [2])

        f, g, h = m.tf.evaluate_fgh((-2,))
        self.assertAlmostEqual(f, -4)
        self.assertStructuredAlmostEqual(g, [4])
        self.assertStructuredAlmostEqual(h, [-2])

        dx = 1e-9
        x = 0.1

        f1, g1, h1 = m.tf.evaluate_fgh((x + dx,))
        f2, g2, h2 = m.tf.evaluate_fgh((x - dx,))
        self.assertAlmostEqual(f1, f2)
        self.assertStructuredAlmostEqual(g1, g2)
        self.assertStructuredAlmostEqual(h1, h2)

        dx = 1e-9
        x = -0.1

        f1, g1, h1 = m.tf.evaluate_fgh((x + dx,))
        f2, g2, h2 = m.tf.evaluate_fgh((x - dx,))
        self.assertAlmostEqual(f1, f2)
        self.assertStructuredAlmostEqual(g1, g2)
        self.assertStructuredAlmostEqual(h1, h2)

    def test_eval_sinc_function_fgh(self):
        m = pyo.ConcreteModel()
        m.tf = pyo.ExternalFunction(library=flib, function="sinc")

        f, g, h = m.tf.evaluate_fgh((0,))
        self.assertAlmostEqual(f, 1.0)
        self.assertAlmostEqual(g, [0])
        self.assertAlmostEqual(h, [-1 / 3.0])

        dx = 1e-10
        f, g, h = m.tf.evaluate_fgh((2,))
        self.assertAlmostEqual(f, math.sin(2) / 2)
        self.assertStructuredAlmostEqual(g, [(math.sin(2 + dx) / (2 + dx) - f) / dx], 5)

        dx = 1e-10
        x = 0.1

        f1, g1, h1 = m.tf.evaluate_fgh((x + dx,))
        f2, g2, h2 = m.tf.evaluate_fgh((x - dx,))
        self.assertAlmostEqual(f1, f2)
        self.assertStructuredAlmostEqual(g1, g2)
        self.assertStructuredAlmostEqual(h1, h2)

        dx = 1e-10
        x = 0.1

        f1, g1, h1 = m.tf.evaluate_fgh((x + dx,))
        f2, g2, h2 = m.tf.evaluate_fgh((x - dx,))
        self.assertAlmostEqual(f1, f2)
        self.assertStructuredAlmostEqual(g1, g2)
        self.assertStructuredAlmostEqual(h1, h2)

    def test_eval_sqnsqrt_c4_function_fgh(self):
        m = pyo.ConcreteModel()
        m.tf = pyo.ExternalFunction(library=flib, function="sgnsqrt_c4")

        f, g, h = m.tf.evaluate_fgh((4,))
        self.assertEqual(f, 2)
        self.assertEqual(g, [0.5 * 4 ** (-0.5)])
        self.assertEqual(h, [-0.25 * 4 ** (-3 / 2)])

        f, g, h = m.tf.evaluate_fgh((-4,))
        self.assertAlmostEqual(f, -2)
        self.assertStructuredAlmostEqual(g, [0.5 * 4 ** (-0.5)])
        self.assertStructuredAlmostEqual(h, [0.25 * 4 ** (-3 / 2)])

        dx = 1e-9
        x = 0.1

        f1, g1, h1 = m.tf.evaluate_fgh((x + dx,))
        f2, g2, h2 = m.tf.evaluate_fgh((x - dx,))
        self.assertAlmostEqual(f1, f2)
        self.assertStructuredAlmostEqual(g1, g2)
        self.assertStructuredAlmostEqual(h1, h2)

        dx = 1e-9
        x = -0.1

        f1, g1, h1 = m.tf.evaluate_fgh((x + dx,))
        f2, g2, h2 = m.tf.evaluate_fgh((x - dx,))
        self.assertAlmostEqual(f1, f2)
        self.assertStructuredAlmostEqual(g1, g2)
        self.assertStructuredAlmostEqual(h1, h2)
