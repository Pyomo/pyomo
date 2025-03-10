#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#
#  Development of this module was conducted as part of the Institute for
#  the Design of Advanced Energy Systems (IDAES) with support through the
#  Simulation-Based Engineering, Crosscutting Research Program within the
#  U.S. Department of Energy’s Office of Fossil Energy and Carbon Management.
#
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________


from io import StringIO
import sys
import logging

import pyomo.common.unittest as unittest

from pyomo.contrib.trustregion.util import IterationLogger, minIgnoreNone, maxIgnoreNone
from pyomo.common.log import LoggingIntercept


class TestUtil(unittest.TestCase):
    def setUp(self):
        self.iterLogger = IterationLogger()
        self.iteration = 0
        self.thetak = 10.0
        self.objk = 5.0
        self.radius = 1.0
        self.stepNorm = 0.25

    def tearDown(self):
        pass

    def test_minIgnoreNone(self):
        a = 1
        b = 2
        self.assertEqual(minIgnoreNone(a, b), a)
        a = None
        self.assertEqual(minIgnoreNone(a, b), b)
        a = 1
        b = None
        self.assertEqual(minIgnoreNone(a, b), a)
        a = None
        self.assertEqual(minIgnoreNone(a, b), None)

    def test_maxIgnoreNone(self):
        a = 1
        b = 2
        self.assertEqual(maxIgnoreNone(a, b), b)
        a = None
        self.assertEqual(maxIgnoreNone(a, b), b)
        a = 1
        b = None
        self.assertEqual(maxIgnoreNone(a, b), a)
        a = None
        self.assertEqual(maxIgnoreNone(a, b), None)

    def test_IterationRecord(self):
        self.iterLogger.newIteration(
            self.iteration, self.thetak, self.objk, self.radius, self.stepNorm
        )
        self.assertEqual(len(self.iterLogger.iterations), 1)
        self.assertEqual(self.iterLogger.iterations[0].objectiveValue, 5.0)

    def test_logIteration(self):
        self.iterLogger.newIteration(
            self.iteration, self.thetak, self.objk, self.radius, self.stepNorm
        )
        OUTPUT = StringIO()
        with LoggingIntercept(OUTPUT, 'pyomo.contrib.trustregion', logging.INFO):
            self.iterLogger.logIteration()
        self.assertIn('Iteration 0', OUTPUT.getvalue())
        self.assertIn('feasibility =', OUTPUT.getvalue())
        self.assertIn('stepNorm =', OUTPUT.getvalue())

    def test_updateIteration(self):
        self.iterLogger.newIteration(
            self.iteration, self.thetak, self.objk, self.radius, self.stepNorm
        )
        self.assertEqual(self.iterLogger.iterations[0].objectiveValue, self.objk)
        self.assertEqual(self.iterLogger.iterations[0].feasibility, self.thetak)
        self.assertEqual(self.iterLogger.iterations[0].trustRadius, self.radius)
        self.assertEqual(self.iterLogger.iterations[0].stepNorm, self.stepNorm)
        self.iterLogger.updateIteration(feasibility=5.0)
        self.assertEqual(self.iterLogger.iterations[0].objectiveValue, self.objk)
        self.assertEqual(self.iterLogger.iterations[0].feasibility, 5.0)
        self.assertEqual(self.iterLogger.iterations[0].trustRadius, self.radius)
        self.assertEqual(self.iterLogger.iterations[0].stepNorm, self.stepNorm)
        self.iterLogger.updateIteration(objectiveValue=0.1)
        self.assertEqual(self.iterLogger.iterations[0].objectiveValue, 0.1)
        self.assertEqual(self.iterLogger.iterations[0].feasibility, 5.0)
        self.assertEqual(self.iterLogger.iterations[0].trustRadius, self.radius)
        self.assertEqual(self.iterLogger.iterations[0].stepNorm, self.stepNorm)
        self.iterLogger.updateIteration(trustRadius=100)
        self.assertEqual(self.iterLogger.iterations[0].objectiveValue, 0.1)
        self.assertEqual(self.iterLogger.iterations[0].feasibility, 5.0)
        self.assertEqual(self.iterLogger.iterations[0].trustRadius, 100)
        self.assertEqual(self.iterLogger.iterations[0].stepNorm, self.stepNorm)
        self.iterLogger.updateIteration(stepNorm=1)
        self.assertEqual(self.iterLogger.iterations[0].objectiveValue, 0.1)
        self.assertEqual(self.iterLogger.iterations[0].feasibility, 5.0)
        self.assertEqual(self.iterLogger.iterations[0].trustRadius, 100)
        self.assertEqual(self.iterLogger.iterations[0].stepNorm, 1)
        self.iterLogger.updateIteration(
            feasibility=10.0, objectiveValue=0.2, trustRadius=1000, stepNorm=10
        )
        self.assertEqual(self.iterLogger.iterations[0].objectiveValue, 0.2)
        self.assertEqual(self.iterLogger.iterations[0].feasibility, 10.0)
        self.assertEqual(self.iterLogger.iterations[0].trustRadius, 1000)
        self.assertEqual(self.iterLogger.iterations[0].stepNorm, 10)

    def test_printIteration(self):
        self.iterLogger.newIteration(
            self.iteration, self.thetak, self.objk, self.radius, self.stepNorm
        )
        OUTPUT = StringIO()
        sys.stdout = OUTPUT
        self.iterLogger.printIteration()
        sys.stdout = sys.__stdout__
        self.assertIn(str(self.radius), OUTPUT.getvalue())
        self.assertIn(str(self.iteration), OUTPUT.getvalue())
        self.assertIn(str(self.thetak), OUTPUT.getvalue())
        self.assertIn(str(self.objk), OUTPUT.getvalue())
        self.assertIn(str(self.stepNorm), OUTPUT.getvalue())
