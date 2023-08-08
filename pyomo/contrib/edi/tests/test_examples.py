#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2023
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

import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.common.dependencies import attempt_import

from pyomo.core.base.units_container import pint_available

import importlib

<<<<<<< HEAD
testIndex = 0

from pyomo.core.base.units_container import pint_available

from pyomo.common.dependencies import numpy, numpy_available
from pyomo.common.dependencies import scipy, scipy_available


@unittest.skipIf(not numpy_available, 'Testing pyomo.contrib.edi requires numpy')
@unittest.skipIf(not scipy_available, 'Testing pyomo.contrib.edi requires scipy')
@unittest.skipIf(not pint_available, 'Testing units requires pint')
=======

testIndex = 0


>>>>>>> parent of f76f4974b (addressing concerns in PR)
class EDIExamples(unittest.TestCase):
    def test_edi_example_placeholder(self):
        "A placeholder"
        pass


def create_new(filename):
    def t_function(self):
        importName = filename[0:-3]
        # filename = ".."+filename
        try:
            importlib.import_module("pyomo.contrib.edi.examples." + importName)
        except:
            self.fail("This example is failing: %s" % (filename))

    return t_function


pythonFileList = ["readme_example.py", "aircraft_gp.py"]

for filename in pythonFileList:
    testName = 'test_DocumentationExample_%d' % (testIndex)
    testIndex += 1
    t_Function = create_new(filename)
    if pint_available:
        setattr(EDIExamples, testName, t_Function)


if __name__ == '__main__':
    unittest.main()
