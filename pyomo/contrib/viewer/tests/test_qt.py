##############################################################################
# Institute for the Design of Advanced Energy Systems Process Systems
# Engineering Framework (IDAES PSE Framework) Copyright (c) 2018-2019, by the
# software owners: The Regents of the University of California, through
# Lawrence Berkeley National Laboratory,  National Technology & Engineering
# Solutions of Sandia, LLC, Carnegie Mellon University, West Virginia
# University Research Corporation, et al. All rights reserved.
#
# This software is distributed under the 3-clause BSD License.
##############################################################################
"""
UI Tests
"""
from subprocess import Popen
import os
import time
from pyomo.environ import *
import pyutilib.th as unittest

test_file = os.path.join(os.path.dirname(__file__), "pytest_qt.py")

try:
    skip_qt_tests=False
    import pytest
    from pyomo.contrib.viewer.qt import qt_available
    assert(qt_available)
except:
    skip_qt_tests=True

def run_subproc_pytest(test_file, test_func, freq=1, timeout=10.0):
    p = Popen(["pytest", "::".join([test_file, test_func])])
    i = 0
    while p.poll() is None:
        time.sleep(1.0/freq)
        i += 1
        if(i > timeout*freq):
            p.kill()
            raise Exception("Test took too long")
    assert(p.poll()==0)

@unittest.skipIf(skip_qt_tests, "Required packages not available")
def test_model_information():
    run_subproc_pytest(test_file, "test_model_information")
