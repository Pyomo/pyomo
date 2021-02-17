#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
#
#  This module was originally developed as part of the IDAES PSE Framework
#
#  Institute for the Design of Advanced Energy Systems Process Systems
#  Engineering Framework (IDAES PSE Framework) Copyright (c) 2018-2019, by the
#  software owners: The Regents of the University of California, through
#  Lawrence Berkeley National Laboratory,  National Technology & Engineering
#  Solutions of Sandia, LLC, Carnegie Mellon University, West Virginia
#  University Research Corporation, et al. All rights reserved.
#
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""
UI Tests
"""
import os
import sys
import pyomo.common.unittest as unittest
from pyomo.common.fileutils import this_file_dir

test_file = os.path.join(this_file_dir(), "pytest_qt.py")

try:
    import pytest
    # Contextvars is required for anyio/sniffio (pytest), but was not
    # added to the standard library until Python 3.7.  If it is not
    # available (either directly or through the 3.6 backport), do not
    # attempt the QT tests.
    import contextvars
    from pyomo.contrib.viewer.qt import qt_available
except:
    qt_available = False

@unittest.skipUnless(qt_available, "Required packages not available")
class TestViewerQT(unittest.TestCase):
    @unittest.timeout(10)
    def test_get_mainwindow(self):
        rc = pytest.main(["%s::%s" % (test_file, 'test_get_mainwindow')])
        self.assertEqual(rc, pytest.ExitCode.OK)

    @unittest.timeout(10)
    def test_model_information(self):
        rc = pytest.main(["%s::%s" % (test_file, 'test_model_information')])
        self.assertEqual(rc, pytest.ExitCode.OK)
