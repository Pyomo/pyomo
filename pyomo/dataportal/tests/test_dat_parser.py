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
#

import os
import time

import pyomo.common.unittest as unittest

import pyomo.dataportal.parse_datacmds as parser


class TestDatParser(unittest.TestCase):
    def test_update_parsetable(self):
        parser.parse_data_commands('')
        self.assertIsNotNone(parser.dat_yaccer)
        _tabfile = parser.dat_yaccer_tabfile
        mtime = os.path.getmtime(_tabfile)
        if _tabfile[-1] == 'c':
            _tabfile = _tabfile[:-1]
        time.sleep(0.01)
        with open(parser.__file__, 'a'):
            os.utime(parser.__file__, None)
        parser.dat_lexer = None
        parser.dat_yaccer = None
        parser.parse_data_commands('')
        self.assertIsNotNone(parser.dat_yaccer)
        self.assertLess(mtime, os.path.getmtime(_tabfile))
