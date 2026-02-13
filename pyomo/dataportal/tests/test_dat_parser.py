# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________
#

import os
import time

import pyomo.common.unittest as unittest

import pyomo.dataportal.parse_datacmds as parser
from pyomo.common.errors import DeveloperError


class TestDatParser(unittest.TestCase):
    def test_arguments(self):
        self.assertIsNone(parser.parse_data_commands(data=None, filename=None))
        self.assertEqual(parser.parse_data_commands(data='', filename=None), {None: []})

    def test_update_parsetable(self):
        def _bad_sig():
            return "122345"

        _orig = parser._get_this_file_signature, parser.dat_lexer, parser.dat_yaccer
        try:
            parser._get_this_file_signature = _bad_sig
            parser.dat_lexer = None
            with self.assertRaisesRegex(
                DeveloperError,
                r"DAT parse tables \(pyomo.dataportal._parse_table_datacmds\) "
                r"out of\s+sync with parser definition",
            ):
                parser.parse_data_commands('')
        finally:
            parser._get_this_file_signature, parser.dat_lexer, parser.dat_yaccer = _orig
