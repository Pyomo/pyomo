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

from pyomo.common import unittest

from pyomo.opt import WriterFactory
from pyomo.repn.plugins import activate_writer_version, active_writer_version

import pyomo.environ


class TestPlugins(unittest.TestCase):
    def test_active(self):
        with self.assertRaises(KeyError):
            active_writer_version('nonexistent_writer')
        ver = active_writer_version('lp')
        self.assertIs(
            WriterFactory.get_class('lp'), WriterFactory.get_class(f'lp_v{ver}')
        )

        class TMP(object):
            pass

        WriterFactory.register('test_writer')(TMP)
        try:
            self.assertIsNone(active_writer_version('test_writer'))
        finally:
            WriterFactory.unregister('test_writer')

    def test_activate(self):
        ver = active_writer_version('lp')
        try:
            activate_writer_version('lp', 2)
            self.assertIs(
                WriterFactory.get_class('lp'), WriterFactory.get_class(f'lp_v2')
            )
            activate_writer_version('lp', 1)
            self.assertIs(
                WriterFactory.get_class('lp'), WriterFactory.get_class(f'lp_v1')
            )
        finally:
            activate_writer_version('lp', ver)
