#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyutilib.th as unittest

from pyomo.pysp.util.config import (PySPConfigValue,
                                    PySPConfigBlock,
                                    safe_register_common_option)
import six
from six import StringIO

class TestPySPConfigValue(unittest.TestCase):

    def test_init(self):
        PySPConfigValue(
            "default",
            domain=str,
            description="a description",
            doc=None,
            visibility=0)

class TestPySPConfigBlock(unittest.TestCase):

    def test_init(self):
        PySPConfigBlock()

    def test_declared(self):
        b = PySPConfigBlock()
        safe_register_common_option(b, "verbose")
        b.display()
        b.display()
        out = StringIO()
        b.display(ostream=out)
        self.assertEqual(out.getvalue(),
                         "verbose: false\n")
        self.assertEqual(b.check_usage(), True)
        self.assertEqual(b.check_usage(error=False), True)
        b.verbose = True
        out = StringIO()
        b.display(ostream=out)
        self.assertEqual(out.getvalue(),
                         "verbose: true\n")
        with self.assertRaises(ValueError):
            b.check_usage()
        with self.assertRaises(ValueError):
            b.check_usage()
        self.assertEqual(b.check_usage(error=False), False)
        b.verbose
        self.assertEqual(b.check_usage(), True)
        self.assertEqual(b.check_usage(error=False), True)
        verbose_about = \
"""PySPConfigValue: verbose
  -    type: <%s 'bool'>
  - default: False
  -    doc: Generate verbose output for both initialization and
            execution.""" % ('class' if six.PY3 else 'type')
        self.assertEqual(b.about("verbose"),
                         verbose_about)

    def test_implicit(self):
        b = PySPConfigBlock()
        b._implicit_declaration = True
        b.name_a = 1
        b.nameb = 2
        b.display()
        out = StringIO()
        b.display(ostream=out)
        self.assertEqual(out.getvalue(),
                         "name_a: 1\nnameb: 2\n")
        with self.assertRaises(ValueError):
            b.check_usage()
        with self.assertRaises(ValueError):
            b.check_usage()
        self.assertEqual(b.check_usage(error=False), False)
        b.name_a
        b.nameb
        self.assertEqual(b.check_usage(), True)
        self.assertEqual(b.check_usage(error=False), True)
        name_a_about = \
"""ConfigValue: name_a
  -    type: None
  - default: 1
  -    doc: None"""
        self.assertEqual(b.about("name_a"),
                         name_a_about)

if __name__ == "__main__":
    unittest.main()
