# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

import pyomo.common.unittest as unittest

from pyomo.common.docutils import copy_docstrings


class TestDocutils(unittest.TestCase):
    def test_copy_docstrings(self):
        class Base:
            #: This isn't really a docstring
            attr = '1'

            def method0(self):
                "Docstring from Base.method0"

            def method1(self):
                "Docstring from Base.method1"

            def method2(self):
                "Docstring from Base.method2"

            def method3(self):
                pass

        @copy_docstrings(Base)
        class Test1:
            attr = 2

            def method1(self):
                "Docstring from Test1.method1"

            def method2(self):
                pass

            def method3(self):
                "Docstring from Test1.method3"

            def method4(self):
                pass

        self.assertFalse(hasattr(Test1, 'method0'))
        self.assertEqual(Test1.attr.__doc__, int.__doc__)
        self.assertEqual(Test1.method1.__doc__, "Docstring from Test1.method1")
        self.assertEqual(Test1.method2.__doc__, "Docstring from Base.method2")
        self.assertEqual(Test1.method3.__doc__, "Docstring from Test1.method3")
        self.assertEqual(Test1.method4.__doc__, None)

        @copy_docstrings(Base, ['method2', 'method3'])
        class Test2:
            attr = 2.0

            def method1(self):
                pass

            def method2(self):
                pass

            def method3(self):
                "Docstring from Test2.method3"

            def method4(self):
                pass

        self.assertFalse(hasattr(Test2, 'method0'))
        self.assertEqual(Test2.attr.__doc__, float.__doc__)
        self.assertEqual(Test2.method1.__doc__, None)
        self.assertEqual(Test2.method2.__doc__, "Docstring from Base.method2")
        self.assertEqual(Test2.method3.__doc__, "Docstring from Test2.method3")
        self.assertEqual(Test2.method4.__doc__, None)
