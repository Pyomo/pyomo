#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
#
#  This module was originally developed as part of the PyUtilib project
#  Copyright (c) 2008 Sandia Corporation.
#  This software is distributed under the 3-clause BSD License.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  ___________________________________________________________________________
#
#  The configuration test case was originally developed as part of the
#  Water Security Toolkit (WST)
#  Copyright (c) 2012 Sandia Corporation.
#  This software is distributed under the Revised (3-clause) BSD License.
#  Under the terms of Contract DE-AC04-94AL85000, there is a non-exclusive
#  license for use of this work by or on behalf of the U.S. government.
#  ___________________________________________________________________________

import argparse
import enum
import os
import os.path
import pickle
import re
import sys
import types
import pyomo.common.unittest as unittest

from io import StringIO

from pyomo.common.dependencies import yaml, yaml_available, yaml_load_args


def yaml_load(arg):
    return yaml.load(arg, **yaml_load_args)


from pyomo.common.config import (
    ConfigDict,
    ConfigValue,
    ConfigList,
    MarkImmutable,
    ImmutableConfigValue,
    Bool,
    Integer,
    PositiveInt,
    NegativeInt,
    NonPositiveInt,
    NonNegativeInt,
    PositiveFloat,
    NegativeFloat,
    NonPositiveFloat,
    NonNegativeFloat,
    In,
    ListOf,
    Module,
    Path,
    PathList,
    ConfigEnum,
    DynamicImplicitDomain,
    ConfigFormatter,
    String_ConfigFormatter,
    document_kwargs_from_configdict,
    add_docstring_list,
    USER_OPTION,
    DEVELOPER_OPTION,
    _UnpickleableDomain,
    _picklable,
)
from pyomo.common.log import LoggingIntercept


# Utility to redirect display() to a string
def _display(obj, *args):
    test = StringIO()
    obj.display(ostream=test, *args)
    return test.getvalue()


class GlobalClass(object):
    "test class for test_known_types"
    pass


def ExampleConfig():
    CONFIG = ConfigDict()
    CONFIG.declare(
        'option_1',
        ConfigValue(default=5, domain=int, doc='The first configuration option'),
    )
    SOLVER = CONFIG.declare('solver_options', ConfigDict())
    SOLVER.declare(
        'solver_option_1',
        ConfigValue(
            default=1,
            domain=float,
            doc='The first solver configuration option',
            visibility=DEVELOPER_OPTION,
        ),
    )
    SOLVER.declare(
        'solver_option_2',
        ConfigValue(
            default=1,
            domain=float,
            doc="""The second solver configuration option

        With a very long line containing
        wrappable text in a long, silly paragraph
        with little actual information.
        #) but a bulleted list
        #) with two bullets
        """,
        ),
    )
    SOLVER.declare(
        'solver_option_3',
        ConfigValue(
            default=1,
            domain=float,
            doc="""
            The third solver configuration option

            This has a leading newline and a very long line containing
            wrappable text in a long, silly paragraph with
            little actual information.

         .. and_a_list::
            #) but a bulleted list
            #) with two bullets """,
        ),
    )
    CONFIG.declare(
        'option_2',
        ConfigValue(
            default=5,
            domain=int,
            doc="""The second solver configuration option
        with a very long line containing
        wrappable text in a long, silly paragraph
        with little actual information.
        """,
        ),
    )
    return CONFIG


class TestConfigDomains(unittest.TestCase):
    def test_Bool(self):
        c = ConfigDict()
        c.declare('a', ConfigValue(True, Bool))
        self.assertEqual(c.a, True)
        c.a = False
        self.assertEqual(c.a, False)
        c.a = 1
        self.assertEqual(c.a, True)
        c.a = 'n'
        self.assertEqual(c.a, False)
        c.a = 'T'
        self.assertEqual(c.a, True)
        c.a = 'no'
        self.assertEqual(c.a, False)
        c.a = '1'
        self.assertEqual(c.a, True)
        c.a = 0.0
        self.assertEqual(c.a, False)
        c.a = True
        self.assertEqual(c.a, True)
        c.a = 0
        self.assertEqual(c.a, False)
        c.a = 'y'
        self.assertEqual(c.a, True)
        c.a = 'F'
        self.assertEqual(c.a, False)
        c.a = 'yes'
        self.assertEqual(c.a, True)
        c.a = '0'
        self.assertEqual(c.a, False)
        c.a = 1.0
        self.assertEqual(c.a, True)

        with self.assertRaises(ValueError):
            c.a = 2
        self.assertEqual(c.a, True)
        with self.assertRaises(ValueError):
            c.a = 'a'
        self.assertEqual(c.a, True)
        with self.assertRaises(ValueError):
            c.a = 0.5
        self.assertEqual(c.a, True)

    def test_Integer(self):
        c = ConfigDict()
        c.declare('a', ConfigValue(5, Integer))
        self.assertEqual(c.a, 5)
        c.a = 4.0
        self.assertEqual(c.a, 4)
        c.a = -6
        self.assertEqual(c.a, -6)
        c.a = '10'
        self.assertEqual(c.a, 10)
        with self.assertRaises(ValueError):
            c.a = 2.6
        self.assertEqual(c.a, 10)
        with self.assertRaises(ValueError):
            c.a = 'a'
        self.assertEqual(c.a, 10)
        with self.assertRaises(ValueError):
            c.a = '1.1'
        self.assertEqual(c.a, 10)

    def test_PositiveInt(self):
        c = ConfigDict()
        c.declare('a', ConfigValue(5, PositiveInt))
        self.assertEqual(c.a, 5)
        c.a = 4.0
        self.assertEqual(c.a, 4)
        c.a = 6
        self.assertEqual(c.a, 6)
        with self.assertRaises(ValueError):
            c.a = 2.6
        self.assertEqual(c.a, 6)
        with self.assertRaises(ValueError):
            c.a = 'a'
        self.assertEqual(c.a, 6)
        with self.assertRaises(ValueError):
            c.a = 0
        self.assertEqual(c.a, 6)
        with self.assertRaises(ValueError):
            c.a = -4
        self.assertEqual(c.a, 6)

    def test_NegativeInt(self):
        c = ConfigDict()
        c.declare('a', ConfigValue(-5, NegativeInt))
        self.assertEqual(c.a, -5)
        c.a = -4.0
        self.assertEqual(c.a, -4)
        c.a = -6
        self.assertEqual(c.a, -6)
        with self.assertRaises(ValueError):
            c.a = -2.6
        self.assertEqual(c.a, -6)
        with self.assertRaises(ValueError):
            c.a = 'a'
        self.assertEqual(c.a, -6)
        with self.assertRaises(ValueError):
            c.a = 0
        self.assertEqual(c.a, -6)
        with self.assertRaises(ValueError):
            c.a = 4
        self.assertEqual(c.a, -6)

    def test_NonPositiveInt(self):
        c = ConfigDict()
        c.declare('a', ConfigValue(-5, NonPositiveInt))
        self.assertEqual(c.a, -5)
        c.a = -4.0
        self.assertEqual(c.a, -4)
        c.a = -6
        self.assertEqual(c.a, -6)
        with self.assertRaises(ValueError):
            c.a = -2.6
        self.assertEqual(c.a, -6)
        with self.assertRaises(ValueError):
            c.a = 'a'
        self.assertEqual(c.a, -6)
        c.a = 0
        self.assertEqual(c.a, 0)
        with self.assertRaises(ValueError):
            c.a = 4
        self.assertEqual(c.a, 0)

    def test_NonNegativeInt(self):
        c = ConfigDict()
        c.declare('a', ConfigValue(5, NonNegativeInt))
        self.assertEqual(c.a, 5)
        c.a = 4.0
        self.assertEqual(c.a, 4)
        c.a = 6
        self.assertEqual(c.a, 6)
        with self.assertRaises(ValueError):
            c.a = 2.6
        self.assertEqual(c.a, 6)
        with self.assertRaises(ValueError):
            c.a = 'a'
        self.assertEqual(c.a, 6)
        c.a = 0
        self.assertEqual(c.a, 0)
        with self.assertRaises(ValueError):
            c.a = -4
        self.assertEqual(c.a, 0)

    def test_PositiveFloat(self):
        c = ConfigDict()
        c.declare('a', ConfigValue(5, PositiveFloat))
        self.assertEqual(c.a, 5)
        c.a = 4.0
        self.assertEqual(c.a, 4)
        c.a = 6
        self.assertEqual(c.a, 6)
        c.a = 2.6
        self.assertEqual(c.a, 2.6)
        with self.assertRaises(ValueError):
            c.a = 'a'
        self.assertEqual(c.a, 2.6)
        with self.assertRaises(ValueError):
            c.a = 0
        self.assertEqual(c.a, 2.6)
        with self.assertRaises(ValueError):
            c.a = -4
        self.assertEqual(c.a, 2.6)

    def test_NegativeFloat(self):
        c = ConfigDict()
        c.declare('a', ConfigValue(-5, NegativeFloat))
        self.assertEqual(c.a, -5)
        c.a = -4.0
        self.assertEqual(c.a, -4)
        c.a = -6
        self.assertEqual(c.a, -6)
        c.a = -2.6
        self.assertEqual(c.a, -2.6)
        with self.assertRaises(ValueError):
            c.a = 'a'
        self.assertEqual(c.a, -2.6)
        with self.assertRaises(ValueError):
            c.a = 0
        self.assertEqual(c.a, -2.6)
        with self.assertRaises(ValueError):
            c.a = 4
        self.assertEqual(c.a, -2.6)

    def test_NonPositiveFloat(self):
        c = ConfigDict()
        c.declare('a', ConfigValue(-5, NonPositiveFloat))
        self.assertEqual(c.a, -5)
        c.a = -4.0
        self.assertEqual(c.a, -4)
        c.a = -6
        self.assertEqual(c.a, -6)
        c.a = -2.6
        self.assertEqual(c.a, -2.6)
        with self.assertRaises(ValueError):
            c.a = 'a'
        self.assertEqual(c.a, -2.6)
        c.a = 0
        self.assertEqual(c.a, 0)
        with self.assertRaises(ValueError):
            c.a = 4
        self.assertEqual(c.a, 0)

    def test_NonNegativeFloat(self):
        c = ConfigDict()
        c.declare('a', ConfigValue(5, NonNegativeFloat))
        self.assertEqual(c.a, 5)
        c.a = 4.0
        self.assertEqual(c.a, 4)
        c.a = 6
        self.assertEqual(c.a, 6)
        c.a = 2.6
        self.assertEqual(c.a, 2.6)
        with self.assertRaises(ValueError):
            c.a = 'a'
        self.assertEqual(c.a, 2.6)
        c.a = 0
        self.assertEqual(c.a, 0)
        with self.assertRaises(ValueError):
            c.a = -4
        self.assertEqual(c.a, 0)

    def test_In(self):
        c = ConfigDict()
        c.declare('a', ConfigValue(None, In([1, 3, 5])))
        self.assertEqual(c.get('a').domain_name(), 'In[1, 3, 5]')
        self.assertEqual(c.a, None)
        c.a = 3
        self.assertEqual(c.a, 3)
        with self.assertRaises(ValueError):
            c.a = 2
        self.assertEqual(c.a, 3)
        with self.assertRaises(ValueError):
            c.a = {}
        self.assertEqual(c.a, 3)
        with self.assertRaises(ValueError):
            c.a = '1'
        self.assertEqual(c.a, 3)

        c.declare('b', ConfigValue(None, In([1, 3, 5], int)))
        self.assertEqual(c.b, None)
        c.b = 3
        self.assertEqual(c.b, 3)
        with self.assertRaises(ValueError):
            c.b = 2
        self.assertEqual(c.b, 3)
        with self.assertRaises(ValueError):
            c.b = {}
        self.assertEqual(c.b, 3)
        c.b = '1'
        self.assertEqual(c.b, 1)

        class Container(object):
            def __init__(self, vals):
                self._vals = vals

            def __str__(self):
                return f'Container{self._vals}'

            def __contains__(self, val):
                return val in self._vals

        c.declare('c', ConfigValue(None, In(Container([1, 3, 5]))))
        self.assertEqual(c.get('c').domain_name(), 'In(Container[1, 3, 5])')
        self.assertEqual(c.c, None)
        c.c = 3
        self.assertEqual(c.c, 3)
        with self.assertRaises(ValueError):
            c.c = 2
        self.assertEqual(c.c, 3)

    def test_In_enum(self):
        class TestEnum(enum.Enum):
            ITEM_ONE = 1
            ITEM_TWO = 'two'

        cfg = ConfigDict()
        cfg.declare('enum', ConfigValue(default=TestEnum.ITEM_TWO, domain=In(TestEnum)))
        self.assertEqual(cfg.get('enum').domain_name(), 'InEnum[TestEnum]')
        self.assertEqual(cfg.enum, TestEnum.ITEM_TWO)
        cfg.enum = 'ITEM_ONE'
        self.assertEqual(cfg.enum, TestEnum.ITEM_ONE)
        cfg.enum = TestEnum.ITEM_TWO
        self.assertEqual(cfg.enum, TestEnum.ITEM_TWO)
        cfg.enum = 1
        self.assertEqual(cfg.enum, TestEnum.ITEM_ONE)
        cfg.enum = 'two'
        self.assertEqual(cfg.enum, TestEnum.ITEM_TWO)
        with self.assertRaisesRegex(ValueError, '.*3 is not a valid'):
            cfg.enum = 3
        with self.assertRaisesRegex(ValueError, '.*invalid value'):
            cfg.enum = 'ITEM_THREE'

    def test_Path(self):
        def norm(x):
            if cwd[1] == ':' and x[0] == '/':
                x = cwd[:2] + x
            return x.replace('/', os.path.sep)

        cwd = os.getcwd() + os.path.sep
        c = ConfigDict()

        c.declare('a', ConfigValue(None, Path()))
        self.assertEqual(c.a, None)
        c.a = "/a/b/c"
        self.assertTrue(os.path.sep in c.a)
        self.assertEqual(c.a, norm('/a/b/c'))
        c.a = "a/b/c"
        self.assertTrue(os.path.sep in c.a)
        self.assertEqual(c.a, norm(cwd + 'a/b/c'))
        c.a = "${CWD}/a/b/c"
        self.assertTrue(os.path.sep in c.a)
        self.assertEqual(c.a, norm(cwd + 'a/b/c'))
        c.a = None
        self.assertIs(c.a, None)

        c.declare('b', ConfigValue(None, Path('rel/path')))
        self.assertEqual(c.b, None)
        c.b = "/a/b/c"
        self.assertTrue(os.path.sep in c.b)
        self.assertEqual(c.b, norm('/a/b/c'))
        c.b = "a/b/c"
        self.assertTrue(os.path.sep in c.b)
        self.assertEqual(c.b, norm(cwd + 'rel/path/a/b/c'))
        c.b = "${CWD}/a/b/c"
        self.assertTrue(os.path.sep in c.b)
        self.assertEqual(c.b, norm(cwd + 'a/b/c'))
        c.b = None
        self.assertIs(c.b, None)

        c.declare('c', ConfigValue(None, Path('/my/dir')))
        self.assertEqual(c.c, None)
        c.c = "/a/b/c"
        self.assertTrue(os.path.sep in c.c)
        self.assertEqual(c.c, norm('/a/b/c'))
        c.c = "a/b/c"
        self.assertTrue(os.path.sep in c.c)
        self.assertEqual(c.c, norm('/my/dir/a/b/c'))
        c.c = "${CWD}/a/b/c"
        self.assertTrue(os.path.sep in c.c)
        self.assertEqual(c.c, norm(cwd + 'a/b/c'))
        c.c = None
        self.assertIs(c.c, None)

        c.declare('d_base', ConfigValue("${CWD}", str))
        c.declare('d', ConfigValue(None, Path(c.get('d_base'))))
        self.assertEqual(c.d, None)
        c.d = "/a/b/c"
        self.assertTrue(os.path.sep in c.d)
        self.assertEqual(c.d, norm('/a/b/c'))
        c.d = "a/b/c"
        self.assertTrue(os.path.sep in c.d)
        self.assertEqual(c.d, norm(cwd + 'a/b/c'))
        c.d = "${CWD}/a/b/c"
        self.assertTrue(os.path.sep in c.d)
        self.assertEqual(c.d, norm(cwd + 'a/b/c'))

        c.d_base = '/my/dir'
        c.d = "/a/b/c"
        self.assertTrue(os.path.sep in c.d)
        self.assertEqual(c.d, norm('/a/b/c'))
        c.d = "a/b/c"
        self.assertTrue(os.path.sep in c.d)
        self.assertEqual(c.d, norm('/my/dir/a/b/c'))
        c.d = "${CWD}/a/b/c"
        self.assertTrue(os.path.sep in c.d)
        self.assertEqual(c.d, norm(cwd + 'a/b/c'))

        c.d_base = 'rel/path'
        c.d = "/a/b/c"
        self.assertTrue(os.path.sep in c.d)
        self.assertEqual(c.d, norm('/a/b/c'))
        c.d = "a/b/c"
        self.assertTrue(os.path.sep in c.d)
        self.assertEqual(c.d, norm(cwd + 'rel/path/a/b/c'))
        c.d = "${CWD}/a/b/c"
        self.assertTrue(os.path.sep in c.d)
        self.assertEqual(c.d, norm(cwd + 'a/b/c'))

        try:
            Path.SuppressPathExpansion = True
            c.d = "/a/b/c"
            self.assertTrue('/' in c.d)
            self.assertTrue('\\' not in c.d)
            self.assertEqual(c.d, '/a/b/c')
            c.d = "a/b/c"
            self.assertTrue('/' in c.d)
            self.assertTrue('\\' not in c.d)
            self.assertEqual(c.d, 'a/b/c')
            c.d = "${CWD}/a/b/c"
            self.assertTrue('/' in c.d)
            self.assertTrue('\\' not in c.d)
            self.assertEqual(c.d, "${CWD}/a/b/c")
        finally:
            Path.SuppressPathExpansion = False

    def test_PathList(self):
        def norm(x):
            if cwd[1] == ':' and x[0] == '/':
                x = cwd[:2] + x
            return x.replace('/', os.path.sep)

        cwd = os.getcwd() + os.path.sep
        c = ConfigDict()

        c.declare('a', ConfigValue(None, PathList()))
        self.assertEqual(c.a, None)
        c.a = "/a/b/c"
        self.assertEqual(len(c.a), 1)
        self.assertTrue(os.path.sep in c.a[0])
        self.assertEqual(c.a[0], norm('/a/b/c'))
        c.a = None
        self.assertIsNone(c.a)

        c.a = ["a/b/c", "/a/b/c", "${CWD}/a/b/c"]
        self.assertEqual(len(c.a), 3)
        self.assertTrue(os.path.sep in c.a[0])
        self.assertEqual(c.a[0], norm(cwd + 'a/b/c'))
        self.assertTrue(os.path.sep in c.a[1])
        self.assertEqual(c.a[1], norm('/a/b/c'))
        self.assertTrue(os.path.sep in c.a[2])
        self.assertEqual(c.a[2], norm(cwd + 'a/b/c'))

        c.a = ()
        self.assertEqual(len(c.a), 0)
        self.assertIs(type(c.a), list)

    def test_ListOf(self):
        c = ConfigDict()
        c.declare('a', ConfigValue(domain=ListOf(int), default=None))
        self.assertEqual(c.get('a').domain_name(), 'ListOf[int]')

        self.assertEqual(c.a, None)
        c.a = 5
        self.assertEqual(c.a, [5])
        c.a = (5, 6.6)
        self.assertEqual(c.a, [5, 6])
        c.a = '7,8'
        self.assertEqual(c.a, [7, 8])

        ref = (
            r"(?m)Failed casting a\s+to ListOf\(int\)\s+"
            r"Error: invalid literal for int\(\) with base 10: 'a'"
        )
        with self.assertRaisesRegex(ValueError, ref):
            c.a = 'a'

        c.declare('b', ConfigValue(domain=ListOf(str), default=None))
        self.assertEqual(c.get('b').domain_name(), 'ListOf[str]')
        self.assertEqual(c.b, None)
        c.b = "'Hello, World'"
        self.assertEqual(c.b, ["Hello, World"])
        c.b = "Hello, World"
        self.assertEqual(c.b, ["Hello", "World"])
        c.b = ("A", 6)
        self.assertEqual(c.b, ["A", "6"])
        with self.assertRaises(ValueError):
            c.b = "'Hello, World"

        c.declare(
            'b1', ConfigValue(domain=ListOf(str, string_lexer=None), default=None)
        )
        self.assertEqual(c.get('b1').domain_name(), 'ListOf[str]')
        self.assertEqual(c.b1, None)
        c.b1 = "'Hello, World'"
        self.assertEqual(c.b1, ["'Hello, World'"])
        c.b1 = "Hello, World"
        self.assertEqual(c.b1, ["Hello, World"])
        c.b1 = ("A", 6)
        self.assertEqual(c.b1, ["A", "6"])
        c.b1 = "'Hello, World"
        self.assertEqual(c.b1, ["'Hello, World"])

        c.declare('c', ConfigValue(domain=ListOf(int, PositiveInt)))
        self.assertEqual(c.get('c').domain_name(), 'ListOf[PositiveInt]')
        self.assertEqual(c.c, None)
        c.c = 6
        self.assertEqual(c.c, [6])

        ref = (
            r"(?m)Failed casting %s\s+to ListOf\(PositiveInt\)\s+"
            r"Error: Expected positive int, but received %s"
        )
        with self.assertRaisesRegex(ValueError, ref % (6.5, 6.5)):
            c.c = 6.5
        with self.assertRaisesRegex(ValueError, ref % (r"\[0\]", "0")):
            c.c = [0]
        c.c = [3, 6, 9]
        self.assertEqual(c.c, [3, 6, 9])

    def test_Module(self):
        c = ConfigDict()

        c.declare('a', ConfigValue(domain=Module(), default=None))
        self.assertEqual(c.a, None)

        # Set using python module name to be imported
        c.a = 'os.path'
        import os.path

        self.assertIs(c.a, os.path)

        # Set to python module object
        import os

        c.a = os
        self.assertIs(c.a, os)

        # Set using path to python file
        this_file = __file__
        this_folder = os.path.dirname(__file__)
        to_import = os.path.join(this_folder, 'test_config.py')
        c.a = to_import
        self.assertEqual(c.a.__file__, to_import)

    def test_ConfigEnum(self):
        out = StringIO()
        with LoggingIntercept(out):

            class TestEnum(ConfigEnum):
                ITEM_ONE = 1
                ITEM_TWO = 2

        self.assertIn('The ConfigEnum base class is deprecated', out.getvalue())
        self.assertEqual(TestEnum.from_enum_or_string(1), TestEnum.ITEM_ONE)
        self.assertEqual(
            TestEnum.from_enum_or_string(TestEnum.ITEM_TWO), TestEnum.ITEM_TWO
        )
        self.assertEqual(TestEnum.from_enum_or_string('ITEM_ONE'), TestEnum.ITEM_ONE)

        cfg = ConfigDict()
        cfg.declare('enum', ConfigValue(default=2, domain=TestEnum.from_enum_or_string))
        self.assertEqual(cfg.enum, TestEnum.ITEM_TWO)
        cfg.enum = 'ITEM_ONE'
        self.assertEqual(cfg.enum, TestEnum.ITEM_ONE)
        cfg.enum = TestEnum.ITEM_TWO
        self.assertEqual(cfg.enum, TestEnum.ITEM_TWO)
        cfg.enum = 1
        self.assertEqual(cfg.enum, TestEnum.ITEM_ONE)
        with self.assertRaisesRegex(ValueError, '.*3 is not a valid'):
            cfg.enum = 3
        with self.assertRaisesRegex(ValueError, '.*invalid value'):
            cfg.enum = 'ITEM_THREE'

    def test_DynamicImplicitDomain(self):
        def _rule(key, val):
            ans = ConfigDict()
            if 'i' in key:
                ans.declare('option_i', ConfigValue(domain=int, default=1))
            if 'f' in key:
                ans.declare('option_f', ConfigValue(domain=float, default=2))
            if 's' in key:
                ans.declare('option_s', ConfigValue(domain=str, default=3))
            if 'l' in key:
                raise ValueError('invalid key: %s' % key)
            return ans(val)

        cfg = ConfigDict(implicit=True, implicit_domain=DynamicImplicitDomain(_rule))
        self.assertEqual(len(cfg), 0)
        test = cfg({'hi': {'option_i': 10}, 'fast': {'option_f': 20}})
        self.assertEqual(len(test), 2)
        self.assertEqual(test.hi.value(), {'option_i': 10})
        self.assertEqual(test.fast.value(), {'option_f': 20, 'option_s': '3'})

        test2 = cfg(test)
        self.assertIsNot(test.hi, test2.hi)
        self.assertIsNot(test.fast, test2.fast)
        self.assertEqual(test.value(), test2.value())

        self.assertEqual(len(test2), 2)
        fit = test2.get('fit', {})
        self.assertEqual(len(test2), 2)
        self.assertEqual(fit.value(), {'option_f': 2, 'option_i': 1})

        with self.assertRaisesRegex(ValueError, "invalid key: fail"):
            test = cfg(
                {
                    'hi': {'option_i': 10},
                    'fast': {'option_f': 20},
                    'fail': {'option_f': 20},
                }
            )


class TestImmutableConfigValue(unittest.TestCase):
    def test_immutable_config_value(self):
        config = ConfigDict()
        config.declare('a', ConfigValue(default=1, domain=int))
        config.declare('b', ConfigValue(default=1, domain=int))
        config.a = 2
        config.b = 3
        self.assertEqual(config.a, 2)
        self.assertEqual(config.b, 3)
        locker = MarkImmutable(config.get('a'), config.get('b'))
        with self.assertRaisesRegex(RuntimeError, 'is currently immutable'):
            config.a = 4
        with self.assertRaisesRegex(RuntimeError, 'is currently immutable'):
            config.b = 5
        config.a = 2
        config.b = 3
        self.assertEqual(config.a, 2)
        self.assertEqual(config.b, 3)
        locker.release_lock()
        config.a = 4
        config.b = 5
        self.assertEqual(config.a, 4)
        self.assertEqual(config.b, 5)
        with self.assertRaisesRegex(
            ValueError, 'Only ConfigValue instances can be marked immutable'
        ):
            locker = MarkImmutable(config.get('a'), config.b)
        self.assertEqual(type(config.get('a')), ConfigValue)
        config.a = 6
        self.assertEqual(config.a, 6)

        config.declare('c', ConfigValue(default=-1, domain=int))
        locker = MarkImmutable(config.get('a'), config.get('b'))
        # Making a copy of an immutable config value results in a
        # *mutable* config value
        config2 = config({'c': -2})
        self.assertEqual(config2.a, 6)
        self.assertEqual(config2.b, 5)
        self.assertEqual(config2.c, -2)
        self.assertIs(type(config2.get('a')), ConfigValue)
        self.assertIs(type(config2.get('b')), ConfigValue)
        self.assertIs(type(config2.get('c')), ConfigValue)
        # you can even update the original, as long as you don't change
        # the immutable value:
        config.set_value(config2)
        self.assertEqual(config.a, 6)
        self.assertEqual(config.b, 5)
        self.assertEqual(config.c, -2)
        self.assertIs(type(config.get('a')), ImmutableConfigValue)
        self.assertIs(type(config.get('b')), ImmutableConfigValue)
        self.assertIs(type(config.get('c')), ConfigValue)

        # Making a copy of an immutable config value results in a
        # *mutable* config value, even if you change the value of
        # something that is currently immutable
        config3 = config({'a': 1})
        self.assertEqual(config3.a, 1)
        self.assertEqual(config3.b, 5)
        self.assertEqual(config3.c, -2)
        self.assertIs(type(config3.get('a')), ConfigValue)
        self.assertIs(type(config3.get('b')), ConfigValue)
        self.assertIs(type(config3.get('c')), ConfigValue)
        # but attempting to update the original will generate an
        # exception
        with self.assertRaisesRegex(RuntimeError, ' is currently immutable'):
            config.set_value(config3)
        locker.release_lock()

        # test reset
        config.reset()
        self.assertEqual(config.a, 1)
        self.assertEqual(config.b, 1)
        with locker:
            # Reset is OK as long as the values are all currently at
            # their defaults
            config.reset()
            self.assertEqual(config.a, 1)
            self.assertEqual(config.b, 1)

        config.a = 2
        with locker:
            # But if reset would change an immutable value you will get
            # an exception
            with self.assertRaisesRegex(RuntimeError, 'is currently immutable'):
                config.reset()


class TestConfig(unittest.TestCase):
    def setUp(self):
        # Save the original environment, then force a fixed column width
        # so tests do not fail on some platforms (notably, OSX)
        self.original_environ, os.environ = os.environ, os.environ.copy()
        os.environ["COLUMNS"] = "80"

        self.config = config = ConfigDict(
            "Basic configuration for Flushing models", implicit=True
        )
        net = config.declare('network', ConfigDict())
        net.declare(
            'epanet file', ConfigValue('Net3.inp', str, 'EPANET network inp file', None)
        ).declare_as_argument(dest='epanet')

        sc = config.declare(
            'scenario',
            ConfigDict("Single scenario block", implicit=True, implicit_domain=str),
        )
        sc.declare(
            'scenario file',
            ConfigValue(
                'Net3.tsg',
                str,
                'Scenario generation file, see the TEVASIM documentation',
                """This is the (long) documentation for the 'scenario file'
            parameter.  It contains multiple lines, and some internal
            formatting; like a bulleted list:
              - item 1
              - item 2
            """,
            ),
        ).declare_as_argument(group='Scenario definition')
        sc.declare(
            'merlion',
            ConfigValue(
                default=False,
                domain=bool,
                description='Water quality model',
                doc="""

            This is the (long) documentation for the 'merlion'
            parameter.  It contains multiple lines, but no apparent internal
            formatting; so the outputter should re-wrap everything.""",
            ),
        ).declare_as_argument(group='Scenario definition')
        sc.declare(
            'detection',
            ConfigValue(
                # Note use of lambda for an "integer list domain"
                [1, 2, 3],
                lambda x: list(int(i) for i in x),
                'Sensor placement list, epanetID',
                None,
            ),
        )

        config.declare('scenarios', ConfigList([], sc, "List of scenario blocks", None))

        config.declare(
            'nodes',
            ConfigList(
                [], ConfigValue(0, int, 'Node ID', None), "List of node IDs", None
            ),
        )

        im = config.declare('impact', ConfigDict())
        im.declare(
            'metric',
            ConfigValue('MC', str, 'Population or network based impact metric', None),
        )

        fl = config.declare('flushing', ConfigDict())
        n = fl.declare('flush nodes', ConfigDict())
        n.declare(
            'feasible nodes',
            ConfigValue('ALL', str, 'ALL, NZD, NONE, list or filename', None),
        )
        n.declare(
            'infeasible nodes',
            ConfigValue('NONE', str, 'ALL, NZD, NONE, list or filename', None),
        )
        n.declare(
            'max nodes', ConfigValue(2, int, 'Maximum number of nodes to flush', None)
        )
        n.declare('rate', ConfigValue(600, float, 'Flushing rate [gallons/min]', None))
        n.declare(
            'response time',
            ConfigValue(60, float, 'Time [min] between detection and flushing', None),
        )
        n.declare('duration', ConfigValue(600, float, 'Time [min] for flushing', None))

        v = fl.declare('close valves', ConfigDict())
        v.declare(
            'feasible pipes',
            ConfigValue(
                'ALL', str, 'ALL, DIAM min max [inch], NONE, list or filename', None
            ),
        )
        v.declare(
            'infeasible pipes',
            ConfigValue(
                'NONE', str, 'ALL, DIAM min max [inch], NONE, list or filename', None
            ),
        )
        v.declare(
            'max pipes', ConfigValue(2, int, 'Maximum number of pipes to close', None)
        )
        v.declare(
            'response time',
            ConfigValue(
                60, float, 'Time [min] between detection and closing valves', None
            ),
        )

        self._reference = {
            'network': {'epanet file': 'Net3.inp'},
            'scenario': {
                'detection': [1, 2, 3],
                'scenario file': 'Net3.tsg',
                'merlion': False,
            },
            'scenarios': [],
            'nodes': [],
            'impact': {'metric': 'MC'},
            'flushing': {
                'close valves': {
                    'infeasible pipes': 'NONE',
                    'max pipes': 2,
                    'feasible pipes': 'ALL',
                    'response time': 60.0,
                },
                'flush nodes': {
                    'feasible nodes': 'ALL',
                    'max nodes': 2,
                    'infeasible nodes': 'NONE',
                    'rate': 600.0,
                    'duration': 600.0,
                    'response time': 60.0,
                },
            },
        }

    def tearDown(self):
        # Restore the original environment
        os.environ = self.original_environ

    # Utility method for generating and validating a template description
    def _validateTemplate(self, config, reference_template, **kwds):
        test = config.generate_yaml_template(**kwds)
        width = kwds.get('width', 80)
        indent = kwds.get('indent_spacing', 2)
        sys.stdout.write(test)
        for l in test.splitlines():
            self.assertLessEqual(len(l), width)
            if l.strip().startswith("#"):
                continue
            self.assertEqual((len(l) - len(l.lstrip())) % indent, 0)
        self.assertEqual(test, reference_template)

    def test_template_default(self):
        reference_template = """# Basic configuration for Flushing models
network:
  epanet file: Net3.inp     # EPANET network inp file
scenario:                   # Single scenario block
  scenario file: Net3.tsg   # Scenario generation file, see the TEVASIM
                            #   documentation
  merlion: false            # Water quality model
  detection: [1, 2, 3]      # Sensor placement list, epanetID
scenarios: []               # List of scenario blocks
nodes: []                   # List of node IDs
impact:
  metric: MC                # Population or network based impact metric
flushing:
  flush nodes:
    feasible nodes: ALL     # ALL, NZD, NONE, list or filename
    infeasible nodes: NONE  # ALL, NZD, NONE, list or filename
    max nodes: 2            # Maximum number of nodes to flush
    rate: 600.0             # Flushing rate [gallons/min]
    response time: 60.0     # Time [min] between detection and flushing
    duration: 600.0         # Time [min] for flushing
  close valves:
    feasible pipes: ALL     # ALL, DIAM min max [inch], NONE, list or filename
    infeasible pipes: NONE  # ALL, DIAM min max [inch], NONE, list or filename
    max pipes: 2            # Maximum number of pipes to close
    response time: 60.0     # Time [min] between detection and closing valves
"""
        self._validateTemplate(self.config, reference_template)

    def test_template_3space(self):
        reference_template = """# Basic configuration for Flushing models
network:
   epanet file: Net3.inp      # EPANET network inp file
scenario:                     # Single scenario block
   scenario file: Net3.tsg    # Scenario generation file, see the TEVASIM
                              #   documentation
   merlion: false             # Water quality model
   detection: [1, 2, 3]       # Sensor placement list, epanetID
scenarios: []                 # List of scenario blocks
nodes: []                     # List of node IDs
impact:
   metric: MC                 # Population or network based impact metric
flushing:
   flush nodes:
      feasible nodes: ALL     # ALL, NZD, NONE, list or filename
      infeasible nodes: NONE  # ALL, NZD, NONE, list or filename
      max nodes: 2            # Maximum number of nodes to flush
      rate: 600.0             # Flushing rate [gallons/min]
      response time: 60.0     # Time [min] between detection and flushing
      duration: 600.0         # Time [min] for flushing
   close valves:
      feasible pipes: ALL     # ALL, DIAM min max [inch], NONE, list or
                              #   filename
      infeasible pipes: NONE  # ALL, DIAM min max [inch], NONE, list or
                              #   filename
      max pipes: 2            # Maximum number of pipes to close
      response time: 60.0     # Time [min] between detection and closing
                              #   valves
"""
        self._validateTemplate(self.config, reference_template, indent_spacing=3)

    def test_template_4space(self):
        reference_template = """# Basic configuration for Flushing models
network:
    epanet file: Net3.inp       # EPANET network inp file
scenario:                       # Single scenario block
    scenario file: Net3.tsg     # Scenario generation file, see the TEVASIM
                                #   documentation
    merlion: false              # Water quality model
    detection: [1, 2, 3]        # Sensor placement list, epanetID
scenarios: []                   # List of scenario blocks
nodes: []                       # List of node IDs
impact:
    metric: MC                  # Population or network based impact metric
flushing:
    flush nodes:
        feasible nodes: ALL     # ALL, NZD, NONE, list or filename
        infeasible nodes: NONE  # ALL, NZD, NONE, list or filename
        max nodes: 2            # Maximum number of nodes to flush
        rate: 600.0             # Flushing rate [gallons/min]
        response time: 60.0     # Time [min] between detection and flushing
        duration: 600.0         # Time [min] for flushing
    close valves:
        feasible pipes: ALL     # ALL, DIAM min max [inch], NONE, list or
                                #   filename
        infeasible pipes: NONE  # ALL, DIAM min max [inch], NONE, list or
                                #   filename
        max pipes: 2            # Maximum number of pipes to close
        response time: 60.0     # Time [min] between detection and closing
                                #   valves
"""
        self._validateTemplate(self.config, reference_template, indent_spacing=4)

    def test_template_3space_narrow(self):
        reference_template = """# Basic configuration for Flushing models
network:
   epanet file: Net3.inp    # EPANET network inp file
scenario:                   # Single scenario block
   scenario file: Net3.tsg  # Scenario generation file, see the TEVASIM
                            #   documentation
   merlion: false           # Water quality model
   detection: [1, 2, 3]     # Sensor placement list, epanetID
scenarios: []               # List of scenario blocks
nodes: []                   # List of node IDs
impact:
   metric: MC               # Population or network based impact metric
flushing:
   flush nodes:
      feasible nodes: ALL     # ALL, NZD, NONE, list or filename
      infeasible nodes: NONE  # ALL, NZD, NONE, list or filename
      max nodes: 2            # Maximum number of nodes to flush
      rate: 600.0             # Flushing rate [gallons/min]
      response time: 60.0     # Time [min] between detection and
                              #   flushing
      duration: 600.0         # Time [min] for flushing
   close valves:
      feasible pipes: ALL     # ALL, DIAM min max [inch], NONE, list or
                              #   filename
      infeasible pipes: NONE  # ALL, DIAM min max [inch], NONE, list or
                              #   filename
      max pipes: 2            # Maximum number of pipes to close
      response time: 60.0     # Time [min] between detection and closing
                              #   valves
"""
        self._validateTemplate(
            self.config, reference_template, indent_spacing=3, width=72
        )

    def test_display_default(self):
        reference = """network:
  epanet file: Net3.inp
scenario:
  scenario file: Net3.tsg
  merlion: false
  detection: [1, 2, 3]
scenarios: []
nodes: []
impact:
  metric: MC
flushing:
  flush nodes:
    feasible nodes: ALL
    infeasible nodes: NONE
    max nodes: 2
    rate: 600.0
    response time: 60.0
    duration: 600.0
  close valves:
    feasible pipes: ALL
    infeasible pipes: NONE
    max pipes: 2
    response time: 60.0
"""
        test = _display(self.config)
        sys.stdout.write(test)
        self.assertEqual(test, reference)

    def test_display_list(self):
        reference = """network:
  epanet file: Net3.inp
scenario:
  scenario file: Net3.tsg
  merlion: false
  detection: [1, 2, 3]
scenarios:
  -
    scenario file: Net3.tsg
    merlion: false
    detection: [1, 2, 3]
  -
    scenario file: Net3.tsg
    merlion: true
    detection: []
nodes: []
impact:
  metric: MC
flushing:
  flush nodes:
    feasible nodes: ALL
    infeasible nodes: NONE
    max nodes: 2
    rate: 600.0
    response time: 60.0
    duration: 600.0
  close valves:
    feasible pipes: ALL
    infeasible pipes: NONE
    max pipes: 2
    response time: 60.0
"""
        self.config['scenarios'].append()
        self.config['scenarios'].append({'merlion': True, 'detection': []})
        test = _display(self.config)
        sys.stdout.write(test)
        self.assertEqual(test, reference)

    def test_display_userdata_default(self):
        test = _display(self.config, 'userdata')
        sys.stdout.write(test)
        self.assertEqual(test, "")

    def test_display_userdata_list(self):
        self.config['scenarios'].append()
        test = _display(self.config, 'userdata')
        sys.stdout.write(test)
        self.assertEqual(
            test,
            """scenarios:
  -
""",
        )

    def test_display_userdata_list_nonDefault(self):
        self.config['scenarios'].append()
        self.config['scenarios'].append({'merlion': True, 'detection': []})
        test = _display(self.config, 'userdata')
        sys.stdout.write(test)
        self.assertEqual(
            test,
            """scenarios:
  -
  -
    merlion: true
    detection: []
""",
        )

    def test_display_userdata_add_block(self):
        self.config.add("foo", ConfigValue(0, int, None, None))
        self.config.add("bar", ConfigDict())
        test = _display(self.config, 'userdata')
        sys.stdout.write(test)
        self.assertEqual(
            test,
            """foo: 0
bar:
""",
        )

    def test_display_userdata_add_block_nonDefault(self):
        self.config.add("foo", ConfigValue(0, int, None, None))
        self.config.add("bar", ConfigDict(implicit=True)).add("baz", ConfigDict())
        test = _display(self.config, 'userdata')
        sys.stdout.write(test)
        self.assertEqual(
            test,
            """foo: 0
bar:
  baz:
""",
        )

    def test_display_userdata_declare_block(self):
        self.config.declare("foo", ConfigValue(0, int, None, None))
        self.config.declare("bar", ConfigDict())
        test = _display(self.config, 'userdata')
        sys.stdout.write(test)
        self.assertEqual(test, "")

    def test_display_userdata_declare_block_nonDefault(self):
        self.config.declare("foo", ConfigValue(0, int, None, None))
        self.config.declare("bar", ConfigDict(implicit=True)).add("baz", ConfigDict())
        test = _display(self.config, 'userdata')
        sys.stdout.write(test)
        self.assertEqual(test, "bar:\n  baz:\n")

    def test_unusedUserValues_default(self):
        test = '\n'.join(x.name(True) for x in self.config.unused_user_values())
        sys.stdout.write(test)
        self.assertEqual(test, "")

    def test_unusedUserValues_scalar(self):
        self.config['scenario']['merlion'] = True
        test = '\n'.join(x.name(True) for x in self.config.unused_user_values())
        sys.stdout.write(test)
        self.assertEqual(test, "scenario.merlion")

    def test_unusedUserValues_list(self):
        self.config['scenarios'].append()
        test = '\n'.join(x.name(True) for x in self.config.unused_user_values())
        sys.stdout.write(test)
        self.assertEqual(test, """scenarios[0]""")

    def test_unusedUserValues_list_nonDefault(self):
        self.config['scenarios'].append()
        self.config['scenarios'].append({'merlion': True, 'detection': []})
        test = '\n'.join(x.name(True) for x in self.config.unused_user_values())
        sys.stdout.write(test)
        self.assertEqual(
            test,
            """scenarios[0]
scenarios[1]
scenarios[1].merlion
scenarios[1].detection""",
        )

    def test_unusedUserValues_list_nonDefault_listAccessed(self):
        self.config['scenarios'].append()
        self.config['scenarios'].append({'merlion': True, 'detection': []})
        for x in self.config['scenarios']:
            pass
        test = '\n'.join(x.name(True) for x in self.config.unused_user_values())
        sys.stdout.write(test)
        self.assertEqual(
            test,
            """scenarios[0]
scenarios[1]
scenarios[1].merlion
scenarios[1].detection""",
        )

    def test_unusedUserValues_list_nonDefault_itemAccessed(self):
        self.config['scenarios'].append()
        self.config['scenarios'].append({'merlion': True, 'detection': []})
        self.config['scenarios'][1]['merlion']
        test = '\n'.join(x.name(True) for x in self.config.unused_user_values())
        sys.stdout.write(test)
        self.assertEqual(
            test,
            """scenarios[0]
scenarios[1].detection""",
        )

    def test_unusedUserValues_add_topBlock(self):
        self.config.add('foo', ConfigDict())
        test = '\n'.join(x.name(True) for x in self.config.unused_user_values())
        sys.stdout.write(test)
        self.assertEqual(test, "foo")
        test = '\n'.join(x.name(True) for x in self.config.foo.unused_user_values())
        sys.stdout.write(test)
        self.assertEqual(test, "foo")

    def test_unusedUserValues_add_subBlock(self):
        self.config['scenario'].add('foo', ConfigDict())
        test = '\n'.join(x.name(True) for x in self.config.unused_user_values())
        sys.stdout.write(test)
        self.assertEqual(test, """scenario.foo""")

    def test_unusedUserValues_declare_topBlock(self):
        self.config.declare('foo', ConfigDict())
        test = '\n'.join(x.name(True) for x in self.config.unused_user_values())
        sys.stdout.write(test)
        self.assertEqual(test, "")

    def test_unusedUserValues_declare_subBlock(self):
        self.config['scenario'].declare('foo', ConfigDict())
        test = '\n'.join(x.name(True) for x in self.config.unused_user_values())
        sys.stdout.write(test)
        self.assertEqual(test, "")

    def test_UserValues_default(self):
        test = '\n'.join(x.name(True) for x in self.config.user_values())
        sys.stdout.write(test)
        self.assertEqual(test, "")

    def test_UserValues_scalar(self):
        self.config['scenario']['merlion'] = True
        test = '\n'.join(x.name(True) for x in self.config.user_values())
        sys.stdout.write(test)
        self.assertEqual(test, "scenario.merlion")

    def test_UserValues_list(self):
        self.config['scenarios'].append()
        test = '\n'.join(x.name(True) for x in self.config.user_values())
        sys.stdout.write(test)
        self.assertEqual(test, """scenarios[0]""")

    def test_UserValues_list_nonDefault(self):
        self.config['scenarios'].append()
        self.config['scenarios'].append({'merlion': True, 'detection': []})
        test = '\n'.join(x.name(True) for x in self.config.user_values())
        sys.stdout.write(test)
        self.assertEqual(
            test,
            """scenarios[0]
scenarios[1]
scenarios[1].merlion
scenarios[1].detection""",
        )

    def test_UserValues_list_nonDefault_listAccessed(self):
        self.config['scenarios'].append()
        self.config['scenarios'].append({'merlion': True, 'detection': []})
        for x in self.config['scenarios']:
            pass
        test = '\n'.join(x.name(True) for x in self.config.user_values())
        sys.stdout.write(test)
        self.assertEqual(
            test,
            """scenarios[0]
scenarios[1]
scenarios[1].merlion
scenarios[1].detection""",
        )

    def test_UserValues_list_nonDefault_itemAccessed(self):
        self.config['scenarios'].append()
        self.config['scenarios'].append({'merlion': True, 'detection': []})
        self.config['scenarios'][1]['merlion']
        test = '\n'.join(x.name(True) for x in self.config.user_values())
        sys.stdout.write(test)
        self.assertEqual(
            test,
            """scenarios[0]
scenarios[1]
scenarios[1].merlion
scenarios[1].detection""",
        )

    def test_UserValues_add_topBlock(self):
        self.config.add('foo', ConfigDict())
        test = '\n'.join(x.name(True) for x in self.config.user_values())
        sys.stdout.write(test)
        self.assertEqual(test, "foo")
        test = '\n'.join(x.name(True) for x in self.config.foo.user_values())
        sys.stdout.write(test)
        self.assertEqual(test, "foo")

    def test_UserValues_add_subBlock(self):
        self.config['scenario'].add('foo', ConfigDict())
        test = '\n'.join(x.name(True) for x in self.config.user_values())
        sys.stdout.write(test)
        self.assertEqual(test, """scenario.foo""")

    def test_UserValues_declare_topBlock(self):
        self.config.declare('foo', ConfigDict())
        test = '\n'.join(x.name(True) for x in self.config.user_values())
        sys.stdout.write(test)
        self.assertEqual(test, "")

    def test_UserValues_declare_subBlock(self):
        self.config['scenario'].declare('foo', ConfigDict())
        test = '\n'.join(x.name(True) for x in self.config.user_values())
        sys.stdout.write(test)
        self.assertEqual(test, "")

    @unittest.skipIf(not yaml_available, "Test requires PyYAML")
    def test_parseDisplayAndValue_default(self):
        test = _display(self.config)
        sys.stdout.write(test)
        self.assertEqual(yaml_load(test), self.config.value())

    @unittest.skipIf(not yaml_available, "Test requires PyYAML")
    def test_parseDisplayAndValue_list(self):
        self.config['scenarios'].append()
        self.config['scenarios'].append({'merlion': True, 'detection': []})
        test = _display(self.config)
        sys.stdout.write(test)
        self.assertEqual(yaml_load(test), self.config.value())

    @unittest.skipIf(not yaml_available, "Test requires PyYAML")
    def test_parseDisplay_userdata_default(self):
        test = _display(self.config, 'userdata')
        sys.stdout.write(test)
        self.assertEqual(yaml_load(test), None)

    @unittest.skipIf(not yaml_available, "Test requires PyYAML")
    def test_parseDisplay_userdata_list(self):
        self.config['scenarios'].append()
        test = _display(self.config, 'userdata')
        sys.stdout.write(test)
        self.assertEqual(yaml_load(test), {'scenarios': [None]})

    @unittest.skipIf(not yaml_available, "Test requires PyYAML")
    def test_parseDisplay_userdata_list_nonDefault(self):
        self.config['scenarios'].append()
        self.config['scenarios'].append({'merlion': True, 'detection': []})
        test = _display(self.config, 'userdata')
        sys.stdout.write(test)
        self.assertEqual(
            yaml_load(test), {'scenarios': [None, {'merlion': True, 'detection': []}]}
        )

    @unittest.skipIf(not yaml_available, "Test requires PyYAML")
    def test_parseDisplay_userdata_add_block(self):
        self.config.add("foo", ConfigValue(0, int, None, None))
        self.config.add("bar", ConfigDict())
        test = _display(self.config, 'userdata')
        sys.stdout.write(test)
        self.assertEqual(yaml_load(test), {'foo': 0, 'bar': None})

    @unittest.skipIf(not yaml_available, "Test requires PyYAML")
    def test_parseDisplay_userdata_add_block_nonDefault(self):
        self.config.add("foo", ConfigValue(0, int, None, None))
        self.config.add("bar", ConfigDict(implicit=True)).add("baz", ConfigDict())
        test = _display(self.config, 'userdata')
        sys.stdout.write(test)
        self.assertEqual(yaml_load(test), {'bar': {'baz': None}, foo: 0})

    @unittest.skipIf(not yaml_available, "Test requires PyYAML")
    def test_parseDisplay_userdata_add_block(self):
        self.config.declare("foo", ConfigValue(0, int, None, None))
        self.config.declare("bar", ConfigDict())
        test = _display(self.config, 'userdata')
        sys.stdout.write(test)
        self.assertEqual(yaml_load(test), None)

    @unittest.skipIf(not yaml_available, "Test requires PyYAML")
    def test_parseDisplay_userdata_add_block_nonDefault(self):
        self.config.declare("foo", ConfigValue(0, int, None, None))
        self.config.declare("bar", ConfigDict(implicit=True)).add("baz", ConfigDict())
        test = _display(self.config, 'userdata')
        sys.stdout.write(test)
        self.assertEqual(yaml_load(test), {'bar': {'baz': None}})

    def test_value_ConfigValue(self):
        val = self.config['flushing']['flush nodes']['rate']
        self.assertIs(type(val), float)
        self.assertEqual(val, 600.0)

    def test_value_ConfigList_empty(self):
        val = self.config['nodes'].value()
        self.assertIs(type(val), list)
        self.assertEqual(val, [])

    def test_value_ConfigList_simplePopulated(self):
        self.config['nodes'].append('1')
        self.config['nodes'].append(3)
        self.config['nodes'].append()
        val = self.config['nodes'].value()
        self.assertIs(type(val), list)
        self.assertEqual(len(val), 3)
        self.assertEqual(val, [1, 3, 0])

    def test_value_ConfigList_complexPopulated(self):
        self.config['scenarios'].append()
        val = self.config['scenarios'].value()
        self.assertIs(type(val), list)
        self.assertEqual(len(val), 1)
        self.assertEqual(
            val,
            [{'detection': [1, 2, 3], 'merlion': False, 'scenario file': 'Net3.tsg'}],
        )

    def test_name(self):
        self.config['scenarios'].append()
        self.assertEqual(self.config.name(), "")
        self.assertEqual(self.config['scenarios'].name(), "scenarios")
        self.assertEqual(self.config['scenarios'][0].name(), "[0]")
        self.assertEqual(self.config['scenarios'][0].get('merlion').name(), "merlion")

    def test_name_fullyQualified(self):
        self.config['scenarios'].append()
        self.assertEqual(self.config.name(True), "")
        self.assertEqual(self.config['scenarios'].name(True), "scenarios")
        self.assertEqual(self.config['scenarios'][0].name(True), "scenarios[0]")
        self.assertEqual(
            self.config['scenarios'][0].get('merlion').name(True),
            "scenarios[0].merlion",
        )

    def test_setValue_scalar(self):
        self.config['flushing']['flush nodes']['rate'] = 50
        val = self.config['flushing']['flush nodes']['rate']
        self.assertIs(type(val), float)
        self.assertEqual(val, 50.0)

    def test_setValue_scalar_badDomain(self):
        with self.assertRaisesRegex(ValueError, 'invalid value for configuration'):
            self.config['flushing']['flush nodes']['rate'] = 'a'
        val = self.config['flushing']['flush nodes']['rate']
        self.assertIs(type(val), float)
        self.assertEqual(val, 600.0)

    def test_setValue_scalarList_empty(self):
        self.config['scenario']['detection'] = []
        val = self.config['scenario']['detection']
        self.assertIs(type(val), list)
        self.assertEqual(val, [])

    def test_setValue_scalarList_withvalue(self):
        self.config['scenario']['detection'] = [6]
        val = self.config['scenario']['detection']
        self.assertIs(type(val), list)
        self.assertEqual(val, [6])

    def test_setValue_scalarList_badDomain(self):
        with self.assertRaisesRegex(ValueError, 'invalid value for configuration'):
            self.config['scenario']['detection'] = 50
        val = self.config['scenario']['detection']
        self.assertIs(type(val), list)
        self.assertEqual(val, [1, 2, 3])

    def test_setValue_scalarList_badSubDomain(self):
        with self.assertRaisesRegex(ValueError, 'invalid value for configuration'):
            self.config['scenario']['detection'] = [5.5, 'a']
        val = self.config['scenario']['detection']
        self.assertIs(type(val), list)
        self.assertEqual(val, [1, 2, 3])

    def test_setValue_list_scalardomain_list(self):
        self.config['nodes'] = [5, 10]
        val = self.config['nodes'].value()
        self.assertIs(type(val), list)
        self.assertEqual(val, [5, 10])

    def test_setValue_list_scalardomain_scalar(self):
        self.config['nodes'] = 10
        val = self.config['nodes'].value()
        self.assertIs(type(val), list)
        self.assertEqual(val, [10])

    def test_setValue_list_badSubDomain(self):
        with self.assertRaisesRegex(ValueError, 'invalid value for configuration'):
            self.config['nodes'] = [5, 'a']
        val = self.config['nodes'].value()
        self.assertIs(type(val), list)
        self.assertEqual(val, [])

    def test_setValue_block_none(self):
        ref = self._reference['scenario']
        self.config['scenario'] = None
        self.assertEqual(ref, self.config['scenario'].value())
        self.config['scenario']['merlion'] = True
        ref['merlion'] = True
        self.assertEqual(ref, self.config['scenario'].value())
        self.config['scenario'] = None
        self.assertEqual(ref, self.config['scenario'].value())

    def test_setValue_block_empty(self):
        ref = self._reference['scenario']
        self.config['scenario'] = {}
        self.assertEqual(ref, self.config['scenario'].value())
        self.config['scenario']['merlion'] = True
        ref['merlion'] = True
        self.assertEqual(ref, self.config['scenario'].value())
        self.config['scenario'] = {}
        self.assertEqual(ref, self.config['scenario'].value())

    def test_setValue_block_simplevalue(self):
        _test = {'merlion': True, 'detection': [1]}
        ref = self._reference['scenario']
        ref.update(_test)
        self.config['scenario'] = _test
        self.assertEqual(ref, self.config['scenario'].value())

    def test_setItem_block_implicit(self):
        ref = self._reference
        ref['foo'] = 1
        self.config['foo'] = 1
        self.assertEqual(ref, self.config.value())
        ref['bar'] = 1
        self.config['bar'] = 1
        self.assertEqual(ref, self.config.value())

    def test_setItem_block_implicit_domain(self):
        ref = self._reference['scenario']
        ref['foo'] = '1'
        self.config['scenario']['foo'] = 1
        self.assertEqual(ref, self.config['scenario'].value())
        ref['bar'] = '1'
        self.config['scenario']['bar'] = 1
        self.assertEqual(ref, self.config['scenario'].value())

    def test_setValue_block_noImplicit(self):
        _test = {'epanet file': 'no_file.inp', 'foo': 1}
        with self.assertRaisesRegex(
            ValueError, "key 'foo' not defined for ConfigDict 'network' and implicit"
        ):
            self.config['network'] = _test
        self.assertEqual(self._reference, self.config.value())

    def test_setValue_block_implicit(self):
        _test = {'scenario': {'merlion': True, 'detection': [1]}, 'foo': 1}
        ref = self._reference
        ref['scenario'].update(_test['scenario'])
        ref['foo'] = 1
        self.config.set_value(_test)
        self.assertEqual(ref, self.config.value())
        _test = {'scenario': {'merlion': True, 'detection': [1]}, 'bar': 1}
        ref['bar'] = 1
        self.config.set_value(_test)
        self.assertEqual(ref, self.config.value())

    def test_setValue_block_implicit_domain(self):
        _test = {'merlion': True, 'detection': [1], 'foo': 1}
        ref = self._reference['scenario']
        ref.update(_test)
        ref['foo'] = '1'
        self.config['scenario'] = _test
        self.assertEqual(ref, self.config['scenario'].value())
        _test = {'merlion': True, 'detection': [1], 'bar': '1'}
        ref['bar'] = '1'
        self.config['scenario'] = _test
        self.assertEqual(ref, self.config['scenario'].value())

    def test_setValue_block_badDomain(self):
        _test = {'merlion': True, 'detection': ['a'], 'foo': 1, 'a': 1}
        with self.assertRaisesRegex(ValueError, 'invalid value for configuration'):
            self.config['scenario'] = _test
        self.assertEqual(self._reference, self.config.value())

        with self.assertRaisesRegex(
            ValueError, 'Expected dict value for scenario.set_value, found list'
        ):
            self.config['scenario'] = []
        self.assertEqual(self._reference, self.config.value())

    def test_default_function(self):
        c = ConfigValue(default=lambda: 10, domain=int)
        self.assertEqual(c.value(), 10)
        c.set_value(5)
        self.assertEqual(c.value(), 5)
        c.reset()
        self.assertEqual(c.value(), 10)

        with self.assertRaisesRegex(TypeError, r"<lambda>\(\) .* argument"):
            c = ConfigValue(default=lambda x: 10 * x, domain=int)

        with self.assertRaisesRegex(ValueError, 'invalid value for configuration'):
            c = ConfigValue('a', domain=int)

    def test_set_default(self):
        c = ConfigValue()
        self.assertIsNone(c.value())
        c.set_default_value(10.5)
        self.assertIsNone(c.value())
        c.reset()
        self.assertIs(type(c.value()), float)
        self.assertEqual(c.value(), 10.5)
        c.set_domain(int)
        self.assertIs(type(c.value()), int)
        self.assertEqual(c.value(), 10)

    def test_getItem_setItem(self):
        # a freshly-initialized object should not be accessed
        self.assertFalse(self.config._userAccessed)
        self.assertFalse(self.config._data['scenario']._userAccessed)
        self.assertFalse(self.config._data['scenario']._data['detection']._userAccessed)

        # Getting a ConfigValue should not access it
        self.assertFalse(self.config['scenario'].get('detection')._userAccessed)

        # ... but should access the parent blocks traversed to get there
        self.assertTrue(self.config._userAccessed)
        self.assertTrue(self.config._data['scenario']._userAccessed)
        self.assertFalse(self.config._data['scenario']._data['detection']._userAccessed)

        # a freshly-initialized object should not be set
        self.assertFalse(self.config._userSet)
        self.assertFalse(self.config._data['scenario']._userSet)
        self.assertFalse(self.config['scenario']._data['detection']._userSet)

        # setting a value should map it to the correct domain
        self.assertEqual(self.config['scenario']['detection'], [1, 2, 3])
        self.config['scenario']['detection'] = [42.5]
        self.assertEqual(self.config['scenario']['detection'], [42])

        # setting a ConfigValue should mark it as userSet, but NOT any parent blocks
        self.assertFalse(self.config._userSet)
        self.assertFalse(self.config._data['scenario']._userSet)
        self.assertTrue(self.config['scenario'].get('detection')._userSet)

    def test_delitem(self):
        config = ConfigDict(implicit=True)
        config.declare('bar', ConfigValue())
        self.assertEqual(sorted(config.keys()), ['bar'])
        config.foo = 5
        self.assertEqual(sorted(config.keys()), ['bar', 'foo'])
        self.assertEqual(sorted(config._declared), ['bar'])
        del config['foo']
        self.assertEqual(sorted(config.keys()), ['bar'])
        self.assertEqual(sorted(config._declared), ['bar'])
        del config['bar']
        self.assertEqual(sorted(config.keys()), [])
        self.assertEqual(sorted(config._declared), [])

        with self.assertRaisesRegex(KeyError, "'get'"):
            del config['get']
        with self.assertRaisesRegex(KeyError, "'foo'"):
            del config['foo']

    def test_delattr(self):
        config = ConfigDict(implicit=True)
        config.declare('bar', ConfigValue())
        self.assertEqual(sorted(config.keys()), ['bar'])
        config.foo = 5
        self.assertEqual(sorted(config.keys()), ['bar', 'foo'])
        self.assertEqual(sorted(config._declared), ['bar'])
        del config.foo
        self.assertEqual(sorted(config._declared), ['bar'])
        self.assertEqual(sorted(config.keys()), ['bar'])
        del config.bar
        self.assertEqual(sorted(config.keys()), [])
        self.assertEqual(sorted(config._declared), [])

        with self.assertRaisesRegex(
            AttributeError, "'ConfigDict' object attribute 'get' is read-only"
        ):
            del config.get
        with self.assertRaisesRegex(
            AttributeError, "'ConfigDict' object has no attribute 'foo'"
        ):
            del config.foo

    def test_generate_custom_documentation(self):
        reference = """startBlock{}
  startItem{network}
  endItem{network}
  startBlock{network}
    startItem{epanet file}
      item{EPANET network inp file}
    endItem{epanet file}
  endBlock{network}
  startItem{scenario}
    item{Single scenario block}
  endItem{scenario}
  startBlock{scenario}
    startItem{scenario file}
      item{This is the (long) documentation for the 'scenario file'
parameter.  It contains multiple lines, and some internal
formatting; like a bulleted list:
  - item 1
  - item 2
}
    endItem{scenario file}
    startItem{merlion}
      item{This is the (long) documentation for the 'merlion' parameter.  It
      contains multiple lines, but no apparent internal formatting; so the
      outputter should re-wrap everything.}
    endItem{merlion}
    startItem{detection}
      item{Sensor placement list, epanetID}
    endItem{detection}
  endBlock{scenario}
  startItem{scenarios}
    item{List of scenario blocks}
  endItem{scenarios}
  startBlock{scenarios}
    startItem{scenario file}
      item{This is the (long) documentation for the 'scenario file'
parameter.  It contains multiple lines, and some internal
formatting; like a bulleted list:
  - item 1
  - item 2
}
    endItem{scenario file}
    startItem{merlion}
      item{This is the (long) documentation for the 'merlion' parameter.  It
      contains multiple lines, but no apparent internal formatting; so the
      outputter should re-wrap everything.}
    endItem{merlion}
    startItem{detection}
      item{Sensor placement list, epanetID}
    endItem{detection}
  endBlock{scenarios}
  startItem{nodes}
    item{List of node IDs}
  endItem{nodes}
  startItem{impact}
  endItem{impact}
  startBlock{impact}
    startItem{metric}
      item{Population or network based impact metric}
    endItem{metric}
  endBlock{impact}
  startItem{flushing}
  endItem{flushing}
  startBlock{flushing}
    startItem{flush nodes}
    endItem{flush nodes}
    startBlock{flush nodes}
      startItem{feasible nodes}
        item{ALL, NZD, NONE, list or filename}
      endItem{feasible nodes}
      startItem{infeasible nodes}
        item{ALL, NZD, NONE, list or filename}
      endItem{infeasible nodes}
      startItem{max nodes}
        item{Maximum number of nodes to flush}
      endItem{max nodes}
      startItem{rate}
        item{Flushing rate [gallons/min]}
      endItem{rate}
      startItem{response time}
        item{Time [min] between detection and flushing}
      endItem{response time}
      startItem{duration}
        item{Time [min] for flushing}
      endItem{duration}
    endBlock{flush nodes}
    startItem{close valves}
    endItem{close valves}
    startBlock{close valves}
      startItem{feasible pipes}
        item{ALL, DIAM min max [inch], NONE, list or filename}
      endItem{feasible pipes}
      startItem{infeasible pipes}
        item{ALL, DIAM min max [inch], NONE, list or filename}
      endItem{infeasible pipes}
      startItem{max pipes}
        item{Maximum number of pipes to close}
      endItem{max pipes}
      startItem{response time}
        item{Time [min] between detection and closing valves}
      endItem{response time}
    endBlock{close valves}
  endBlock{flushing}
endBlock{}
"""
        with LoggingIntercept() as LOG:
            test = self.config.generate_documentation(
                block_start="startBlock{%s}\n",
                block_end="endBlock{%s}\n",
                item_start="startItem{%s}\n",
                item_body="item{%s}\n",
                item_end="endItem{%s}\n",
            )
        LOG = LOG.getvalue().replace('\n', ' ')
        for name in ('block_start', 'block_end', 'item_start', 'item_end', 'item_body'):
            self.assertIn(
                f"Overriding '{name}' by passing strings to "
                "generate_documentation is deprecated.",
                LOG,
            )
        self.maxDiff = None
        # print(test)
        self.assertEqual(test, reference)

        with LoggingIntercept() as LOG:
            test = self.config.generate_documentation(
                format=String_ConfigFormatter(
                    block_start="startBlock{%s}\n",
                    block_end="endBlock{%s}\n",
                    item_start="startItem{%s}\n",
                    item_body="item{%s}\n",
                    item_end="endItem{%s}\n",
                )
            )
        self.assertEqual(LOG.getvalue(), "")
        self.maxDiff = None
        # print(test)
        self.assertEqual(test, reference)

        with LoggingIntercept() as LOG:
            test = self.config.generate_documentation(
                block_start="startBlock\n",
                block_end="endBlock\n",
                item_start="startItem\n",
                item_body="item\n",
                item_end="endItem\n",
            )

        stripped_reference = re.sub(r'\{[^\}]*\}', '', reference, flags=re.M)
        # print(test)
        self.assertEqual(test, stripped_reference)

        reference = """startBlock{}
  startBlock{network}
  startBlock{scenario}
  startBlock{scenarios}
  startBlock{impact}
  startBlock{flushing}
    startBlock{flush nodes}
    startBlock{close valves}
"""
        with LoggingIntercept() as LOG:
            test = self.config.generate_documentation(
                block_start="startBlock{%s}\n",
                block_end="",
                item_start="",
                item_body="",
            )
        LOG = LOG.getvalue().replace('\n', ' ')
        for name in ('block_start', 'block_end', 'item_start', 'item_body'):
            self.assertIn(
                f"Overriding '{name}' by passing strings to "
                "generate_documentation is deprecated.",
                LOG,
            )
        for name in 'item_end':
            self.assertNotIn(
                f"Overriding '{name}' by passing strings to "
                "generate_documentation is deprecated.",
                LOG,
            )
        self.maxDiff = None
        # print(test)
        self.assertEqual(test, reference)

    def test_generate_latex_documentation(self):
        cfg = ConfigDict()
        cfg.declare(
            'int',
            ConfigValue(domain=int, default=10, doc="This is an integer parameter"),
        )
        cfg.declare(
            'in',
            ConfigValue(
                domain=In([1, 3, 5]),
                default=1,
                description="This parameter must be in {1,3,5}",
            ),
        )
        cfg.declare(
            'lambda',
            ConfigValue(
                domain=lambda x: int(x),
                default=1,
                description="This is a float",
                doc="This parameter is actually a float, but for testing "
                "purposes we will use a lambda function for validation",
            ),
        )
        cfg.declare(
            'list', ConfigList(domain=str, description="A simple list of strings")
        )
        self.assertEqual(
            cfg.generate_documentation(format='latex').strip(),
            """
\\begin{description}[topsep=0pt,parsep=0.5em,itemsep=-0.4em]
  \\item[{int}]\\hfill
    \\\\This is an integer parameter
  \\item[{in}]\\hfill
    \\\\This parameter must be in {1,3,5}
  \\item[{lambda}]\\hfill
    \\\\This parameter is actually a float, but for testing purposes we will use
    a lambda function for validation
  \\item[{list}]\\hfill
    \\\\A simple list of strings
\\end{description}
            """.strip(),
        )

    def test_empty_ConfigFormatter(self):
        cfg = ConfigDict()
        cfg.declare('field', ConfigValue())
        with self.assertRaisesRegex(
            ValueError, "Unrecognized documentation formatter, 'unknown'"
        ):
            cfg.generate_documentation(format="unknown")

        self.assertEqual(cfg.generate_documentation(format=ConfigFormatter()), '')

    def test_generate_documentation_StringFormatter(self):
        # This test verifies behavior with simple StringFormatters (in
        # particular, the handling of newlines and indentation reported
        # in #IDAES/idaes-pse#1191)
        CONFIG = ExampleConfig()
        doc = CONFIG.generate_documentation(
            format=String_ConfigFormatter(
                block_start="",  # %s\n",
                block_end="",
                item_start="%s\n",
                item_body="%s",
                item_end="\n",
            ),
            indent_spacing=4,
            width=66,
        )

        # print(doc)
        ref = """    option_1
        The first configuration option

    solver_options

        solver_option_1
            The first solver configuration option

        solver_option_2
            The second solver configuration option

        With a very long line containing
        wrappable text in a long, silly paragraph
        with little actual information.
        #) but a bulleted list
        #) with two bullets

        solver_option_3
            The third solver configuration option

            This has a leading newline and a very long line containing
            wrappable text in a long, silly paragraph with
            little actual information.

         .. and_a_list::
            #) but a bulleted list
            #) with two bullets

    option_2
        The second solver configuration option with a very long
        line containing wrappable text in a long, silly paragraph
        with little actual information.

"""
        self.assertEqual(
            [_.rstrip() for _ in ref.splitlines()],
            [_.rstrip() for _ in doc.splitlines()],
        )

    def test_block_get(self):
        self.assertTrue('scenario' in self.config)
        self.assertNotEqual(self.config.get('scenario', 'bogus').value(), 'bogus')
        self.assertFalse('fubar' in self.config)
        self.assertEqual(self.config.get('fubar', 'bogus').value(), 'bogus')

        cfg = ConfigDict()
        cfg.declare('foo', ConfigValue(1, int))
        self.assertEqual(cfg.get('foo', 5).value(), 1)
        self.assertEqual(len(cfg), 1)
        self.assertIs(cfg.get('bar'), None)
        self.assertEqual(cfg.get('bar', None).value(), None)
        self.assertEqual(len(cfg), 1)

        cfg = ConfigDict(implicit=True)
        cfg.declare('foo', ConfigValue(1, int))
        self.assertEqual(cfg.get('foo', 5).value(), 1)
        self.assertEqual(len(cfg), 1)
        self.assertEqual(cfg.get('bar', 5).value(), 5)
        self.assertEqual(len(cfg), 1)
        self.assertIs(cfg.get('baz'), None)
        self.assertIs(cfg.get('baz', None).value(), None)
        self.assertEqual(len(cfg), 1)

        cfg = ConfigDict(implicit=True, implicit_domain=ConfigList(domain=str))
        cfg.declare('foo', ConfigValue(1, int))
        self.assertEqual(cfg.get('foo', 5).value(), 1)
        self.assertEqual(len(cfg), 1)
        self.assertEqual(cfg.get('bar', [5]).value(), ['5'])
        self.assertEqual(len(cfg), 1)
        self.assertIs(cfg.get('baz'), None)
        self.assertEqual(cfg.get('baz', None).value(), [])
        self.assertEqual(len(cfg), 1)

    def test_setdefault(self):
        cfg = ConfigDict()
        cfg.declare('foo', ConfigValue(1, int))
        self.assertEqual(cfg.setdefault('foo', 5).value(), 1)
        self.assertEqual(len(cfg), 1)
        self.assertRaisesRegex(
            ValueError, '.*disallows implicit entries', cfg.setdefault, 'bar', 0
        )
        self.assertEqual(len(cfg), 1)

        cfg = ConfigDict(implicit=True)
        cfg.declare('foo', ConfigValue(1, int))
        self.assertEqual(cfg.setdefault('foo', 5).value(), 1)
        self.assertEqual(len(cfg), 1)
        self.assertEqual(cfg.setdefault('bar', 5).value(), 5)
        self.assertEqual(len(cfg), 2)
        self.assertEqual(cfg.setdefault('baz').value(), None)
        self.assertEqual(len(cfg), 3)

        cfg = ConfigDict(implicit=True, implicit_domain=ConfigList(domain=str))
        cfg.declare('foo', ConfigValue(1, int))
        self.assertEqual(cfg.setdefault('foo', 5).value(), 1)
        self.assertEqual(len(cfg), 1)
        self.assertEqual(cfg.setdefault('bar', [5]).value(), ['5'])
        self.assertEqual(len(cfg), 2)
        self.assertEqual(cfg.setdefault('baz').value(), [])
        self.assertEqual(len(cfg), 3)

    def test_block_keys(self):
        ref = ['scenario file', 'merlion', 'detection']

        # keys iterator
        keys = self.config['scenario'].keys()
        # lists are independent
        self.assertIsNot(keys, self.config['scenario'].keys())
        self.assertIsNot(type(keys), list)
        self.assertEqual(list(keys), ref)

        # (deprecated) python 2 iterator
        out = StringIO()
        with LoggingIntercept(out):
            keyiter = self.config['scenario'].iterkeys()
            # iterators are independent
            self.assertIsNot(keyiter, self.config['scenario'].iterkeys())
        self.assertIn("The iterkeys method is deprecated", out.getvalue())
        self.assertIsNot(type(keyiter), list)
        self.assertEqual(list(keyiter), ref)

        # default iterator
        keyiter = self.config['scenario'].__iter__()
        self.assertIsNot(type(keyiter), list)
        self.assertEqual(list(keyiter), ref)
        # iterators are independent
        self.assertIsNot(keyiter, self.config['scenario'].__iter__())

    def test_block_values(self):
        ref = ['Net3.tsg', False, [1, 2, 3]]

        # values iterator
        values = self.config['scenario'].values()
        self.assertIsNot(type(values), list)
        self.assertEqual(list(values), ref)
        # lists are independent
        self.assertIsNot(values, self.config['scenario'].values())

        # (deprecated) python 2 iterator
        out = StringIO()
        with LoggingIntercept(out):
            valueiter = self.config['scenario'].itervalues()
            # iterators are independent
            self.assertIsNot(valueiter, self.config['scenario'].itervalues())
        self.assertIn("The itervalues method is deprecated", out.getvalue())
        self.assertIsNot(type(valueiter), list)
        self.assertEqual(list(valueiter), ref)

    def test_block_items(self):
        ref = [
            ('scenario file', 'Net3.tsg'),
            ('merlion', False),
            ('detection', [1, 2, 3]),
        ]

        # items iterator
        items = self.config['scenario'].items()
        self.assertIsNot(type(items), list)
        self.assertEqual(list(items), ref)
        # lists are independent
        self.assertIsNot(items, self.config['scenario'].items())

        # (deprecated) python 2 iterator
        out = StringIO()
        with LoggingIntercept(out):
            itemiter = self.config['scenario'].iteritems()
            # iterators are independent
            self.assertIsNot(itemiter, self.config['scenario'].iteritems())
        self.assertIn("The iteritems method is deprecated", out.getvalue())
        self.assertIsNot(type(itemiter), list)
        self.assertEqual(list(itemiter), ref)

    def test_value(self):
        # print(self.config.value())
        self.assertEqual(self._reference, self.config.value())

    def test_list_manipulation(self):
        self.assertEqual(len(self.config['scenarios']), 0)
        self.config['scenarios'].append()
        os = StringIO()
        with LoggingIntercept(os):
            self.config['scenarios'].add()
        self.assertIn(
            "ConfigList.add() has been deprecated.  Use append()", os.getvalue()
        )
        self.assertEqual(len(self.config['scenarios']), 2)
        self.config['scenarios'].append({'merlion': True, 'detection': []})
        self.assertEqual(len(self.config['scenarios']), 3)
        test = _display(self.config, 'userdata')
        sys.stdout.write(test)
        self.assertEqual(
            test,
            """scenarios:
  -
  -
  -
    merlion: true
    detection: []
""",
        )
        self.config['scenarios'][0] = {'merlion': True, 'detection': []}
        self.assertEqual(len(self.config['scenarios']), 3)
        test = _display(self.config, 'userdata')
        sys.stdout.write(test)
        self.assertEqual(
            test,
            """scenarios:
  -
    merlion: true
    detection: []
  -
  -
    merlion: true
    detection: []
""",
        )
        test = _display(self.config['scenarios'])
        sys.stdout.write(test)
        self.assertEqual(
            test,
            """-
  scenario file: Net3.tsg
  merlion: true
  detection: []
-
  scenario file: Net3.tsg
  merlion: false
  detection: [1, 2, 3]
-
  scenario file: Net3.tsg
  merlion: true
  detection: []
""",
        )

    def test_list_get(self):
        X = ConfigDict(implicit=True)
        X.declare('config', ConfigList())
        self.assertEqual(_display(X, 'userdata'), "")
        with self.assertRaisesRegex(IndexError, 'list index out of range'):
            self.assertIs(X.config.get(0), None)
        self.assertIs(X.config.get(0, None).value(), None)
        val = X.config.get(0, 1)
        self.assertIsInstance(val, ConfigValue)
        self.assertEqual(val.value(), 1)
        self.assertRaisesRegex(IndexError, '.*out of range', X.config.__getitem__, 0)
        # get() shouldn't change the userdata flag...
        self.assertEqual(_display(X, 'userdata'), "")

        X = ConfigDict(implicit=True)
        X.declare('config', ConfigList([42], int))
        self.assertEqual(_display(X, 'userdata'), "")
        val = X.config.get(0)
        self.assertEqual(val.value(), 42)
        self.assertIs(type(val), ConfigValue)
        # get() shouldn't change the userdata flag...
        self.assertEqual(_display(X, 'userdata'), "")
        val = X.config[0]
        self.assertIs(type(val), int)
        self.assertEqual(val, 42)
        # get() shouldn't change the userdata flag...
        self.assertEqual(_display(X, 'userdata'), "")

        with self.assertRaisesRegex(IndexError, 'list index out of range'):
            self.assertIs(X.config.get(1), None)
        self.assertRaisesRegex(IndexError, '.*out of range', X.config.__getitem__, 1)

        # this should ONLY change the userSet flag on the item (and not
        # the list)
        X.config.get(0).set_value(20)
        self.assertEqual(_display(X, 'userdata'), "config:\n  - 20\n")
        self.assertEqual([_.name(True) for _ in X.user_values()], ["config[0]"])

        # this should ONLY change the userSet flag on the item (and not
        # the list)
        X = ConfigDict(implicit=True)
        X.declare('config', ConfigList([42], int))
        X.config[0] = 20
        self.assertEqual(_display(X, 'userdata'), "config:\n  - 20\n")
        self.assertEqual([_.name(True) for _ in X.user_values()], ["config[0]"])

        # this should ONLY change the userSet flag on the item (and not
        # the list)
        X = ConfigDict(implicit=True)
        X.declare('config', ConfigList([42], int))
        X.config.append(20)
        self.assertEqual(_display(X, 'userdata'), "config:\n  - 20\n")
        self.assertEqual([_.name(True) for _ in X.user_values()], ["config[1]"])

        # This should change both... because the [42] was "declared" as
        # the default for the List, it will *not* be a user-set value
        X = ConfigDict(implicit=True)
        X.add('config', ConfigList([42], int))
        X.config.append(20)
        self.assertEqual(_display(X, 'userdata'), "config:\n  - 20\n")
        self.assertEqual(
            [_.name(True) for _ in X.user_values()], ["config", "config[1]"]
        )

    def test_implicit_entries(self):
        config = ConfigDict()
        with self.assertRaisesRegex(
            ValueError,
            "Key 'test' not defined in ConfigDict '' "
            "and Dict disallows implicit entries",
        ):
            config['test'] = 5

        config = ConfigDict(implicit=True)
        config['implicit_1'] = 5
        config.declare('formal', ConfigValue(42, int))
        config['implicit_2'] = 5
        self.assertEqual(3, len(config))
        self.assertEqual(['implicit_1', 'formal', 'implicit_2'], list(config.keys()))
        config.reset()
        self.assertEqual(1, len(config))
        self.assertEqual(['formal'], list(config.keys()))

    def test_argparse_help(self):
        parser = argparse.ArgumentParser(prog='tester')
        self.config.initialize_argparse(parser)
        help = parser.format_help()
        self.assertIn(
            """  -h, --help            show this help message and exit
  --epanet-file EPANET  EPANET network inp file

Scenario definition:
  --scenario-file STR   Scenario generation file, see the TEVASIM
                        documentation
  --merlion             Water quality model
""",
            help,
        )

    def test_argparse_help_implicit_disable(self):
        self.config['scenario'].declare(
            'epanet',
            ConfigValue(True, bool, 'Use EPANET as the Water quality model', None),
        ).declare_as_argument(group='Scenario definition')
        parser = argparse.ArgumentParser(prog='tester')
        self.config.initialize_argparse(parser)
        help = parser.format_help()
        self.maxDiff = None
        self.assertIn(
            """
  -h, --help            show this help message and exit
  --epanet-file EPANET  EPANET network inp file

Scenario definition:
  --scenario-file STR   Scenario generation file, see the TEVASIM
                        documentation
  --merlion             Water quality model
  --disable-epanet      [DON'T] Use EPANET as the Water quality model
""",
            help,
        )

    def test_argparse_import(self):
        parser = argparse.ArgumentParser(prog='tester')
        self.config.initialize_argparse(parser)

        args = parser.parse_args([])
        self.assertEqual(0, len(vars(args)))
        leftovers = self.config.import_argparse(args)
        self.assertEqual(0, len(vars(args)))
        self.assertEqual([], [x.name(True) for x in self.config.user_values()])

        args = parser.parse_args(['--merlion'])
        self.config.reset()
        self.assertFalse(self.config['scenario']['merlion'])
        self.assertEqual(1, len(vars(args)))
        leftovers = self.config.import_argparse(args)
        self.assertEqual(0, len(vars(args)))
        self.assertEqual(
            ['scenario.merlion'], [x.name(True) for x in self.config.user_values()]
        )

        args = parser.parse_args(['--merlion', '--epanet-file', 'foo'])
        self.config.reset()
        self.assertFalse(self.config['scenario']['merlion'])
        self.assertEqual('Net3.inp', self.config['network']['epanet file'])
        self.assertEqual(2, len(vars(args)))
        leftovers = self.config.import_argparse(args)
        self.assertEqual(1, len(vars(args)))
        self.assertEqual(
            ['network.epanet file', 'scenario.merlion'],
            [x.name(True) for x in self.config.user_values()],
        )
        self.assertTrue(self.config['scenario']['merlion'])
        self.assertEqual('foo', self.config['network']['epanet file'])

    def test_argparse_subparsers(self):
        parser = argparse.ArgumentParser(prog='tester')
        subp = parser.add_subparsers(title="Subcommands").add_parser('flushing')

        # Declare an argument by passing in the name of the subparser
        self.config['flushing']['flush nodes'].get('duration').declare_as_argument(
            group='flushing'
        )
        # Declare an argument by passing in the name of the subparser
        # and an implicit group
        self.config['flushing']['flush nodes'].get(
            'feasible nodes'
        ).declare_as_argument(group=('flushing', 'Node information'))
        # Declare an argument by passing in the subparser and a group name
        self.config['flushing']['flush nodes'].get(
            'infeasible nodes'
        ).declare_as_argument(group=(subp, 'Node information'))
        self.config.initialize_argparse(parser)

        # Note that the output for argparse changes in different versions
        # (in particular, "options:" vs "optional arguments:").  We will
        # only test for a subset of the output that should stay consistent.
        help = parser.format_help()
        self.assertIn(
            """
  -h, --help            show this help message and exit
  --epanet-file EPANET  EPANET network inp file

Subcommands:
  {flushing}

Scenario definition:
  --scenario-file STR   Scenario generation file, see the TEVASIM
                        documentation
  --merlion             Water quality model
""",
            help,
        )

        help = subp.format_help()
        self.assertIn(
            """
  -h, --help            show this help message and exit
  --duration FLOAT      Time [min] for flushing

Node information:
  --feasible-nodes STR  ALL, NZD, NONE, list or filename
  --infeasible-nodes STR
                        ALL, NZD, NONE, list or filename
""",
            help,
        )

    def test_argparse_lists(self):
        c = ConfigDict()
        self.assertEqual(c.domain_name(), '')
        sub_dict = c.declare('sub_dict', ConfigDict())
        sub_dict.declare('a', ConfigValue(domain=int))
        sub_dict.declare('b', ConfigValue())
        self.assertEqual(c.sub_dict.domain_name(), 'sub-dict')
        self.assertEqual(c.sub_dict.get('a').domain_name(), 'int')
        self.assertEqual(c.sub_dict.get('b').domain_name(), '')
        c.declare('lst', ConfigList(domain=int)).declare_as_argument(action='append')
        c.declare('sub', ConfigList(domain=c.sub_dict)).declare_as_argument(
            action='append'
        )
        c.declare('listof', ConfigValue(domain=ListOf(int))).declare_as_argument()

        parser = argparse.ArgumentParser(prog='tester')
        c.initialize_argparse(parser)

        # Note that the output for argparse changes in different versions
        # (in particular, "options:" vs "optional arguments:").  We will
        # only test for a subset of the output that should stay consistent.
        self.assertIn(
            """
  -h, --help            show this help message and exit
  --lst INT
  --sub SUB-DICT
  --listof LISTOF[INT]""".strip(),
            parser.format_help(),
        )

        args = parser.parse_args(
            [
                '--lst',
                '42',
                '--lst',
                '1',
                '--sub',
                'a=4',
                '--sub',
                'b=12,a:0',
                '--listof',
                '3,2 4',
            ]
        )
        leftovers = c.import_argparse(args)
        self.assertEqual(c.lst.value(), [42, 1])
        self.assertEqual(c.sub.value(), [{'a': 4, 'b': None}, {'a': 0, 'b': '12'}])
        self.assertEqual(c.listof, [3, 2, 4])

        args = parser.parse_args(['--sub', 'b=12,a 0'])
        with self.assertRaisesRegex(
            ValueError,
            r"(?s)invalid value for configuration 'sub':.*"
            r"Expected ':' or '=' but found '0' at Line 1 Column 8",
        ):
            leftovers = c.import_argparse(args)
        args = parser.parse_args(['--sub', 'b='])
        with self.assertRaisesRegex(
            ValueError,
            r"(?s)Expected value following '=' but encountered end of string",
        ):
            leftovers = c.import_argparse(args)
        args = parser.parse_args(['--sub', 'b'])
        with self.assertRaisesRegex(
            ValueError, r"(?s)Expected ':' or '=' but encountered end of string"
        ):
            leftovers = c.import_argparse(args)

    def test_getattr_setattr(self):
        config = ConfigDict()
        foo = config.declare('foo', ConfigDict(implicit=True, implicit_domain=int))
        foo.declare('explicit_bar', ConfigValue(0, int))

        self.assertEqual(1, len(foo))
        self.assertEqual(0, foo['explicit_bar'])
        self.assertEqual(0, foo.explicit_bar)
        foo.explicit_bar = 10
        self.assertEqual(1, len(foo))
        self.assertEqual(10, foo['explicit_bar'])
        self.assertEqual(10, foo.explicit_bar)

        foo.implicit_bar = 20
        self.assertEqual(2, len(foo))
        self.assertEqual(20, foo['implicit bar'])
        self.assertEqual(20, foo.implicit_bar)

        with self.assertRaisesRegex(
            ValueError,
            "Key 'baz' not defined in ConfigDict '' "
            "and Dict disallows implicit entries",
        ):
            config.baz = 10

        with self.assertRaisesRegex(AttributeError, "Unknown attribute 'baz'"):
            a = config.baz

    def test_nonString_keys(self):
        config = ConfigDict(implicit=True)
        config.declare(5, ConfigValue(50, int))
        self.assertIn(5, config)
        self.assertIn('5', config)
        self.assertEqual(config[5], 50)
        self.assertEqual(config['5'], 50)
        self.assertEqual(config.get(5).value(), 50)
        self.assertEqual(config.get('5').value(), 50)

        config[5] = 500
        self.assertIn(5, config)
        self.assertIn('5', config)
        self.assertEqual(config[5], 500)
        self.assertEqual(config['5'], 500)
        self.assertEqual(config.get(5).value(), 500)
        self.assertEqual(config.get('5').value(), 500)

        config[1] = 10
        self.assertIn(1, config)
        self.assertIn('1', config)
        self.assertEqual(config[1], 10)
        self.assertEqual(config['1'], 10)
        self.assertEqual(config.get(1).value(), 10)
        self.assertEqual(config.get('1').value(), 10)

        self.assertEqual(_display(config), "5: 500\n1: 10\n")

        config.set_value({5: 5000})
        self.assertIn(1, config)
        self.assertIn('1', config)
        self.assertEqual(config[1], 10)
        self.assertEqual(config['1'], 10)
        self.assertEqual(config.get(1).value(), 10)
        self.assertEqual(config.get('1').value(), 10)
        self.assertIn(5, config)
        self.assertIn('5', config)
        self.assertEqual(config[5], 5000)
        self.assertEqual(config['5'], 5000)
        self.assertEqual(config.get(5).value(), 5000)
        self.assertEqual(config.get('5').value(), 5000)

    def test_set_value(self):
        config = ConfigDict()
        config.declare('a b', ConfigValue())
        config.declare('a_c', ConfigValue())
        config.declare('a d e', ConfigValue())

        config.set_value({'a_b': 10, 'a_c': 20, 'a_d_e': 30})
        self.assertEqual(config.a_b, 10)
        self.assertEqual(config.a_c, 20)
        self.assertEqual(config.a_d_e, 30)

    def test_name_mapping(self):
        config = ConfigDict(implicit=True)

        config.a_b = 5
        self.assertEqual(list(config), ['a_b'])
        self.assertIs(config.get('a_b'), config.get('a b'))
        config['a b'] = 10
        self.assertEqual(config.a_b, 10)
        self.assertEqual(list(config), ['a_b'])
        self.assertIn('a b', config)
        self.assertIn('a_b', config)

        config['c d'] = 1
        self.assertEqual(list(config), ['a_b', 'c d'])
        self.assertIs(config.get('c_d'), config.get('c d'))
        config.c_d = 2
        self.assertEqual(config['c d'], 2)
        self.assertEqual(list(config), ['a_b', 'c d'])
        self.assertIn('c d', config)
        self.assertIn('c_d', config)

        config.declare('e_f', ConfigValue(5, domain=int))
        self.assertEqual(list(config), ['a_b', 'c d', 'e_f'])
        self.assertIs(config.get('e_f'), config.get('e f'))
        config['e f'] = 10
        self.assertEqual(config.e_f, 10)
        self.assertEqual(list(config), ['a_b', 'c d', 'e_f'])
        self.assertIn('e f', config)
        self.assertIn('e_f', config)

        config['g h'] = 1
        self.assertEqual(list(config), ['a_b', 'c d', 'e_f', 'g h'])
        self.assertIs(config.get('g_h'), config.get('g h'))
        config.g_h = 2
        self.assertEqual(config['g h'], 2)
        self.assertEqual(list(config), ['a_b', 'c d', 'e_f', 'g h'])
        self.assertIn('g h', config)
        self.assertIn('g_h', config)

    def test_call_options(self):
        config = ConfigDict(
            description="base description", doc="base doc", visibility=1, implicit=True
        )
        config.declare("a", ConfigValue(domain=int, doc="a doc", default=1))
        config.declare("b", config.get("a")(2))
        config.declare("c", config.get("a")(domain=float, doc="c doc"))
        config.d = 0
        config.e = ConfigDict(implicit=True)
        config.e.a = 0

        reference_template = """# base description
"""
        self._validateTemplate(config, reference_template)
        reference_template = """# base description
a: 1
b: 2
c: 1.0
d: 0
e:
  a: 0
"""
        self._validateTemplate(config, reference_template, visibility=1)

        # Preserving implicit values should leave the copy the same as
        # the original
        implicit_copy = config(preserve_implicit=True)
        self._validateTemplate(config, reference_template, visibility=1)

        # Simple copies strip out the implicitly-declared values
        reference_template = """# base description
a: 1
b: 2
c: 1.0
"""
        simple_copy = config()
        self._validateTemplate(simple_copy, reference_template, visibility=1)
        self.assertEqual(simple_copy._doc, "base doc")
        self.assertEqual(simple_copy._description, "base description")
        self.assertEqual(simple_copy._visibility, 1)

        mod_copy = config(description="new description", doc="new doc", visibility=0)
        reference_template = """# new description
a: 1
b: 2
c: 1.0
"""
        self._validateTemplate(mod_copy, reference_template, visibility=0)
        self.assertEqual(mod_copy._doc, "new doc")
        self.assertEqual(mod_copy._description, "new description")
        self.assertEqual(mod_copy._visibility, 0)

    def test_pickle(self):
        def anon_domain(domain):
            def cast(x):
                return domain(x)

            return cast

        cfg = ConfigDict()
        cfg.declare('int', ConfigValue(domain=int, default=10))
        cfg.declare('in', ConfigValue(domain=In([1, 3, 5]), default=1))
        cfg.declare('anon', ConfigValue(domain=anon_domain(int), default=1))
        cfg.declare('lambda', ConfigValue(domain=lambda x: int(x), default=1))
        cfg.declare('list', ConfigList(domain=str))

        out = StringIO()
        with LoggingIntercept(out, module=None):
            cfg.set_value(
                {'int': 100, 'in': 3, 'anon': 2.5, 'lambda': 1.5, 'list': [2, 'a']}
            )
            self.assertEqual(
                cfg.value(),
                {'int': 100, 'in': 3, 'anon': 2, 'lambda': 1, 'list': ['2', 'a']},
            )

            cfg2 = pickle.loads(pickle.dumps(cfg))
            self.assertEqual(
                cfg2.value(),
                {'int': 100, 'in': 3, 'anon': 2, 'lambda': 1, 'list': ['2', 'a']},
            )

            cfg2.list.append(10)
            self.assertEqual(
                cfg2.value(),
                {'int': 100, 'in': 3, 'anon': 2, 'lambda': 1, 'list': ['2', 'a', '10']},
            )
        # No warnings due to anything above.
        self.assertEqual(out.getvalue(), "")

        # On some platforms (notably, pypy3) if dill has been imported,
        # then lambda and anonymous functions are actually picklable
        # using the standard pickle.dumps() method.  We will check for
        # one of two cases: either the domain was not picklable and was
        # replaced by a passthrough "_UnpickleableDomain" object, OR it
        # was picklable and the original domain was enforced.

        out = StringIO()
        with LoggingIntercept(out, module=None):
            cfg2['anon'] = 5.5
        if type(cfg2.get('anon')._domain) is _UnpickleableDomain:
            self.assertIn(
                "ConfigValue 'anon' was pickled with an unpicklable domain",
                out.getvalue(),
            )
            self.assertEqual(cfg2['anon'], 5.5)
        else:
            self.assertEqual(out.getvalue(), "")
            self.assertIn('dill', sys.modules)
            self.assertEqual(cfg2['anon'], 5)

        out = StringIO()
        with LoggingIntercept(out, module=None):
            cfg2['lambda'] = 6.5
        if type(cfg2.get('lambda')._domain) is _UnpickleableDomain:
            self.assertIn(
                "ConfigValue 'lambda' was pickled with an unpicklable domain",
                out.getvalue(),
            )
            self.assertEqual(cfg2['lambda'], 6.5)
        else:
            self.assertEqual(out.getvalue(), "")
            self.assertIn('dill', sys.modules)
            self.assertEqual(cfg2['lambda'], 6)

    def test_unknowable_types(self):
        obj = ConfigValue()

        def local_fcn():
            pass

        try:
            pickle.dumps(local_fcn)
            local_picklable = True
        except:
            local_picklable = False

        # Test that _picklable does not cache the picklability of
        # function types
        self.assertIs(_picklable(_display, obj), _display)
        if local_picklable:
            self.assertIs(_picklable(local_fcn, obj), local_fcn)
        else:
            self.assertIsNot(_picklable(local_fcn, obj), local_fcn)

        # Twice: implicit test that the result is not cached
        self.assertIs(_picklable(_display, obj), _display)
        if local_picklable:
            self.assertIs(_picklable(local_fcn, obj), local_fcn)
        else:
            self.assertIsNot(_picklable(local_fcn, obj), local_fcn)

        self.assertIn(types.FunctionType, _picklable.unknowable_types)
        self.assertNotIn(types.FunctionType, _picklable.known)

    def test_known_types(self):
        def local_fcn():
            class LocalClass(object):
                pass

            return LocalClass

        local_class = local_fcn()

        self.assertIsNone(_picklable.known.get(local_class, None))
        self.assertIsNone(_picklable.known.get(GlobalClass, None))

        obj = ConfigValue()

        # Test that a global class is picklable
        self.assertIs(_picklable(GlobalClass, obj), GlobalClass)
        self.assertEqual(_picklable.known.get(GlobalClass, None), True)

        # Test that a local class is (most likely) not picklable
        try:
            pickle.dumps(local_class)
            local_picklable = True
        except:
            local_picklable = False
        if local_picklable:
            self.assertIs(_picklable(local_class, obj), local_class)
            self.assertEqual(_picklable.known.get(local_class, None), True)
        else:
            self.assertIsNot(_picklable(local_class, obj), local_class)
            self.assertEqual(_picklable.known.get(local_class, None), False)

        # Ensure that none of the above added the type `type` to the
        # "known" dict
        self.assertNotIn(type, _picklable.known)

    def test_self_assignment(self):
        cfg = ConfigDict()
        self.assertNotIn('d', dir(cfg))
        cfg.d = cfg.declare('d', ConfigValue(10, int))
        self.assertIn('d', dir(cfg))
        cfg.aa = cfg.declare('aa', ConfigValue(1, int))
        self.assertIn('aa', dir(cfg))
        # test that dir is sorted
        self.assertEqual(dir(cfg), sorted(dir(cfg)))
        # check that inconsistent name is flagged
        with self.assertRaisesRegex(ValueError, "Key 'b' not defined in ConfigDict ''"):
            cfg.b = cfg.declare('bb', ConfigValue(2, int))

    def test_declaration_errors(self):
        cfg = ConfigDict()
        cfg.b = cfg.declare('b', ConfigValue(2, int))
        with self.assertRaisesRegex(
            ValueError, "duplicate config 'b' defined for ConfigDict ''"
        ):
            cfg.b = cfg.declare('b', ConfigValue(2, int))
        with self.assertRaisesRegex(
            ValueError, "config 'dd' is already assigned to ConfigDict ''"
        ):
            cfg.declare('dd', cfg.get('b'))

    def test_declare_from(self):
        cfg = ConfigDict()
        cfg.declare('a', ConfigValue(default=1, domain=int))
        cfg.declare('b', ConfigValue(default=2, domain=int))
        cfg2 = ConfigDict()
        cfg2.declare_from(cfg)
        self.assertEqual(cfg.value(), cfg2.value())
        self.assertIsNot(cfg.get('a'), cfg2.get('a'))
        self.assertIsNot(cfg.get('b'), cfg2.get('b'))

        cfg2 = ConfigDict()
        cfg2.declare_from(cfg, skip={'a'})
        self.assertEqual(cfg.value()['b'], cfg2.value()['b'])
        self.assertNotIn('a', cfg2)

        with self.assertRaisesRegex(
            ValueError, "passed a block with a duplicate field, 'b'"
        ):
            cfg2.declare_from(cfg)

        with self.assertRaisesRegex(ValueError, "only accepts other ConfigDicts"):
            cfg2.declare_from({})

    def test_docstring_decorator(self):
        self.maxDiff = None

        @document_kwargs_from_configdict('CONFIG')
        class ExampleClass(object):
            CONFIG = ExampleConfig()

            @document_kwargs_from_configdict(CONFIG)
            def __init__(self):
                "A simple docstring"

            @document_kwargs_from_configdict(
                CONFIG, doc="A simple docstring\n", visibility=USER_OPTION
            )
            def fcn(self):
                pass

        ref = """
Keyword Arguments
-----------------
option_1: int, default=5
    The first configuration option

solver_options: dict, optional

    solver_option_1: float, default=1
        [DEVELOPER option]

        The first solver configuration option

    solver_option_2: float, default=1
        The second solver configuration option

        With a very long line containing wrappable text in a long, silly
        paragraph with little actual information.
        #) but a bulleted list
        #) with two bullets

    solver_option_3: float, default=1
        The third solver configuration option

           This has a leading newline and a very long line containing
           wrappable text in a long, silly paragraph with little actual
           information.

        .. and_a_list::
           #) but a bulleted list
           #) with two bullets

option_2: int, default=5
    The second solver configuration option with a very long line
    containing wrappable text in a long, silly paragraph with little
    actual information."""
        self.assertEqual(ExampleClass.__doc__, ref.lstrip())
        self.assertEqual(ExampleClass.__init__.__doc__, "A simple docstring\n" + ref)

        ref = """
Keyword Arguments
-----------------
option_1: int, default=5
    The first configuration option

solver_options: dict, optional

    solver_option_2: float, default=1
        The second solver configuration option

        With a very long line containing wrappable text in a long, silly
        paragraph with little actual information.
        #) but a bulleted list
        #) with two bullets

    solver_option_3: float, default=1
        The third solver configuration option

           This has a leading newline and a very long line containing
           wrappable text in a long, silly paragraph with little actual
           information.

        .. and_a_list::
           #) but a bulleted list
           #) with two bullets

option_2: int, default=5
    The second solver configuration option with a very long line
    containing wrappable text in a long, silly paragraph with little
    actual information."""
        self.assertEqual(ExampleClass.fcn.__doc__, "A simple docstring\n" + ref)

        ref = """
Keyword Arguments
-----------------
option_1: int, default=5
    The first configuration option

solver_options: dict, optional

    solver_option_2: float, default=1
        The second solver configuration option

        With a very long line containing wrappable text in a long, silly paragraph with little actual information.
        #) but a bulleted list
        #) with two bullets

    solver_option_3: float, default=1
        The third solver configuration option

           This has a leading newline and a very long line containing wrappable text in a long, silly paragraph with little actual information.

        .. and_a_list::
           #) but a bulleted list
           #) with two bullets

option_2: int, default=5
    The second solver configuration option with a very long line containing wrappable text in a long, silly paragraph with little actual information."""
        with LoggingIntercept() as LOG:
            self.assertEqual(add_docstring_list("", ExampleClass.CONFIG), ref)
        self.assertIn('add_docstring_list is deprecated', LOG.getvalue())


if __name__ == "__main__":
    unittest.main()
