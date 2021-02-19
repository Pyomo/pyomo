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

from pyomo.common import PyomoAPIData, pyomo_api, PyomoAPIFactory
from pyomo.common.log import LoggingIntercept

from six import StringIO

class TestData(unittest.TestCase):

    def test_print_PyomoAPIData_string(self):
        #"""Print PyomoAPIData string"""
        data = PyomoAPIData()
        data.a = 1
        data.b = [1,2]
        data.c = PyomoAPIData()
        data.c.x = 1
        data.c.y = 2
        data['aa'] = 'here is more'
        data.clean()
        self.assertEqual(sorted(data.unused()), ['a','aa','b','c'])
        self.assertEqual(
            str(data),
            """a: 1
aa: here is more
b: [1, 2]
c:
    x: 1
    y: 2""")
        self.assertEqual(len(data._dirty_), 0)

    def test_print_PyomoAPIData_repr(self):
        #"""Print PyomoAPIData representation"""
        data = PyomoAPIData()
        data.a = 1
        data.b = [1,2]
        data.c = PyomoAPIData()
        data.c.x = 1
        data.c.y = 2
        data['aa'] = 'here is more'
        data.clean()
        # Because the PyomoAPIData is a dict, the repr() is subject to
        # change between python versions.  Cast back to a dict and
        # verify.
        self.assertEqual(
            eval(repr(data)),
            eval("{'a': 1, 'aa': 'here is more', 'b': [1, 2], "
                 "'c': {'y': 2, 'x': 1}}"))
        self.assertEqual(len(data._dirty_), 0)

    def test_err_unknown_attr(self):
        #"""Unknown attribute"""
        data = PyomoAPIData()
        with self.assertRaisesRegexp(AttributeError, 'Unknown attribute _x'):
            data._x

    def test_err_undeclared_attr(self):
        #"""Undeclared attribute"""
        data = PyomoAPIData()
        data.declare('a')
        with self.assertRaisesRegexp( AttributeError,
                                      "Undeclared attribute 'x'" ):
            data.x

    def test_err_undeclared_list_attr(self):
        #"""Undeclared attribute"""
        data = PyomoAPIData()
        data.declare(['a'])
        with self.assertRaisesRegexp( AttributeError,
                                      "Undeclared attribute 'x'" ):
            data.x


class TestAPI(unittest.TestCase):

    def test1_no_kwds_no_return(self):
        #"""Simple test: no keyword arguments or return values"""
        @pyomo_api
        def test1(data):
            """
            Required:
                data: input data
            """
            data.a = 2
            data.b[0] = 2
        #
        options = PyomoAPIData()
        options.a = 1
        options.b = [1,2]
        retval = test1(options)
        self.assertEqual(retval.data.a, 2)
        self.assertEqual(retval.data.b, [2,2])
        retval = test1(data=options)
        self.assertEqual(retval.data.a, 2)
        self.assertEqual(retval.data.b, [2,2])
        #
        self.assertTrue('test1' in PyomoAPIFactory.services())

    def test1a_no_kwds_single_return(self):
        #"""Simple test: no keyword arguments or return values"""
        @pyomo_api
        def test1a(data):
            """
            Required:
                data: input data
            Return:
                data: output data
            """
            data.a = 2
            data.b[0] = 2
        #
        options = PyomoAPIData()
        options.a = 1
        options.b = [1,2]
        retval = test1a(options)
        self.assertEqual(retval.data.a, 2)
        self.assertEqual(retval.data.b, [2,2])
        retval = test1a(data=options)
        self.assertEqual(retval.data.a, 2)
        self.assertEqual(retval.data.b, [2,2])

    def test1b_data_kwds_no_return(self):
        #"""Simple test: data keyword argument, no return values"""
        @pyomo_api
        def test1b(data=None):
            data.a = 2
            data.b[0] = 2
        #
        options = PyomoAPIData()
        options.a = 1
        options.b = [1,2]
        retval = test1b(options)
        self.assertEqual(retval.data.a, 2)
        self.assertEqual(retval.data.b, [2,2])
        retval = test1b(data=options)
        self.assertEqual(retval.data.a, 2)
        self.assertEqual(retval.data.b, [2,2])

    def test2_implicit_no_kwds_return_data(self):
        #"""Simple test: no keyword arguments, returning data"""
        @pyomo_api
        def test2(data):
            data.a = 2
            data.b[0] = 2
            return data
        #
        options = PyomoAPIData()
        options.a = 1
        options.b = [1,2]
        retval = test2(options)
        self.assertEqual(retval.data.a, 2)
        self.assertEqual(retval.data.b, [2,2])
        retval = test2(data=options)
        self.assertEqual(retval.data.a, 2)
        self.assertEqual(retval.data.b, [2,2])

    def test2a_no_kwds_return_data(self):
        #"""Simple test: no keyword arguments, returning data"""
        @pyomo_api
        def test2a(data):
            """
            Required:
                data: input data
            Return:
                data: output data
            """
            data.a = 2
            data.b[0] = 2
            return data
        #
        options = PyomoAPIData()
        options.a = 1
        options.b = [1,2]
        retval = test2a(options)
        self.assertEqual(retval.data.a, 2)
        self.assertEqual(retval.data.b, [2,2])
        retval = test2a(data=options)
        self.assertEqual(retval.data.a, 2)
        self.assertEqual(retval.data.b, [2,2])

    def test3_kwds_no_return(self):
        #"""Simple test: keyword arguments, no return values"""
        @pyomo_api
        def test3(data, x=1, y=2):
            """
            Required:
                data: input data
                x: integer
            Optional:
                y: integer
            """
            data.a = y
            data.b[0] = x
        #
        options = PyomoAPIData()
        options.a = 1
        options.b = [1,2]
        retval = test3(options, x=2)
        self.assertEqual(retval.data.a, 2)
        self.assertEqual(retval.data.b, [2,2])
        retval = test3(data=options, x=2)
        self.assertEqual(retval.data.a, 2)
        self.assertEqual(retval.data.b, [2,2])

    def test3a_kwds_no_return(self):
        #"""Simple test: keyword arguments, no return values"""
        @pyomo_api
        def test3a(x=1, y=2, data=None):
            """
            Required:
                data: input data
                x: integer
            Optional:
                y: integer
            """
            data.a = y
            data.b[0] = x
        #
        options = PyomoAPIData()
        options.a = 1
        options.b = [1,2]
        retval = test3a(options, x=2)
        self.assertEqual(retval.data.a, 2)
        self.assertEqual(retval.data.b, [2,2])
        retval = test3a(data=options, x=2)
        self.assertEqual(retval.data.a, 2)
        self.assertEqual(retval.data.b, [2,2])

    def test4_kwds_return_data(self):
        #"""Simple test: keyword arguments, simple return value"""
        @pyomo_api
        def test4(data, x=1, y=2):
            """
            Required:
                data: input data
                x: integer
            Optional:
                y: integer
            Return:
                data: output data
            """
            data.a = y
            data.b[0] = x
            return data
        #
        options = PyomoAPIData()
        options.a = 1
        options.b = [1,2]
        retval = test4(options, x=2)
        self.assertEqual(retval.data.a, 2)
        self.assertEqual(retval.data.b, [2,2])
        retval = test4(data=options, x=2)
        self.assertEqual(retval.data.a, 2)
        self.assertEqual(retval.data.b, [2,2])

    def test5_kwds_return_apidata(self):
        #"""Simple test: keyword arguments, non-data return values"""
        @pyomo_api(outputs=('z'))
        def test5(data, x=1, y=2):
            """
            Required:
                data: input data
                x: integer
            Optional:
                y: integer
            Return:
                data: output data
                z: integer
            """
            data.a = y
            data.b[0] = x
            return PyomoAPIData(z=x)
        #
        options = PyomoAPIData()
        options.a = 1
        options.b = [1,2]
        retval = test5(options, x=2)
        self.assertEqual(retval.data.a, 2)
        self.assertEqual(retval.data.b, [2,2])
        retval = test5(data=options, x=2)
        self.assertEqual(retval.data.a, 2)
        self.assertEqual(retval.data.b, [2,2])
        self.assertEqual(retval.z, 2)

    def test6_kwds_return_apidata(self):
        #"""Simple test: keyword arguments, non-data return values with data"""
        @pyomo_api(outputs=('z'))
        def test6(data, x=1, y=2):
            """
            Required:
                data: input data
                x: integer
            Optional:
                y: integer
            Return:
                data: output data
                z: integer
            """
            data.a = y
            data.b[0] = x
            return PyomoAPIData(data=data, z=x)
        #
        options = PyomoAPIData()
        options.a = 1
        options.b = [1,2]
        retval = test6(options, x=2)
        self.assertEqual(retval.data.a, 2)
        self.assertEqual(retval.data.b, [2,2])
        retval = test6(data=options, x=2)
        self.assertEqual(retval.data.a, 2)
        self.assertEqual(retval.data.b, [2,2])
        self.assertEqual(retval.z, 2)

    def test5a_kwds_return_apidata(self):
        #"""Outputs specified in docstring: keyword arguments, non-data return values"""
        @pyomo_api
        def test5a(data, x=1, y=2):
            """
            Required:
                data: input data
                x: integer
            Optional:
                y: integer
            Return:
                data: output data
                z: integer
            """
            data.a = y
            data.b[0] = x
            return PyomoAPIData(z=x)
        #
        options = PyomoAPIData()
        options.a = 1
        options.b = [1,2]
        retval = test5a(options, x=2)
        self.assertEqual(retval.data.a, 2)
        self.assertEqual(retval.data.b, [2,2])
        retval = test5a(data=options, x=2)
        self.assertEqual(retval.data.a, 2)
        self.assertEqual(retval.data.b, [2,2])
        self.assertEqual(retval.z, 2)

    def test6a_kwds_return_apidata(self):
        #"""Outputs specified in docstring: keyword arguments, non-data return values with data"""
        @pyomo_api
        def test6a(data, x=1, y=2):
            """
            Required:
                data: input data
                x: integer
            Optional:
                y: integer
            Return:
                data: output data
                z: integer
            """
            data.a = y
            data.b[0] = x
            return PyomoAPIData(data=data, z=x)
        #
        options = PyomoAPIData()
        options.a = 1
        options.b = [1,2]
        retval = test6a(options, x=2)
        self.assertEqual(retval.data.a, 2)
        self.assertEqual(retval.data.b, [2,2])
        retval = test6a(data=options, x=2)
        self.assertEqual(retval.data.a, 2)
        self.assertEqual(retval.data.b, [2,2])
        self.assertEqual(retval.z, 2)

    def test7a_dict_data(self):
        #"""Test with dict data"""
        @pyomo_api
        def test7a(data, x=1, y=2):
            """
            Required:
                data: input data
                x: integer
            Optional:
                y: integer
            """
            data.a = y
            data.b[0] = x
        #
        options = {}
        options['a'] = 1
        options['b'] = [1,2]
        retval = test7a(options, x=2)
        self.assertEqual(retval.data.a, 2)
        self.assertEqual(retval.data.b, [2,2])
        retval = test7a(data=options, x=2)
        self.assertEqual(retval.data.a, 2)
        self.assertEqual(retval.data.b, [2,2])

    def test7b_dict_data_return_dict(self):
        #"""Test with dict data and return a dictionary"""
        @pyomo_api
        def test7b(data, x=1, y=2):
            """
            Required:
                data: input data
                x: integer
            Optional:
                y: integer
            """
            data.a = y
            data.b[0] = x
            return {'data':data}
        #
        options = {}
        options['a'] = 1
        options['b'] = [1,2]
        retval = test7b(options, x=2)
        self.assertEqual(retval.data.a, 2)
        self.assertEqual(retval.data.b, [2,2])
        retval = test7b(data=options, x=2)
        self.assertEqual(retval.data.a, 2)
        self.assertEqual(retval.data.b, [2,2])

    def test7c_dict_data_return_dict(self):
        #"""Test with dict data and return a dictionary"""
        @pyomo_api
        def test7c(data, x=1, y=2):
            """
            Required:
                data: input data
                x: integer
            Optional:
                y: integer
            Return:
                z: integer
            """
            data.a = y
            data.b[0] = x
            return {'data':data, 'z':x}
        #
        options = {}
        options['a'] = 1
        options['b'] = [1,2]
        retval = test7c(options, x=2)
        self.assertEqual(retval.data.a, 2)
        self.assertEqual(retval.data.b, [2,2])
        retval = test7c(data=options, x=2)
        self.assertEqual(retval.data.a, 2)
        self.assertEqual(retval.data.b, [2,2])
        self.assertEqual(retval.z, 2)

    def test8_nested_data(self):
        #"""Simple test with required nested data"""
        @pyomo_api
        def test8(data):
            """
            Required:
                data: input data
                data.foo.bar:
            """
            data.foo.foo = 3
            #data.a = 2
            #data.b[0] = 2
        #
        options = PyomoAPIData()
        options.foo = PyomoAPIData()
        options.foo.bar = 1
        options.a = 1
        options.b = [1,2]
        retval = test8(options)
        self.assertEqual(retval.data.a, 1)
        self.assertEqual(retval.data.b, [1,2])
        self.assertEqual(retval.data.foo.foo, 3)
        #retval = test8(data=options)
        #self.assertEqual(retval.data.a, 2)
        #self.assertEqual(retval.data.b, [2,2])

    def test9_complex_documentation(self):
        #"""Simple test: no keyword arguments or return values"""
        @pyomo_api
        def test9(data):
            """
            This is the
            short
            documentation.

            This

            is

            the

            long documentation.
            Required:
                data: multiline
                      description of
                      input data object
                data.a: More data
            """
            data.a = 2
            data.b[0] = 2
        #
        options = PyomoAPIData()
        options.a = 1
        options.b = [1,2]
        retval = test9(options)
        self.assertEqual(retval.data.a, 2)
        self.assertEqual(retval.data.b, [2,2])
        retval = test9(data=options)
        self.assertEqual(retval.data.a, 2)
        self.assertEqual(retval.data.b, [2,2])
        #
        self.assertTrue('test1' in PyomoAPIFactory.services())
        self.assertEqual(test9.__short_doc__, 'This is the\nshort\ndocumentation.')
        self.assertEqual(test9.__long_doc__, 'This\n\nis\n\nthe\n\nlong documentation.')

    def test10_missing_required_arg(self):
        #"""Simple test: no keyword arguments or return values"""
        @pyomo_api
        def test10(data=None, x=1):
            """
            Required:
                x:
            Optional:
                data:
            """
            return PyomoAPIData(z=2*x)
        #
        with self.assertRaisesRegexp(
                RuntimeError,
                "Cannot return value 'z' that is not a predefined output" ):
            retval = test10(x=3)

    def test_err1_varargs(self):
        #"""Expect an error when variable length arguments are supported"""
        buf = StringIO()
        with LoggingIntercept(buf, 'pyomo.common'):
            @pyomo_api
            def err1(*args): pass
        self.assertIn(
            "Attempting to declare Pyomo task with function 'err1' "
            "that contains variable arguments", buf.getvalue())

    def test_err2_kwds(self):
        #"""Expect an error when variable length keyword arguments are supported"""
        buf = StringIO()
        with LoggingIntercept(buf, 'pyomo.common'):
            @pyomo_api
            def err2(**kwargs): pass
        self.assertIn(
            "Attempting to declare Pyomo task with function 'err2' "
            "that contains variable keyword arguments", buf.getvalue())

    def test_err3_invalid_return(self):
        #"""Expect an error when return value is not None or Options()"""
        @pyomo_api
        def err3(data):
            return 1

        with self.assertRaisesRegexp(
                RuntimeError,
                "A Pyomo task function must return either None, a "
                "PyomoAPIData object, or an instance of dict"):
            err3(PyomoAPIData())

    def test_err4_missing_input_data(self):
        #"""Expect an error when no data argument is specified"""
        @pyomo_api
        def err4(data):
            data.a = 2
            data.b[0] = 2
        with self.assertRaisesRegexp(
                RuntimeError,
                "A PyomoTask instance must be executed with at "
                "'data' argument"):
            err4()

    def test_err5_unexpected_return(self):
        #"""Expect an error when an unspecified return value is given"""
        @pyomo_api
        def err5(data):
            return PyomoAPIData(z=None)
        with self.assertRaisesRegexp(
                RuntimeError,
                "Cannot return value 'z' that is not a predefined output "
                "of a Pyomo task" ):
            err5(PyomoAPIData())

    def test_err6_missing_input_data(self):
        #"""Expect an error when no data argument is specified"""

        buf = StringIO()
        with LoggingIntercept(buf, 'pyomo.common'):
            @pyomo_api
            def err6(): pass
        self.assertIn("A Pyomo functor 'err6' must have a 'data argument",
                      buf.getvalue())

        # Note: the TypeError message changes in Python 3.6, so we need
        # a weaker regexp.
        with self.assertRaisesRegexp(
                TypeError,
                "err6\(\) takes .* arguments .* given" ):
            err6(PyomoAPIData())

    def test_err7a_arg_missing_from_docstring(self):
        #"""Argument missing from docstring"""
        buf = StringIO()
        with LoggingIntercept(buf, 'pyomo.common'):
            @pyomo_api
            def err7a(data, x=1, y=2):
                """
                Optional:
                   y: integer
                """
                pass
        self.assertIn("Argument 'x' is not specified in the docstring",
                      buf.getvalue())

    def test_err7b_arg_missing_from_docstring(self):
        #"""Argument missing from docstring"""
        buf = StringIO()
        with LoggingIntercept(buf, 'pyomo.common'):
            @pyomo_api
            def err7b(data, x=1, y=2):
                """
                Required:
                    x: integer
               """
                pass
        self.assertIn("Argument 'y' is not specified in the docstring",
                      buf.getvalue())

    def test_err7c_return_missing_from_docstring(self):
        #"""Argument missing from docstring"""
        buf = StringIO()
        with LoggingIntercept(buf, 'pyomo.common'):
            @pyomo_api(outputs=('z'))
            def err7c(data, x=1, y=2):
                """
                Required:
                    x: integer
                Optional:
                    y: integer
                """
                return PyomoAPIData(z=1)
        self.assertIn("Return value 'z' is not specified", buf.getvalue())

    def test_err7A_extra_required_arg(self):
        #"""Unexpected required argument"""
        buf = StringIO()
        with LoggingIntercept(buf, 'pyomo.common'):
            @pyomo_api
            def err7A(data, x=1, y=2):
                """
                Required:
                    x: integer
                    bad: integer
                Optional:
                    y: integer
                """
                pass
        self.assertIn("Unexpected name 'bad' in list of required inputs",
                      buf.getvalue())

    def test_err7B_extra_optional_arg(self):
        #"""Argument missing from docstring"""
        buf = StringIO()
        with LoggingIntercept(buf, 'pyomo.common'):
            @pyomo_api
            def err7B(data, x=1, y=2):
                """
                Required:
                    x: integer
                Optional:
                    y: integer
                    bad: integer
                """
                pass
        self.assertIn("Unexpected name 'bad' in list of optional inputs",
                      buf.getvalue())

    def test_err7C_extra_return_arg(self):
        #"""Argument missing from docstring"""
        buf = StringIO()
        with LoggingIntercept(buf, 'pyomo.common'):
            @pyomo_api(outputs=('z'))
            def err7C(data, x=1, y=2):
                """
                Required:
                    x: integer
                Optional:
                    y: integer
                Return:
                    z: integer
                    bad: integer
                """
                return PyomoAPIData(z=1)
        self.assertIn("Unexpected name 'bad' in list of outputs",
                      buf.getvalue())

    def test_err8a_missing_nested_apidata(self):
        #"""Missing nested value"""
        @pyomo_api
        def err8a(data):
            """
            Required:
                data.x: integer
            """
            pass
        with self.assertRaisesRegexp(
                RuntimeError,
                "None value found for nested attribute 'data.x'" ):
            err8a(PyomoAPIData())

    def test_err8b_missing_nested_arg(self):
        #"""Nested value with None value"""
        @pyomo_api
        def err8b(data):
            """
            Required:
                data.x: integer
            """
            pass
        with self.assertRaisesRegexp(
                RuntimeError,
                "None value found for nested attribute 'data.x'" ):
            err8b(PyomoAPIData(data=PyomoAPIData()))

    def test_err8c_missing_nested_dict_arg(self):
        #"""Nested value with None value"""
        @pyomo_api
        def err8c(data):
            """
            Required:
                data.x.y: integer
            """
            pass
        with self.assertRaisesRegexp(
                RuntimeError,
                "Failed to verify existence of nested attribute 'data.x.y'" ):
            err8c(PyomoAPIData(x={}))

    def test_err10_test_with_nested_data(self):
        #"""Simple test with required nested data"""
        @pyomo_api
        def err10(data):
            """
            Required:
                data: input data
                data.foo.bar:
            """
            data.foo.foo = 3
            with self.assertRaisesRegexp(
                    AttributeError, "Undeclared attribute 'a'" ):
                data.a = 2
        #
        options = PyomoAPIData()
        options.foo = PyomoAPIData()
        options.foo.bar = 1
        options.a = 1
        options.b = [1,2]
        retval = err10(options)
        self.assertEqual(retval.data.a, 1)
        self.assertEqual(retval.data.b, [1,2])
        self.assertEqual(retval.data.foo.foo, 3)

    def test_err10a_duplicate_function(self):
        #"""Expect an error when the same functor is defined twice"""
        buf = StringIO()
        with LoggingIntercept(buf, 'pyomo.common'):
            @pyomo_api
            def err10a(data): pass

            @pyomo_api
            def err10a(data): pass
        self.assertIn("Cannot define API err10a, since this API name "
                      "is already defined", buf.getvalue())

    def test_err10b_duplicate_namespace_function(self):
        #"""Expect an error when the same functor is defined twice"""
        buf = StringIO()
        with LoggingIntercept(buf, 'pyomo.common'):
            @pyomo_api(namespace='foo')
            def err10b(data): pass

            @pyomo_api(namespace='foo')
            def err10b(data): pass
        self.assertIn("Cannot define API foo.err10b, since this API name "
                      "is already defined", buf.getvalue())

    def test_err10c_duplicate_namespace_scoping(self):
        #"""Expect an error when the same functor is defined twice"""
        @pyomo_api(namespace='foo')
        def err10c(data): pass

        @pyomo_api()
        def err10c(data): pass

    def test_err11_missing_data(self):
        #"""Expect an error when 'data' is not defined when it is required"""
        @pyomo_api
        def err11(data=None, x=1):
            """
            Required:
                data:
                x:
            """
        with self.assertRaisesRegexp(
                RuntimeError,
                "A PyomoTask instance must be executed with at 'data' "
                "argument" ):
            err11(x=2)

    def test_err12_extra_data(self):
        #"""Expect an error when multiple data options are provided"""
        @pyomo_api
        def err12(data): pass
        with self.assertRaisesRegexp(
                RuntimeError,
                "A PyomoTask instance can only be executed with a single "
                "non-keyword argument" ):
            err12({}, {})

    def test_err13_bad_return_type(self):
        #"""Expect an error when returning something other than None, PyomoAPIData or a dict object"""
        @pyomo_api
        def err13(data):
            return set()
        with self.assertRaisesRegexp(
                RuntimeError,
                "A Pyomo task function must return either None, a "
                "PyomoAPIData object, or an instance of dict" ):
            err13({})

if __name__ == "__main__":
    unittest.main()
