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
import os
import shutil
import sys
from io import StringIO

import pyomo.common.unittest as unittest

from pyomo.common.gsl import find_GSL
from pyomo.common.log import LoggingIntercept
from pyomo.common.tempfiles import TempfileManager
from pyomo.environ import (
    ConcreteModel,
    Block,
    Var,
    Objective,
    Expression,
    SolverFactory,
    value,
    Param,
)
from pyomo.core.base.external import (
    PythonCallbackFunction,
    ExternalFunction,
    AMPLExternalFunction,
)
from pyomo.core.base.units_container import pint_available, units
from pyomo.core.expr.numeric_expr import (
    ExternalFunctionExpression,
    NPV_ExternalFunctionExpression,
)
from pyomo.opt import check_available_solvers


def _count(*args):
    return len(args)


def _sum(*args):
    return 2 + sum(args)


def _f(x, y, z):
    return x**2 + 3 * x * y + x * y * z**2


def _g(args, fixed):
    x, y, z = args[:3]
    return [2 * x + 3 * y + y * z**2, 3 * x + x * z**2, 2 * x * y * z]


def _h(args, fixed):
    x, y, z = args[:3]
    return [2, 3 + z**2, 0, 2 * y * z, 2 * x * z, 2 * x * y]


def _g_bad(args, fixed):
    x, y, z = args[:3]
    return [2 * x + 3 * y + y * z**2, 3 * x + x * z**2, 2 * x * y * z, 0]


def _h_bad(args, fixed):
    x, y, z = args[:3]
    return [
        # 2,
        3 + z**2,
        0,
        2 * y * z,
        2 * x * z,
        2 * x * y,
    ]


def _fgh(args, fixed, fgh):
    return _f(*args), _g(args, fixed), _h(args, fixed)


class TestPythonCallbackFunction(unittest.TestCase):
    def test_constructor_errors(self):
        m = ConcreteModel()
        with self.assertRaisesRegex(
            ValueError,
            "Duplicate definition of external function "
            r"through positional and keyword \('function='\)",
        ):
            m.f = ExternalFunction(_count, function=_count)
        with self.assertRaisesRegex(
            ValueError,
            "PythonCallbackFunction constructor only "
            "supports 0 - 3 positional arguments",
        ):
            m.f = ExternalFunction(1, 2, 3, 4)
        with self.assertRaisesRegex(
            ValueError,
            "Cannot specify 'fgh' with any of {'function', 'gradient', hessian'}",
        ):
            m.f = ExternalFunction(_count, fgh=_fgh)

    def test_call_countArgs(self):
        m = ConcreteModel()
        m.f = ExternalFunction(_count)
        self.assertIsInstance(m.f, PythonCallbackFunction)
        self.assertEqual(value(m.f()), 0)
        self.assertEqual(value(m.f(2)), 1)
        self.assertEqual(value(m.f(2, 3)), 2)

    def test_call_sumfcn(self):
        m = ConcreteModel()
        m.f = ExternalFunction(_sum)
        self.assertIsInstance(m.f, PythonCallbackFunction)
        self.assertEqual(value(m.f()), 2.0)
        self.assertEqual(value(m.f(1)), 3.0)
        self.assertEqual(value(m.f(1, 2)), 5.0)

    def test_evaluate_fgh_fgh(self):
        m = ConcreteModel()
        m.f = ExternalFunction(fgh=_fgh)
        f, g, h = m.f.evaluate_fgh((5, 7, 11, m.f._fcn_id))
        self.assertEqual(f, 5**2 + 3 * 5 * 7 + 5 * 7 * 11**2)
        self.assertEqual(
            g, [2 * 5 + 3 * 7 + 7 * 11**2, 3 * 5 + 5 * 11**2, 2 * 5 * 7 * 11, 0]
        )
        self.assertEqual(
            h, [2, 3 + 11**2, 0, 2 * 7 * 11, 2 * 5 * 11, 2 * 5 * 7, 0, 0, 0, 0]
        )

        f, g, h = m.f.evaluate_fgh((5, 7, 11, m.f._fcn_id), fixed=[0, 1, 0, 1])
        self.assertEqual(f, 5**2 + 3 * 5 * 7 + 5 * 7 * 11**2)
        self.assertEqual(g, [2 * 5 + 3 * 7 + 7 * 11**2, 0, 2 * 5 * 7 * 11, 0])
        self.assertEqual(h, [2, 0, 0, 2 * 7 * 11, 0, 2 * 5 * 7, 0, 0, 0, 0])

    def test_evaluate_fgh_f_g_h(self):
        m = ConcreteModel()
        m.f = ExternalFunction(_f, _g, _h)
        f, g, h = m.f.evaluate_fgh((5, 7, 11, m.f._fcn_id))
        self.assertEqual(f, 5**2 + 3 * 5 * 7 + 5 * 7 * 11**2)
        self.assertEqual(
            g, [2 * 5 + 3 * 7 + 7 * 11**2, 3 * 5 + 5 * 11**2, 2 * 5 * 7 * 11, 0]
        )
        self.assertEqual(
            h, [2, 3 + 11**2, 0, 2 * 7 * 11, 2 * 5 * 11, 2 * 5 * 7, 0, 0, 0, 0]
        )

        f, g, h = m.f.evaluate_fgh((5, 7, 11, m.f._fcn_id), fixed=[0, 1, 0, 1])
        self.assertEqual(f, 5**2 + 3 * 5 * 7 + 5 * 7 * 11**2)
        self.assertEqual(g, [2 * 5 + 3 * 7 + 7 * 11**2, 0, 2 * 5 * 7 * 11, 0])
        self.assertEqual(h, [2, 0, 0, 2 * 7 * 11, 0, 2 * 5 * 7, 0, 0, 0, 0])

        f, g, h = m.f.evaluate_fgh((5, 7, 11, m.f._fcn_id), fgh=1)
        self.assertEqual(f, 5**2 + 3 * 5 * 7 + 5 * 7 * 11**2)
        self.assertEqual(
            g, [2 * 5 + 3 * 7 + 7 * 11**2, 3 * 5 + 5 * 11**2, 2 * 5 * 7 * 11, 0]
        )
        self.assertIsNone(h)

        f, g, h = m.f.evaluate_fgh((5, 7, 11, m.f._fcn_id), fgh=0)
        self.assertEqual(f, 5**2 + 3 * 5 * 7 + 5 * 7 * 11**2)
        self.assertIsNone(g)
        self.assertIsNone(h)

    def test_evaluate_fgh_f_g(self):
        m = ConcreteModel()
        m.f = ExternalFunction(_f, _g)
        with self.assertRaisesRegex(
            RuntimeError,
            "ExternalFunction 'f' was not defined with a Hessian callback.",
        ):
            f, g, h = m.f.evaluate_fgh((5, 7, 11, m.f._fcn_id))

        f, g, h = m.f.evaluate_fgh((5, 7, 11, m.f._fcn_id), fgh=1)
        self.assertEqual(f, 5**2 + 3 * 5 * 7 + 5 * 7 * 11**2)
        self.assertEqual(
            g, [2 * 5 + 3 * 7 + 7 * 11**2, 3 * 5 + 5 * 11**2, 2 * 5 * 7 * 11, 0]
        )
        self.assertIsNone(h)

        f, g, h = m.f.evaluate_fgh((5, 7, 11, m.f._fcn_id), fgh=0)
        self.assertEqual(f, 5**2 + 3 * 5 * 7 + 5 * 7 * 11**2)
        self.assertIsNone(g)
        self.assertIsNone(h)

    def test_evaluate_fgh_f(self):
        m = ConcreteModel()
        m.f = ExternalFunction(_f)
        with self.assertRaisesRegex(
            RuntimeError,
            "ExternalFunction 'f' was not defined with a gradient callback.",
        ):
            f, g, h = m.f.evaluate_fgh((5, 7, 11, m.f._fcn_id))

        with self.assertRaisesRegex(
            RuntimeError,
            "ExternalFunction 'f' was not defined with a gradient callback.",
        ):
            f, g, h = m.f.evaluate_fgh((5, 7, 11, m.f._fcn_id), fgh=1)

        f, g, h = m.f.evaluate_fgh((5, 7, 11, m.f._fcn_id), fgh=0)
        self.assertEqual(f, 5**2 + 3 * 5 * 7 + 5 * 7 * 11**2)
        self.assertIsNone(g)
        self.assertIsNone(h)

    def test_evaluate_errors(self):
        m = ConcreteModel()
        m.f = ExternalFunction(_f, _g_bad, _h_bad)
        f = m.f.evaluate((5, 7, 11, m.f._fcn_id))
        self.assertEqual(f, 5**2 + 3 * 5 * 7 + 5 * 7 * 11**2)

        with self.assertRaisesRegex(
            RuntimeError, "PythonCallbackFunction called with invalid Global ID"
        ):
            f = m.f.evaluate((5, 7, 11, -1))

        with self.assertRaisesRegex(
            RuntimeError,
            "External function 'f' returned an invalid "
            r"derivative vector \(expected 4, received 5\)",
        ):
            f = m.f.evaluate_fgh((5, 7, 11, m.f._fcn_id), fgh=1)

        with self.assertRaisesRegex(
            RuntimeError,
            "External function 'f' returned an invalid "
            r"Hessian matrix \(expected 10, received 9\)",
        ):
            f = m.f.evaluate_fgh((5, 7, 11, m.f._fcn_id), fgh=2)

    def test_getname(self):
        m = ConcreteModel()
        m.f = ExternalFunction(_sum)
        self.assertIsInstance(m.f, PythonCallbackFunction)
        self.assertEqual(m.f.name, "f")
        self.assertEqual(m.f.local_name, "f")
        self.assertEqual(m.f.getname(), "f")
        self.assertEqual(m.f.getname(True), "f")

        M = ConcreteModel()
        M.m = m
        self.assertEqual(M.m.f.name, "m.f")
        self.assertEqual(M.m.f.local_name, "f")
        self.assertEqual(M.m.f.getname(), "f")
        self.assertEqual(M.m.f.getname(True), "m.f")

    def test_extra_kwargs(self):
        m = ConcreteModel()
        with self.assertRaises(ValueError):
            m.f = ExternalFunction(_count, this_should_raise_error='foo')

    def test_clone(self):
        m = ConcreteModel()
        m.f = ExternalFunction(_sum)
        m.x = Var(initialize=3)
        m.y = Var(initialize=5)
        m.e = Expression(expr=m.f(m.x, m.y))
        self.assertIsInstance(m.f, PythonCallbackFunction)
        self.assertEqual(m.f._fcn_id, m.e.arg(0).arg(-1).value)
        self.assertEqual(value(m.e), 10)

        i = m.clone()
        self.assertIsInstance(i.f, PythonCallbackFunction)
        self.assertIsNot(i.f, m.f)
        self.assertIsNot(i.e, m.e)
        self.assertIsNot(i.e.arg(0), m.e.arg(0))
        self.assertEqual(i.f._fcn_id, i.e.arg(0).arg(-1).value)
        self.assertNotEqual(i.f._fcn_id, m.f._fcn_id)
        self.assertEqual(value(i.e), 10)

    def test_partial_clone(self):
        m = ConcreteModel()
        m.f = ExternalFunction(_sum)
        m.x = Var(initialize=3)
        m.y = Var(initialize=5)
        m.b = Block()
        m.b.e = Expression(expr=m.f(m.x, m.y))
        self.assertIsInstance(m.f, PythonCallbackFunction)
        self.assertEqual(m.f._fcn_id, m.b.e.arg(0).arg(-1).value)
        self.assertEqual(value(m.b.e), 10)

        m.c = m.b.clone()
        self.assertIsNot(m.b.e, m.c.e)
        self.assertIsNot(m.b.e.arg(0), m.c.e.arg(0))
        self.assertEqual(m.f._fcn_id, m.b.e.arg(0).arg(-1).value)
        self.assertEqual(m.f._fcn_id, m.c.e.arg(0).arg(-1).value)
        self.assertEqual(value(m.c.e), 10)

        # save / restore should not change the fcn_id
        _fcn_id = m.f._fcn_id
        m.f.__setstate__(m.f.__getstate__())
        self.assertEqual(m.f._fcn_id, _fcn_id)

        self.assertIsNot(m.b.e, m.c.e)
        self.assertIsNot(m.b.e.arg(0), m.c.e.arg(0))
        self.assertEqual(m.f._fcn_id, m.b.e.arg(0).arg(-1).value)
        self.assertEqual(m.f._fcn_id, m.c.e.arg(0).arg(-1).value)
        self.assertEqual(value(m.c.e), 10)

    def test_properties(self):
        m = ConcreteModel()
        m.f = ExternalFunction(_sum)
        e = m.f()
        self.assertIsInstance(e, NPV_ExternalFunctionExpression)
        self.assertFalse(e.is_constant())
        self.assertTrue(e.is_fixed())
        self.assertFalse(e.is_potentially_variable())

        # This is just a check for coverage: the expression is_constant
        # catches that the function ID is a NumericConstant and doesn't
        # call the method.
        self.assertFalse(e.arg(0).is_constant())
        self.assertTrue(e.arg(0).is_fixed())
        self.assertFalse(e.arg(0).is_potentially_variable())

        m.p = Param(initialize=1)
        e = m.f(m.p)
        self.assertIsInstance(e, NPV_ExternalFunctionExpression)
        self.assertFalse(e.is_constant())
        self.assertTrue(e.is_fixed())
        self.assertFalse(e.is_potentially_variable())

        m.x = Var(initialize=1)
        e = m.f(m.p, m.x)
        self.assertIsInstance(e, ExternalFunctionExpression)
        self.assertFalse(e.is_constant())
        self.assertFalse(e.is_fixed())
        self.assertTrue(e.is_potentially_variable())

    def test_pprint(self):
        m = ConcreteModel()
        m.h = ExternalFunction(_count)

        out = StringIO()
        m.pprint(ostream=out)
        self.assertEqual(
            out.getvalue().strip(),
            """
1 ExternalFunction Declarations
    h : function=_count, units=None, arg_units=None

1 Declarations: h
        """.strip(),
        )

        if not pint_available:
            return
        m.i = ExternalFunction(
            function=_sum, units=units.kg, arg_units=[units.m, units.s]
        )
        out = StringIO()
        m.pprint(ostream=out)
        self.assertEqual(
            out.getvalue().strip(),
            """
2 ExternalFunction Declarations
    h : function=_count, units=None, arg_units=None
    i : function=_sum, units=kg, arg_units=['m', 's']

2 Declarations: h i
        """.strip(),
        )

    def test_pprint(self):
        m = ConcreteModel()
        m.h = ExternalFunction(_g)
        out = StringIO()
        m.pprint()
        m.pprint(ostream=out)
        self.assertEqual(
            out.getvalue().strip(),
            """
1 ExternalFunction Declarations
    h : function=_g, units=None, arg_units=None

1 Declarations: h
        """.strip(),
        )

        if not pint_available:
            return
        m.i = ExternalFunction(
            function=_h, units=units.kg, arg_units=[units.m, units.s]
        )
        out = StringIO()
        m.pprint(ostream=out)
        self.assertEqual(
            out.getvalue().strip(),
            """
2 ExternalFunction Declarations
    h : function=_g, units=None, arg_units=None
    i : function=_h, units=kg, arg_units=['m', 's']

2 Declarations: h i
        """.strip(),
        )


class TestAMPLExternalFunction(unittest.TestCase):
    def assertListsAlmostEqual(self, first, second, places=7, msg=None):
        self.assertEqual(len(first), len(second))
        msg = "lists %s and %s differ at item " % (first, second)
        for i, a in enumerate(first):
            self.assertAlmostEqual(a, second[i], places, msg + str(i))

    def test_getname(self):
        m = ConcreteModel()
        m.f = ExternalFunction(library="junk.so", function="junk")
        self.assertIsInstance(m.f, AMPLExternalFunction)
        self.assertEqual(m.f.name, "f")
        self.assertEqual(m.f.local_name, "f")
        self.assertEqual(m.f.getname(), "f")
        self.assertEqual(m.f.getname(True), "f")

        M = ConcreteModel()
        M.m = m
        self.assertEqual(M.m.f.name, "m.f")
        self.assertEqual(M.m.f.local_name, "f")
        self.assertEqual(M.m.f.getname(), "f")
        self.assertEqual(M.m.f.getname(True), "m.f")

    @unittest.skipIf(
        sys.platform.lower().startswith('win'),
        "Cannot (easily) unload a DLL in Windows, so "
        "cannot clean up the 'temporary' DLL",
    )
    def test_load_local_asl_library(self):
        DLL = find_GSL()
        if not DLL:
            self.skipTest("Could not find the amplgsl.dll library")

        LIB = 'test_pyomo_external_gsl.dll'

        model = ConcreteModel()
        model.gamma = ExternalFunction(library=LIB, function="gsl_sf_gamma")
        model.x = Var(initialize=3, bounds=(1e-5, None))
        model.o = Objective(expr=model.gamma(model.x))

        with TempfileManager.new_context() as tempfile:
            dname = tempfile.mkdtemp()
            shutil.copyfile(DLL, os.path.join(dname, LIB))
            # Without changing directories, the load should fail
            with self.assertRaises(OSError):
                value(model.o)
            # Changing directories should pick up the library
            try:
                orig_dir = os.getcwd()
                os.chdir(dname)
                self.assertAlmostEqual(value(model.o), 2.0, 7)
            finally:
                os.chdir(orig_dir)

    def test_unknown_library(self):
        m = ConcreteModel()
        with LoggingIntercept() as LOG:
            m.ef = ExternalFunction(
                library='unknown_pyomo_external_testing_function', function='f'
            )
        self.assertEqual(
            LOG.getvalue(),
            'Defining AMPL external function, but cannot locate '
            'specified library "unknown_pyomo_external_testing_function"\n',
        )

    def test_eval_gsl_function(self):
        DLL = find_GSL()
        if not DLL:
            self.skipTest("Could not find the amplgsl.dll library")
        model = ConcreteModel()
        model.gamma = ExternalFunction(library=DLL, function="gsl_sf_gamma")
        model.bessel = ExternalFunction(library=DLL, function="gsl_sf_bessel_Jnu")
        model.x = Var(initialize=3, bounds=(1e-5, None))
        model.o = Objective(expr=model.gamma(model.x))
        self.assertAlmostEqual(value(model.o), 2.0, 7)

        f = model.bessel.evaluate((0.5, 2.0))
        self.assertAlmostEqual(f, 0.5130161365618272, 7)

    def test_eval_gsl_error(self):
        DLL = find_GSL()
        if not DLL:
            self.skipTest("Could not find the amplgsl.dll library")
        model = ConcreteModel()
        model.bogus = ExternalFunction(library=DLL, function="bogus_function")
        with self.assertRaisesRegex(
            RuntimeError,
            "Error: external function 'bogus_function' was "
            "not registered within external library(?s:.*)gsl_sf_gamma",
        ):
            f = model.bogus.evaluate((1,))

    def test_eval_fgh_gsl_function(self):
        DLL = find_GSL()
        if not DLL:
            self.skipTest("Could not find the amplgsl.dll library")
        model = ConcreteModel()
        model.gamma = ExternalFunction(library=DLL, function="gsl_sf_gamma")
        model.beta = ExternalFunction(library=DLL, function="gsl_sf_beta")
        model.bessel = ExternalFunction(library=DLL, function="gsl_sf_bessel_Jnu")

        f, g, h = model.gamma.evaluate_fgh((2.0,))
        self.assertAlmostEqual(f, 1.0, 7)
        self.assertListsAlmostEqual(g, [0.422784335098467], 7)
        self.assertListsAlmostEqual(h, [0.8236806608528794], 7)

        f, g, h = model.beta.evaluate_fgh((2.5, 2.0), fixed=[1, 1])
        self.assertAlmostEqual(f, 0.11428571428571432, 7)
        self.assertListsAlmostEqual(g, [0.0, 0.0], 7)
        self.assertListsAlmostEqual(h, [0.0, 0.0, 0.0], 7)

        f, g, h = model.beta.evaluate_fgh((2.5, 2.0), fixed=[0, 1])
        self.assertAlmostEqual(f, 0.11428571428571432, 7)
        self.assertListsAlmostEqual(g, [-0.07836734693877555, 0.0], 7)
        self.assertListsAlmostEqual(h, [0.08135276967930034, 0.0, 0.0], 7)

        f, g, h = model.beta.evaluate_fgh((2.5, 2.0))
        self.assertAlmostEqual(f, 0.11428571428571432, 7)
        self.assertListsAlmostEqual(g, [-0.07836734693877555, -0.11040989614412142], 7)
        self.assertListsAlmostEqual(
            h, [0.08135276967930034, 0.0472839170086535, 0.15194654464270113], 7
        )

        f, g, h = model.beta.evaluate_fgh((2.5, 2.0), fgh=1)
        self.assertAlmostEqual(f, 0.11428571428571432, 7)
        self.assertListsAlmostEqual(g, [-0.07836734693877555, -0.11040989614412142], 7)
        self.assertIsNone(h)

        f, g, h = model.beta.evaluate_fgh((2.5, 2.0), fgh=0)
        self.assertAlmostEqual(f, 0.11428571428571432, 7)
        self.assertIsNone(g)
        self.assertIsNone(h)

        f, g, h = model.bessel.evaluate_fgh((2.5, 2.0), fixed=[1, 0])
        self.assertAlmostEqual(f, 0.223924531469, 7)
        self.assertListsAlmostEqual(g, [0.0, 0.21138811435101745], 7)
        self.assertListsAlmostEqual(h, [0.0, 0.0, 0.02026349177575621], 7)

        # Note: Not all AMPL-GSL functions honor the fixed flag
        # (notably, gamma and bessel do not as of 12/2021).  We will
        # test that our interface corrects that

        f, g, h = model.gamma.evaluate_fgh((2.0,), fixed=[1])
        self.assertAlmostEqual(f, 1.0, 7)
        self.assertListsAlmostEqual(g, [0.0], 7)
        self.assertListsAlmostEqual(h, [0.0], 7)

        f, g, h = model.bessel.evaluate_fgh((2.5, 2.0), fixed=[1, 1])
        self.assertAlmostEqual(f, 0.223924531469, 7)
        self.assertListsAlmostEqual(g, [0.0, 0.0], 7)
        self.assertListsAlmostEqual(h, [0.0, 0.0, 0.0], 7)

    @unittest.skipIf(
        not check_available_solvers('ipopt'), "The 'ipopt' solver is not available"
    )
    def test_solve_gsl_function(self):
        DLL = find_GSL()
        if not DLL:
            self.skipTest("Could not find the amplgsl.dll library")
        model = ConcreteModel()
        model.z_func = ExternalFunction(library=DLL, function="gsl_sf_gamma")
        model.x = Var(initialize=3, bounds=(1e-5, None))
        model.o = Objective(expr=model.z_func(model.x))
        opt = SolverFactory('ipopt')
        res = opt.solve(model, tee=True)
        self.assertAlmostEqual(value(model.o), 0.885603194411, 7)

    @unittest.skipIf(
        not check_available_solvers('ipopt'), "The 'ipopt' solver is not available"
    )
    def test_solve_gsl_function_const_arg(self):
        DLL = find_GSL()
        if not DLL:
            self.skipTest("Could not find the amplgsl.dll library")
        model = ConcreteModel()
        model.z_func = ExternalFunction(library=DLL, function="gsl_sf_beta")
        model.x = Var(initialize=1, bounds=(0.1, None))
        model.o = Objective(expr=-model.z_func(1, model.x))
        opt = SolverFactory('ipopt')
        res = opt.solve(model, tee=True)
        self.assertAlmostEqual(value(model.x), 0.1, 5)

    @unittest.skipIf(
        not check_available_solvers('ipopt'), "The 'ipopt' solver is not available"
    )
    def test_clone_gsl_function(self):
        DLL = find_GSL()
        if not DLL:
            self.skipTest("Could not find the amplgsl.dll library")
        m = ConcreteModel()
        m.z_func = ExternalFunction(library=DLL, function="gsl_sf_gamma")
        self.assertIsInstance(m.z_func, AMPLExternalFunction)
        m.x = Var(initialize=3, bounds=(1e-5, None))
        m.o = Objective(expr=m.z_func(m.x))

        opt = SolverFactory('ipopt')

        # Test a simple clone...
        model2 = m.clone()
        res = opt.solve(model2, tee=True)
        self.assertAlmostEqual(value(model2.o), 0.885603194411, 7)

        # Trigger the library to be loaded.  This tests that the CDLL
        # objects that are created when the SO/DLL are loaded do not
        # interfere with cloning the model.
        self.assertAlmostEqual(value(m.o), 2)
        model3 = m.clone()
        res = opt.solve(model3, tee=True)
        self.assertAlmostEqual(value(model3.o), 0.885603194411, 7)

    def test_pprint(self):
        m = ConcreteModel()
        m.f = ExternalFunction(library="junk.so", function="junk")

        out = StringIO()
        m.pprint(ostream=out)
        self.assertEqual(
            out.getvalue().strip(),
            """
1 ExternalFunction Declarations
    f : function=junk, library=junk.so, units=None, arg_units=None

1 Declarations: f
        """.strip(),
        )

        if not pint_available:
            return
        m.g = ExternalFunction(
            library="junk.so",
            function="junk",
            units=units.kg,
            arg_units=[units.m, units.s],
        )
        out = StringIO()
        m.pprint(ostream=out)
        self.assertEqual(
            out.getvalue().strip(),
            """
2 ExternalFunction Declarations
    f : function=junk, library=junk.so, units=None, arg_units=None
    g : function=junk, library=junk.so, units=kg, arg_units=['m', 's']

2 Declarations: f g
        """.strip(),
        )


if __name__ == "__main__":
    unittest.main()
