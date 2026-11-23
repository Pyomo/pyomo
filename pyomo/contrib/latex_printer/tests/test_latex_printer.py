# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

import io
import subprocess
import sys
from textwrap import dedent

import pyomo.common.unittest as unittest
import pyomo.core.tests.examples.pmedian_concrete as pmedian_concrete
import pyomo.environ as pyo

from pyomo.contrib.latex_printer import latex_printer
from pyomo.common.dependencies import numpy as np, numpy_available
from pyomo.common.tempfiles import TempfileManager
from pyomo.common.collections.component_map import ComponentMap
from pyomo.environ import (
    Reals,
    PositiveReals,
    NonPositiveReals,
    NegativeReals,
    NonNegativeReals,
    Integers,
    PositiveIntegers,
    NonPositiveIntegers,
    NegativeIntegers,
    NonNegativeIntegers,
    Boolean,
    Binary,
    Any,
    # AnyWithNone,
    EmptySet,
    UnitInterval,
    PercentFraction,
    # RealInterval,
    # IntegerInterval,
)


def generate_model():
    import pyomo.environ as pyo
    from pyomo.core.expr import Expr_if
    from pyomo.core.base import ExternalFunction

    m = pyo.ConcreteModel(name='basicFormulation')
    m.x = pyo.Var()
    m.y = pyo.Var()
    m.z = pyo.Var()
    m.objective_1 = pyo.Objective(expr=m.x + m.y + m.z)
    m.constraint_1 = pyo.Constraint(
        expr=m.x**2 + m.y**-2.0 - m.x * m.y * m.z + 1 == 2.0
    )
    m.constraint_2 = pyo.Constraint(expr=abs(m.x / m.z**-2) * (m.x + m.y) <= 2.0)
    m.constraint_3 = pyo.Constraint(expr=pyo.sqrt(m.x / m.z**-2) <= 2.0)
    m.constraint_4 = pyo.Constraint(expr=(1, m.x, 2))
    m.constraint_5 = pyo.Constraint(expr=Expr_if(m.x <= 1.0, m.z, m.y) <= 1.0)

    def blackbox(a, b):
        return sin(a - b)

    m.bb = ExternalFunction(blackbox)
    m.constraint_6 = pyo.Constraint(expr=m.x + m.bb(m.x, m.y) == 2)

    m.I = pyo.Set(initialize=[1, 2, 3, 4, 5])
    m.J = pyo.Set(initialize=[1, 2, 3])
    m.K = pyo.Set(initialize=[1, 3, 5])
    m.u = pyo.Var(m.I * m.I)
    m.v = pyo.Var(m.I)
    m.w = pyo.Var(m.J)
    m.p = pyo.Var(m.K)

    m.express = pyo.Expression(expr=m.x**2 + m.y**2)

    def ruleMaker(m, j):
        return (m.x + m.y) * sum(m.v[i] + m.u[i, j] ** 2 for i in m.I) <= 0

    m.constraint_7 = pyo.Constraint(m.I, rule=ruleMaker)

    def ruleMaker(m):
        return sum(m.p[k] for k in m.K) == 1

    m.constraint_8 = pyo.Constraint(rule=ruleMaker)

    def ruleMaker(m):
        return (m.x + m.y) * sum(m.w[j] for j in m.J)

    m.objective_2 = pyo.Objective(rule=ruleMaker)

    m.objective_3 = pyo.Objective(expr=m.x + m.y + m.z, sense=-1)

    return m


def generate_simple_model():
    import pyomo.environ as pyo

    m = pyo.ConcreteModel(name='basicFormulation')
    m.x = pyo.Var()
    m.y = pyo.Var()
    m.objective_1 = pyo.Objective(expr=m.x + m.y)
    m.constraint_1 = pyo.Constraint(expr=m.x**2 + m.y**2.0 <= 1.0)
    m.constraint_2 = pyo.Constraint(expr=m.x >= 0.0)

    m.I = pyo.Set(initialize=[1, 2, 3, 4, 5])
    m.J = pyo.Set(initialize=[1, 2, 3])
    m.K = pyo.Set(initialize=[1, 3, 5])
    m.u = pyo.Var(m.I * m.I)
    m.v = pyo.Var(m.I)
    m.w = pyo.Var(m.J)
    m.p = pyo.Var(m.K)

    def ruleMaker(m, j):
        return (m.x + m.y) * sum(m.v[i] + m.u[i, j] ** 2 for i in m.I) <= 0

    m.constraint_7 = pyo.Constraint(m.I, rule=ruleMaker)

    def ruleMaker(m):
        return sum(m.p[k] for k in m.K) == 1

    m.constraint_8 = pyo.Constraint(rule=ruleMaker)

    return m


def generate_simple_model_2():
    import pyomo.environ as pyo

    m = pyo.ConcreteModel(name='basicFormulation')
    m.x_dot = pyo.Var()
    m.x_bar = pyo.Var()
    m.x_star = pyo.Var()
    m.x_hat = pyo.Var()
    m.x_hat_1 = pyo.Var()
    m.y_sub1_sub2_sub3 = pyo.Var()
    m.objective_1 = pyo.Objective(expr=m.y_sub1_sub2_sub3)
    m.constraint_1 = pyo.Constraint(
        expr=(m.x_dot + m.x_bar + m.x_star + m.x_hat + m.x_hat_1) ** 2
        <= m.y_sub1_sub2_sub3
    )
    m.constraint_2 = pyo.Constraint(
        expr=(m.x_dot + m.x_bar) ** -(m.x_star + m.x_hat) <= m.y_sub1_sub2_sub3
    )
    m.constraint_3 = pyo.Constraint(
        expr=-(m.x_dot + m.x_bar) + -(m.x_star + m.x_hat) <= m.y_sub1_sub2_sub3
    )

    return m


class TestLatexPrinter(unittest.TestCase):
    def test_latexPrinter_simpleDocTests(self):
        # Ex 1 -----------------------
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var()
        m.y = pyo.Var()
        pstr = latex_printer(m.x + m.y)
        bstr = dedent(r"""
        \begin{equation} 
             x + y 
        \end{equation} 
        """)
        self.assertEqual('\n' + pstr + '\n', bstr)

        # Ex 2 -----------------------
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.expression_1 = pyo.Expression(expr=m.x**2 + m.y**2)
        pstr = latex_printer(m.expression_1)
        bstr = dedent(r"""
        \begin{equation} 
             x^{2} + y^{2} 
        \end{equation} 
        """)
        self.assertEqual('\n' + pstr + '\n', bstr)

        # Ex 3 -----------------------
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.constraint_1 = pyo.Constraint(expr=m.x**2 + m.y**2 <= 1.0)
        pstr = latex_printer(m.constraint_1)
        bstr = dedent(r"""
        \begin{equation} 
             x^{2} + y^{2} \leq 1 
        \end{equation} 
        """)
        self.assertEqual('\n' + pstr + '\n', bstr)

        # Ex 4 -----------------------
        m = pyo.ConcreteModel(name='basicFormulation')
        m.I = pyo.Set(initialize=[1, 2, 3, 4, 5])
        m.v = pyo.Var(m.I)

        def ruleMaker(m):
            return sum(m.v[i] for i in m.I) <= 0

        m.constraint = pyo.Constraint(rule=ruleMaker)
        pstr = latex_printer(m.constraint)
        bstr = dedent(r"""
        \begin{equation} 
             \sum_{ i \in I  } v_{i} \leq 0 
        \end{equation} 
        """)
        self.assertEqual('\n' + pstr + '\n', bstr)

        # Ex 5 -----------------------
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.z = pyo.Var()
        m.c = pyo.Param(initialize=1.0, mutable=True)
        m.objective = pyo.Objective(expr=m.x + m.y + m.z)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 + m.y**2.0 - m.z**2.0 <= m.c)
        pstr = latex_printer(m)
        bstr = dedent(r"""
        \begin{align} 
            & \min 
            & & x + y + z & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} + y^{2} - z^{2} \leq c & \label{con:basicFormulation_constraint_1} 
        \end{align} 
        """)
        self.assertEqual('\n' + pstr + '\n', bstr)

        # Ex 6 -----------------------
        m = pyo.ConcreteModel(name='basicFormulation')
        m.I = pyo.Set(initialize=[1, 2, 3, 4, 5])
        m.v = pyo.Var(m.I)

        def ruleMaker(m):
            return sum(m.v[i] for i in m.I) <= 0

        m.constraint = pyo.Constraint(rule=ruleMaker)
        lcm = ComponentMap()
        lcm[m.v] = 'x'
        lcm[m.I] = ['\\mathcal{A}', ['j', 'k']]
        pstr = latex_printer(m.constraint, latex_component_map=lcm)
        bstr = dedent(r"""
        \begin{equation} 
             \sum_{ j \in \mathcal{A}  } x_{j} \leq 0 
        \end{equation} 
        """)
        self.assertEqual('\n' + pstr + '\n', bstr)

    def test_latexPrinter_objective(self):
        m = generate_model()
        pstr = latex_printer(m.objective_1)
        bstr = dedent(r"""
        \begin{equation} 
            & \min 
            & & x + y + z 
        \end{equation} 
        """)
        self.assertEqual('\n' + pstr + '\n', bstr)

        pstr = latex_printer(m.objective_3)
        bstr = dedent(r"""
        \begin{equation} 
            & \max 
            & & x + y + z 
        \end{equation} 
        """)
        self.assertEqual('\n' + pstr + '\n', bstr)

    def test_latexPrinter_constraint(self):
        m = generate_model()
        pstr = latex_printer(m.constraint_1)

        bstr = dedent(r"""
        \begin{equation} 
             x^{2} + y^{-2} - x y z + 1 = 2 
        \end{equation} 
        """)

        self.assertEqual('\n' + pstr + '\n', bstr)

    def test_latexPrinter_expression(self):
        m = generate_model()

        m.express = pyo.Expression(expr=m.x + m.y)

        pstr = latex_printer(m.express)

        bstr = dedent(r"""
        \begin{equation} 
             x + y 
        \end{equation} 
        """)

        self.assertEqual('\n' + pstr + '\n', bstr)

    @unittest.skipUnless(numpy_available, "numpy is not available")
    def test_latexPrinter_numpyConstants(self):
        # Regression test for #3325: NumPy scalar constants (e.g.,
        # from models built using pandas data) are valid constant
        # leaf nodes in Pyomo expressions and should be printed as
        # numbers instead of raising a DeveloperError.
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var()

        # numpy integer constant as a standalone term in a sum
        m.con_int = pyo.Constraint(expr=m.x + np.int64(7) >= 1)
        pstr = latex_printer(m.con_int)
        bstr = dedent(r"""
        \begin{equation} 
             1 \leq x + 7 
        \end{equation} 
        """)
        self.assertEqual('\n' + pstr + '\n', bstr)

        # numpy float constant as a standalone term in a sum
        m.con_float = pyo.Constraint(expr=m.y + np.float32(1.5) <= 10)
        pstr = latex_printer(m.con_float)
        bstr = dedent(r"""
        \begin{equation} 
             y + 1.5 \leq 10 
        \end{equation} 
        """)
        self.assertEqual('\n' + pstr + '\n', bstr)

        # numpy integer constant as an exponent
        m.con_pow = pyo.Constraint(expr=m.x ** np.int64(2) <= 4)
        pstr = latex_printer(m.con_pow)
        bstr = dedent(r"""
        \begin{equation} 
             x^{2} \leq 4 
        \end{equation} 
        """)
        self.assertEqual('\n' + pstr + '\n', bstr)

        # numpy integer constant as a monomial coefficient: on the
        # left of a product (np.int64(3) * x) Pyomo's expression
        # construction converts the coefficient to a native Python
        # int, but on the right it reaches the printer as a NumPy
        # type (this is how the model in #3325 was built)
        m.con_monomial = pyo.Constraint(expr=m.x * np.int64(3) <= 10)
        pstr = latex_printer(m.con_monomial)
        bstr = dedent(r"""
        \begin{equation} 
             3 x \leq 10 
        \end{equation} 
        """)
        self.assertEqual('\n' + pstr + '\n', bstr)

        # the remaining NumPy scalar numeric types should also print
        for numeric_type in (
            np.int8,
            np.int16,
            np.int32,
            np.uint8,
            np.uint64,
            np.float16,
            np.float64,
        ):
            m.con = pyo.Constraint(expr=m.x + numeric_type(3) >= 0)
            self.assertIn('x + 3', latex_printer(m.con))
            del m.con

        # extended-precision floats (np.longdouble, aliased as
        # np.float128 on some platforms) are registered as well
        m.con = pyo.Constraint(expr=m.x + np.longdouble(1.5) >= 0)
        self.assertIn('x + 1.5', latex_printer(m.con))
        del m.con

    @unittest.skipUnless(numpy_available, "numpy is not available")
    def test_latexPrinter_lateNumpyRegistration(self):
        # When printing a block, the printer builds its expression
        # visitor before templatizing the constraints, and a
        # constraint rule is only executed (importing NumPy) by that
        # templatization -- so NumPy types can enter
        # native_numeric_types after the visitor was constructed.
        # They must still be printable.  This needs a fresh
        # interpreter: once NumPy has been imported, its types are
        # registered before the visitor is built and the timing
        # cannot be observed.
        cmd = """
import pyomo.environ as pyo
from pyomo.common.numeric_types import native_numeric_types
from pyomo.contrib.latex_printer import latex_printer

# Precondition: no NumPy types are registered yet in this interpreter
assert not any(t.__module__ == 'numpy' for t in native_numeric_types)

m = pyo.ConcreteModel()
m.S = pyo.Set(dimen=1)
m.x = pyo.Var()

def _rule(mdl, i):
    import numpy as np
    return mdl.x + np.float64(1.5) <= 10

m.c = pyo.Constraint(m.S, rule=_rule)
print(latex_printer(m))
"""
        rc = subprocess.run(
            [sys.executable, '-c', cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.assertEqual(rc.returncode, 0, msg=rc.stderr)
        self.assertIn('x + 1.5', rc.stdout)

    def test_latexPrinter_customRegisteredNumericType(self):
        # Types registered by users with RegisterNumericType() (with
        # no NumPy involved) must also be printable.  Run in a fresh
        # interpreter so the registration, which is process-global,
        # cannot leak into other tests.
        cmd = """
import pyomo.environ as pyo
from pyomo.common.numeric_types import RegisterNumericType
from pyomo.contrib.latex_printer import latex_printer

class CustomScalar:
    def __init__(self, value):
        self._value = float(value)
    def __float__(self):
        return self._value
    def __str__(self):
        return str(self._value)

RegisterNumericType(CustomScalar)

m = pyo.ConcreteModel()
m.x = pyo.Var()
m.con = pyo.Constraint(expr=m.x + CustomScalar(7) >= 1)
print(latex_printer(m.con))
"""
        rc = subprocess.run(
            [sys.executable, '-c', cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.assertEqual(rc.returncode, 0, msg=rc.stderr)
        self.assertIn('x + 7.0', rc.stdout)

    def test_latexPrinter_simpleExpression(self):
        m = generate_model()

        pstr = latex_printer(m.x - m.y)
        bstr = dedent(r"""
        \begin{equation} 
             x - y 
        \end{equation} 
        """)
        self.assertEqual('\n' + pstr + '\n', bstr)

        pstr = latex_printer(m.x - 2 * m.y)
        bstr = dedent(r"""
        \begin{equation} 
             x - 2 y 
        \end{equation} 
        """)
        self.assertEqual('\n' + pstr + '\n', bstr)

    def test_latexPrinter_unary(self):
        m = generate_model()

        pstr = latex_printer(m.constraint_2)
        bstr = dedent(r"""
        \begin{equation} 
              \left| \frac{x}{z^{-2}} \right|   \left( x + y \right)  \leq 2 
        \end{equation} 
        """)
        self.assertEqual('\n' + pstr + '\n', bstr)

        pstr = latex_printer(pyo.Constraint(expr=pyo.sin(m.x) == 1))
        bstr = dedent(r"""
        \begin{equation} 
             \sin \left( x \right)  = 1 
        \end{equation} 
        """)
        self.assertEqual('\n' + pstr + '\n', bstr)

        pstr = latex_printer(pyo.Constraint(expr=pyo.log10(m.x) == 1))
        bstr = dedent(r"""
        \begin{equation} 
             \log_{10} \left( x \right)  = 1 
        \end{equation} 
        """)
        self.assertEqual('\n' + pstr + '\n', bstr)

        pstr = latex_printer(pyo.Constraint(expr=pyo.sqrt(m.x) == 1))
        bstr = dedent(r"""
        \begin{equation} 
             \sqrt { x } = 1 
        \end{equation} 
        """)
        self.assertEqual('\n' + pstr + '\n', bstr)

    def test_latexPrinter_rangedConstraint(self):
        m = generate_model()

        pstr = latex_printer(m.constraint_4)
        bstr = dedent(r"""
        \begin{equation} 
             1 \leq x \leq 2 
        \end{equation} 
        """)
        self.assertEqual('\n' + pstr + '\n', bstr)

    def test_latexPrinter_exprIf(self):
        m = generate_model()

        pstr = latex_printer(m.constraint_5)
        bstr = dedent(r"""
        \begin{equation} 
             f_{\text{exprIf}}(x \leq 1,z,y) \leq 1 
        \end{equation} 
        """)
        self.assertEqual('\n' + pstr + '\n', bstr)

    def test_latexPrinter_blackBox(self):
        m = generate_model()

        pstr = latex_printer(m.constraint_6)
        bstr = dedent(r"""
        \begin{equation} 
             x + f\_1(x,y) = 2 
        \end{equation} 
        """)
        self.assertEqual('\n' + pstr + '\n', bstr)

    def test_latexPrinter_iteratedConstraints(self):
        m = generate_model()

        pstr = latex_printer(m.constraint_7)
        bstr = dedent(r"""
        \begin{equation} 
              \left( x + y \right)  \sum_{ i \in I  } v_{i} + u_{i,j}^{2} \leq 0  \qquad \forall j \in I 
        \end{equation} 
        """)
        self.assertEqual('\n' + pstr + '\n', bstr)

        pstr = latex_printer(m.constraint_8)
        bstr = dedent(r"""
        \begin{equation} 
             \sum_{ i \in K  } p_{i} = 1 
        \end{equation} 
        """)
        self.assertEqual('\n' + pstr + '\n', bstr)

    def test_latexPrinter_fileWriter(self):
        m = generate_simple_model()

        with TempfileManager.new_context() as tempfile:
            fd, fname = tempfile.mkstemp()
            pstr = latex_printer(m, ostream=fname)

            f = open(fname)
            bstr = f.read()
            f.close()

            bstr_split = bstr.split('\n')
            bstr_stripped = bstr_split[8:-2]
            bstr = '\n'.join(bstr_stripped) + '\n'

            self.assertEqual(pstr + '\n', bstr)

    def test_latexPrinter_inputError(self):
        self.assertRaises(
            ValueError, latex_printer, **{'pyomo_component': 'errorString'}
        )

    def test_latexPrinter_fileWriter(self):
        m = generate_simple_model()

        with TempfileManager.new_context() as tempfile:
            fd, fname = tempfile.mkstemp()
            pstr = latex_printer(m, ostream=fname)

            f = open(fname)
            bstr = f.read()
            f.close()

            bstr_split = bstr.split('\n')
            bstr_stripped = bstr_split[8:-2]
            bstr = '\n'.join(bstr_stripped) + '\n'

            self.assertEqual(pstr + '\n', bstr)

        self.assertRaises(
            ValueError, latex_printer, **{'pyomo_component': m, 'ostream': 2.0}
        )

    def test_latexPrinter_overwriteError(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.I = pyo.Set(initialize=[1, 2, 3, 4, 5])
        m.v = pyo.Var(m.I)

        def ruleMaker(m):
            return sum(m.v[i] for i in m.I) <= 0

        m.constraint = pyo.Constraint(rule=ruleMaker)
        lcm = ComponentMap()
        lcm[m.v] = 'x'
        lcm[m.I] = ['\\mathcal{A}', ['j', 'k']]
        lcm['err'] = 1.0

        self.assertRaises(
            ValueError,
            latex_printer,
            **{'pyomo_component': m.constraint, 'latex_component_map': lcm},
        )

    def test_latexPrinter_indexedParam(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.I = pyo.Set(initialize=[1, 2, 3, 4, 5])
        m.x = pyo.Var(m.I * m.I)
        m.c = pyo.Param(m.I * m.I, initialize=1.0, mutable=True)

        def ruleMaker_1(m):
            return sum(m.c[i, j] * m.x[i, j] for i in m.I for j in m.I)

        def ruleMaker_2(m):
            return sum(m.x[i, j] ** 2 for i in m.I for j in m.I) <= 1

        m.objective = pyo.Objective(rule=ruleMaker_1)
        m.constraint_1 = pyo.Constraint(rule=ruleMaker_2)
        pstr = latex_printer(m)

        bstr = dedent(r"""
        \begin{align} 
            & \min 
            & & \sum_{ i \in I  } \sum_{ j \in I  } c_{i,j} x_{i,j} & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & \sum_{ i \in I  } \sum_{ j \in I  } x_{i,j}^{2} \leq 1 & \label{con:basicFormulation_constraint_1} 
        \end{align} 
        """)

        self.assertEqual('\n' + pstr + '\n', bstr)

        lcm = ComponentMap()
        lcm[m.I] = ['\\mathcal{A}', ['j']]
        self.assertRaises(
            ValueError,
            latex_printer,
            **{'pyomo_component': m, 'latex_component_map': lcm},
        )

    def test_latexPrinter_involvedModel(self):
        m = generate_model()
        pstr = latex_printer(m)
        print(pstr)

        bstr = dedent(r"""
        \begin{align} 
            & \min 
            & & x + y + z & \label{obj:basicFormulation_objective_1} \\ 
            & \min 
            & &  \left( x + y \right)  \sum_{ i \in J  } w_{i} & \label{obj:basicFormulation_objective_2} \\ 
            & \max 
            & & x + y + z & \label{obj:basicFormulation_objective_3} \\ 
            & \text{s.t.} 
            & & x^{2} + y^{-2} - x y z + 1 = 2 & \label{con:basicFormulation_constraint_1} \\ 
            &&&  \left| \frac{x}{z^{-2}} \right|   \left( x + y \right)  \leq 2 & \label{con:basicFormulation_constraint_2} \\ 
            &&& \sqrt { \frac{x}{z^{-2}} } \leq 2 & \label{con:basicFormulation_constraint_3} \\ 
            &&& 1 \leq x \leq 2 & \label{con:basicFormulation_constraint_4} \\ 
            &&& f_{\text{exprIf}}(x \leq 1,z,y) \leq 1 & \label{con:basicFormulation_constraint_5} \\ 
            &&& x + f\_1(x,y) = 2 & \label{con:basicFormulation_constraint_6} \\ 
            &&&  \left( x + y \right)  \sum_{ i \in I  } v_{i} + u_{i,j}^{2} \leq 0 &  \qquad \forall j \in I \label{con:basicFormulation_constraint_7} \\ 
            &&& \sum_{ i \in K  } p_{i} = 1 & \label{con:basicFormulation_constraint_8} 
        \end{align} 
        """)

        self.assertEqual('\n' + pstr + '\n', bstr)

    def test_latexPrinter_continuousSet(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.I = pyo.Set(initialize=[1, 2, 3, 4, 5])
        m.v = pyo.Var(m.I)

        def ruleMaker(m):
            return sum(m.v[i] for i in m.I) <= 0

        m.constraint = pyo.Constraint(rule=ruleMaker)
        pstr = latex_printer(m.constraint, explicit_set_summation=True)

        bstr = dedent(r"""
        \begin{equation} 
             \sum_{ i = 1 }^{5} v_{i} \leq 0 
        \end{equation} 
        """)
        self.assertEqual('\n' + pstr + '\n', bstr)

    def test_latexPrinter_notContinuousSet(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.I = pyo.Set(initialize=[1, 3, 4, 5])
        m.v = pyo.Var(m.I)

        def ruleMaker(m):
            return sum(m.v[i] for i in m.I) <= 0

        m.constraint = pyo.Constraint(rule=ruleMaker)
        pstr = latex_printer(m.constraint, explicit_set_summation=True)

        bstr = dedent(r"""
        \begin{equation} 
             \sum_{ i \in I  } v_{i} \leq 0 
        \end{equation} 
        """)
        self.assertEqual('\n' + pstr + '\n', bstr)

    def test_latexPrinter_autoIndex(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.I = pyo.Set(initialize=[1, 2, 3, 4, 5])
        m.v = pyo.Var(m.I)

        def ruleMaker(m):
            return sum(m.v[i] for i in m.I) <= 0

        m.constraint = pyo.Constraint(rule=ruleMaker)
        lcm = ComponentMap()
        lcm[m.v] = 'x'
        lcm[m.I] = ['\\mathcal{A}', []]
        pstr = latex_printer(m.constraint, latex_component_map=lcm)
        bstr = dedent(r"""
        \begin{equation} 
             \sum_{ i \in \mathcal{A}  } x_{i} \leq 0 
        \end{equation} 
        """)
        self.assertEqual('\n' + pstr + '\n', bstr)

    def test_latexPrinter_equationEnvironment(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.z = pyo.Var()
        m.c = pyo.Param(initialize=1.0, mutable=True)
        m.objective = pyo.Objective(expr=m.x + m.y + m.z)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 + m.y**2.0 - m.z**2.0 <= m.c)
        pstr = latex_printer(m, use_equation_environment=True)

        bstr = dedent(r"""
        \begin{equation} 
            \begin{aligned} 
                & \min 
                & & x + y + z \\ 
                & \text{s.t.} 
                & & x^{2} + y^{2} - z^{2} \leq c 
            \end{aligned} 
            \label{basicFormulation} 
        \end{equation} 
        """)
        self.assertEqual('\n' + pstr + '\n', bstr)

    def test_latexPrinter_manyVariablesWithDomains(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=Integers, bounds=(-10, 10))
        m.y = pyo.Var(domain=Binary, bounds=(-10, 10))
        m.z = pyo.Var(domain=PositiveReals, bounds=(-10, 10))
        m.u = pyo.Var(domain=NonNegativeIntegers, bounds=(-10, 10))
        m.v = pyo.Var(domain=NegativeReals, bounds=(-10, 10))
        m.w = pyo.Var(domain=PercentFraction, bounds=(-10, 10))
        m.objective = pyo.Objective(expr=m.x + m.y + m.z + m.u + m.v + m.w)
        pstr = latex_printer(m)

        bstr = dedent(r"""
        \begin{align} 
            & \min 
            & & x + y + z + u + v + w & \label{obj:basicFormulation_objective} \\ 
            & \text{w.b.} 
            & & -10 \leq x \leq 10 & \qquad \in \mathds{Z} \label{con:basicFormulation_x_bound} \\ 
            &&& y & \qquad \in \left\{ 0 , 1 \right \} \label{con:basicFormulation_y_bound} \\ 
            &&&  0 < z \leq 10 & \qquad \in \mathds{R}_{> 0} \label{con:basicFormulation_z_bound} \\ 
            &&&  0 \leq u \leq 10 & \qquad \in \mathds{Z}_{\geq 0} \label{con:basicFormulation_u_bound} \\ 
            &&& -10 \leq v < 0  & \qquad \in \mathds{R}_{< 0} \label{con:basicFormulation_v_bound} \\ 
            &&&  0 \leq w \leq 1  & \qquad \in \mathds{R} \label{con:basicFormulation_w_bound} 
        \end{align} 
        """)

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_manyVariablesWithDomains_eqn(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=Integers, bounds=(-10, 10))
        m.y = pyo.Var(domain=Binary, bounds=(-10, 10))
        m.z = pyo.Var(domain=PositiveReals, bounds=(-10, 10))
        m.u = pyo.Var(domain=NonNegativeIntegers, bounds=(-10, 10))
        m.v = pyo.Var(domain=NegativeReals, bounds=(-10, 10))
        m.w = pyo.Var(domain=PercentFraction, bounds=(-10, 10))
        m.objective = pyo.Objective(expr=m.x + m.y + m.z + m.u + m.v + m.w)
        pstr = latex_printer(m, use_equation_environment=True)

        bstr = dedent(r"""
        \begin{equation} 
            \begin{aligned} 
                & \min 
                & & x + y + z + u + v + w \\ 
                & \text{w.b.} 
                & & -10 \leq x \leq 10 \qquad \in \mathds{Z}\\ 
                &&& y \qquad \in \left\{ 0 , 1 \right \}\\ 
                &&&  0 < z \leq 10 \qquad \in \mathds{R}_{> 0}\\ 
                &&&  0 \leq u \leq 10 \qquad \in \mathds{Z}_{\geq 0}\\ 
                &&& -10 \leq v < 0  \qquad \in \mathds{R}_{< 0}\\ 
                &&&  0 \leq w \leq 1  \qquad \in \mathds{R}
            \end{aligned} 
            \label{basicFormulation} 
        \end{equation} 
        """)

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_indexedParamSingle(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.I = pyo.Set(initialize=[1, 2, 3, 4, 5])
        m.x = pyo.Var(m.I * m.I)
        m.c = pyo.Param(m.I * m.I, initialize=1.0, mutable=True)

        def ruleMaker_1(m):
            return sum(m.c[i, j] * m.x[i, j] for i in m.I for j in m.I)

        def ruleMaker_2(m):
            return sum(m.c[i, j] * m.x[i, j] ** 2 for i in m.I for j in m.I) <= 1

        m.objective = pyo.Objective(rule=ruleMaker_1)
        m.constraint_1 = pyo.Constraint(rule=ruleMaker_2)
        pstr = latex_printer(m.constraint_1)
        print(pstr)

        bstr = dedent(r"""
        \begin{equation} 
             \sum_{ i \in I  } \sum_{ j \in I  } c_{i,j} x_{i,j}^{2} \leq 1 
        \end{equation} 
        """)

        self.assertEqual('\n' + pstr + '\n', bstr)

    def test_latexPrinter_throwTemplatizeError(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.I = pyo.Set(initialize=[1, 2, 3, 4, 5])
        m.x = pyo.Var(m.I, bounds=[-10, 10])
        m.c = pyo.Param(m.I, initialize=1.0, mutable=True)

        def ruleMaker_1(m):
            return sum(m.c[i] * m.x[i] for i in m.I)

        def ruleMaker_2(m, i):
            if i >= 2:
                return m.x[i] <= 1
            else:
                return pyo.Constraint.Skip

        m.objective = pyo.Objective(rule=ruleMaker_1)
        m.constraint_1 = pyo.Constraint(m.I, rule=ruleMaker_2)
        self.assertRaises(
            RuntimeError,
            latex_printer,
            **{'pyomo_component': m, 'throw_templatization_error': True},
        )
        pstr = latex_printer(m)
        bstr = dedent(r"""
        \begin{align} 
            & \min 
            & & \sum_{ i \in I  } c_{i} x_{i} & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x[2] \leq 1 & \label{con:basicFormulation_constraint_1} \\ 
            & & x[3] \leq 1 & \label{con:basicFormulation_constraint_1} \\ 
            & & x[4] \leq 1 & \label{con:basicFormulation_constraint_1} \\ 
            & & x[5] \leq 1 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & & -10 \leq x \leq 10 & \qquad \in \mathds{R} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """)

        self.assertEqual('\n' + pstr + '\n', bstr)

    def test_latexPrinter_pmedian_verbose(self):
        m = pmedian_concrete.create_model()
        self.assertEqual(
            latex_printer(m).strip(),
            r"""
\begin{align} 
    & \min 
    & & \sum_{ i \in Locations  } \sum_{ j \in Customers  } cost_{i,j} serve\_customer\_from\_location_{i,j} & \label{obj:M1_obj} \\ 
    & \text{s.t.} 
    & & \sum_{ i \in Locations  } serve\_customer\_from\_location_{i,j} = 1 &  \qquad \forall j \in Customers \label{con:M1_single_x} \\ 
    &&& serve\_customer\_from\_location_{i,j} \leq select\_location_{i} &  \qquad \forall i,j \in Locations \times Customers \label{con:M1_bound_y} \\ 
    &&& \sum_{ i \in Locations  } select\_location_{i} = P & \label{con:M1_num_facilities} \\ 
    & \text{w.b.} 
    & & 0.0 \leq serve\_customer\_from\_location \leq 1.0 & \qquad \in \mathds{R} \label{con:M1_serve_customer_from_location_bound} \\ 
    &&& select\_location & \qquad \in \left\{ 0 , 1 \right \} \label{con:M1_select_location_bound} 
\end{align}
            """.strip(),
        )

    def test_latexPrinter_pmedian_concise(self):
        m = pmedian_concrete.create_model()
        lcm = ComponentMap()
        lcm[m.Locations] = ['L', ['n']]
        lcm[m.Customers] = ['C', ['m']]
        lcm[m.cost] = 'd'
        lcm[m.serve_customer_from_location] = 'x'
        lcm[m.select_location] = 'y'
        self.assertEqual(
            latex_printer(m, latex_component_map=lcm).strip(),
            r"""
\begin{align} 
    & \min 
    & & \sum_{ n \in L  } \sum_{ m \in C  } d_{n,m} x_{n,m} & \label{obj:M1_obj} \\ 
    & \text{s.t.} 
    & & \sum_{ n \in L  } x_{n,m} = 1 &  \qquad \forall m \in C \label{con:M1_single_x} \\ 
    &&& x_{n,m} \leq y_{n} &  \qquad \forall n,m \in L \times C \label{con:M1_bound_y} \\ 
    &&& \sum_{ n \in L  } y_{n} = P & \label{con:M1_num_facilities} \\ 
    & \text{w.b.} 
    & & 0.0 \leq x \leq 1.0 & \qquad \in \mathds{R} \label{con:M1_x_bound} \\ 
    &&& y & \qquad \in \left\{ 0 , 1 \right \} \label{con:M1_y_bound} 
\end{align}
            """.strip(),
        )


if __name__ == '__main__':
    unittest.main()
