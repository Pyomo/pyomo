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

import io
from textwrap import dedent

import pyomo.common.unittest as unittest
import pyomo.core.tests.examples.pmedian_concrete as pmedian_concrete
import pyomo.environ as pyo

from pyomo.contrib.latex_printer import latex_printer
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
        bstr = dedent(
            r"""
        \begin{equation} 
             x + y 
        \end{equation} 
        """
        )
        self.assertEqual('\n' + pstr + '\n', bstr)

        # Ex 2 -----------------------
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.expression_1 = pyo.Expression(expr=m.x**2 + m.y**2)
        pstr = latex_printer(m.expression_1)
        bstr = dedent(
            r"""
        \begin{equation} 
             x^{2} + y^{2} 
        \end{equation} 
        """
        )
        self.assertEqual('\n' + pstr + '\n', bstr)

        # Ex 3 -----------------------
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.constraint_1 = pyo.Constraint(expr=m.x**2 + m.y**2 <= 1.0)
        pstr = latex_printer(m.constraint_1)
        bstr = dedent(
            r"""
        \begin{equation} 
             x^{2} + y^{2} \leq 1 
        \end{equation} 
        """
        )
        self.assertEqual('\n' + pstr + '\n', bstr)

        # Ex 4 -----------------------
        m = pyo.ConcreteModel(name='basicFormulation')
        m.I = pyo.Set(initialize=[1, 2, 3, 4, 5])
        m.v = pyo.Var(m.I)

        def ruleMaker(m):
            return sum(m.v[i] for i in m.I) <= 0

        m.constraint = pyo.Constraint(rule=ruleMaker)
        pstr = latex_printer(m.constraint)
        bstr = dedent(
            r"""
        \begin{equation} 
             \sum_{ i \in I  } v_{i} \leq 0 
        \end{equation} 
        """
        )
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
        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x + y + z & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} + y^{2} - z^{2} \leq c & \label{con:basicFormulation_constraint_1} 
        \end{align} 
        """
        )
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
        bstr = dedent(
            r"""
        \begin{equation} 
             \sum_{ j \in \mathcal{A}  } x_{j} \leq 0 
        \end{equation} 
        """
        )
        self.assertEqual('\n' + pstr + '\n', bstr)

    def test_latexPrinter_objective(self):
        m = generate_model()
        pstr = latex_printer(m.objective_1)
        bstr = dedent(
            r"""
        \begin{equation} 
            & \min 
            & & x + y + z 
        \end{equation} 
        """
        )
        self.assertEqual('\n' + pstr + '\n', bstr)

        pstr = latex_printer(m.objective_3)
        bstr = dedent(
            r"""
        \begin{equation} 
            & \max 
            & & x + y + z 
        \end{equation} 
        """
        )
        self.assertEqual('\n' + pstr + '\n', bstr)

    def test_latexPrinter_constraint(self):
        m = generate_model()
        pstr = latex_printer(m.constraint_1)

        bstr = dedent(
            r"""
        \begin{equation} 
             x^{2} + y^{-2} - x y z + 1 = 2 
        \end{equation} 
        """
        )

        self.assertEqual('\n' + pstr + '\n', bstr)

    def test_latexPrinter_expression(self):
        m = generate_model()

        m.express = pyo.Expression(expr=m.x + m.y)

        pstr = latex_printer(m.express)

        bstr = dedent(
            r"""
        \begin{equation} 
             x + y 
        \end{equation} 
        """
        )

        self.assertEqual('\n' + pstr + '\n', bstr)

    def test_latexPrinter_simpleExpression(self):
        m = generate_model()

        pstr = latex_printer(m.x - m.y)
        bstr = dedent(
            r"""
        \begin{equation} 
             x - y 
        \end{equation} 
        """
        )
        self.assertEqual('\n' + pstr + '\n', bstr)

        pstr = latex_printer(m.x - 2 * m.y)
        bstr = dedent(
            r"""
        \begin{equation} 
             x - 2 y 
        \end{equation} 
        """
        )
        self.assertEqual('\n' + pstr + '\n', bstr)

    def test_latexPrinter_unary(self):
        m = generate_model()

        pstr = latex_printer(m.constraint_2)
        bstr = dedent(
            r"""
        \begin{equation} 
              \left| \frac{x}{z^{-2}} \right|   \left( x + y \right)  \leq 2 
        \end{equation} 
        """
        )
        self.assertEqual('\n' + pstr + '\n', bstr)

        pstr = latex_printer(pyo.Constraint(expr=pyo.sin(m.x) == 1))
        bstr = dedent(
            r"""
        \begin{equation} 
             \sin \left( x \right)  = 1 
        \end{equation} 
        """
        )
        self.assertEqual('\n' + pstr + '\n', bstr)

        pstr = latex_printer(pyo.Constraint(expr=pyo.log10(m.x) == 1))
        bstr = dedent(
            r"""
        \begin{equation} 
             \log_{10} \left( x \right)  = 1 
        \end{equation} 
        """
        )
        self.assertEqual('\n' + pstr + '\n', bstr)

        pstr = latex_printer(pyo.Constraint(expr=pyo.sqrt(m.x) == 1))
        bstr = dedent(
            r"""
        \begin{equation} 
             \sqrt { x } = 1 
        \end{equation} 
        """
        )
        self.assertEqual('\n' + pstr + '\n', bstr)

    def test_latexPrinter_rangedConstraint(self):
        m = generate_model()

        pstr = latex_printer(m.constraint_4)
        bstr = dedent(
            r"""
        \begin{equation} 
             1 \leq x \leq 2 
        \end{equation} 
        """
        )
        self.assertEqual('\n' + pstr + '\n', bstr)

    def test_latexPrinter_exprIf(self):
        m = generate_model()

        pstr = latex_printer(m.constraint_5)
        bstr = dedent(
            r"""
        \begin{equation} 
             f_{\text{exprIf}}(x \leq 1,z,y) \leq 1 
        \end{equation} 
        """
        )
        self.assertEqual('\n' + pstr + '\n', bstr)

    def test_latexPrinter_blackBox(self):
        m = generate_model()

        pstr = latex_printer(m.constraint_6)
        bstr = dedent(
            r"""
        \begin{equation} 
             x + f\_1(x,y) = 2 
        \end{equation} 
        """
        )
        self.assertEqual('\n' + pstr + '\n', bstr)

    def test_latexPrinter_iteratedConstraints(self):
        m = generate_model()

        pstr = latex_printer(m.constraint_7)
        bstr = dedent(
            r"""
        \begin{equation} 
              \left( x + y \right)  \sum_{ i \in I  } v_{i} + u_{i,j}^{2} \leq 0  \qquad \forall j \in I 
        \end{equation} 
        """
        )
        self.assertEqual('\n' + pstr + '\n', bstr)

        pstr = latex_printer(m.constraint_8)
        bstr = dedent(
            r"""
        \begin{equation} 
             \sum_{ i \in K  } p_{i} = 1 
        \end{equation} 
        """
        )
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

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & \sum_{ i \in I  } \sum_{ j \in I  } c_{i,j} x_{i,j} & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & \sum_{ i \in I  } \sum_{ j \in I  } x_{i,j}^{2} \leq 1 & \label{con:basicFormulation_constraint_1} 
        \end{align} 
        """
        )

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

        bstr = dedent(
            r"""
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
        """
        )

        self.assertEqual('\n' + pstr + '\n', bstr)

    def test_latexPrinter_continuousSet(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.I = pyo.Set(initialize=[1, 2, 3, 4, 5])
        m.v = pyo.Var(m.I)

        def ruleMaker(m):
            return sum(m.v[i] for i in m.I) <= 0

        m.constraint = pyo.Constraint(rule=ruleMaker)
        pstr = latex_printer(m.constraint, explicit_set_summation=True)

        bstr = dedent(
            r"""
        \begin{equation} 
             \sum_{ i = 1 }^{5} v_{i} \leq 0 
        \end{equation} 
        """
        )
        self.assertEqual('\n' + pstr + '\n', bstr)

    def test_latexPrinter_notContinuousSet(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.I = pyo.Set(initialize=[1, 3, 4, 5])
        m.v = pyo.Var(m.I)

        def ruleMaker(m):
            return sum(m.v[i] for i in m.I) <= 0

        m.constraint = pyo.Constraint(rule=ruleMaker)
        pstr = latex_printer(m.constraint, explicit_set_summation=True)

        bstr = dedent(
            r"""
        \begin{equation} 
             \sum_{ i \in I  } v_{i} \leq 0 
        \end{equation} 
        """
        )
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
        bstr = dedent(
            r"""
        \begin{equation} 
             \sum_{ i \in \mathcal{A}  } x_{i} \leq 0 
        \end{equation} 
        """
        )
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

        bstr = dedent(
            r"""
        \begin{equation} 
            \begin{aligned} 
                & \min 
                & & x + y + z \\ 
                & \text{s.t.} 
                & & x^{2} + y^{2} - z^{2} \leq c 
            \end{aligned} 
            \label{basicFormulation} 
        \end{equation} 
        """
        )
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

        bstr = dedent(
            r"""
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
        """
        )

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

        bstr = dedent(
            r"""
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
        """
        )

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

        bstr = dedent(
            r"""
        \begin{equation} 
             \sum_{ i \in I  } \sum_{ j \in I  } c_{i,j} x_{i,j}^{2} \leq 1 
        \end{equation} 
        """
        )

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
        bstr = dedent(
            r"""
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
        """
        )

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
