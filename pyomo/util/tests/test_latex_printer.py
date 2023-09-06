#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2023
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyomo.common.unittest as unittest
from pyomo.util.latex_printer import latex_printer
import pyomo.environ as pyo
from textwrap import dedent
from pyomo.common.tempfiles import TempfileManager


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
    def test_latexPrinter_objective(self):
        m = generate_model()
        pstr = latex_printer(m.objective_1)
        bstr = dedent(
            r"""
        \begin{equation} 
            & \text{minimize} 
            & & x + y + z 
        \end{equation} 
        """
        )
        self.assertEqual('\n' + pstr, bstr)

        pstr = latex_printer(m.objective_3)
        bstr = dedent(
            r"""
        \begin{equation} 
            & \text{maximize} 
            & & x + y + z 
        \end{equation} 
        """
        )
        self.assertEqual('\n' + pstr, bstr)

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

        self.assertEqual('\n' + pstr, bstr)

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

        self.assertEqual('\n' + pstr, bstr)

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
        self.assertEqual('\n' + pstr, bstr)

        pstr = latex_printer(m.x - 2 * m.y)
        bstr = dedent(
            r"""
        \begin{equation} 
             x - 2 y 
        \end{equation} 
        """
        )
        self.assertEqual('\n' + pstr, bstr)

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
        self.assertEqual('\n' + pstr, bstr)

        pstr = latex_printer(pyo.Constraint(expr=pyo.sin(m.x) == 1))
        bstr = dedent(
            r"""
        \begin{equation} 
             \sin \left( x \right)  = 1 
        \end{equation} 
        """
        )
        self.assertEqual('\n' + pstr, bstr)

        pstr = latex_printer(pyo.Constraint(expr=pyo.log10(m.x) == 1))
        bstr = dedent(
            r"""
        \begin{equation} 
             \log_{10} \left( x \right)  = 1 
        \end{equation} 
        """
        )
        self.assertEqual('\n' + pstr, bstr)

        pstr = latex_printer(pyo.Constraint(expr=pyo.sqrt(m.x) == 1))
        bstr = dedent(
            r"""
        \begin{equation} 
             \sqrt { x } = 1 
        \end{equation} 
        """
        )
        self.assertEqual('\n' + pstr, bstr)

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
        self.assertEqual('\n' + pstr, bstr)

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
        self.assertEqual('\n' + pstr, bstr)

    def test_latexPrinter_blackBox(self):
        m = generate_model()

        pstr = latex_printer(m.constraint_6)
        bstr = dedent(
            r"""
        \begin{equation} 
             x + f(x,y) = 2 
        \end{equation} 
        """
        )
        self.assertEqual('\n' + pstr, bstr)

    def test_latexPrinter_iteratedConstraints(self):
        m = generate_model()

        pstr = latex_printer(m.constraint_7)
        bstr = dedent(
            r"""
        \begin{equation} 
              \left( x + y \right)  \sum_{i \in I} v_{i} + u_{i,j}^{2} \leq 0 , \quad j \in I 
        \end{equation} 
        """
        )
        self.assertEqual('\n' + pstr, bstr)

        pstr = latex_printer(m.constraint_8)
        bstr = dedent(
            r"""
        \begin{equation} 
             \sum_{k \in K} p_{k} = 1 
        \end{equation} 
        """
        )
        self.assertEqual('\n' + pstr, bstr)

    def test_latexPrinter_model(self):
        m = generate_simple_model()

        pstr = latex_printer(m)
        bstr = dedent(
            r"""
        \begin{equation} 
            \begin{aligned} 
                & \text{minimize} 
                & & x + y \\ 
                & \text{subject to} 
                & & x^{2} + y^{2} \leq 1 \\ 
                &&& 0 \leq x \\ 
                &&&  \left( x + y \right)  \sum_{i \in I} v_{i} + u_{i,j}^{2} \leq 0 , \quad j \in I \\ 
                &&& \sum_{k \in K} p_{k} = 1 
            \end{aligned} 
            \label{basicFormulation} 
        \end{equation} 
        """
        )
        self.assertEqual('\n' + pstr, bstr)

        pstr = latex_printer(m, None, True)
        bstr = dedent(
            r"""
        \begin{align} 
            & \text{minimize} 
            & & x + y \label{obj:basicFormulation_objective_1} \\ 
            & \text{subject to} 
            & & x^{2} + y^{2} \leq 1 \label{con:basicFormulation_constraint_1} \\ 
            &&& 0 \leq x \label{con:basicFormulation_constraint_2} \\ 
            &&&  \left( x + y \right)  \sum_{i \in I} v_{i} + u_{i,j}^{2} \leq 0 , \quad j \in I \label{con:basicFormulation_constraint_7} \\ 
            &&& \sum_{k \in K} p_{k} = 1 \label{con:basicFormulation_constraint_8} 
        \end{align} 
        """
        )
        self.assertEqual('\n' + pstr, bstr)

        pstr = latex_printer(m, None, False, True)
        bstr = dedent(
            r"""
        \begin{equation} 
            \begin{aligned} 
                & \text{minimize} 
                & & x + y \\ 
                & \text{subject to} 
                & & x^{2} + y^{2} \leq 1 \\ 
                &&& 0 \leq x \\ 
                &&&  \left( x + y \right)  \sum_{i = 1}^{5} v_{i} + u_{i,j}^{2} \leq 0 , \quad j \in I \\ 
                &&& \sum_{k \in K} p_{k} = 1 
            \end{aligned} 
            \label{basicFormulation} 
        \end{equation} 
        """
        )
        self.assertEqual('\n' + pstr, bstr)

        pstr = latex_printer(m, None, True, True)
        bstr = dedent(
            r"""
        \begin{align} 
            & \text{minimize} 
            & & x + y \label{obj:basicFormulation_objective_1} \\ 
            & \text{subject to} 
            & & x^{2} + y^{2} \leq 1 \label{con:basicFormulation_constraint_1} \\ 
            &&& 0 \leq x \label{con:basicFormulation_constraint_2} \\ 
            &&&  \left( x + y \right)  \sum_{i = 1}^{5} v_{i} + u_{i,j}^{2} \leq 0 , \quad j \in I \label{con:basicFormulation_constraint_7} \\ 
            &&& \sum_{k \in K} p_{k} = 1 \label{con:basicFormulation_constraint_8} 
        \end{align} 
        """
        )
        self.assertEqual('\n' + pstr, bstr)

    def test_latexPrinter_advancedVariables(self):
        m = generate_simple_model_2()

        pstr = latex_printer(m)
        bstr = dedent(
            r"""
        \begin{equation} 
            \begin{aligned} 
                & \text{minimize} 
                & & y_{sub1_{sub2_{sub3}}} \\ 
                & \text{subject to} 
                & &  \left( \dot{x} + \bar{x} + x_{star} + \hat{x} + \hat{x}_{1} \right) ^{2} \leq y_{sub1_{sub2_{sub3}}} \\ 
                &&&  \left( \dot{x} + \bar{x} \right) ^{ \left( - \left( x_{star} + \hat{x} \right)  \right) } \leq y_{sub1_{sub2_{sub3}}} \\ 
                &&& - \left( \dot{x} + \bar{x} \right)  -  \left( x_{star} + \hat{x} \right)  \leq y_{sub1_{sub2_{sub3}}} 
            \end{aligned} 
            \label{basicFormulation} 
        \end{equation} 
        """
        )
        self.assertEqual('\n' + pstr, bstr)

    def test_latexPrinter_fileWriter(self):
        m = generate_simple_model()

        with TempfileManager.new_context() as tempfile:
            fd, fname = tempfile.mkstemp()
            pstr = latex_printer(m, fname)

            f = open(fname)
            bstr = f.read()
            f.close()

            bstr_split = bstr.split('\n')
            bstr_stripped = bstr_split[3:-2]
            bstr = '\n'.join(bstr_stripped) + '\n'

            self.assertEqual(pstr, bstr)

    def test_latexPrinter_inputError(self):
        self.assertRaises(ValueError, latex_printer, **{'pyomoElement': 'errorString'})


if __name__ == '__main__':
    unittest.main()
