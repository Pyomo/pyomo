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
import pyomo.environ as pe


def generate_model():
    import pyomo.environ as pe
    from pyomo.core.expr import Expr_if
    from pyomo.core.base import ExternalFunction

    m = pe.ConcreteModel(name='basicFormulation')
    m.x = pe.Var()
    m.y = pe.Var()
    m.z = pe.Var()
    m.objective_1 = pe.Objective(expr=m.x + m.y + m.z)
    m.constraint_1 = pe.Constraint(
        expr=m.x**2 + m.y**-2.0 - m.x * m.y * m.z + 1 == 2.0
    )
    m.constraint_2 = pe.Constraint(expr=abs(m.x / m.z**-2) * (m.x + m.y) <= 2.0)
    m.constraint_3 = pe.Constraint(expr=pe.sqrt(m.x / m.z**-2) <= 2.0)
    m.constraint_4 = pe.Constraint(expr=(1, m.x, 2))
    m.constraint_5 = pe.Constraint(expr=Expr_if(m.x <= 1.0, m.z, m.y) <= 1.0)

    def blackbox(a, b):
        return sin(a - b)

    m.bb = ExternalFunction(blackbox)
    m.constraint_6 = pe.Constraint(expr=m.x + m.bb(m.x, m.y) == 2)

    m.I = pe.Set(initialize=[1, 2, 3, 4, 5])
    m.J = pe.Set(initialize=[1, 2, 3])
    m.K = pe.Set(initialize=[1, 3, 5])
    m.u = pe.Var(m.I * m.I)
    m.v = pe.Var(m.I)
    m.w = pe.Var(m.J)
    m.p = pe.Var(m.K)

    m.express = pe.Expression(expr=m.x**2 + m.y**2)

    def ruleMaker(m, j):
        return (m.x + m.y) * sum(m.v[i] + m.u[i, j] ** 2 for i in m.I) <= 0

    m.constraint_7 = pe.Constraint(m.I, rule=ruleMaker)

    def ruleMaker(m):
        return sum(m.p[k] for k in m.K) == 1

    m.constraint_8 = pe.Constraint(rule=ruleMaker)

    def ruleMaker(m):
        return (m.x + m.y) * sum(m.w[j] for j in m.J)

    m.objective_2 = pe.Objective(rule=ruleMaker)

    m.objective_3 = pe.Objective(expr=m.x + m.y + m.z, sense=-1)

    return m


def generate_simple_model():
    import pyomo.environ as pe

    m = pe.ConcreteModel(name='basicFormulation')
    m.x = pe.Var()
    m.y = pe.Var()
    m.objective_1 = pe.Objective(expr=m.x + m.y)
    m.constraint_1 = pe.Constraint(expr=m.x**2 + m.y**2.0 <= 1.0)
    m.constraint_2 = pe.Constraint(expr=m.x >= 0.0)

    m.I = pe.Set(initialize=[1, 2, 3, 4, 5])
    m.J = pe.Set(initialize=[1, 2, 3])
    m.K = pe.Set(initialize=[1, 3, 5])
    m.u = pe.Var(m.I * m.I)
    m.v = pe.Var(m.I)
    m.w = pe.Var(m.J)
    m.p = pe.Var(m.K)

    def ruleMaker(m, j):
        return (m.x + m.y) * sum(m.v[i] + m.u[i, j] ** 2 for i in m.I) <= 0

    m.constraint_7 = pe.Constraint(m.I, rule=ruleMaker)

    def ruleMaker(m):
        return sum(m.p[k] for k in m.K) == 1

    m.constraint_8 = pe.Constraint(rule=ruleMaker)

    return m


def generate_simple_model_2():
    import pyomo.environ as pe
    
    m = pe.ConcreteModel(name = 'basicFormulation')
    m.x_dot = pe.Var()
    m.x_bar = pe.Var()
    m.x_star = pe.Var()
    m.x_hat = pe.Var()
    m.x_hat_1 = pe.Var()
    m.y_sub1_sub2_sub3 = pe.Var()
    m.objective_1  = pe.Objective( expr = m.y_sub1_sub2_sub3  )
    m.constraint_1 = pe.Constraint(expr = (m.x_dot + m.x_bar + m.x_star + m.x_hat + m.x_hat_1)**2 <= m.y_sub1_sub2_sub3 )
    m.constraint_2 = pe.Constraint(expr = (m.x_dot + m.x_bar)**-(m.x_star + m.x_hat) <= m.y_sub1_sub2_sub3 )
    m.constraint_3 = pe.Constraint(expr = -(m.x_dot + m.x_bar)+ -(m.x_star + m.x_hat) <= m.y_sub1_sub2_sub3 )

    return m


class TestLatexPrinter(unittest.TestCase):
    def test_latexPrinter_objective(self):
        m = generate_model()
        pstr = latex_printer(m.objective_1)
        bstr = ''
        bstr += '\\begin{equation} \n'
        bstr += '    & \\text{minimize} \n'
        bstr += '    & & x + y + z \n'
        bstr += '\end{equation} \n'
        self.assertEqual(pstr, bstr)

        pstr = latex_printer(m.objective_3)
        bstr = ''
        bstr += '\\begin{equation} \n'
        bstr += '    & \\text{maximize} \n'
        bstr += '    & & x + y + z \n'
        bstr += '\end{equation} \n'
        self.assertEqual(pstr, bstr)

    def test_latexPrinter_constraint(self):
        m = generate_model()
        pstr = latex_printer(m.constraint_1)

        bstr = ''
        bstr += '\\begin{equation} \n'
        bstr += '     x^{2} + y^{-2} - xyz + 1 = 2 \n'
        bstr += '\end{equation} \n'

        self.assertEqual(pstr, bstr)

    def test_latexPrinter_expression(self):
        m = generate_model()

        m.express = pe.Expression(expr=m.x + m.y)

        pstr = latex_printer(m.express)

        bstr = ''
        bstr += '\\begin{equation} \n'
        bstr += '     x + y \n'
        bstr += '\end{equation} \n'

        self.assertEqual(pstr, bstr)

    def test_latexPrinter_simpleExpression(self):
        m = generate_model()

        pstr = latex_printer(m.x - m.y)
        bstr = ''
        bstr += '\\begin{equation} \n'
        bstr += '     x - y \n'
        bstr += '\end{equation} \n'
        self.assertEqual(pstr, bstr)

        pstr = latex_printer(m.x - 2 * m.y)
        bstr = ''
        bstr += '\\begin{equation} \n'
        bstr += '     x - 2y \n'
        bstr += '\end{equation} \n'
        self.assertEqual(pstr, bstr)

    def test_latexPrinter_unary(self):
        m = generate_model()

        pstr = latex_printer(m.constraint_2)
        bstr = ''
        bstr += '\\begin{equation} \n'
        bstr += (
            '      \left| \\frac{x}{z^{-2}} \\right|  \left( x + y \\right)  \leq 2 \n'
        )
        bstr += '\end{equation} \n'
        self.assertEqual(pstr, bstr)

        pstr = latex_printer(pe.Constraint(expr=pe.sin(m.x) == 1))
        bstr = ''
        bstr += '\\begin{equation} \n'
        bstr += '     \sin \left( x \\right)  = 1 \n'
        bstr += '\end{equation} \n'
        self.assertEqual(pstr, bstr)

        pstr = latex_printer(pe.Constraint(expr=pe.log10(m.x) == 1))
        bstr = ''
        bstr += '\\begin{equation} \n'
        bstr += '     \log_{10} \left( x \\right)  = 1 \n'
        bstr += '\end{equation} \n'
        self.assertEqual(pstr, bstr)

        pstr = latex_printer(pe.Constraint(expr=pe.sqrt(m.x) == 1))
        bstr = ''
        bstr += '\\begin{equation} \n'
        bstr += '     \sqrt { x } = 1 \n'
        bstr += '\end{equation} \n'
        self.assertEqual(pstr, bstr)

    def test_latexPrinter_rangedConstraint(self):
        m = generate_model()

        pstr = latex_printer(m.constraint_4)
        bstr = ''
        bstr += '\\begin{equation} \n'
        bstr += '     1 \leq x \leq 2 \n'
        bstr += '\end{equation} \n'
        self.assertEqual(pstr, bstr)

    def test_latexPrinter_exprIf(self):
        m = generate_model()

        pstr = latex_printer(m.constraint_5)
        bstr = ''
        bstr += '\\begin{equation} \n'
        bstr += '     f_{\\text{exprIf}}(x \leq 1,z,y) \leq 1 \n'
        bstr += '\end{equation} \n'
        self.assertEqual(pstr, bstr)

    def test_latexPrinter_blackBox(self):
        m = generate_model()

        pstr = latex_printer(m.constraint_6)
        bstr = ''
        bstr += '\\begin{equation} \n'
        bstr += '     x + f(x,y) = 2 \n'
        bstr += '\end{equation} \n'
        self.assertEqual(pstr, bstr)

    def test_latexPrinter_iteratedConstraints(self):
        m = generate_model()

        pstr = latex_printer(m.constraint_7)
        bstr = ''
        bstr += '\\begin{equation} \n'
        bstr += '      \left( x + y \\right) \sum_{i \in I} v_{i} + u_{i,j}^{2} \leq 0 , \quad j \in I \n'
        bstr += '\end{equation} \n'
        self.assertEqual(pstr, bstr)

        pstr = latex_printer(m.constraint_8)
        bstr = ''
        bstr += '\\begin{equation} \n'
        bstr += '     \sum_{k \in K} p_{k} = 1 \n'
        bstr += '\end{equation} \n'
        self.assertEqual(pstr, bstr)

    def test_latexPrinter_model(self):
        m = generate_simple_model()

        pstr = latex_printer(m)
        bstr = ''
        bstr += '\\begin{equation} \n'
        bstr += '    \\begin{aligned} \n'
        bstr += '        & \\text{minimize} \n'
        bstr += '        & & x + y \\\\ \n'
        bstr += '        & \\text{subject to} \n'
        bstr += '        & & x^{2} + y^{2} \leq 1 \\\\ \n'
        bstr += '        &&& 0 \leq x \\\\ \n'
        bstr += '        &&&  \left( x + y \\right) \sum_{i \\in I} v_{i} + u_{i,j}^{2} \leq 0 , \quad j \\in I \\\\ \n'
        bstr += '        &&& \sum_{k \\in K} p_{k} = 1 \n'
        bstr += '    \end{aligned} \n'
        bstr += '    \label{basicFormulation} \n'
        bstr += '\end{equation} \n'
        self.assertEqual(pstr, bstr)

        pstr = latex_printer(m, None, True)
        bstr = ''
        bstr += '\\begin{align} \n'
        bstr += '    & \\text{minimize} \n'
        bstr += '    & & x + y \label{obj:basicFormulation_objective_1} \\\\ \n'
        bstr += '    & \\text{subject to} \n'
        bstr += '    & & x^{2} + y^{2} \leq 1 \label{con:basicFormulation_constraint_1} \\\\ \n'
        bstr += '    &&& 0 \leq x \label{con:basicFormulation_constraint_2} \\\\ \n'
        bstr += '    &&&  \left( x + y \\right) \sum_{i \\in I} v_{i} + u_{i,j}^{2} \leq 0 , \quad j \\in I \label{con:basicFormulation_constraint_7} \\\\ \n'
        bstr += '    &&& \sum_{k \\in K} p_{k} = 1 \label{con:basicFormulation_constraint_8} \n'
        bstr += '\end{align} \n'
        self.assertEqual(pstr, bstr)

        pstr = latex_printer(m, None, False, True)
        bstr = ''
        bstr += '\\begin{equation} \n'
        bstr += '    \\begin{aligned} \n'
        bstr += '        & \\text{minimize} \n'
        bstr += '        & & x + y \\\\ \n'
        bstr += '        & \\text{subject to} \n'
        bstr += '        & & x^{2} + y^{2} \leq 1 \\\\ \n'
        bstr += '        &&& 0 \leq x \\\\ \n'
        bstr += '        &&&  \left( x + y \\right) \sum_{i = 1}^{5} v_{i} + u_{i,j}^{2} \leq 0 , \quad j \\in I \\\\ \n'
        bstr += '        &&& \sum_{k \\in K} p_{k} = 1 \n'
        bstr += '    \end{aligned} \n'
        bstr += '    \label{basicFormulation} \n'
        bstr += '\end{equation} \n'
        self.assertEqual(pstr, bstr)

        pstr = latex_printer(m, None, True, True)
        bstr = ''
        bstr += '\\begin{align} \n'
        bstr += '    & \\text{minimize} \n'
        bstr += '    & & x + y \label{obj:basicFormulation_objective_1} \\\\ \n'
        bstr += '    & \\text{subject to} \n'
        bstr += '    & & x^{2} + y^{2} \leq 1 \label{con:basicFormulation_constraint_1} \\\\ \n'
        bstr += '    &&& 0 \leq x \label{con:basicFormulation_constraint_2} \\\\ \n'
        bstr += '    &&&  \left( x + y \\right) \sum_{i = 1}^{5} v_{i} + u_{i,j}^{2} \leq 0 , \quad j \\in I \label{con:basicFormulation_constraint_7} \\\\ \n'
        bstr += '    &&& \sum_{k \in K} p_{k} = 1 \label{con:basicFormulation_constraint_8} \n'
        bstr += '\end{align} \n'
        self.assertEqual(pstr, bstr)

    def test_latexPrinter_advancedVariables(self):
        m = generate_simple_model_2()

        pstr = latex_printer(m)
        bstr = ''
        bstr += '\\begin{equation} \n'
        bstr += '    \\begin{aligned} \n'
        bstr += '        & \\text{minimize} \n'
        bstr += '        & & y_{sub1_{sub2_{sub3}}} \\\\ \n'
        bstr += '        & \\text{subject to} \n'
        bstr += '        & &  \left( \dot{x} + \\bar{x} + x^{*} + \hat{x} + \hat{x}_{1} \\right) ^{2} \leq y_{sub1_{sub2_{sub3}}} \\\\ \n'
        bstr += '        &&&  \left( \dot{x} + \\bar{x} \\right) ^{ \left( - \left( x^{*} + \hat{x} \\right)  \\right) } \leq y_{sub1_{sub2_{sub3}}} \\\\ \n'
        bstr += '        &&& - \left( \dot{x} + \\bar{x} \\right)  -  \left( x^{*} + \hat{x} \\right)  \leq y_{sub1_{sub2_{sub3}}} \n'
        bstr += '    \\end{aligned} \n'
        bstr += '    \label{basicFormulation} \n'
        bstr += '\\end{equation} \n'
        self.assertEqual(pstr, bstr)



if __name__ == '__main__':
    unittest.main()
