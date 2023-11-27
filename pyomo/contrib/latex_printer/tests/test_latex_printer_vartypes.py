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
from pyomo.contrib.latex_printer import latex_printer
import pyomo.environ as pyo
from textwrap import dedent
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

from pyomo.common.errors import InfeasibleConstraintException


class TestLatexPrinterVariableTypes(unittest.TestCase):
    def test_latexPrinter_variableType_Reals_1(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=Reals)
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_Reals_2(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=Reals, bounds=(None, None))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_Reals_3(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=Reals, bounds=(-10, 10))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & & -10 \leq x \leq 10 & \qquad \in \mathds{R} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_Reals_4(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=Reals, bounds=(-10, 0))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & & -10 \leq x \leq 0 & \qquad \in \mathds{R} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_Reals_5(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=Reals, bounds=(-10, -2))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & & -10 \leq x \leq -2 & \qquad \in \mathds{R} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_Reals_6(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=Reals, bounds=(0, 0))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & & 0 \leq x \leq 0 & \qquad \in \mathds{R} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_Reals_7(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=Reals, bounds=(0, 10))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & & 0 \leq x \leq 10 & \qquad \in \mathds{R} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_Reals_8(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=Reals, bounds=(2, 10))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & & 2 \leq x \leq 10 & \qquad \in \mathds{R} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_Reals_9(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=Reals, bounds=(0, 1))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & & 0 \leq x \leq 1 & \qquad \in \mathds{R} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_Reals_10(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=Reals, bounds=(1, 10))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & & 1 \leq x \leq 10 & \qquad \in \mathds{R} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_Reals_11(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=Reals, bounds=(0.25, 0.75))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & & 0.25 \leq x \leq 0.75 & \qquad \in \mathds{R} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_PositiveReals_1(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=PositiveReals)
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & &  0 < x & \qquad \in \mathds{R}_{> 0} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_PositiveReals_2(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=PositiveReals, bounds=(None, None))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & &  0 < x & \qquad \in \mathds{R}_{> 0} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_PositiveReals_3(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=PositiveReals, bounds=(-10, 10))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & &  0 < x \leq 10 & \qquad \in \mathds{R}_{> 0} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_PositiveReals_4(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=PositiveReals, bounds=(-10, 0))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        self.assertRaises(
            InfeasibleConstraintException, latex_printer, **{'pyomo_component': m}
        )

    def test_latexPrinter_variableType_PositiveReals_5(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=PositiveReals, bounds=(-10, -2))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        self.assertRaises(
            InfeasibleConstraintException, latex_printer, **{'pyomo_component': m}
        )

    def test_latexPrinter_variableType_PositiveReals_6(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=PositiveReals, bounds=(0, 0))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        self.assertRaises(
            InfeasibleConstraintException, latex_printer, **{'pyomo_component': m}
        )

    def test_latexPrinter_variableType_PositiveReals_7(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=PositiveReals, bounds=(0, 10))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & &  0 < x \leq 10 & \qquad \in \mathds{R}_{> 0} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_PositiveReals_8(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=PositiveReals, bounds=(2, 10))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & & 2 \leq x \leq 10 & \qquad \in \mathds{R}_{> 0} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_PositiveReals_9(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=PositiveReals, bounds=(0, 1))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & &  0 < x \leq 1 & \qquad \in \mathds{R}_{> 0} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_PositiveReals_10(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=PositiveReals, bounds=(1, 10))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & & 1 \leq x \leq 10 & \qquad \in \mathds{R}_{> 0} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_PositiveReals_11(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=PositiveReals, bounds=(0.25, 0.75))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & & 0.25 \leq x \leq 0.75 & \qquad \in \mathds{R}_{> 0} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_NonPositiveReals_1(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=NonPositiveReals)
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & & x \leq 0  & \qquad \in \mathds{R}_{\leq 0} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_NonPositiveReals_2(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=NonPositiveReals, bounds=(None, None))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & & x \leq 0  & \qquad \in \mathds{R}_{\leq 0} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_NonPositiveReals_3(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=NonPositiveReals, bounds=(-10, 10))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & & -10 \leq x \leq 0  & \qquad \in \mathds{R}_{\leq 0} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_NonPositiveReals_4(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=NonPositiveReals, bounds=(-10, 0))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & & -10 \leq x \leq 0  & \qquad \in \mathds{R}_{\leq 0} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_NonPositiveReals_5(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=NonPositiveReals, bounds=(-10, -2))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & & -10 \leq x \leq -2 & \qquad \in \mathds{R}_{\leq 0} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_NonPositiveReals_6(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=NonPositiveReals, bounds=(0, 0))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & &  0 = x \leq 0  & \qquad \in \mathds{R}_{\leq 0} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_NonPositiveReals_7(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=NonPositiveReals, bounds=(0, 10))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & &  0 = x \leq 0  & \qquad \in \mathds{R}_{\leq 0} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_NonPositiveReals_8(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=NonPositiveReals, bounds=(2, 10))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        self.assertRaises(
            InfeasibleConstraintException, latex_printer, **{'pyomo_component': m}
        )

    def test_latexPrinter_variableType_NonPositiveReals_9(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=NonPositiveReals, bounds=(0, 1))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & &  0 = x \leq 0  & \qquad \in \mathds{R}_{\leq 0} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_NonPositiveReals_10(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=NonPositiveReals, bounds=(1, 10))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        self.assertRaises(
            InfeasibleConstraintException, latex_printer, **{'pyomo_component': m}
        )

    def test_latexPrinter_variableType_NonPositiveReals_11(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=NonPositiveReals, bounds=(0.25, 0.75))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        self.assertRaises(
            InfeasibleConstraintException, latex_printer, **{'pyomo_component': m}
        )

    def test_latexPrinter_variableType_NegativeReals_1(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=NegativeReals)
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & & x < 0  & \qquad \in \mathds{R}_{< 0} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_NegativeReals_2(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=NegativeReals, bounds=(None, None))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & & x < 0  & \qquad \in \mathds{R}_{< 0} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_NegativeReals_3(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=NegativeReals, bounds=(-10, 10))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & & -10 \leq x < 0  & \qquad \in \mathds{R}_{< 0} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_NegativeReals_4(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=NegativeReals, bounds=(-10, 0))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & & -10 \leq x < 0  & \qquad \in \mathds{R}_{< 0} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_NegativeReals_5(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=NegativeReals, bounds=(-10, -2))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & & -10 \leq x \leq -2 & \qquad \in \mathds{R}_{< 0} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_NegativeReals_6(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=NegativeReals, bounds=(0, 0))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        self.assertRaises(
            InfeasibleConstraintException, latex_printer, **{'pyomo_component': m}
        )

    def test_latexPrinter_variableType_NegativeReals_7(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=NegativeReals, bounds=(0, 10))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        self.assertRaises(
            InfeasibleConstraintException, latex_printer, **{'pyomo_component': m}
        )

    def test_latexPrinter_variableType_NegativeReals_8(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=NegativeReals, bounds=(2, 10))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        self.assertRaises(
            InfeasibleConstraintException, latex_printer, **{'pyomo_component': m}
        )

    def test_latexPrinter_variableType_NegativeReals_9(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=NegativeReals, bounds=(0, 1))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        self.assertRaises(
            InfeasibleConstraintException, latex_printer, **{'pyomo_component': m}
        )

    def test_latexPrinter_variableType_NegativeReals_10(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=NegativeReals, bounds=(1, 10))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        self.assertRaises(
            InfeasibleConstraintException, latex_printer, **{'pyomo_component': m}
        )

    def test_latexPrinter_variableType_NegativeReals_11(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=NegativeReals, bounds=(0.25, 0.75))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        self.assertRaises(
            InfeasibleConstraintException, latex_printer, **{'pyomo_component': m}
        )

    def test_latexPrinter_variableType_NonNegativeReals_1(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=NonNegativeReals)
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & &  0 \leq x & \qquad \in \mathds{R}_{\geq 0} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_NonNegativeReals_2(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=NonNegativeReals, bounds=(None, None))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & &  0 \leq x & \qquad \in \mathds{R}_{\geq 0} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_NonNegativeReals_3(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=NonNegativeReals, bounds=(-10, 10))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & &  0 \leq x \leq 10 & \qquad \in \mathds{R}_{\geq 0} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_NonNegativeReals_4(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=NonNegativeReals, bounds=(-10, 0))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & &  0 \leq x = 0  & \qquad \in \mathds{R}_{\geq 0} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_NonNegativeReals_5(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=NonNegativeReals, bounds=(-10, -2))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        self.assertRaises(
            InfeasibleConstraintException, latex_printer, **{'pyomo_component': m}
        )

    def test_latexPrinter_variableType_NonNegativeReals_6(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=NonNegativeReals, bounds=(0, 0))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & &  0 \leq x = 0  & \qquad \in \mathds{R}_{\geq 0} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_NonNegativeReals_7(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=NonNegativeReals, bounds=(0, 10))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & &  0 \leq x \leq 10 & \qquad \in \mathds{R}_{\geq 0} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_NonNegativeReals_8(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=NonNegativeReals, bounds=(2, 10))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & & 2 \leq x \leq 10 & \qquad \in \mathds{R}_{\geq 0} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_NonNegativeReals_9(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=NonNegativeReals, bounds=(0, 1))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & &  0 \leq x \leq 1 & \qquad \in \mathds{R}_{\geq 0} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_NonNegativeReals_10(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=NonNegativeReals, bounds=(1, 10))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & & 1 \leq x \leq 10 & \qquad \in \mathds{R}_{\geq 0} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_NonNegativeReals_11(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=NonNegativeReals, bounds=(0.25, 0.75))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & & 0.25 \leq x \leq 0.75 & \qquad \in \mathds{R}_{\geq 0} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_Integers_1(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=Integers)
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & & x & \qquad \in \mathds{Z} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_Integers_2(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=Integers, bounds=(None, None))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & & x & \qquad \in \mathds{Z} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_Integers_3(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=Integers, bounds=(-10, 10))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & & -10 \leq x \leq 10 & \qquad \in \mathds{Z} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_Integers_4(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=Integers, bounds=(-10, 0))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & & -10 \leq x \leq 0 & \qquad \in \mathds{Z} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_Integers_5(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=Integers, bounds=(-10, -2))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & & -10 \leq x \leq -2 & \qquad \in \mathds{Z} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_Integers_6(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=Integers, bounds=(0, 0))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & & 0 \leq x \leq 0 & \qquad \in \mathds{Z} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_Integers_7(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=Integers, bounds=(0, 10))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & & 0 \leq x \leq 10 & \qquad \in \mathds{Z} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_Integers_8(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=Integers, bounds=(2, 10))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & & 2 \leq x \leq 10 & \qquad \in \mathds{Z} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_Integers_9(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=Integers, bounds=(0, 1))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & & 0 \leq x \leq 1 & \qquad \in \mathds{Z} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_Integers_10(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=Integers, bounds=(1, 10))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & & 1 \leq x \leq 10 & \qquad \in \mathds{Z} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_Integers_11(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=Integers, bounds=(0.25, 0.75))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & & 0.25 \leq x \leq 0.75 & \qquad \in \mathds{Z} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_PositiveIntegers_1(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=PositiveIntegers)
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & & 1 \leq x & \qquad \in \mathds{Z}_{> 0} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_PositiveIntegers_2(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=PositiveIntegers, bounds=(None, None))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & & 1 \leq x & \qquad \in \mathds{Z}_{> 0} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_PositiveIntegers_3(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=PositiveIntegers, bounds=(-10, 10))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & & 1 \leq x \leq 10 & \qquad \in \mathds{Z}_{> 0} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_PositiveIntegers_4(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=PositiveIntegers, bounds=(-10, 0))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        self.assertRaises(
            InfeasibleConstraintException, latex_printer, **{'pyomo_component': m}
        )

    def test_latexPrinter_variableType_PositiveIntegers_5(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=PositiveIntegers, bounds=(-10, -2))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        self.assertRaises(
            InfeasibleConstraintException, latex_printer, **{'pyomo_component': m}
        )

    def test_latexPrinter_variableType_PositiveIntegers_6(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=PositiveIntegers, bounds=(0, 0))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        self.assertRaises(
            InfeasibleConstraintException, latex_printer, **{'pyomo_component': m}
        )

    def test_latexPrinter_variableType_PositiveIntegers_7(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=PositiveIntegers, bounds=(0, 10))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & & 1 \leq x \leq 10 & \qquad \in \mathds{Z}_{> 0} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_PositiveIntegers_8(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=PositiveIntegers, bounds=(2, 10))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & & 2 \leq x \leq 10 & \qquad \in \mathds{Z}_{> 0} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_PositiveIntegers_9(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=PositiveIntegers, bounds=(0, 1))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & & 1 \leq x \leq 1 & \qquad \in \mathds{Z}_{> 0} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_PositiveIntegers_10(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=PositiveIntegers, bounds=(1, 10))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & & 1 \leq x \leq 10 & \qquad \in \mathds{Z}_{> 0} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_PositiveIntegers_11(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=PositiveIntegers, bounds=(0.25, 0.75))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & & 1 \leq x \leq 0.75 & \qquad \in \mathds{Z}_{> 0} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_NonPositiveIntegers_1(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=NonPositiveIntegers)
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & & x \leq 0  & \qquad \in \mathds{Z}_{\leq 0} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_NonPositiveIntegers_2(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=NonPositiveIntegers, bounds=(None, None))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & & x \leq 0  & \qquad \in \mathds{Z}_{\leq 0} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_NonPositiveIntegers_3(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=NonPositiveIntegers, bounds=(-10, 10))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & & -10 \leq x \leq 0  & \qquad \in \mathds{Z}_{\leq 0} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_NonPositiveIntegers_4(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=NonPositiveIntegers, bounds=(-10, 0))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & & -10 \leq x \leq 0  & \qquad \in \mathds{Z}_{\leq 0} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_NonPositiveIntegers_5(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=NonPositiveIntegers, bounds=(-10, -2))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & & -10 \leq x \leq -2 & \qquad \in \mathds{Z}_{\leq 0} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_NonPositiveIntegers_6(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=NonPositiveIntegers, bounds=(0, 0))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & &  0 = x \leq 0  & \qquad \in \mathds{Z}_{\leq 0} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_NonPositiveIntegers_7(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=NonPositiveIntegers, bounds=(0, 10))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & &  0 = x \leq 0  & \qquad \in \mathds{Z}_{\leq 0} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_NonPositiveIntegers_8(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=NonPositiveIntegers, bounds=(2, 10))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        self.assertRaises(
            InfeasibleConstraintException, latex_printer, **{'pyomo_component': m}
        )

    def test_latexPrinter_variableType_NonPositiveIntegers_9(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=NonPositiveIntegers, bounds=(0, 1))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & &  0 = x \leq 0  & \qquad \in \mathds{Z}_{\leq 0} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_NonPositiveIntegers_10(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=NonPositiveIntegers, bounds=(1, 10))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        self.assertRaises(
            InfeasibleConstraintException, latex_printer, **{'pyomo_component': m}
        )

    def test_latexPrinter_variableType_NonPositiveIntegers_11(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=NonPositiveIntegers, bounds=(0.25, 0.75))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        self.assertRaises(
            InfeasibleConstraintException, latex_printer, **{'pyomo_component': m}
        )

    def test_latexPrinter_variableType_NegativeIntegers_1(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=NegativeIntegers)
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & & x \leq -1 & \qquad \in \mathds{Z}_{< 0} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_NegativeIntegers_2(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=NegativeIntegers, bounds=(None, None))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & & x \leq -1 & \qquad \in \mathds{Z}_{< 0} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_NegativeIntegers_3(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=NegativeIntegers, bounds=(-10, 10))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & & -10 \leq x \leq -1 & \qquad \in \mathds{Z}_{< 0} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_NegativeIntegers_4(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=NegativeIntegers, bounds=(-10, 0))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & & -10 \leq x \leq -1 & \qquad \in \mathds{Z}_{< 0} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_NegativeIntegers_5(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=NegativeIntegers, bounds=(-10, -2))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & & -10 \leq x \leq -2 & \qquad \in \mathds{Z}_{< 0} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_NegativeIntegers_6(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=NegativeIntegers, bounds=(0, 0))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        self.assertRaises(
            InfeasibleConstraintException, latex_printer, **{'pyomo_component': m}
        )

    def test_latexPrinter_variableType_NegativeIntegers_7(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=NegativeIntegers, bounds=(0, 10))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        self.assertRaises(
            InfeasibleConstraintException, latex_printer, **{'pyomo_component': m}
        )

    def test_latexPrinter_variableType_NegativeIntegers_8(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=NegativeIntegers, bounds=(2, 10))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        self.assertRaises(
            InfeasibleConstraintException, latex_printer, **{'pyomo_component': m}
        )

    def test_latexPrinter_variableType_NegativeIntegers_9(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=NegativeIntegers, bounds=(0, 1))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        self.assertRaises(
            InfeasibleConstraintException, latex_printer, **{'pyomo_component': m}
        )

    def test_latexPrinter_variableType_NegativeIntegers_10(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=NegativeIntegers, bounds=(1, 10))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        self.assertRaises(
            InfeasibleConstraintException, latex_printer, **{'pyomo_component': m}
        )

    def test_latexPrinter_variableType_NegativeIntegers_11(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=NegativeIntegers, bounds=(0.25, 0.75))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        self.assertRaises(
            InfeasibleConstraintException, latex_printer, **{'pyomo_component': m}
        )

    def test_latexPrinter_variableType_NonNegativeIntegers_1(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=NonNegativeIntegers)
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & &  0 \leq x & \qquad \in \mathds{Z}_{\geq 0} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_NonNegativeIntegers_2(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=NonNegativeIntegers, bounds=(None, None))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & &  0 \leq x & \qquad \in \mathds{Z}_{\geq 0} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_NonNegativeIntegers_3(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=NonNegativeIntegers, bounds=(-10, 10))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & &  0 \leq x \leq 10 & \qquad \in \mathds{Z}_{\geq 0} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_NonNegativeIntegers_4(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=NonNegativeIntegers, bounds=(-10, 0))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & &  0 \leq x = 0  & \qquad \in \mathds{Z}_{\geq 0} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_NonNegativeIntegers_5(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=NonNegativeIntegers, bounds=(-10, -2))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        self.assertRaises(
            InfeasibleConstraintException, latex_printer, **{'pyomo_component': m}
        )

    def test_latexPrinter_variableType_NonNegativeIntegers_6(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=NonNegativeIntegers, bounds=(0, 0))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & &  0 \leq x = 0  & \qquad \in \mathds{Z}_{\geq 0} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_NonNegativeIntegers_7(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=NonNegativeIntegers, bounds=(0, 10))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & &  0 \leq x \leq 10 & \qquad \in \mathds{Z}_{\geq 0} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_NonNegativeIntegers_8(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=NonNegativeIntegers, bounds=(2, 10))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & & 2 \leq x \leq 10 & \qquad \in \mathds{Z}_{\geq 0} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_NonNegativeIntegers_9(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=NonNegativeIntegers, bounds=(0, 1))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & &  0 \leq x \leq 1 & \qquad \in \mathds{Z}_{\geq 0} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_NonNegativeIntegers_10(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=NonNegativeIntegers, bounds=(1, 10))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & & 1 \leq x \leq 10 & \qquad \in \mathds{Z}_{\geq 0} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_NonNegativeIntegers_11(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=NonNegativeIntegers, bounds=(0.25, 0.75))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & & 0.25 \leq x \leq 0.75 & \qquad \in \mathds{Z}_{\geq 0} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_Boolean_1(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=Boolean)
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & & x & \qquad \in \left\{ \text{True} , \text{False} \right \} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_Boolean_2(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=Boolean, bounds=(None, None))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & & x & \qquad \in \left\{ \text{True} , \text{False} \right \} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_Boolean_3(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=Boolean, bounds=(-10, 10))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & & x & \qquad \in \left\{ \text{True} , \text{False} \right \} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_Boolean_4(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=Boolean, bounds=(-10, 0))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & & x & \qquad \in \left\{ \text{True} , \text{False} \right \} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_Boolean_5(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=Boolean, bounds=(-10, -2))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & & x & \qquad \in \left\{ \text{True} , \text{False} \right \} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_Boolean_6(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=Boolean, bounds=(0, 0))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & & x & \qquad \in \left\{ \text{True} , \text{False} \right \} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_Boolean_7(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=Boolean, bounds=(0, 10))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & & x & \qquad \in \left\{ \text{True} , \text{False} \right \} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_Boolean_8(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=Boolean, bounds=(2, 10))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & & x & \qquad \in \left\{ \text{True} , \text{False} \right \} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_Boolean_9(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=Boolean, bounds=(0, 1))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & & x & \qquad \in \left\{ \text{True} , \text{False} \right \} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_Boolean_10(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=Boolean, bounds=(1, 10))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & & x & \qquad \in \left\{ \text{True} , \text{False} \right \} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_Boolean_11(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=Boolean, bounds=(0.25, 0.75))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & & x & \qquad \in \left\{ \text{True} , \text{False} \right \} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_Binary_1(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=Binary)
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & & x & \qquad \in \left\{ 0 , 1 \right \} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_Binary_2(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=Binary, bounds=(None, None))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & & x & \qquad \in \left\{ 0 , 1 \right \} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_Binary_3(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=Binary, bounds=(-10, 10))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & & x & \qquad \in \left\{ 0 , 1 \right \} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_Binary_4(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=Binary, bounds=(-10, 0))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & & x & \qquad \in \left\{ 0 , 1 \right \} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_Binary_5(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=Binary, bounds=(-10, -2))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & & x & \qquad \in \left\{ 0 , 1 \right \} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_Binary_6(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=Binary, bounds=(0, 0))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & & x & \qquad \in \left\{ 0 , 1 \right \} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_Binary_7(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=Binary, bounds=(0, 10))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & & x & \qquad \in \left\{ 0 , 1 \right \} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_Binary_8(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=Binary, bounds=(2, 10))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & & x & \qquad \in \left\{ 0 , 1 \right \} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_Binary_9(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=Binary, bounds=(0, 1))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & & x & \qquad \in \left\{ 0 , 1 \right \} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_Binary_10(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=Binary, bounds=(1, 10))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & & x & \qquad \in \left\{ 0 , 1 \right \} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_Binary_11(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=Binary, bounds=(0.25, 0.75))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & & x & \qquad \in \left\{ 0 , 1 \right \} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_EmptySet_1(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=EmptySet)
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & & x & \qquad \in \varnothing \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_EmptySet_2(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=EmptySet, bounds=(None, None))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & & x & \qquad \in \varnothing \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_EmptySet_3(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=EmptySet, bounds=(-10, 10))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & & x & \qquad \in \varnothing \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_EmptySet_4(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=EmptySet, bounds=(-10, 0))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & & x & \qquad \in \varnothing \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_EmptySet_5(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=EmptySet, bounds=(-10, -2))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & & x & \qquad \in \varnothing \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_EmptySet_6(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=EmptySet, bounds=(0, 0))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & & x & \qquad \in \varnothing \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_EmptySet_7(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=EmptySet, bounds=(0, 10))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & & x & \qquad \in \varnothing \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_EmptySet_8(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=EmptySet, bounds=(2, 10))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & & x & \qquad \in \varnothing \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_EmptySet_9(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=EmptySet, bounds=(0, 1))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & & x & \qquad \in \varnothing \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_EmptySet_10(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=EmptySet, bounds=(1, 10))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & & x & \qquad \in \varnothing \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_EmptySet_11(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=EmptySet, bounds=(0.25, 0.75))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & & x & \qquad \in \varnothing \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_UnitInterval_1(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=UnitInterval)
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & &  0 \leq x \leq 1  & \qquad \in \mathds{R} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_UnitInterval_2(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=UnitInterval, bounds=(None, None))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & &  0 \leq x \leq 1  & \qquad \in \mathds{R} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_UnitInterval_3(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=UnitInterval, bounds=(-10, 10))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & &  0 \leq x \leq 1  & \qquad \in \mathds{R} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_UnitInterval_4(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=UnitInterval, bounds=(-10, 0))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & &  0 \leq x = 0  & \qquad \in \mathds{R} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_UnitInterval_5(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=UnitInterval, bounds=(-10, -2))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        self.assertRaises(
            InfeasibleConstraintException, latex_printer, **{'pyomo_component': m}
        )

    def test_latexPrinter_variableType_UnitInterval_6(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=UnitInterval, bounds=(0, 0))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & &  0 \leq x = 0  & \qquad \in \mathds{R} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_UnitInterval_7(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=UnitInterval, bounds=(0, 10))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & &  0 \leq x \leq 1  & \qquad \in \mathds{R} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_UnitInterval_8(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=UnitInterval, bounds=(2, 10))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        self.assertRaises(
            InfeasibleConstraintException, latex_printer, **{'pyomo_component': m}
        )

    def test_latexPrinter_variableType_UnitInterval_9(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=UnitInterval, bounds=(0, 1))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & &  0 \leq x \leq 1  & \qquad \in \mathds{R} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_UnitInterval_10(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=UnitInterval, bounds=(1, 10))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & &  = 1 x \leq 1  & \qquad \in \mathds{R} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_UnitInterval_11(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=UnitInterval, bounds=(0.25, 0.75))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & & 0.25 \leq x \leq 0.75 & \qquad \in \mathds{R} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_PercentFraction_1(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=PercentFraction)
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & &  0 \leq x \leq 1  & \qquad \in \mathds{R} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_PercentFraction_2(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=PercentFraction, bounds=(None, None))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & &  0 \leq x \leq 1  & \qquad \in \mathds{R} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_PercentFraction_3(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=PercentFraction, bounds=(-10, 10))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & &  0 \leq x \leq 1  & \qquad \in \mathds{R} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_PercentFraction_4(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=PercentFraction, bounds=(-10, 0))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & &  0 \leq x = 0  & \qquad \in \mathds{R} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_PercentFraction_5(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=PercentFraction, bounds=(-10, -2))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        self.assertRaises(
            InfeasibleConstraintException, latex_printer, **{'pyomo_component': m}
        )

    def test_latexPrinter_variableType_PercentFraction_6(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=PercentFraction, bounds=(0, 0))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & &  0 \leq x = 0  & \qquad \in \mathds{R} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_PercentFraction_7(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=PercentFraction, bounds=(0, 10))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & &  0 \leq x \leq 1  & \qquad \in \mathds{R} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_PercentFraction_8(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=PercentFraction, bounds=(2, 10))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        self.assertRaises(
            InfeasibleConstraintException, latex_printer, **{'pyomo_component': m}
        )

    def test_latexPrinter_variableType_PercentFraction_9(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=PercentFraction, bounds=(0, 1))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & &  0 \leq x \leq 1  & \qquad \in \mathds{R} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_PercentFraction_10(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=PercentFraction, bounds=(1, 10))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & &  = 1 x \leq 1  & \qquad \in \mathds{R} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)

    def test_latexPrinter_variableType_PercentFraction_11(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=PercentFraction, bounds=(0.25, 0.75))
        m.objective = pyo.Objective(expr=m.x)
        m.constraint_1 = pyo.Constraint(expr=m.x**2 <= 5.0)
        pstr = latex_printer(m)

        bstr = dedent(
            r"""
        \begin{align} 
            & \min 
            & & x & \label{obj:basicFormulation_objective} \\ 
            & \text{s.t.} 
            & & x^{2} \leq 5 & \label{con:basicFormulation_constraint_1} \\ 
            & \text{w.b.} 
            & & 0.25 \leq x \leq 0.75 & \qquad \in \mathds{R} \label{con:basicFormulation_x_bound} 
        \end{align} 
        """
        )

        self.assertEqual("\n" + pstr + "\n", bstr)


if __name__ == '__main__':
    unittest.main()
