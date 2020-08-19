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

import pyomo.environ as pyo
from pyomo.pysp.annotations import (locate_annotations,
                                    StageCostAnnotation,
                                    PySP_StageCostAnnotation,
                                    VariableStageAnnotation,
                                    PySP_VariableStageAnnotation,
                                    _ConstraintStageAnnotation,
                                    ConstraintStageAnnotation,
                                    PySP_ConstraintStageAnnotation,
                                    StochasticDataAnnotation,
                                    PySP_StochasticDataAnnotation,
                                    StochasticConstraintBoundsAnnotation,
                                    PySP_StochasticRHSAnnotation,
                                    StochasticConstraintBodyAnnotation,
                                    PySP_StochasticMatrixAnnotation,
                                    StochasticObjectiveAnnotation,
                                    PySP_StochasticObjectiveAnnotation,
                                    StochasticVariableBoundsAnnotation)

class TestAnnotations(unittest.TestCase):

    def test_deprecated(self):
        self.assertIs(StageCostAnnotation,
                      type(PySP_StageCostAnnotation()))
        self.assertIs(VariableStageAnnotation,
                      type(PySP_VariableStageAnnotation()))
        self.assertIs(_ConstraintStageAnnotation,
                      type(PySP_ConstraintStageAnnotation()))
        self.assertIs(_ConstraintStageAnnotation,
                      type(ConstraintStageAnnotation()))
        self.assertIs(StochasticDataAnnotation,
                      type(PySP_StochasticDataAnnotation()))
        self.assertIs(StochasticConstraintBoundsAnnotation,
                      type(PySP_StochasticRHSAnnotation()))
        self.assertIs(StochasticConstraintBodyAnnotation,
                      type(PySP_StochasticMatrixAnnotation()))
        self.assertIs(StochasticObjectiveAnnotation,
                      type(PySP_StochasticObjectiveAnnotation()))

    def _populate_block_with_vars_expressions(self, b):
        b.x = pyo.Var()
        b.X1 = pyo.Var([1])
        b.X2 = pyo.Var([1])
        b.e = pyo.Expression()
        b.E1 = pyo.Expression([1])
        b.E2 = pyo.Expression([1])

    def _populate_block_with_vars(self, b):
        b.x = pyo.Var()
        b.X1 = pyo.Var([1])
        b.X2 = pyo.Var([1])

    def _populate_block_with_constraints(self, b):
        b.x = pyo.Var()
        b.c = pyo.Constraint(expr= b.x == 1)
        b.C1 = pyo.Constraint([1], rule=lambda m, i: m.x == 1)
        b.C2 = pyo.Constraint([1], rule=lambda m, i: m.x == 1)
        b.C3 = pyo.ConstraintList()
        b.C3.add(b.x == 1)

    def _populate_block_with_objectives(self, b):
        b.x = pyo.Var()
        b.o = pyo.Objective(expr= b.x + 1)
        b.O1 = pyo.Objective([1], rule=lambda m, i: m.x + 1)
        b.O2 = pyo.Objective([1], rule=lambda m, i: m.x + 1)

    def _populate_block_with_params(self, b):
        b.p = pyo.Param(mutable=True ,initialize=0)
        b.P1 = pyo.Param([1], mutable=True, initialize=0)
        b.P2 = pyo.Param([1], mutable=True, initialize=0)

    def test_multiple_declarations(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        a = StageCostAnnotation()
        a.declare(m, 1)
        a.declare(m.x, 1)
        with self.assertRaises(RuntimeError) as cm:
            a.expand_entries()

    def test_locate_annotations(self):
        m = pyo.ConcreteModel()
        m.a = StageCostAnnotation()
        m.b = pyo.Block()
        m.b.a = StageCostAnnotation()
        self.assertEqual(locate_annotations(m, StageCostAnnotation),
                         [('a', m.a), ('a', m.b.a)])
        with self.assertRaises(ValueError):
            locate_annotations(m, StageCostAnnotation, max_allowed=1)
        m.b.deactivate()
        self.assertEqual(locate_annotations(m, StageCostAnnotation),
                         [('a', m.a)])
        self.assertEqual(locate_annotations(m, VariableStageAnnotation),
                         [])

    def test_stage_cost(self):
        m = pyo.ConcreteModel()
        self._populate_block_with_vars_expressions(m)
        m.b = pyo.Block()
        self._populate_block_with_vars_expressions(m.b)
        m.b_inactive = pyo.Block()
        self._populate_block_with_vars_expressions(m.b_inactive)
        m.b_inactive.deactivate()
        m.B = pyo.Block([1],
                        rule=lambda b: \
            self._populate_block_with_vars_expressions(b))

        a = StageCostAnnotation()
        self.assertEqual(a.default, None)
        self.assertEqual(a.has_declarations, False)
        a.declare(m.x, 1)
        self.assertEqual(a.has_declarations, True)
        a.declare(m.X1, 1)
        a.declare(m.X2[1], 1)
        a.declare(m.e, 1)
        a.declare(m.E1, 1)
        a.declare(m.E2[1], 1)
        with self.assertRaises(TypeError):
            a.declare(m.b, None)
        a.declare(m.b, 1)
        a.declare(m.b_inactive, 1)
        a.declare(m.B, 2)
        with self.assertRaises(TypeError):
            a.declare(None, 1)
        self.assertEqual(
            set([(v[0].name, v[1]) for v in a.expand_entries()]),
            set([('x', 1), ('X1[1]', 1), ('X2[1]', 1),
                 ('e', 1), ('E1[1]', 1), ('E2[1]', 1),
                 ('b.x', 1), ('b.X1[1]', 1), ('b.X2[1]', 1),
                 ('b.e', 1), ('b.E1[1]', 1), ('b.E2[1]', 1),
                 ('B[1].x', 2), ('B[1].X1[1]', 2), ('B[1].X2[1]', 2),
                 ('B[1].e', 2), ('B[1].E1[1]', 2), ('B[1].E2[1]', 2)]))
        self.assertEqual(
            set([(v[0].name, v[1]) for v in a.expand_entries(expand_containers=False)]),
            set([('x', 1), ('X1', 1), ('X2[1]', 1),
                 ('e', 1), ('E1', 1), ('E2[1]', 1),
                 ('b.x', 1), ('b.X1', 1), ('b.X2', 1),
                 ('b.e', 1), ('b.E1', 1), ('b.E2', 1),
                 ('B[1].x', 2), ('B[1].X1', 2), ('B[1].X2', 2),
                 ('B[1].e', 2), ('B[1].E1', 2), ('B[1].E2', 2)]))

    def test_variable_stage(self):
        m = pyo.ConcreteModel()
        self._populate_block_with_vars_expressions(m)
        m.b = pyo.Block()
        self._populate_block_with_vars_expressions(m.b)
        m.b_inactive = pyo.Block()
        self._populate_block_with_vars_expressions(m.b_inactive)
        m.b_inactive.deactivate()
        m.B = pyo.Block([1],
                        rule=lambda b: \
            self._populate_block_with_vars_expressions(b))

        a = VariableStageAnnotation()
        self.assertEqual(a.default, None)
        self.assertEqual(a.has_declarations, False)
        a.declare(m.x, 1)
        self.assertEqual(a.has_declarations, True)
        a.declare(m.X1, 1)
        a.declare(m.X2[1], 1)
        a.declare(m.e, 1)
        a.declare(m.E1, 1)
        a.declare(m.E2[1], 1)
        with self.assertRaises(TypeError):
            a.declare(m.b, None)
        a.declare(m.b, 1)
        a.declare(m.b_inactive, 1)
        a.declare(m.B, 2, derived=True)
        with self.assertRaises(TypeError):
            a.declare(None, 1)
        self.assertEqual(
            set([(v[0].name, v[1]) for v in a.expand_entries()]),
            set([('x', (1,False)), ('X1[1]', (1,False)), ('X2[1]', (1,False)),
                 ('e', (1,False)), ('E1[1]', (1,False)), ('E2[1]', (1,False)),
                 ('b.x', (1,False)), ('b.X1[1]', (1,False)), ('b.X2[1]', (1,False)),
                 ('b.e', (1,False)), ('b.E1[1]', (1,False)), ('b.E2[1]', (1,False)),
                 ('B[1].x', (2,True)), ('B[1].X1[1]', (2,True)), ('B[1].X2[1]', (2,True)),
                 ('B[1].e', (2,True)), ('B[1].E1[1]', (2,True)), ('B[1].E2[1]', (2,True))]))
        self.assertEqual(
            set([(v[0].name, v[1]) for v in a.expand_entries(expand_containers=False)]),
            set([('x', (1,False)), ('X1', (1,False)), ('X2[1]', (1,False)),
                 ('e', (1,False)), ('E1', (1,False)), ('E2[1]', (1,False)),
                 ('b.x', (1,False)), ('b.X1', (1,False)), ('b.X2', (1,False)),
                 ('b.e', (1,False)), ('b.E1', (1,False)), ('b.E2', (1,False)),
                 ('B[1].x', (2,True)), ('B[1].X1', (2,True)), ('B[1].X2', (2,True)),
                 ('B[1].e', (2,True)), ('B[1].E1', (2,True)), ('B[1].E2', (2,True))]))

    def test_constraint_stage(self):
        m = pyo.ConcreteModel()
        self._populate_block_with_constraints(m)
        m.b = pyo.Block()
        self._populate_block_with_constraints(m.b)
        m.b_inactive = pyo.Block()
        self._populate_block_with_constraints(m.b_inactive)
        m.b_inactive.deactivate()
        m.B = pyo.Block([1],
                        rule=lambda b: \
            self._populate_block_with_constraints(b))

        a = _ConstraintStageAnnotation()
        self.assertEqual(a.default, None)
        self.assertEqual(a.has_declarations, False)
        a.declare(m.c, 1)
        self.assertEqual(a.has_declarations, True)
        a.declare(m.C1, 1)
        a.declare(m.C2[1], 1)
        a.declare(m.C3, 1)
        with self.assertRaises(TypeError):
            a.declare(m.b, None)
        a.declare(m.b, 1)
        a.declare(m.b_inactive, 1)
        a.declare(m.B, 2)
        with self.assertRaises(TypeError):
            a.declare(None, 1)
        self.assertEqual(
            set([(v[0].name, v[1]) for v in a.expand_entries()]),
            set([('c', 1), ('C1[1]', 1), ('C2[1]', 1), ('C3[1]', 1),
                 ('b.c', 1), ('b.C1[1]', 1), ('b.C2[1]', 1), ('b.C3[1]', 1),
                 ('B[1].c', 2), ('B[1].C1[1]', 2), ('B[1].C2[1]', 2), ('B[1].C3[1]', 2)]))
        self.assertEqual(
            set([(v[0].name, v[1]) for v in a.expand_entries(expand_containers=False)]),
            set([('c', 1), ('C1', 1), ('C2[1]', 1), ('C3', 1),
                 ('b.c', 1), ('b.C1', 1), ('b.C2', 1), ('b.C3', 1),
                 ('B[1].c', 2), ('B[1].C1', 2), ('B[1].C2', 2), ('B[1].C3', 2)]))

    def test_stochastic_data(self):
        m = pyo.ConcreteModel()
        self._populate_block_with_params(m)
        m.b = pyo.Block()
        self._populate_block_with_params(m.b)
        m.b_inactive = pyo.Block()
        self._populate_block_with_params(m.b_inactive)
        m.b_inactive.deactivate()
        m.B = pyo.Block([1],
                        rule=lambda b: \
            self._populate_block_with_params(b))

        a = StochasticDataAnnotation()
        self.assertEqual(a.default, None)
        self.assertEqual(a.has_declarations, False)
        a.declare(m.p, 1)
        self.assertEqual(a.has_declarations, True)
        a.declare(m.P1, 1)
        a.declare(m.P2[1], 1)
        a.declare(m.b, 1)
        a.declare(m.b_inactive, 1)
        a.declare(m.B, 2)
        with self.assertRaises(TypeError):
            a.declare(None, 1)
        self.assertEqual(
            set([(v[0].name, v[1]) for v in a.expand_entries()]),
            set([('p', 1), ('P1[1]', 1), ('P2[1]', 1),
                 ('b.p', 1), ('b.P1[1]', 1), ('b.P2[1]', 1),
                 ('B[1].p', 2), ('B[1].P1[1]', 2), ('B[1].P2[1]', 2)]))
        self.assertEqual(
            set([(v[0].name, v[1]) for v in a.expand_entries(expand_containers=False)]),
            set([('p', 1), ('P1', 1), ('P2[1]', 1),
                 ('b.p', 1), ('b.P1', 1), ('b.P2', 1),
                 ('B[1].p', 2), ('B[1].P1', 2), ('B[1].P2', 2)]))

    def test_constraint_bounds(self):
        m = pyo.ConcreteModel()
        self._populate_block_with_constraints(m)
        m.b = pyo.Block()
        self._populate_block_with_constraints(m.b)
        m.b_inactive = pyo.Block()
        self._populate_block_with_constraints(m.b_inactive)
        m.b_inactive.deactivate()
        m.B = pyo.Block([1],
                        rule=lambda b: \
            self._populate_block_with_constraints(b))

        a = StochasticConstraintBoundsAnnotation()
        self.assertEqual(a.default, True)
        self.assertEqual(a.has_declarations, False)
        a.declare(m.c)
        self.assertEqual(a.has_declarations, True)
        a.declare(m.C1)
        a.declare(m.C2[1])
        a.declare(m.C3)
        a.declare(m.b)
        a.declare(m.b_inactive)
        a.declare(m.B, lb=False, ub=True)
        with self.assertRaises(TypeError):
            a.declare(None, 1)
        self.assertEqual(
            set([(v[0].name, v[1]) for v in a.expand_entries()]),
            set([('c', True), ('C1[1]', True), ('C2[1]', True), ('C3[1]', True),
                 ('b.c', True), ('b.C1[1]', True), ('b.C2[1]', True), ('b.C3[1]', True),
                 ('B[1].c', (False,True)), ('B[1].C1[1]', (False,True)),
                 ('B[1].C2[1]', (False,True)), ('B[1].C3[1]', (False,True))]))
        self.assertEqual(
            set([(v[0].name, v[1]) for v in a.expand_entries(expand_containers=False)]),
            set([('c', True), ('C1', True), ('C2[1]', True), ('C3', True),
                 ('b.c', True), ('b.C1', True), ('b.C2', True), ('b.C3', True),
                 ('B[1].c', (False,True)), ('B[1].C1', (False,True)),
                 ('B[1].C2', (False,True)), ('B[1].C3', (False,True))]))

    def test_stochastic_constraint_body(self):
        m = pyo.ConcreteModel()
        self._populate_block_with_constraints(m)
        m.b = pyo.Block()
        self._populate_block_with_constraints(m.b)
        m.b_inactive = pyo.Block()
        self._populate_block_with_constraints(m.b_inactive)
        m.b_inactive.deactivate()
        m.B = pyo.Block([1],
                        rule=lambda b: \
            self._populate_block_with_constraints(b))

        a = StochasticConstraintBodyAnnotation()
        self.assertEqual(a.default, None)
        self.assertEqual(a.has_declarations, False)
        a.declare(m.c)
        self.assertEqual(a.has_declarations, True)
        a.declare(m.C1)
        a.declare(m.C2[1])
        a.declare(m.C3)
        a.declare(m.b)
        a.declare(m.b_inactive)
        a.declare(m.B, variables=2)
        with self.assertRaises(TypeError):
            a.declare(None, 1)
        self.assertEqual(
            set([(v[0].name, v[1]) for v in a.expand_entries()]),
            set([('c', None), ('C1[1]', None), ('C2[1]', None), ('C3[1]', None),
                 ('b.c', None), ('b.C1[1]', None), ('b.C2[1]', None), ('b.C3[1]', None),
                 ('B[1].c', 2), ('B[1].C1[1]', 2), ('B[1].C2[1]', 2), ('B[1].C3[1]', 2)]))
        self.assertEqual(
            set([(v[0].name, v[1]) for v in a.expand_entries(expand_containers=False)]),
            set([('c', None), ('C1', None), ('C2[1]', None), ('C3', None),
                 ('b.c', None), ('b.C1', None), ('b.C2', None), ('b.C3', None),
                 ('B[1].c', 2), ('B[1].C1', 2), ('B[1].C2', 2), ('B[1].C3', 2)]))

    def test_stochastic_objective(self):
        m = pyo.ConcreteModel()
        self._populate_block_with_objectives(m)
        m.b = pyo.Block()
        self._populate_block_with_objectives(m.b)
        m.b_inactive = pyo.Block()
        self._populate_block_with_objectives(m.b_inactive)
        m.b_inactive.deactivate()
        m.B = pyo.Block([1],
                        rule=lambda b: \
            self._populate_block_with_objectives(b))

        a = StochasticObjectiveAnnotation()
        self.assertEqual(a.default, (None, True))
        self.assertEqual(a.has_declarations, False)
        a.declare(m.o)
        self.assertEqual(a.has_declarations, True)
        a.declare(m.O1)
        a.declare(m.O2[1])
        a.declare(m.b)
        a.declare(m.b_inactive)
        a.declare(m.B, variables=1, include_constant=False)
        with self.assertRaises(TypeError):
            a.declare(None, 1)
        self.assertEqual(
            set([(v[0].name, v[1]) for v in a.expand_entries()]),
            set([('o', (None,True)), ('O1[1]', (None,True)), ('O2[1]', (None,True)),
                 ('b.o', (None,True)), ('b.O1[1]', (None,True)), ('b.O2[1]', (None,True)),
                 ('B[1].o', (1,False)), ('B[1].O1[1]', (1,False)), ('B[1].O2[1]', (1,False))]))
        self.assertEqual(
            set([(v[0].name, v[1]) for v in a.expand_entries(expand_containers=False)]),
            set([('o', (None,True)), ('O1', (None,True)), ('O2[1]', (None,True)),
                 ('b.o', (None,True)), ('b.O1', (None,True)), ('b.O2', (None,True)),
                 ('B[1].o', (1,False)), ('B[1].O1', (1,False)), ('B[1].O2', (1,False))]))

    def test_stochastic_variable_bounds(self):
        m = pyo.ConcreteModel()
        self._populate_block_with_vars(m)
        m.b = pyo.Block()
        self._populate_block_with_vars(m.b)
        m.b_inactive = pyo.Block()
        self._populate_block_with_vars(m.b_inactive)
        m.b_inactive.deactivate()
        m.B = pyo.Block([1],
                        rule=lambda b: \
            self._populate_block_with_vars(b))

        a = StochasticVariableBoundsAnnotation()
        self.assertEqual(a.default, (True, True))
        self.assertEqual(a.has_declarations, False)
        a.declare(m.x)
        self.assertEqual(a.has_declarations, True)
        a.declare(m.X1)
        a.declare(m.X2[1])
        a.declare(m.b)
        a.declare(m.b_inactive)
        a.declare(m.B, lb=False, ub=True)
        with self.assertRaises(TypeError):
            a.declare(None, 1)
        self.assertEqual(
            set([(v[0].name, v[1]) for v in a.expand_entries()]),
            set([('x', (True,True)), ('X1[1]', (True,True)), ('X2[1]', (True,True)),
                 ('b.x', (True,True)), ('b.X1[1]', (True,True)), ('b.X2[1]', (True,True)),
                 ('B[1].x', (False,True)), ('B[1].X1[1]', (False,True)), ('B[1].X2[1]', (False,True))]))
        self.assertEqual(
            set([(v[0].name, v[1]) for v in a.expand_entries(expand_containers=False)]),
            set([('x', (True,True)), ('X1', (True,True)), ('X2[1]', (True,True)),
                 ('b.x', (True,True)), ('b.X1', (True,True)), ('b.X2', (True,True)),
                 ('B[1].x', (False,True)), ('B[1].X1', (False,True)), ('B[1].X2', (False,True))]))

if __name__ == "__main__":
    unittest.main()
