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
#
# Test the canonical expressions
#

import os
import random

from filecmp import cmp
import pyomo.common.unittest as unittest

from pyomo.environ import (
    ConcreteModel,
    Var,
    Objective,
    Constraint,
    ComponentMap,
    minimize,
    Binary,
    NonNegativeReals,
    NonNegativeIntegers,
)

thisdir = os.path.dirname(os.path.abspath(__file__))


class TestMPSOrdering(unittest.TestCase):
    def _cleanup(self, fname):
        try:
            os.remove(fname)
        except OSError:
            pass

    def _get_fnames(self):
        class_name, test_name = self.id().split('.')[-2:]
        prefix = os.path.join(thisdir, test_name.replace("test_", "", 1))
        return prefix + ".mps.baseline", prefix + ".mps.out"

    def _check_baseline(self, model, **kwds):
        int_marker = kwds.pop("int_marker", False)
        baseline_fname, test_fname = self._get_fnames()
        self._cleanup(test_fname)
        io_options = {"symbolic_solver_labels": True}
        io_options.update(kwds)
        model.write(
            test_fname, format="mps", io_options=io_options, int_marker=int_marker
        )

        self.assertTrue(
            cmp(test_fname, baseline_fname),
            msg="Files %s and %s differ" % (test_fname, baseline_fname),
        )
        self._cleanup(test_fname)

    # generates an expression in a randomized way so that
    # we can test for consistent ordering of expressions
    # in the MPS file
    def _gen_expression(self, terms):
        terms = list(terms)
        random.shuffle(terms)
        expr = 0.0
        for term in terms:
            if type(term) is tuple:
                prodterms = list(term)
                random.shuffle(prodterms)
                prodexpr = 1.0
                for x in prodterms:
                    prodexpr *= x
                expr += prodexpr
            else:
                expr += term
        return expr

    def test_no_column_ordering_quadratic(self):
        model = ConcreteModel()
        model.a = Var()
        model.b = Var()
        model.c = Var()

        terms = [
            model.a,
            model.b,
            model.c,
            (model.a, model.a),
            (model.b, model.b),
            (model.c, model.c),
            (model.a, model.b),
            (model.a, model.c),
            (model.b, model.c),
        ]
        model.obj = Objective(expr=self._gen_expression(terms))
        model.con = Constraint(expr=self._gen_expression(terms) <= 1)
        self._check_baseline(model)

    def test_column_ordering_quadratic(self):
        model = ConcreteModel()
        model.a = Var()
        model.b = Var()
        model.c = Var()

        terms = [
            model.a,
            model.b,
            model.c,
            (model.a, model.a),
            (model.b, model.b),
            (model.c, model.c),
            (model.a, model.b),
            (model.a, model.c),
            (model.b, model.c),
        ]
        model.obj = Objective(expr=self._gen_expression(terms))
        model.con = Constraint(expr=self._gen_expression(terms) <= 1)
        # reverse the symbolic ordering
        column_order = ComponentMap()
        column_order[model.a] = 2
        column_order[model.b] = 1
        column_order[model.c] = 0
        self._check_baseline(model, column_order=column_order)

    def test_no_column_ordering_linear(self):
        model = ConcreteModel()
        model.a = Var()
        model.b = Var()
        model.c = Var()

        terms = [model.a, model.b, model.c]
        model.obj = Objective(expr=self._gen_expression(terms))
        model.con = Constraint(expr=self._gen_expression(terms) <= 1)
        self._check_baseline(model)

    def test_column_ordering_linear(self):
        model = ConcreteModel()
        model.a = Var()
        model.b = Var()
        model.c = Var()

        terms = [model.a, model.b, model.c]
        model.obj = Objective(expr=self._gen_expression(terms))
        model.con = Constraint(expr=self._gen_expression(terms) <= 1)
        # reverse the symbolic ordering
        column_order = ComponentMap()
        column_order[model.a] = 2
        column_order[model.b] = 1
        column_order[model.c] = 0
        self._check_baseline(model, column_order=column_order)

    def test_no_row_ordering(self):
        model = ConcreteModel()
        model.a = Var()

        components = {}
        components["obj"] = Objective(expr=model.a)
        components["con1"] = Constraint(expr=model.a >= 0)
        components["con2"] = Constraint(expr=model.a <= 1)
        components["con3"] = Constraint(expr=(0, model.a, 1))
        components["con4"] = Constraint([1, 2], rule=lambda m, i: model.a == i)

        # add components in random order
        random_order = list(components.keys())
        random.shuffle(random_order)
        for key in random_order:
            model.add_component(key, components[key])

        self._check_baseline(model, file_determinism=2)

    def test_row_ordering(self):
        model = ConcreteModel()
        model.a = Var()

        components = {}
        components["obj"] = Objective(expr=model.a)
        components["con1"] = Constraint(expr=model.a >= 0)
        components["con2"] = Constraint(expr=model.a <= 1)
        components["con3"] = Constraint(expr=(0, model.a, 1))
        components["con4"] = Constraint([1, 2], rule=lambda m, i: model.a == i)

        # add components in random order
        random_order = list(components.keys())
        random.shuffle(random_order)
        for key in random_order:
            model.add_component(key, components[key])

        # reverse the symbol and index order
        row_order = ComponentMap()
        row_order[model.con1] = 100
        row_order[model.con2] = 2
        row_order[model.con3] = 1
        row_order[model.con4[1]] = 0
        row_order[model.con4[2]] = -1
        self._check_baseline(model, row_order=row_order)

    def test_knapsack_problem_binary_variable_declaration_with_marker(self):
        elements_size = [30, 24, 11, 35, 29, 8, 31, 18]
        elements_weight = [3, 2, 2, 4, 5, 4, 3, 1]
        capacity = 60

        model = ConcreteModel("knapsack problem")
        var_names = [f"{i + 1}" for i in range(len(elements_size))]

        model.x = Var(var_names, within=Binary)

        model.obj = Objective(
            expr=sum(
                model.x[var_names[i]] * elements_weight[i]
                for i in range(len(elements_size))
            ),
            sense=minimize,
            name="obj",
        )

        model.const1 = Constraint(
            expr=sum(
                model.x[var_names[i]] * elements_size[i]
                for i in range(len(elements_size))
            )
            >= capacity,
            name="const",
        )

        self._check_baseline(model, int_marker=True)

    def test_integer_variable_declaration_with_marker(self):
        model = ConcreteModel("Example-mix-integer-linear-problem")

        # Define the decision variables
        model.x1 = Var(within=NonNegativeIntegers)  # Integer variable
        model.x2 = Var(within=NonNegativeReals)  # Continuous variable

        # Define the objective function
        model.obj = Objective(expr=3 * model.x1 + 2 * model.x2, sense=minimize)

        # Define the constraints
        model.const1 = Constraint(expr=4 * model.x1 + 3 * model.x2 >= 10)
        model.const2 = Constraint(expr=model.x1 + 2 * model.x2 <= 7)

        self._check_baseline(model, int_marker=True)


if __name__ == "__main__":
    unittest.main()
