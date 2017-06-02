#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
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

import pyutilib.th as unittest

from pyomo.environ import *
import pyomo.opt

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
        return prefix+".mps.baseline", prefix+".mps.out"

    def _check_baseline(self, model, **kwds):
        baseline_fname, test_fname = self._get_fnames()
        self._cleanup(test_fname)
        io_options = {"symbolic_solver_labels": True}
        io_options.update(kwds)
        model.write(test_fname,
                    format="mps",
                    io_options=io_options)
        self.assertFileEqualsBaseline(
            test_fname,
            baseline_fname,
            delete=True)

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

        terms = [model.a, model.b, model.c,
                 (model.a, model.a), (model.b, model.b), (model.c, model.c),
                 (model.a, model.b), (model.a, model.c), (model.b, model.c)]
        model.obj = Objective(expr=self._gen_expression(terms))
        model.con = Constraint(expr=self._gen_expression(terms) <= 1)
        self._check_baseline(model)

    def test_column_ordering_quadratic(self):
        model = ConcreteModel()
        model.a = Var()
        model.b = Var()
        model.c = Var()

        terms = [model.a, model.b, model.c,
                 (model.a, model.a), (model.b, model.b), (model.c, model.c),
                 (model.a, model.b), (model.a, model.c), (model.b, model.c)]
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
        components["con3"] = Constraint(expr=0 <= model.a <= 1)
        components["con4"] = Constraint([1,2], rule=lambda m, i: model.a == i)

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
        components["con3"] = Constraint(expr=0 <= model.a <= 1)
        components["con4"] = Constraint([1,2], rule=lambda m, i: model.a == i)

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

if __name__ == "__main__":
    unittest.main()
