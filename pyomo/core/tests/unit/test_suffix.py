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
# Unit Tests for Suffix
#

import os
import itertools
import logging
import pickle
from os.path import abspath, dirname

currdir = dirname(abspath(__file__)) + os.sep

import pyomo.common.unittest as unittest
from pyomo.common.collections import ComponentMap
from pyomo.common.log import LoggingIntercept
from pyomo.core.base.suffix import (
    active_export_suffix_generator,
    export_suffix_generator,
    active_import_suffix_generator,
    import_suffix_generator,
    active_local_suffix_generator,
    local_suffix_generator,
    active_suffix_generator,
    suffix_generator,
    SuffixFinder,
)
from pyomo.environ import (
    ConcreteModel,
    Suffix,
    Var,
    Param,
    Set,
    Objective,
    Constraint,
    Block,
    sum_product,
)

from io import StringIO


def simple_con_rule(model, i):
    return model.x[i] == 1


def simple_obj_rule(model, i):
    return model.x[i]


class TestSuffixMethods(unittest.TestCase):
    def test_suffix_debug(self):
        with LoggingIntercept(level=logging.DEBUG) as OUT:
            m = ConcreteModel()
            m.s = Suffix()
            m.foo = Suffix(rule=[])
        print(OUT.getvalue())
        self.assertEqual(
            OUT.getvalue(),
            "Constructing ConcreteModel 'ConcreteModel', from data=None\n"
            "Constructing Suffix 'Suffix'\n"
            "Constructing AbstractSuffix 'foo' on [Model] from data=None\n"
            "Constructing Suffix 'foo'\n"
            "Constructed component ''[Model].foo'':\n"
            "foo : Direction=LOCAL, Datatype=FLOAT\n"
            "    Key : Value\n\n",
        )

    def test_suffix_rule(self):
        m = ConcreteModel()
        m.I = Set(initialize=[1, 2, 3])
        m.x = Var(m.I)
        m.y = Var(m.I)
        m.c = Constraint(m.I, rule=lambda m, i: m.x[i] >= i)
        m.d = Constraint(m.I, rule=lambda m, i: m.x[i] <= -i)

        _dict = {m.c[1]: 10, m.c[2]: 20, m.c[3]: 30, m.d: 100}
        m.suffix_dict = Suffix(initialize=_dict)
        self.assertEqual(len(m.suffix_dict), 6)
        self.assertEqual(m.suffix_dict[m.c[1]], 10)
        self.assertEqual(m.suffix_dict[m.c[2]], 20)
        self.assertEqual(m.suffix_dict[m.c[3]], 30)
        self.assertEqual(m.suffix_dict[m.d[1]], 100)
        self.assertEqual(m.suffix_dict[m.d[2]], 100)
        self.assertEqual(m.suffix_dict[m.d[3]], 100)

        # check double-construction
        _dict[m.c[1]] = 1000
        m.suffix_dict.construct()
        self.assertEqual(len(m.suffix_dict), 6)
        self.assertEqual(m.suffix_dict[m.c[1]], 10)

        m.suffix_cmap = Suffix(
            initialize=ComponentMap(
                [(m.x[1], 10), (m.x[2], 20), (m.x[3], 30), (m.y, 100)]
            )
        )
        self.assertEqual(len(m.suffix_dict), 6)
        self.assertEqual(m.suffix_cmap[m.x[1]], 10)
        self.assertEqual(m.suffix_cmap[m.x[2]], 20)
        self.assertEqual(m.suffix_cmap[m.x[3]], 30)
        self.assertEqual(m.suffix_cmap[m.y[1]], 100)
        self.assertEqual(m.suffix_cmap[m.y[2]], 100)
        self.assertEqual(m.suffix_cmap[m.y[3]], 100)

        m.suffix_list = Suffix(
            initialize=[(m.x[1], 10), (m.x[2], 20), (m.x[3], 30), (m.y, 100)]
        )
        self.assertEqual(len(m.suffix_dict), 6)
        self.assertEqual(m.suffix_list[m.x[1]], 10)
        self.assertEqual(m.suffix_list[m.x[2]], 20)
        self.assertEqual(m.suffix_list[m.x[3]], 30)
        self.assertEqual(m.suffix_list[m.y[1]], 100)
        self.assertEqual(m.suffix_list[m.y[2]], 100)
        self.assertEqual(m.suffix_list[m.y[3]], 100)

        def gen_init():
            yield (m.x[1], 10)
            yield (m.x[2], 20)
            yield (m.x[3], 30)
            yield (m.y, 100)

        m.suffix_generator = Suffix(initialize=gen_init())
        self.assertEqual(len(m.suffix_dict), 6)
        self.assertEqual(m.suffix_generator[m.x[1]], 10)
        self.assertEqual(m.suffix_generator[m.x[2]], 20)
        self.assertEqual(m.suffix_generator[m.x[3]], 30)
        self.assertEqual(m.suffix_generator[m.y[1]], 100)
        self.assertEqual(m.suffix_generator[m.y[2]], 100)
        self.assertEqual(m.suffix_generator[m.y[3]], 100)

        def genfcn_init(m, i):
            yield (m.x[1], 10)
            yield (m.x[2], 20)
            yield (m.x[3], 30)
            yield (m.y, 100)

        m.suffix_generator_fcn = Suffix(initialize=genfcn_init)
        self.assertEqual(len(m.suffix_dict), 6)
        self.assertEqual(m.suffix_generator_fcn[m.x[1]], 10)
        self.assertEqual(m.suffix_generator_fcn[m.x[2]], 20)
        self.assertEqual(m.suffix_generator_fcn[m.x[3]], 30)
        self.assertEqual(m.suffix_generator_fcn[m.y[1]], 100)
        self.assertEqual(m.suffix_generator_fcn[m.y[2]], 100)
        self.assertEqual(m.suffix_generator_fcn[m.y[3]], 100)

    # test import_enabled
    def test_import_enabled(self):
        model = ConcreteModel()
        model.test_implicit = Suffix()
        self.assertFalse(model.test_implicit.import_enabled())

        model.test_local = Suffix(direction=Suffix.LOCAL)
        self.assertFalse(model.test_local.import_enabled())

        model.test_out = Suffix(direction=Suffix.IMPORT)
        self.assertTrue(model.test_out.import_enabled())

        model.test_in = Suffix(direction=Suffix.EXPORT)
        self.assertFalse(model.test_in.import_enabled())

        model.test_inout = Suffix(direction=Suffix.IMPORT_EXPORT)
        self.assertTrue(model.test_inout.import_enabled())

    # test export_enabled
    def test_export_enabled(self):
        model = ConcreteModel()

        model.test_implicit = Suffix()
        self.assertFalse(model.test_implicit.export_enabled())

        model.test_local = Suffix(direction=Suffix.LOCAL)
        self.assertFalse(model.test_local.export_enabled())

        model.test_out = Suffix(direction=Suffix.IMPORT)
        self.assertFalse(model.test_out.export_enabled())

        model.test_in = Suffix(direction=Suffix.EXPORT)
        self.assertTrue(model.test_in.export_enabled())

        model.test_inout = Suffix(direction=Suffix.IMPORT_EXPORT)
        self.assertTrue(model.test_inout.export_enabled())

    # test set_value and getValue
    # and if Var arrays are correctly expanded
    def test_set_value_getValue_Var1(self):
        model = ConcreteModel()
        model.junk = Suffix()
        model.x = Var()
        model.X = Var([1, 2, 3], dense=True)

        model.junk.set_value(model.X, 1.0)
        model.junk.set_value(model.X[1], 2.0)

        self.assertEqual(model.junk.get(model.X), None)
        self.assertEqual(model.junk.get(model.X[1]), 2.0)
        self.assertEqual(model.junk.get(model.X[2]), 1.0)
        self.assertEqual(model.junk.get(model.x), None)

        model.junk.set_value(model.x, 3.0)
        model.junk.set_value(model.X[2], 3.0)

        self.assertEqual(model.junk.get(model.X), None)
        self.assertEqual(model.junk.get(model.X[1]), 2.0)
        self.assertEqual(model.junk.get(model.X[2]), 3.0)
        self.assertEqual(model.junk.get(model.x), 3.0)

        model.junk.set_value(model.X, 1.0, expand=False)

        self.assertEqual(model.junk.get(model.X), 1.0)

    # test set_value and getValue
    # and if Var arrays are correctly expanded
    def test_set_value_getValue_Var2(self):
        model = ConcreteModel()
        model.junk = Suffix()
        model.x = Var()
        model.X = Var([1, 2, 3], dense=True)

        model.X.set_suffix_value('junk', 1.0)
        model.X[1].set_suffix_value('junk', 2.0)

        self.assertEqual(model.X.get_suffix_value('junk'), None)
        self.assertEqual(model.X[1].get_suffix_value('junk'), 2.0)
        self.assertEqual(model.X[2].get_suffix_value('junk'), 1.0)
        self.assertEqual(model.x.get_suffix_value('junk'), None)

        model.x.set_suffix_value('junk', 3.0)
        model.X[2].set_suffix_value('junk', 3.0)

        self.assertEqual(model.X.get_suffix_value('junk'), None)
        self.assertEqual(model.X[1].get_suffix_value('junk'), 2.0)
        self.assertEqual(model.X[2].get_suffix_value('junk'), 3.0)
        self.assertEqual(model.x.get_suffix_value('junk'), 3.0)

        model.X.set_suffix_value('junk', 1.0, expand=False)

        self.assertEqual(model.X.get_suffix_value('junk'), 1.0)

    # test set_value and getValue
    # and if Var arrays are correctly expanded
    def test_set_value_getValue_Var3(self):
        model = ConcreteModel()
        model.junk = Suffix()
        model.x = Var()
        model.X = Var([1, 2, 3], dense=True)

        model.X.set_suffix_value(model.junk, 1.0)
        model.X[1].set_suffix_value(model.junk, 2.0)

        self.assertEqual(model.X.get_suffix_value(model.junk), None)
        self.assertEqual(model.X[1].get_suffix_value(model.junk), 2.0)
        self.assertEqual(model.X[2].get_suffix_value(model.junk), 1.0)
        self.assertEqual(model.x.get_suffix_value(model.junk), None)

        model.x.set_suffix_value(model.junk, 3.0)
        model.X[2].set_suffix_value(model.junk, 3.0)

        self.assertEqual(model.X.get_suffix_value(model.junk), None)
        self.assertEqual(model.X[1].get_suffix_value(model.junk), 2.0)
        self.assertEqual(model.X[2].get_suffix_value(model.junk), 3.0)
        self.assertEqual(model.x.get_suffix_value(model.junk), 3.0)

        model.X.set_suffix_value(model.junk, 1.0, expand=False)

        self.assertEqual(model.X.get_suffix_value(model.junk), 1.0)

    # test set_value and getValue
    # and if Constraint arrays are correctly expanded
    def test_set_value_getValue_Constraint1(self):
        model = ConcreteModel()
        model.junk = Suffix()
        model.x = Var()
        model.X = Var([1, 2, 3])
        model.c = Constraint(expr=model.x >= 1)
        model.C = Constraint([1, 2, 3], rule=lambda model, i: model.X[i] >= 1)

        model.junk.set_value(model.C, 1.0)
        model.junk.set_value(model.C[1], 2.0)

        self.assertEqual(model.junk.get(model.C), None)
        self.assertEqual(model.junk.get(model.C[1]), 2.0)
        self.assertEqual(model.junk.get(model.C[2]), 1.0)

        self.assertEqual(model.junk.get(model.c), None)

        model.junk.set_value(model.c, 3.0)
        model.junk.set_value(model.C[2], 3.0)

        self.assertEqual(model.junk.get(model.C), None)
        self.assertEqual(model.junk.get(model.C[1]), 2.0)
        self.assertEqual(model.junk.get(model.C[2]), 3.0)
        self.assertEqual(model.junk.get(model.c), 3.0)

        model.junk.set_value(model.C, 1.0, expand=False)

        self.assertEqual(model.junk.get(model.C), 1.0)

    # test set_value and getValue
    # and if Constraint arrays are correctly expanded
    def test_set_value_getValue_Constraint2(self):
        model = ConcreteModel()
        model.junk = Suffix()
        model.x = Var()
        model.X = Var([1, 2, 3])
        model.c = Constraint(expr=model.x >= 1)
        model.C = Constraint([1, 2, 3], rule=lambda model, i: model.X[i] >= 1)

        model.C.set_suffix_value('junk', 1.0)
        model.C[1].set_suffix_value('junk', 2.0)

        self.assertEqual(model.C.get_suffix_value('junk'), None)
        self.assertEqual(model.C[1].get_suffix_value('junk'), 2.0)
        self.assertEqual(model.C[2].get_suffix_value('junk'), 1.0)
        self.assertEqual(model.c.get_suffix_value('junk'), None)

        model.c.set_suffix_value('junk', 3.0)
        model.C[2].set_suffix_value('junk', 3.0)

        self.assertEqual(model.C.get_suffix_value('junk'), None)
        self.assertEqual(model.C[1].get_suffix_value('junk'), 2.0)
        self.assertEqual(model.C[2].get_suffix_value('junk'), 3.0)
        self.assertEqual(model.c.get_suffix_value('junk'), 3.0)

        model.C.set_suffix_value('junk', 1.0, expand=False)

        self.assertEqual(model.C.get_suffix_value('junk'), 1.0)

    # test set_value and getValue
    # and if Constraint arrays are correctly expanded
    def test_set_value_getValue_Constraint3(self):
        model = ConcreteModel()
        model.junk = Suffix()
        model.x = Var()
        model.X = Var([1, 2, 3])
        model.c = Constraint(expr=model.x >= 1)
        model.C = Constraint([1, 2, 3], rule=lambda model, i: model.X[i] >= 1)

        model.C.set_suffix_value(model.junk, 1.0)
        model.C[1].set_suffix_value(model.junk, 2.0)

        self.assertEqual(model.C.get_suffix_value(model.junk), None)
        self.assertEqual(model.C[1].get_suffix_value(model.junk), 2.0)
        self.assertEqual(model.C[2].get_suffix_value(model.junk), 1.0)
        self.assertEqual(model.c.get_suffix_value(model.junk), None)

        model.c.set_suffix_value(model.junk, 3.0)
        model.C[2].set_suffix_value(model.junk, 3.0)

        self.assertEqual(model.C.get_suffix_value(model.junk), None)
        self.assertEqual(model.C[1].get_suffix_value(model.junk), 2.0)
        self.assertEqual(model.C[2].get_suffix_value(model.junk), 3.0)
        self.assertEqual(model.c.get_suffix_value(model.junk), 3.0)

        model.C.set_suffix_value(model.junk, 1.0, expand=False)

        self.assertEqual(model.C.get_suffix_value(model.junk), 1.0)

    # test set_value and getValue
    # and if Objective arrays are correctly expanded
    def test_set_value_getValue_Objective1(self):
        model = ConcreteModel()
        model.junk = Suffix()
        model.x = Var()
        model.X = Var([1, 2, 3])
        model.obj = Objective(expr=sum_product(model.X) + model.x)
        model.OBJ = Objective([1, 2, 3], rule=lambda model, i: model.X[i])

        model.junk.set_value(model.OBJ, 1.0)
        model.junk.set_value(model.OBJ[1], 2.0)

        self.assertEqual(model.junk.get(model.OBJ), None)
        self.assertEqual(model.junk.get(model.OBJ[1]), 2.0)
        self.assertEqual(model.junk.get(model.OBJ[2]), 1.0)
        self.assertEqual(model.junk.get(model.obj), None)

        model.junk.set_value(model.obj, 3.0)
        model.junk.set_value(model.OBJ[2], 3.0)

        self.assertEqual(model.junk.get(model.OBJ), None)
        self.assertEqual(model.junk.get(model.OBJ[1]), 2.0)
        self.assertEqual(model.junk.get(model.OBJ[2]), 3.0)
        self.assertEqual(model.junk.get(model.obj), 3.0)

        model.junk.set_value(model.OBJ, 1.0, expand=False)

        self.assertEqual(model.junk.get(model.OBJ), 1.0)

    # test set_value and getValue
    # and if Objective arrays are correctly expanded
    def test_set_value_getValue_Objective2(self):
        model = ConcreteModel()
        model.junk = Suffix()
        model.x = Var()
        model.X = Var([1, 2, 3])
        model.obj = Objective(expr=sum_product(model.X) + model.x)
        model.OBJ = Objective([1, 2, 3], rule=lambda model, i: model.X[i])

        model.OBJ.set_suffix_value('junk', 1.0)
        model.OBJ[1].set_suffix_value('junk', 2.0)

        self.assertEqual(model.OBJ.get_suffix_value('junk'), None)
        self.assertEqual(model.OBJ[1].get_suffix_value('junk'), 2.0)
        self.assertEqual(model.OBJ[2].get_suffix_value('junk'), 1.0)
        self.assertEqual(model.obj.get_suffix_value('junk'), None)

        model.obj.set_suffix_value('junk', 3.0)
        model.OBJ[2].set_suffix_value('junk', 3.0)

        self.assertEqual(model.OBJ.get_suffix_value('junk'), None)
        self.assertEqual(model.OBJ[1].get_suffix_value('junk'), 2.0)
        self.assertEqual(model.OBJ[2].get_suffix_value('junk'), 3.0)
        self.assertEqual(model.obj.get_suffix_value('junk'), 3.0)

        model.OBJ.set_suffix_value('junk', 1.0, expand=False)

        self.assertEqual(model.OBJ.get_suffix_value('junk'), 1.0)

    # test set_value and getValue
    # and if Objective arrays are correctly expanded
    def test_set_value_getValue_Objective3(self):
        model = ConcreteModel()
        model.junk = Suffix()
        model.x = Var()
        model.X = Var([1, 2, 3])
        model.obj = Objective(expr=sum_product(model.X) + model.x)
        model.OBJ = Objective([1, 2, 3], rule=lambda model, i: model.X[i])

        model.OBJ.set_suffix_value(model.junk, 1.0)
        model.OBJ[1].set_suffix_value(model.junk, 2.0)

        self.assertEqual(model.OBJ.get_suffix_value(model.junk), None)
        self.assertEqual(model.OBJ[1].get_suffix_value(model.junk), 2.0)
        self.assertEqual(model.OBJ[2].get_suffix_value(model.junk), 1.0)
        self.assertEqual(model.obj.get_suffix_value(model.junk), None)

        model.obj.set_suffix_value(model.junk, 3.0)
        model.OBJ[2].set_suffix_value(model.junk, 3.0)

        self.assertEqual(model.OBJ.get_suffix_value(model.junk), None)
        self.assertEqual(model.OBJ[1].get_suffix_value(model.junk), 2.0)
        self.assertEqual(model.OBJ[2].get_suffix_value(model.junk), 3.0)
        self.assertEqual(model.obj.get_suffix_value(model.junk), 3.0)

        model.OBJ.set_suffix_value(model.junk, 1.0, expand=False)

        self.assertEqual(model.OBJ.get_suffix_value(model.junk), 1.0)

    # test set_value and getValue
    # and if mutable Param arrays are correctly expanded
    def test_set_value_getValue_mutableParam1(self):
        model = ConcreteModel()
        model.junk = Suffix()
        model.x = Var()
        model.X = Var([1, 2, 3])
        model.p = Param(initialize=1.0, mutable=True)
        model.P = Param([1, 2, 3], initialize=1.0, mutable=True)

        model.junk.set_value(model.P, 1.0)
        model.junk.set_value(model.P[1], 2.0)

        self.assertEqual(model.junk.get(model.P), None)
        self.assertEqual(model.junk.get(model.P[1]), 2.0)
        self.assertEqual(model.junk.get(model.P[2]), 1.0)
        self.assertEqual(model.junk.get(model.p), None)

        model.junk.set_value(model.p, 3.0)
        model.junk.set_value(model.P[2], 3.0)

        self.assertEqual(model.junk.get(model.P), None)
        self.assertEqual(model.junk.get(model.P[1]), 2.0)
        self.assertEqual(model.junk.get(model.P[2]), 3.0)
        self.assertEqual(model.junk.get(model.p), 3.0)

        model.junk.set_value(model.P, 1.0, expand=False)

        self.assertEqual(model.junk.get(model.P), 1.0)

    # test set_value and getValue
    # and if mutable Param arrays are correctly expanded
    def test_set_value_getValue_mutableParam2(self):
        model = ConcreteModel()
        model.junk = Suffix()
        model.x = Var()
        model.X = Var([1, 2, 3])
        model.p = Param(initialize=1.0, mutable=True)
        model.P = Param([1, 2, 3], initialize=1.0, mutable=True)

        model.P.set_suffix_value('junk', 1.0)
        model.P[1].set_suffix_value('junk', 2.0)

        self.assertEqual(model.P.get_suffix_value('junk'), None)
        self.assertEqual(model.P[1].get_suffix_value('junk'), 2.0)
        self.assertEqual(model.P[2].get_suffix_value('junk'), 1.0)
        self.assertEqual(model.p.get_suffix_value('junk'), None)

        model.p.set_suffix_value('junk', 3.0)
        model.P[2].set_suffix_value('junk', 3.0)

        self.assertEqual(model.P.get_suffix_value('junk'), None)
        self.assertEqual(model.P[1].get_suffix_value('junk'), 2.0)
        self.assertEqual(model.P[2].get_suffix_value('junk'), 3.0)
        self.assertEqual(model.p.get_suffix_value('junk'), 3.0)

        model.P.set_suffix_value('junk', 1.0, expand=False)

        self.assertEqual(model.P.get_suffix_value('junk'), 1.0)

    # test set_value and getValue
    # and if mutable Param arrays are correctly expanded
    def test_set_value_getValue_mutableParam3(self):
        model = ConcreteModel()
        model.junk = Suffix()
        model.x = Var()
        model.X = Var([1, 2, 3])
        model.p = Param(initialize=1.0, mutable=True)
        model.P = Param([1, 2, 3], initialize=1.0, mutable=True)

        model.P.set_suffix_value(model.junk, 1.0)
        model.P[1].set_suffix_value(model.junk, 2.0)

        self.assertEqual(model.P.get_suffix_value(model.junk), None)
        self.assertEqual(model.P[1].get_suffix_value(model.junk), 2.0)
        self.assertEqual(model.P[2].get_suffix_value(model.junk), 1.0)
        self.assertEqual(model.p.get_suffix_value(model.junk), None)

        model.p.set_suffix_value(model.junk, 3.0)
        model.P[2].set_suffix_value(model.junk, 3.0)

        self.assertEqual(model.P.get_suffix_value(model.junk), None)
        self.assertEqual(model.P[1].get_suffix_value(model.junk), 2.0)
        self.assertEqual(model.P[2].get_suffix_value(model.junk), 3.0)
        self.assertEqual(model.p.get_suffix_value(model.junk), 3.0)

        model.P.set_suffix_value(model.junk, 1.0, expand=False)

        self.assertEqual(model.P.get_suffix_value(model.junk), 1.0)

    # test set_value and getValue
    # and if immutable Param arrays are correctly expanded
    def test_set_value_getValue_immutableParam1(self):
        model = ConcreteModel()
        model.junk = Suffix()
        model.x = Var()
        model.X = Var([1, 2, 3])
        model.p = Param(initialize=1.0, mutable=False)
        model.P = Param([1, 2, 3], initialize=1.0, mutable=False)

        self.assertEqual(model.junk.get(model.P), None)

        model.junk.set_value(model.P, 1.0, expand=False)

        self.assertEqual(model.junk.get(model.P), 1.0)

    # test set_value and getValue
    # and if immutable Param arrays are correctly expanded
    def test_set_value_getValue_immutableParam2(self):
        model = ConcreteModel()
        model.junk = Suffix()
        model.x = Var()
        model.X = Var([1, 2, 3])
        model.p = Param(initialize=1.0, mutable=False)
        model.P = Param([1, 2, 3], initialize=1.0, mutable=False)

        self.assertEqual(model.P.get_suffix_value('junk'), None)

        model.P.set_suffix_value('junk', 1.0, expand=False)

        self.assertEqual(model.P.get_suffix_value('junk'), 1.0)

    # test set_value and getValue
    # and if immutable Param arrays are correctly expanded
    def test_set_value_getValue_immutableParam3(self):
        model = ConcreteModel()
        model.junk = Suffix()
        model.x = Var()
        model.X = Var([1, 2, 3])
        model.p = Param(initialize=1.0, mutable=False)
        model.P = Param([1, 2, 3], initialize=1.0, mutable=False)

        self.assertEqual(model.P.get_suffix_value(model.junk), None)

        model.P.set_suffix_value(model.junk, 1.0, expand=False)

        self.assertEqual(model.P.get_suffix_value(model.junk), 1.0)

    # test set_value and getValue
    # and if Set arrays are correctly expanded
    def test_set_value_getValue_Set1(self):
        model = ConcreteModel()
        model.junk = Suffix()
        model.x = Var()
        model.X = Var([1, 2, 3])
        model.s = Set(initialize=[1, 2, 3])
        model.S = Set([1, 2, 3], initialize={1: [1, 2, 3], 2: [1, 2, 3], 3: [1, 2, 3]})

        model.junk.set_value(model.S, 1.0)
        model.junk.set_value(model.S[1], 2.0)

        self.assertEqual(model.junk.get(model.S), None)
        self.assertEqual(model.junk.get(model.S[1]), 2.0)
        self.assertEqual(model.junk.get(model.S[2]), 1.0)
        self.assertEqual(model.junk.get(model.s), None)

        model.junk.set_value(model.s, 3.0)
        model.junk.set_value(model.S[2], 3.0)

        self.assertEqual(model.junk.get(model.S), None)
        self.assertEqual(model.junk.get(model.S[1]), 2.0)
        self.assertEqual(model.junk.get(model.S[2]), 3.0)
        self.assertEqual(model.junk.get(model.s), 3.0)

        model.junk.set_value(model.S, 1.0, expand=False)

        self.assertEqual(model.junk.get(model.S), 1.0)

    # test set_value and getValue
    # and if Set arrays are correctly expanded
    def test_set_value_getValue_Set2(self):
        model = ConcreteModel()
        model.junk = Suffix()
        model.x = Var()
        model.X = Var([1, 2, 3])
        model.s = Set(initialize=[1, 2, 3])
        model.S = Set([1, 2, 3], initialize={1: [1, 2, 3], 2: [1, 2, 3], 3: [1, 2, 3]})

        model.S.set_suffix_value('junk', 1.0)
        model.S[1].set_suffix_value('junk', 2.0)

        self.assertEqual(model.S.get_suffix_value('junk'), None)
        self.assertEqual(model.S[1].get_suffix_value('junk'), 2.0)
        self.assertEqual(model.S[2].get_suffix_value('junk'), 1.0)
        self.assertEqual(model.s.get_suffix_value('junk'), None)

        model.s.set_suffix_value('junk', 3.0)
        model.S[2].set_suffix_value('junk', 3.0)

        self.assertEqual(model.S.get_suffix_value('junk'), None)
        self.assertEqual(model.S[1].get_suffix_value('junk'), 2.0)
        self.assertEqual(model.S[2].get_suffix_value('junk'), 3.0)
        self.assertEqual(model.s.get_suffix_value('junk'), 3.0)

        model.S.set_suffix_value('junk', 1.0, expand=False)

        self.assertEqual(model.S.get_suffix_value('junk'), 1.0)

    # test set_value and getValue
    # and if Set arrays are correctly expanded
    def test_set_value_getValue_Set3(self):
        model = ConcreteModel()
        model.junk = Suffix()
        model.x = Var()
        model.X = Var([1, 2, 3])
        model.s = Set(initialize=[1, 2, 3])
        model.S = Set([1, 2, 3], initialize={1: [1, 2, 3], 2: [1, 2, 3], 3: [1, 2, 3]})

        model.S.set_suffix_value(model.junk, 1.0)
        model.S[1].set_suffix_value(model.junk, 2.0)

        self.assertEqual(model.S.get_suffix_value(model.junk), None)
        self.assertEqual(model.S[1].get_suffix_value(model.junk), 2.0)
        self.assertEqual(model.S[2].get_suffix_value(model.junk), 1.0)
        self.assertEqual(model.s.get_suffix_value(model.junk), None)

        model.s.set_suffix_value(model.junk, 3.0)
        model.S[2].set_suffix_value(model.junk, 3.0)

        self.assertEqual(model.S.get_suffix_value(model.junk), None)
        self.assertEqual(model.S[1].get_suffix_value(model.junk), 2.0)
        self.assertEqual(model.S[2].get_suffix_value(model.junk), 3.0)
        self.assertEqual(model.s.get_suffix_value(model.junk), 3.0)

        model.S.set_suffix_value(model.junk, 1.0, expand=False)

        self.assertEqual(model.S.get_suffix_value(model.junk), 1.0)

    # test set_value and getValue
    # and if Block arrays are correctly expanded
    def test_set_value_getValue_Block1(self):
        model = ConcreteModel()
        model.junk = Suffix()
        model.b = Block()
        model.B = Block([1, 2, 3])

        # make sure each BlockData gets constructed
        model.B[1].x = 1
        model.B[2].x = 2
        model.B[3].x = 3

        model.junk.set_value(model.B, 1.0)
        model.junk.set_value(model.B[1], 2.0)

        self.assertEqual(model.junk.get(model.B), None)
        self.assertEqual(model.junk.get(model.B[1]), 2.0)
        self.assertEqual(model.junk.get(model.B[2]), 1.0)
        self.assertEqual(model.junk.get(model.b), None)

        model.junk.set_value(model.b, 3.0)
        model.junk.set_value(model.B[2], 3.0)

        self.assertEqual(model.junk.get(model.B), None)
        self.assertEqual(model.junk.get(model.B[1]), 2.0)
        self.assertEqual(model.junk.get(model.B[2]), 3.0)
        self.assertEqual(model.junk.get(model.b), 3.0)

        model.junk.set_value(model.B, 1.0, expand=False)

        self.assertEqual(model.junk.get(model.B), 1.0)

    # test set_value and getValue
    # and if Block arrays are correctly expanded
    def test_set_value_getValue_Block2(self):
        model = ConcreteModel()
        model.junk = Suffix()
        model.b = Block()
        model.B = Block([1, 2, 3])

        # make sure each BlockData gets constructed
        model.B[1].x = 1
        model.B[2].x = 2
        model.B[3].x = 3

        model.B.set_suffix_value('junk', 1.0)
        model.B[1].set_suffix_value('junk', 2.0)

        self.assertEqual(model.B.get_suffix_value('junk'), None)
        self.assertEqual(model.B[1].get_suffix_value('junk'), 2.0)
        self.assertEqual(model.B[2].get_suffix_value('junk'), 1.0)
        self.assertEqual(model.b.get_suffix_value('junk'), None)

        model.b.set_suffix_value('junk', 3.0)
        model.B[2].set_suffix_value('junk', 3.0)

        self.assertEqual(model.B.get_suffix_value('junk'), None)
        self.assertEqual(model.B[1].get_suffix_value('junk'), 2.0)
        self.assertEqual(model.B[2].get_suffix_value('junk'), 3.0)
        self.assertEqual(model.b.get_suffix_value('junk'), 3.0)

        model.B.set_suffix_value('junk', 1.0, expand=False)

        self.assertEqual(model.B.get_suffix_value('junk'), 1.0)

    # test set_value and getValue
    # and if Block arrays are correctly expanded
    def test_set_value_getValue_Block3(self):
        model = ConcreteModel()
        model.junk = Suffix()
        model.b = Block()
        model.B = Block([1, 2, 3])

        # make sure each BlockData gets constructed
        model.B[1].x = 1
        model.B[2].x = 2
        model.B[3].x = 3

        model.B.set_suffix_value(model.junk, 1.0)
        model.B[1].set_suffix_value(model.junk, 2.0)

        self.assertEqual(model.B.get_suffix_value(model.junk), None)
        self.assertEqual(model.B[1].get_suffix_value(model.junk), 2.0)
        self.assertEqual(model.B[2].get_suffix_value(model.junk), 1.0)
        self.assertEqual(model.b.get_suffix_value(model.junk), None)

        model.b.set_suffix_value(model.junk, 3.0)
        model.B[2].set_suffix_value(model.junk, 3.0)

        self.assertEqual(model.B.get_suffix_value(model.junk), None)
        self.assertEqual(model.B[1].get_suffix_value(model.junk), 2.0)
        self.assertEqual(model.B[2].get_suffix_value(model.junk), 3.0)
        self.assertEqual(model.b.get_suffix_value(model.junk), 3.0)

        model.B.set_suffix_value(model.junk, 1.0, expand=False)

        self.assertEqual(model.B.get_suffix_value(model.junk), 1.0)

    # test set_value with no component argument
    def test_set_all_values1(self):
        model = ConcreteModel()
        model.junk = Suffix()
        model.x = Var()
        model.y = Var([1, 2, 3], dense=True)
        model.z = Var([1, 2, 3], dense=True)

        model.junk.set_value(model.y[2], 1.0)
        model.junk.set_value(model.z, 2.0)

        self.assertTrue(model.junk.get(model.x) is None)
        self.assertTrue(model.junk.get(model.y) is None)
        self.assertTrue(model.junk.get(model.y[1]) is None)
        self.assertEqual(model.junk.get(model.y[2]), 1.0)
        self.assertEqual(model.junk.get(model.z), None)
        self.assertEqual(model.junk.get(model.z[1]), 2.0)

        model.junk.set_all_values(3.0)

        self.assertTrue(model.junk.get(model.x) is None)
        self.assertTrue(model.junk.get(model.y) is None)
        self.assertTrue(model.junk.get(model.y[1]) is None)
        self.assertEqual(model.junk.get(model.y[2]), 3.0)
        self.assertEqual(model.junk.get(model.z), None)
        self.assertEqual(model.junk.get(model.z[1]), 3.0)

    # test set_value with no component argument
    def test_set_all_values2(self):
        model = ConcreteModel()
        model.junk = Suffix()
        model.x = Var()
        model.y = Var([1, 2, 3], dense=True)
        model.z = Var([1, 2, 3], dense=True)

        model.y[2].set_suffix_value('junk', 1.0)
        model.z.set_suffix_value('junk', 2.0)

        self.assertTrue(model.x.get_suffix_value('junk') is None)
        self.assertTrue(model.y.get_suffix_value('junk') is None)
        self.assertTrue(model.y[1].get_suffix_value('junk') is None)
        self.assertEqual(model.y[2].get_suffix_value('junk'), 1.0)
        self.assertEqual(model.z.get_suffix_value('junk'), None)
        self.assertEqual(model.z[1].get_suffix_value('junk'), 2.0)

        model.junk.set_all_values(3.0)

        self.assertTrue(model.x.get_suffix_value('junk') is None)
        self.assertTrue(model.y.get_suffix_value('junk') is None)
        self.assertTrue(model.y[1].get_suffix_value('junk') is None)
        self.assertEqual(model.y[2].get_suffix_value('junk'), 3.0)
        self.assertEqual(model.z.get_suffix_value('junk'), None)
        self.assertEqual(model.z[1].get_suffix_value('junk'), 3.0)

    # test set_value with no component argument
    def test_set_all_values3(self):
        model = ConcreteModel()
        model.junk = Suffix()
        model.x = Var()
        model.y = Var([1, 2, 3], dense=True)
        model.z = Var([1, 2, 3], dense=True)

        model.y[2].set_suffix_value(model.junk, 1.0)
        model.z.set_suffix_value(model.junk, 2.0)

        self.assertTrue(model.x.get_suffix_value(model.junk) is None)
        self.assertTrue(model.y.get_suffix_value(model.junk) is None)
        self.assertTrue(model.y[1].get_suffix_value(model.junk) is None)
        self.assertEqual(model.y[2].get_suffix_value(model.junk), 1.0)
        self.assertEqual(model.z.get_suffix_value(model.junk), None)
        self.assertEqual(model.z[1].get_suffix_value(model.junk), 2.0)

        model.junk.set_all_values(3.0)

        self.assertTrue(model.x.get_suffix_value(model.junk) is None)
        self.assertTrue(model.y.get_suffix_value(model.junk) is None)
        self.assertTrue(model.y[1].get_suffix_value(model.junk) is None)
        self.assertEqual(model.y[2].get_suffix_value(model.junk), 3.0)
        self.assertEqual(model.z.get_suffix_value(model.junk), None)
        self.assertEqual(model.z[1].get_suffix_value(model.junk), 3.0)

    # test update_values
    def test_update_values(self):
        model = ConcreteModel()
        model.junk = Suffix()
        model.x = Var()
        model.y = Var()
        model.z = Var([1, 2])
        model.junk.set_value(model.x, 0.0)
        self.assertEqual(model.junk.get(model.x), 0.0)
        self.assertEqual(model.junk.get(model.y), None)
        self.assertEqual(model.junk.get(model.z), None)
        self.assertEqual(model.junk.get(model.z[1]), None)
        self.assertEqual(model.junk.get(model.z[2]), None)
        model.junk.update_values([(model.x, 1.0), (model.y, 2.0), (model.z, 3.0)])
        self.assertEqual(model.junk.get(model.x), 1.0)
        self.assertEqual(model.junk.get(model.y), 2.0)
        self.assertEqual(model.junk.get(model.z), None)
        self.assertEqual(model.junk.get(model.z[1]), 3.0)
        self.assertEqual(model.junk.get(model.z[2]), 3.0)
        model.junk.clear()
        model.junk.update_values(
            [(model.x, 1.0), (model.y, 2.0), (model.z, 3.0)], expand=False
        )
        self.assertEqual(model.junk.get(model.x), 1.0)
        self.assertEqual(model.junk.get(model.y), 2.0)
        self.assertEqual(model.junk.get(model.z), 3.0)
        self.assertEqual(model.junk.get(model.z[1]), None)
        self.assertEqual(model.junk.get(model.z[2]), None)

    # test clear_value
    def test_clear_value(self):
        model = ConcreteModel()
        model.junk = Suffix()
        model.x = Var()
        model.y = Var([1, 2, 3], dense=True)
        model.z = Var([1, 2, 3], dense=True)

        model.junk.set_value(model.x, -1.0)
        model.junk.set_value(model.y, -2.0)
        model.junk.set_value(model.y[2], 1.0)
        model.junk.set_value(model.z, 2.0)
        model.junk.set_value(model.z[1], 4.0)

        self.assertEqual(model.junk.get(model.x), -1.0)
        self.assertEqual(model.junk.get(model.y), None)
        self.assertEqual(model.junk.get(model.y[1]), -2.0)
        self.assertEqual(model.junk.get(model.y[2]), 1.0)
        self.assertEqual(model.junk.get(model.z), None)
        self.assertEqual(model.junk.get(model.z[2]), 2.0)
        self.assertEqual(model.junk.get(model.z[1]), 4.0)

        model.junk.clear_value(model.y)

        self.assertEqual(model.junk.get(model.x), -1.0)
        self.assertEqual(model.junk.get(model.y), None)
        self.assertEqual(model.junk.get(model.y[1]), None)
        self.assertEqual(model.junk.get(model.y[2]), None)
        self.assertEqual(model.junk.get(model.z), None)
        self.assertEqual(model.junk.get(model.z[2]), 2.0)
        self.assertEqual(model.junk.get(model.z[1]), 4.0)

        model.junk.clear_value(model.x)

        self.assertEqual(model.junk.get(model.x), None)
        self.assertEqual(model.junk.get(model.y), None)
        self.assertEqual(model.junk.get(model.y[1]), None)
        self.assertEqual(model.junk.get(model.y[2]), None)
        self.assertEqual(model.junk.get(model.z), None)
        self.assertEqual(model.junk.get(model.z[2]), 2.0)
        self.assertEqual(model.junk.get(model.z[1]), 4.0)

        # Clearing a scalar that is not there does not raise an error
        model.junk.clear_value(model.x)

        self.assertEqual(model.junk.get(model.x), None)
        self.assertEqual(model.junk.get(model.y), None)
        self.assertEqual(model.junk.get(model.y[1]), None)
        self.assertEqual(model.junk.get(model.y[2]), None)
        self.assertEqual(model.junk.get(model.z), None)
        self.assertEqual(model.junk.get(model.z[2]), 2.0)
        self.assertEqual(model.junk.get(model.z[1]), 4.0)

        model.junk.clear_value(model.z[1])

        self.assertEqual(model.junk.get(model.x), None)
        self.assertEqual(model.junk.get(model.y), None)
        self.assertEqual(model.junk.get(model.y[1]), None)
        self.assertEqual(model.junk.get(model.y[2]), None)
        self.assertEqual(model.junk.get(model.z), None)
        self.assertEqual(model.junk.get(model.z[2]), 2.0)
        self.assertEqual(model.junk.get(model.z[1]), None)

        # Clearing an indexed component with missing indices does not raise an error
        model.junk.clear_value(model.z)

        self.assertEqual(model.junk.get(model.x), None)
        self.assertEqual(model.junk.get(model.y), None)
        self.assertEqual(model.junk.get(model.y[1]), None)
        self.assertEqual(model.junk.get(model.y[2]), None)
        self.assertEqual(model.junk.get(model.z), None)
        self.assertEqual(model.junk.get(model.z[2]), None)
        self.assertEqual(model.junk.get(model.z[1]), None)

    # test clear_value no args
    def test_clear_all_values(self):
        model = ConcreteModel()
        model.junk = Suffix()
        model.x = Var()
        model.y = Var([1, 2, 3], dense=True)
        model.z = Var([1, 2, 3], dense=True)

        model.junk.set_value(model.y[2], 1.0)
        model.junk.set_value(model.z, 2.0)

        self.assertTrue(model.junk.get(model.x) is None)
        self.assertTrue(model.junk.get(model.y) is None)
        self.assertTrue(model.junk.get(model.y[1]) is None)
        self.assertEqual(model.junk.get(model.y[2]), 1.0)
        self.assertEqual(model.junk.get(model.z), None)
        self.assertEqual(model.junk.get(model.z[1]), 2.0)

        model.junk.clear_all_values()

        self.assertTrue(model.junk.get(model.x) is None)
        self.assertTrue(model.junk.get(model.y) is None)
        self.assertTrue(model.junk.get(model.y[1]) is None)
        self.assertTrue(model.junk.get(model.y[2]) is None)
        self.assertTrue(model.junk.get(model.z) is None)
        self.assertTrue(model.junk.get(model.z[1]) is None)

    # test set_datatype and get_datatype
    def test_set_datatype_get_datatype(self):
        model = ConcreteModel()
        model.junk = Suffix(datatype=Suffix.FLOAT)
        self.assertEqual(model.junk.datatype, Suffix.FLOAT)
        model.junk.datatype = Suffix.INT
        self.assertEqual(model.junk.datatype, Suffix.INT)
        model.junk.datatype = None
        self.assertEqual(model.junk.datatype, None)
        model.junk.datatype = 'FLOAT'
        self.assertEqual(model.junk.datatype, Suffix.FLOAT)
        model.junk.datatype = 'INT'
        self.assertEqual(model.junk.datatype, Suffix.INT)
        model.junk.datatype = 4
        self.assertEqual(model.junk.datatype, Suffix.FLOAT)
        model.junk.datatype = 0
        self.assertEqual(model.junk.datatype, Suffix.INT)

        with LoggingIntercept() as LOG:
            model.junk.set_datatype(None)
        self.assertEqual(model.junk.datatype, None)
        self.assertRegex(
            LOG.getvalue().replace("\n", " "),
            "^DEPRECATED: Suffix.set_datatype is replaced with the "
            "Suffix.datatype property",
        )

        model.junk.datatype = 'FLOAT'
        with LoggingIntercept() as LOG:
            self.assertEqual(model.junk.get_datatype(), Suffix.FLOAT)
        self.assertRegex(
            LOG.getvalue().replace("\n", " "),
            "^DEPRECATED: Suffix.get_datatype is replaced with the "
            "Suffix.datatype property",
        )

        with self.assertRaisesRegex(ValueError, "1.0 is not a valid SuffixDataType"):
            model.junk.datatype = 1.0

    # test set_direction and get_direction
    def test_set_direction_get_direction(self):
        model = ConcreteModel()
        model.junk = Suffix(direction=Suffix.LOCAL)
        self.assertEqual(model.junk.direction, Suffix.LOCAL)
        model.junk.direction = Suffix.EXPORT
        self.assertEqual(model.junk.direction, Suffix.EXPORT)
        model.junk.direction = Suffix.IMPORT
        self.assertEqual(model.junk.direction, Suffix.IMPORT)
        model.junk.direction = Suffix.IMPORT_EXPORT
        self.assertEqual(model.junk.direction, Suffix.IMPORT_EXPORT)

        with LoggingIntercept() as LOG:
            model.junk.set_direction(1)
        self.assertEqual(model.junk.direction, Suffix.EXPORT)
        self.assertRegex(
            LOG.getvalue().replace("\n", " "),
            "^DEPRECATED: Suffix.set_direction is replaced with the "
            "Suffix.direction property",
        )

        model.junk.direction = 'IMPORT'
        with LoggingIntercept() as LOG:
            self.assertEqual(model.junk.get_direction(), Suffix.IMPORT)
        self.assertRegex(
            LOG.getvalue().replace("\n", " "),
            "^DEPRECATED: Suffix.get_direction is replaced with the "
            "Suffix.direction property",
        )

        with self.assertRaisesRegex(ValueError, "'a' is not a valid SuffixDirection"):
            model.junk.direction = 'a'
        # None is allowed for datatype, but not direction
        with self.assertRaisesRegex(ValueError, "None is not a valid SuffixDirection"):
            model.junk.direction = None

    # test __str__
    def test_str(self):
        model = ConcreteModel()
        model.junk = Suffix()
        self.assertEqual(model.junk.__str__(), "junk")

    # test pprint()
    def test_pprint(self):
        model = ConcreteModel()
        model.junk = Suffix(direction=Suffix.EXPORT)
        output = StringIO()
        model.junk.pprint(ostream=output)
        self.assertEqual(
            output.getvalue(),
            "junk : Direction=EXPORT, Datatype=FLOAT\n    Key : Value\n",
        )
        model.junk.direction = Suffix.IMPORT
        output = StringIO()
        model.junk.pprint(ostream=output)
        self.assertEqual(
            output.getvalue(),
            "junk : Direction=IMPORT, Datatype=FLOAT\n    Key : Value\n",
        )
        model.junk.direction = Suffix.LOCAL
        model.junk.datatype = None
        output = StringIO()
        model.junk.pprint(ostream=output)
        self.assertEqual(
            output.getvalue(),
            "junk : Direction=LOCAL, Datatype=None\n    Key : Value\n",
        )
        model.junk.direction = Suffix.IMPORT_EXPORT
        model.junk.datatype = Suffix.INT
        output = StringIO()
        model.junk.pprint(ostream=output)
        self.assertEqual(
            output.getvalue(),
            "junk : Direction=IMPORT_EXPORT, Datatype=INT\n    Key : Value\n",
        )
        output = StringIO()
        model.pprint(ostream=output)
        self.assertEqual(
            output.getvalue(),
            """1 Suffix Declarations
    junk : Direction=IMPORT_EXPORT, Datatype=INT
        Key : Value

1 Declarations: junk
""",
        )

    # test pprint(verbose=True)
    def test_pprint_verbose(self):
        model = ConcreteModel()
        model.junk = Suffix()
        model.s = Block()
        model.s.b = Block()
        model.s.B = Block([1, 2, 3])

        model.junk.set_value(model.s.B, 1.0)
        model.junk.set_value(model.s.B[1], 2.0)

        model.junk.set_value(model.s.b, 3.0)
        model.junk.set_value(model.s.B[2], 3.0)

        output = StringIO()
        model.junk.pprint(ostream=output, verbose=True)
        self.assertEqual(
            output.getvalue(),
            """junk : Direction=LOCAL, Datatype=FLOAT
    Key    : Value
    s.B[1] :   2.0
    s.B[2] :   3.0
    s.B[3] :   1.0
       s.b :   3.0
""",
        )

    def test_active_export_suffix_generator(self):
        model = ConcreteModel()
        model.junk_EXPORT_int = Suffix(direction=Suffix.EXPORT, datatype=Suffix.INT)
        model.junk_EXPORT_float = Suffix(direction=Suffix.EXPORT, datatype=Suffix.FLOAT)
        model.junk_IMPORT_EXPORT_float = Suffix(
            direction=Suffix.IMPORT_EXPORT, datatype=Suffix.FLOAT
        )
        model.junk_IMPORT = Suffix(direction=Suffix.IMPORT, datatype=None)
        model.junk_LOCAL = Suffix(direction=Suffix.LOCAL, datatype=None)

        suffixes = dict(active_export_suffix_generator(model))
        self.assertTrue('junk_EXPORT_int' in suffixes)
        self.assertTrue('junk_EXPORT_float' in suffixes)
        self.assertTrue('junk_IMPORT_EXPORT_float' in suffixes)
        self.assertTrue('junk_IMPORT' not in suffixes)
        self.assertTrue('junk_LOCAL' not in suffixes)

        model.junk_EXPORT_float.deactivate()
        suffixes = dict(active_export_suffix_generator(model))
        self.assertTrue('junk_EXPORT_int' in suffixes)
        self.assertTrue('junk_EXPORT_float' not in suffixes)
        self.assertTrue('junk_IMPORT_EXPORT_float' in suffixes)
        self.assertTrue('junk_IMPORT' not in suffixes)
        self.assertTrue('junk_LOCAL' not in suffixes)
        model.junk_EXPORT_float.activate()

        suffixes = dict(active_export_suffix_generator(model, datatype=Suffix.FLOAT))
        self.assertTrue('junk_EXPORT_int' not in suffixes)
        self.assertTrue('junk_EXPORT_float' in suffixes)
        self.assertTrue('junk_IMPORT_EXPORT_float' in suffixes)
        self.assertTrue('junk_IMPORT' not in suffixes)
        self.assertTrue('junk_LOCAL' not in suffixes)

        model.junk_EXPORT_float.deactivate()
        suffixes = dict(active_export_suffix_generator(model, datatype=Suffix.FLOAT))
        self.assertTrue('junk_EXPORT_int' not in suffixes)
        self.assertTrue('junk_EXPORT_float' not in suffixes)
        self.assertTrue('junk_IMPORT_EXPORT_float' in suffixes)
        self.assertTrue('junk_IMPORT' not in suffixes)
        self.assertTrue('junk_LOCAL' not in suffixes)

    def test_export_suffix_generator(self):
        model = ConcreteModel()
        model.junk_EXPORT_int = Suffix(direction=Suffix.EXPORT, datatype=Suffix.INT)
        model.junk_EXPORT_float = Suffix(direction=Suffix.EXPORT, datatype=Suffix.FLOAT)
        model.junk_IMPORT_EXPORT_float = Suffix(
            direction=Suffix.IMPORT_EXPORT, datatype=Suffix.FLOAT
        )
        model.junk_IMPORT = Suffix(direction=Suffix.IMPORT, datatype=None)
        model.junk_LOCAL = Suffix(direction=Suffix.LOCAL, datatype=None)

        suffixes = dict(export_suffix_generator(model))
        self.assertTrue('junk_EXPORT_int' in suffixes)
        self.assertTrue('junk_EXPORT_float' in suffixes)
        self.assertTrue('junk_IMPORT_EXPORT_float' in suffixes)
        self.assertTrue('junk_IMPORT' not in suffixes)
        self.assertTrue('junk_LOCAL' not in suffixes)

        model.junk_EXPORT_float.deactivate()
        suffixes = dict(export_suffix_generator(model))
        self.assertTrue('junk_EXPORT_int' in suffixes)
        self.assertTrue('junk_EXPORT_float' in suffixes)
        self.assertTrue('junk_IMPORT_EXPORT_float' in suffixes)
        self.assertTrue('junk_IMPORT' not in suffixes)
        self.assertTrue('junk_LOCAL' not in suffixes)
        model.junk_EXPORT_float.activate()

        suffixes = dict(export_suffix_generator(model, datatype=Suffix.FLOAT))
        self.assertTrue('junk_EXPORT_int' not in suffixes)
        self.assertTrue('junk_EXPORT_float' in suffixes)
        self.assertTrue('junk_IMPORT_EXPORT_float' in suffixes)
        self.assertTrue('junk_IMPORT' not in suffixes)
        self.assertTrue('junk_LOCAL' not in suffixes)

        model.junk_EXPORT_float.deactivate()
        suffixes = dict(export_suffix_generator(model, datatype=Suffix.FLOAT))
        self.assertTrue('junk_EXPORT_int' not in suffixes)
        self.assertTrue('junk_EXPORT_float' in suffixes)
        self.assertTrue('junk_IMPORT_EXPORT_float' in suffixes)
        self.assertTrue('junk_IMPORT' not in suffixes)
        self.assertTrue('junk_LOCAL' not in suffixes)

    def test_active_import_suffix_generator(self):
        model = ConcreteModel()
        model.junk_IMPORT_int = Suffix(direction=Suffix.IMPORT, datatype=Suffix.INT)
        model.junk_IMPORT_float = Suffix(direction=Suffix.IMPORT, datatype=Suffix.FLOAT)
        model.junk_IMPORT_EXPORT_float = Suffix(
            direction=Suffix.IMPORT_EXPORT, datatype=Suffix.FLOAT
        )
        model.junk_EXPORT = Suffix(direction=Suffix.EXPORT, datatype=None)
        model.junk_LOCAL = Suffix(direction=Suffix.LOCAL, datatype=None)

        suffixes = dict(active_import_suffix_generator(model))
        self.assertTrue('junk_IMPORT_int' in suffixes)
        self.assertTrue('junk_IMPORT_float' in suffixes)
        self.assertTrue('junk_IMPORT_EXPORT_float' in suffixes)
        self.assertTrue('junk_EXPORT' not in suffixes)
        self.assertTrue('junk_LOCAL' not in suffixes)

        model.junk_IMPORT_float.deactivate()
        suffixes = dict(active_import_suffix_generator(model))
        self.assertTrue('junk_IMPORT_int' in suffixes)
        self.assertTrue('junk_IMPORT_float' not in suffixes)
        self.assertTrue('junk_IMPORT_EXPORT_float' in suffixes)
        self.assertTrue('junk_EXPORT' not in suffixes)
        self.assertTrue('junk_LOCAL' not in suffixes)
        model.junk_IMPORT_float.activate()

        suffixes = dict(active_import_suffix_generator(model, datatype=Suffix.FLOAT))
        self.assertTrue('junk_IMPORT_int' not in suffixes)
        self.assertTrue('junk_IMPORT_float' in suffixes)
        self.assertTrue('junk_IMPORT_EXPORT_float' in suffixes)
        self.assertTrue('junk_EXPORT' not in suffixes)
        self.assertTrue('junk_LOCAL' not in suffixes)

        model.junk_IMPORT_float.deactivate()
        suffixes = dict(active_import_suffix_generator(model, datatype=Suffix.FLOAT))
        self.assertTrue('junk_IMPORT_int' not in suffixes)
        self.assertTrue('junk_IMPORT_float' not in suffixes)
        self.assertTrue('junk_IMPORT_EXPORT_float' in suffixes)
        self.assertTrue('junk_EXPORT' not in suffixes)
        self.assertTrue('junk_LOCAL' not in suffixes)

    def test_import_suffix_generator(self):
        model = ConcreteModel()
        model.junk_IMPORT_int = Suffix(direction=Suffix.IMPORT, datatype=Suffix.INT)
        model.junk_IMPORT_float = Suffix(direction=Suffix.IMPORT, datatype=Suffix.FLOAT)
        model.junk_IMPORT_EXPORT_float = Suffix(
            direction=Suffix.IMPORT_EXPORT, datatype=Suffix.FLOAT
        )
        model.junk_EXPORT = Suffix(direction=Suffix.EXPORT, datatype=None)
        model.junk_LOCAL = Suffix(direction=Suffix.LOCAL, datatype=None)

        suffixes = dict(import_suffix_generator(model))
        self.assertTrue('junk_IMPORT_int' in suffixes)
        self.assertTrue('junk_IMPORT_float' in suffixes)
        self.assertTrue('junk_IMPORT_EXPORT_float' in suffixes)
        self.assertTrue('junk_EXPORT' not in suffixes)
        self.assertTrue('junk_LOCAL' not in suffixes)

        model.junk_IMPORT_float.deactivate()
        suffixes = dict(import_suffix_generator(model))
        self.assertTrue('junk_IMPORT_int' in suffixes)
        self.assertTrue('junk_IMPORT_float' in suffixes)
        self.assertTrue('junk_IMPORT_EXPORT_float' in suffixes)
        self.assertTrue('junk_EXPORT' not in suffixes)
        self.assertTrue('junk_LOCAL' not in suffixes)
        model.junk_IMPORT_float.activate()

        suffixes = dict(import_suffix_generator(model, datatype=Suffix.FLOAT))
        self.assertTrue('junk_IMPORT_int' not in suffixes)
        self.assertTrue('junk_IMPORT_float' in suffixes)
        self.assertTrue('junk_IMPORT_EXPORT_float' in suffixes)
        self.assertTrue('junk_EXPORT' not in suffixes)
        self.assertTrue('junk_LOCAL' not in suffixes)

        model.junk_IMPORT_float.deactivate()
        suffixes = dict(import_suffix_generator(model, datatype=Suffix.FLOAT))
        self.assertTrue('junk_IMPORT_int' not in suffixes)
        self.assertTrue('junk_IMPORT_float' in suffixes)
        self.assertTrue('junk_IMPORT_EXPORT_float' in suffixes)
        self.assertTrue('junk_EXPORT' not in suffixes)
        self.assertTrue('junk_LOCAL' not in suffixes)

    def test_active_local_suffix_generator(self):
        model = ConcreteModel()
        model.junk_LOCAL_int = Suffix(direction=Suffix.LOCAL, datatype=Suffix.INT)
        model.junk_LOCAL_float = Suffix(direction=Suffix.LOCAL, datatype=Suffix.FLOAT)
        model.junk_IMPORT_EXPORT = Suffix(direction=Suffix.IMPORT_EXPORT, datatype=None)
        model.junk_EXPORT = Suffix(direction=Suffix.EXPORT, datatype=None)
        model.junk_IMPORT = Suffix(direction=Suffix.IMPORT, datatype=None)

        suffixes = dict(active_local_suffix_generator(model))
        self.assertTrue('junk_LOCAL_int' in suffixes)
        self.assertTrue('junk_LOCAL_float' in suffixes)
        self.assertTrue('junk_IMPORT_EXPORT' not in suffixes)
        self.assertTrue('junk_EXPORT' not in suffixes)
        self.assertTrue('junk_IMPORT' not in suffixes)

        model.junk_LOCAL_float.deactivate()
        suffixes = dict(active_local_suffix_generator(model))
        self.assertTrue('junk_LOCAL_int' in suffixes)
        self.assertTrue('junk_LOCAL_float' not in suffixes)
        self.assertTrue('junk_IMPORT_EXPORT' not in suffixes)
        self.assertTrue('junk_EXPORT' not in suffixes)
        self.assertTrue('junk_IMPORT' not in suffixes)
        model.junk_LOCAL_float.activate()

        suffixes = dict(active_local_suffix_generator(model, datatype=Suffix.FLOAT))
        self.assertTrue('junk_LOCAL_int' not in suffixes)
        self.assertTrue('junk_LOCAL_float' in suffixes)
        self.assertTrue('junk_IMPORT_EXPORT' not in suffixes)
        self.assertTrue('junk_EXPORT' not in suffixes)
        self.assertTrue('junk_IMPORT' not in suffixes)

        model.junk_LOCAL_float.deactivate()
        suffixes = dict(active_local_suffix_generator(model, datatype=Suffix.FLOAT))
        self.assertTrue('junk_LOCAL_int' not in suffixes)
        self.assertTrue('junk_LOCAL_float' not in suffixes)
        self.assertTrue('junk_IMPORT_EXPORT' not in suffixes)
        self.assertTrue('junk_EXPORT' not in suffixes)
        self.assertTrue('junk_IMPORT' not in suffixes)

    def test_local_suffix_generator(self):
        model = ConcreteModel()
        model.junk_LOCAL_int = Suffix(direction=Suffix.LOCAL, datatype=Suffix.INT)
        model.junk_LOCAL_float = Suffix(direction=Suffix.LOCAL, datatype=Suffix.FLOAT)
        model.junk_IMPORT_EXPORT = Suffix(direction=Suffix.IMPORT_EXPORT, datatype=None)
        model.junk_EXPORT = Suffix(direction=Suffix.EXPORT, datatype=None)
        model.junk_IMPORT = Suffix(direction=Suffix.IMPORT, datatype=None)

        suffixes = dict(local_suffix_generator(model))
        self.assertTrue('junk_LOCAL_int' in suffixes)
        self.assertTrue('junk_LOCAL_float' in suffixes)
        self.assertTrue('junk_IMPORT_EXPORT' not in suffixes)
        self.assertTrue('junk_EXPORT' not in suffixes)
        self.assertTrue('junk_IMPORT' not in suffixes)

        model.junk_LOCAL_float.deactivate()
        suffixes = dict(local_suffix_generator(model))
        self.assertTrue('junk_LOCAL_int' in suffixes)
        self.assertTrue('junk_LOCAL_float' in suffixes)
        self.assertTrue('junk_IMPORT_EXPORT' not in suffixes)
        self.assertTrue('junk_EXPORT' not in suffixes)
        self.assertTrue('junk_IMPORT' not in suffixes)
        model.junk_LOCAL_float.activate()

        suffixes = dict(local_suffix_generator(model, datatype=Suffix.FLOAT))
        self.assertTrue('junk_LOCAL_int' not in suffixes)
        self.assertTrue('junk_LOCAL_float' in suffixes)
        self.assertTrue('junk_IMPORT_EXPORT' not in suffixes)
        self.assertTrue('junk_EXPORT' not in suffixes)
        self.assertTrue('junk_IMPORT' not in suffixes)

        model.junk_LOCAL_float.deactivate()
        suffixes = dict(local_suffix_generator(model, datatype=Suffix.FLOAT))
        self.assertTrue('junk_LOCAL_int' not in suffixes)
        self.assertTrue('junk_LOCAL_float' in suffixes)
        self.assertTrue('junk_IMPORT_EXPORT' not in suffixes)
        self.assertTrue('junk_EXPORT' not in suffixes)
        self.assertTrue('junk_IMPORT' not in suffixes)

    def test_active_suffix_generator(self):
        model = ConcreteModel()
        model.junk_LOCAL_int = Suffix(direction=Suffix.LOCAL, datatype=Suffix.INT)
        model.junk_LOCAL_float = Suffix(direction=Suffix.LOCAL, datatype=Suffix.FLOAT)
        model.junk_IMPORT_EXPORT = Suffix(direction=Suffix.IMPORT_EXPORT, datatype=None)
        model.junk_EXPORT = Suffix(direction=Suffix.EXPORT, datatype=None)
        model.junk_IMPORT = Suffix(direction=Suffix.IMPORT, datatype=None)

        suffixes = dict(active_suffix_generator(model))
        self.assertTrue('junk_LOCAL_int' in suffixes)
        self.assertTrue('junk_LOCAL_float' in suffixes)
        self.assertTrue('junk_IMPORT_EXPORT' in suffixes)
        self.assertTrue('junk_EXPORT' in suffixes)
        self.assertTrue('junk_IMPORT' in suffixes)

        model.junk_LOCAL_float.deactivate()
        suffixes = dict(active_suffix_generator(model))
        self.assertTrue('junk_LOCAL_int' in suffixes)
        self.assertTrue('junk_LOCAL_float' not in suffixes)
        self.assertTrue('junk_IMPORT_EXPORT' in suffixes)
        self.assertTrue('junk_EXPORT' in suffixes)
        self.assertTrue('junk_IMPORT' in suffixes)
        model.junk_LOCAL_float.activate()

        suffixes = dict(active_suffix_generator(model, datatype=Suffix.FLOAT))
        self.assertTrue('junk_LOCAL_int' not in suffixes)
        self.assertTrue('junk_LOCAL_float' in suffixes)
        self.assertTrue('junk_IMPORT_EXPORT' not in suffixes)
        self.assertTrue('junk_EXPORT' not in suffixes)
        self.assertTrue('junk_IMPORT' not in suffixes)

        model.junk_LOCAL_float.deactivate()
        suffixes = dict(active_suffix_generator(model, datatype=Suffix.FLOAT))
        self.assertTrue('junk_LOCAL_int' not in suffixes)
        self.assertTrue('junk_LOCAL_float' not in suffixes)
        self.assertTrue('junk_IMPORT_EXPORT' not in suffixes)
        self.assertTrue('junk_EXPORT' not in suffixes)
        self.assertTrue('junk_IMPORT' not in suffixes)

    def test_suffix_generator(self):
        model = ConcreteModel()
        model.junk_LOCAL_int = Suffix(direction=Suffix.LOCAL, datatype=Suffix.INT)
        model.junk_LOCAL_float = Suffix(direction=Suffix.LOCAL, datatype=Suffix.FLOAT)
        model.junk_IMPORT_EXPORT = Suffix(direction=Suffix.IMPORT_EXPORT, datatype=None)
        model.junk_EXPORT = Suffix(direction=Suffix.EXPORT, datatype=None)
        model.junk_IMPORT = Suffix(direction=Suffix.IMPORT, datatype=None)

        suffixes = dict(suffix_generator(model))
        self.assertTrue('junk_LOCAL_int' in suffixes)
        self.assertTrue('junk_LOCAL_float' in suffixes)
        self.assertTrue('junk_IMPORT_EXPORT' in suffixes)
        self.assertTrue('junk_EXPORT' in suffixes)
        self.assertTrue('junk_IMPORT' in suffixes)

        model.junk_LOCAL_float.deactivate()
        suffixes = dict(suffix_generator(model))
        self.assertTrue('junk_LOCAL_int' in suffixes)
        self.assertTrue('junk_LOCAL_float' in suffixes)
        self.assertTrue('junk_IMPORT_EXPORT' in suffixes)
        self.assertTrue('junk_EXPORT' in suffixes)
        self.assertTrue('junk_IMPORT' in suffixes)
        model.junk_LOCAL_float.activate()

        suffixes = dict(suffix_generator(model, datatype=Suffix.FLOAT))
        self.assertTrue('junk_LOCAL_int' not in suffixes)
        self.assertTrue('junk_LOCAL_float' in suffixes)
        self.assertTrue('junk_IMPORT_EXPORT' not in suffixes)
        self.assertTrue('junk_EXPORT' not in suffixes)
        self.assertTrue('junk_IMPORT' not in suffixes)

        model.junk_LOCAL_float.deactivate()
        suffixes = dict(suffix_generator(model, datatype=Suffix.FLOAT))
        self.assertTrue('junk_LOCAL_int' not in suffixes)
        self.assertTrue('junk_LOCAL_float' in suffixes)
        self.assertTrue('junk_IMPORT_EXPORT' not in suffixes)
        self.assertTrue('junk_EXPORT' not in suffixes)
        self.assertTrue('junk_IMPORT' not in suffixes)


class TestSuffixCloneUsage(unittest.TestCase):
    def test_clone_VarElement(self):
        model = ConcreteModel()
        model.x = Var()
        model.junk = Suffix()
        self.assertEqual(model.junk.get(model.x), None)
        model.junk.set_value(model.x, 1.0)
        self.assertEqual(model.junk.get(model.x), 1.0)
        inst = model.clone()
        self.assertEqual(inst.junk.get(model.x), None)
        self.assertEqual(inst.junk.get(inst.x), 1.0)

    def test_clone_VarArray(self):
        model = ConcreteModel()
        model.x = Var([1, 2, 3], dense=True)
        model.junk = Suffix()
        self.assertEqual(model.junk.get(model.x), None)
        self.assertEqual(model.junk.get(model.x[1]), None)
        model.junk.set_value(model.x, 1.0)
        self.assertEqual(model.junk.get(model.x), None)
        self.assertEqual(model.junk.get(model.x[1]), 1.0)
        inst = model.clone()
        self.assertEqual(inst.junk.get(model.x[1]), None)
        self.assertEqual(inst.junk.get(inst.x[1]), 1.0)

    def test_clone_VarData(self):
        model = ConcreteModel()
        model.x = Var([1, 2, 3], dense=True)
        model.junk = Suffix()
        self.assertEqual(model.junk.get(model.x[1]), None)
        model.junk.set_value(model.x[1], 1.0)
        self.assertEqual(model.junk.get(model.x[1]), 1.0)
        inst = model.clone()
        self.assertEqual(inst.junk.get(model.x[1]), None)
        self.assertEqual(inst.junk.get(inst.x[1]), 1.0)

    def test_clone_ConstraintElement(self):
        model = ConcreteModel()
        model.x = Var()
        model.c = Constraint(expr=model.x == 1.0)
        model.junk = Suffix()
        self.assertEqual(model.junk.get(model.c), None)
        model.junk.set_value(model.c, 1.0)
        self.assertEqual(model.junk.get(model.c), 1.0)
        inst = model.clone()
        self.assertEqual(inst.junk.get(model.c), None)
        self.assertEqual(inst.junk.get(inst.c), 1.0)

    def test_clone_ConstraintArray(self):
        model = ConcreteModel()
        model.x = Var([1, 2, 3], dense=True)
        model.c = Constraint([1, 2, 3], rule=lambda model, i: model.x[i] == 1.0)
        model.junk = Suffix()
        self.assertEqual(model.junk.get(model.c), None)
        self.assertEqual(model.junk.get(model.c[1]), None)
        model.junk.set_value(model.c, 1.0)
        self.assertEqual(model.junk.get(model.c), None)
        self.assertEqual(model.junk.get(model.c[1]), 1.0)
        inst = model.clone()
        self.assertEqual(inst.junk.get(model.c[1]), None)
        self.assertEqual(inst.junk.get(inst.c[1]), 1.0)

    def test_clone_ConstraintData(self):
        model = ConcreteModel()
        model.x = Var([1, 2, 3], dense=True)
        model.c = Constraint([1, 2, 3], rule=lambda model, i: model.x[i] == 1.0)
        model.junk = Suffix()
        self.assertEqual(model.junk.get(model.c[1]), None)
        model.junk.set_value(model.c[1], 1.0)
        self.assertEqual(model.junk.get(model.c[1]), 1.0)
        inst = model.clone()
        self.assertEqual(inst.junk.get(model.c[1]), None)
        self.assertEqual(inst.junk.get(inst.c[1]), 1.0)

    def test_clone_ObjectiveElement(self):
        model = ConcreteModel()
        model.x = Var()
        model.obj = Objective(expr=model.x)
        model.junk = Suffix()
        self.assertEqual(model.junk.get(model.obj), None)
        model.junk.set_value(model.obj, 1.0)
        self.assertEqual(model.junk.get(model.obj), 1.0)
        inst = model.clone()
        self.assertEqual(inst.junk.get(model.obj), None)
        self.assertEqual(inst.junk.get(inst.obj), 1.0)

    def test_clone_ObjectiveArray(self):
        model = ConcreteModel()
        model.x = Var([1, 2, 3], dense=True)
        model.obj = Objective([1, 2, 3], rule=lambda model, i: model.x[i])
        model.junk = Suffix()
        self.assertEqual(model.junk.get(model.obj), None)
        self.assertEqual(model.junk.get(model.obj[1]), None)
        model.junk.set_value(model.obj, 1.0)
        self.assertEqual(model.junk.get(model.obj), None)
        self.assertEqual(model.junk.get(model.obj[1]), 1.0)
        inst = model.clone()
        self.assertEqual(inst.junk.get(model.obj[1]), None)
        self.assertEqual(inst.junk.get(inst.obj[1]), 1.0)

    def test_clone_ObjectiveData(self):
        model = ConcreteModel()
        model.x = Var([1, 2, 3], dense=True)
        model.obj = Objective([1, 2, 3], rule=lambda model, i: model.x[i])
        model.junk = Suffix()
        self.assertEqual(model.junk.get(model.obj[1]), None)
        model.junk.set_value(model.obj[1], 1.0)
        self.assertEqual(model.junk.get(model.obj[1]), 1.0)
        inst = model.clone()
        self.assertEqual(inst.junk.get(model.obj[1]), None)
        self.assertEqual(inst.junk.get(inst.obj[1]), 1.0)

    def test_clone_SimpleBlock(self):
        model = ConcreteModel()
        model.b = Block()
        model.junk = Suffix()
        self.assertEqual(model.junk.get(model.b), None)
        model.junk.set_value(model.b, 1.0)
        self.assertEqual(model.junk.get(model.b), 1.0)
        inst = model.clone()
        self.assertEqual(inst.junk.get(model.b), None)
        self.assertEqual(inst.junk.get(inst.b), 1.0)

    def test_clone_IndexedBlock(self):
        model = ConcreteModel()
        model.b = Block([1, 2, 3])
        model.junk = Suffix()
        self.assertEqual(model.junk.get(model.b), None)
        self.assertEqual(model.junk.get(model.b[1]), None)
        model.junk.set_value(model.b, 1.0)
        self.assertEqual(model.junk.get(model.b), None)
        self.assertEqual(model.junk.get(model.b[1]), 1.0)
        inst = model.clone()
        self.assertEqual(inst.junk.get(model.b[1]), None)
        self.assertEqual(inst.junk.get(inst.b[1]), 1.0)

    def test_clone_BlockData(self):
        model = ConcreteModel()
        model.b = Block([1, 2, 3])
        model.junk = Suffix()
        self.assertEqual(model.junk.get(model.b[1]), None)
        model.junk.set_value(model.b[1], 1.0)
        self.assertEqual(model.junk.get(model.b[1]), 1.0)
        inst = model.clone()
        self.assertEqual(inst.junk.get(model.b[1]), None)
        self.assertEqual(inst.junk.get(inst.b[1]), 1.0)

    def test_clone_model(self):
        model = ConcreteModel()
        model.junk = Suffix()
        self.assertEqual(model.junk.get(model), None)
        model.junk.set_value(model, 1.0)
        self.assertEqual(model.junk.get(model), 1.0)
        inst = model.clone()
        self.assertEqual(inst.junk.get(model), None)
        self.assertEqual(inst.junk.get(inst), 1.0)


class TestSuffixPickleUsage(unittest.TestCase):
    def test_pickle_VarElement(self):
        model = ConcreteModel()
        model.x = Var()
        model.junk = Suffix()
        self.assertEqual(model.junk.get(model.x), None)
        model.junk.set_value(model.x, 1.0)
        self.assertEqual(model.junk.get(model.x), 1.0)
        inst = pickle.loads(pickle.dumps(model))
        self.assertEqual(inst.junk.get(model.x), None)
        self.assertEqual(inst.junk.get(inst.x), 1.0)

    def test_pickle_VarArray(self):
        model = ConcreteModel()
        model.x = Var([1, 2, 3], dense=True)
        model.junk = Suffix()
        self.assertEqual(model.junk.get(model.x), None)
        self.assertEqual(model.junk.get(model.x[1]), None)
        model.junk.set_value(model.x, 1.0)
        self.assertEqual(model.junk.get(model.x), None)
        self.assertEqual(model.junk.get(model.x[1]), 1.0)
        inst = pickle.loads(pickle.dumps(model))
        self.assertEqual(inst.junk.get(model.x[1]), None)
        self.assertEqual(inst.junk.get(inst.x[1]), 1.0)

    def test_pickle_VarData(self):
        model = ConcreteModel()
        model.x = Var([1, 2, 3], dense=True)
        model.junk = Suffix()
        self.assertEqual(model.junk.get(model.x[1]), None)
        model.junk.set_value(model.x[1], 1.0)
        self.assertEqual(model.junk.get(model.x[1]), 1.0)
        inst = pickle.loads(pickle.dumps(model))
        self.assertEqual(inst.junk.get(model.x[1]), None)
        self.assertEqual(inst.junk.get(inst.x[1]), 1.0)

    def test_pickle_ConstraintElement(self):
        model = ConcreteModel()
        model.x = Var()
        model.c = Constraint(expr=model.x == 1.0)
        model.junk = Suffix()
        self.assertEqual(model.junk.get(model.c), None)
        model.junk.set_value(model.c, 1.0)
        self.assertEqual(model.junk.get(model.c), 1.0)
        inst = pickle.loads(pickle.dumps(model))
        self.assertEqual(inst.junk.get(model.c), None)
        self.assertEqual(inst.junk.get(inst.c), 1.0)

    def test_pickle_ConstraintArray(self):
        model = ConcreteModel()
        model.x = Var([1, 2, 3], dense=True)
        model.c = Constraint([1, 2, 3], rule=simple_con_rule)
        model.junk = Suffix()
        self.assertEqual(model.junk.get(model.c), None)
        self.assertEqual(model.junk.get(model.c[1]), None)
        model.junk.set_value(model.c, 1.0)
        self.assertEqual(model.junk.get(model.c), None)
        self.assertEqual(model.junk.get(model.c[1]), 1.0)
        inst = pickle.loads(pickle.dumps(model))
        self.assertEqual(inst.junk.get(model.c[1]), None)
        self.assertEqual(inst.junk.get(inst.c[1]), 1.0)

    def test_pickle_ConstraintData(self):
        model = ConcreteModel()
        model.x = Var([1, 2, 3], dense=True)
        model.c = Constraint([1, 2, 3], rule=simple_con_rule)
        model.junk = Suffix()
        self.assertEqual(model.junk.get(model.c[1]), None)
        model.junk.set_value(model.c[1], 1.0)
        self.assertEqual(model.junk.get(model.c[1]), 1.0)
        inst = pickle.loads(pickle.dumps(model))
        self.assertEqual(inst.junk.get(model.c[1]), None)
        self.assertEqual(inst.junk.get(inst.c[1]), 1.0)

    def test_pickle_ObjectiveElement(self):
        model = ConcreteModel()
        model.x = Var()
        model.obj = Objective(expr=model.x)
        model.junk = Suffix()
        self.assertEqual(model.junk.get(model.obj), None)
        model.junk.set_value(model.obj, 1.0)
        self.assertEqual(model.junk.get(model.obj), 1.0)
        inst = pickle.loads(pickle.dumps(model))
        self.assertEqual(inst.junk.get(model.obj), None)
        self.assertEqual(inst.junk.get(inst.obj), 1.0)

    def test_pickle_ObjectiveArray(self):
        model = ConcreteModel()
        model.x = Var([1, 2, 3], dense=True)
        model.obj = Objective([1, 2, 3], rule=simple_obj_rule)
        model.junk = Suffix()
        self.assertEqual(model.junk.get(model.obj), None)
        self.assertEqual(model.junk.get(model.obj[1]), None)
        model.junk.set_value(model.obj, 1.0)
        self.assertEqual(model.junk.get(model.obj), None)
        self.assertEqual(model.junk.get(model.obj[1]), 1.0)
        inst = pickle.loads(pickle.dumps(model))
        self.assertEqual(inst.junk.get(model.obj[1]), None)
        self.assertEqual(inst.junk.get(inst.obj[1]), 1.0)

    def test_pickle_ObjectiveData(self):
        model = ConcreteModel()
        model.x = Var([1, 2, 3], dense=True)
        model.obj = Objective([1, 2, 3], rule=simple_obj_rule)
        model.junk = Suffix()
        self.assertEqual(model.junk.get(model.obj[1]), None)
        model.junk.set_value(model.obj[1], 1.0)
        self.assertEqual(model.junk.get(model.obj[1]), 1.0)
        inst = pickle.loads(pickle.dumps(model))
        self.assertEqual(inst.junk.get(model.obj[1]), None)
        self.assertEqual(inst.junk.get(inst.obj[1]), 1.0)

    def test_pickle_SimpleBlock(self):
        model = ConcreteModel()
        model.b = Block()
        model.junk = Suffix()
        self.assertEqual(model.junk.get(model.b), None)
        model.junk.set_value(model.b, 1.0)
        self.assertEqual(model.junk.get(model.b), 1.0)
        inst = pickle.loads(pickle.dumps(model))
        self.assertEqual(inst.junk.get(model.b), None)
        self.assertEqual(inst.junk.get(inst.b), 1.0)

    def test_pickle_IndexedBlock(self):
        model = ConcreteModel()
        model.b = Block([1, 2, 3])
        model.junk = Suffix()
        self.assertEqual(model.junk.get(model.b), None)
        self.assertEqual(model.junk.get(model.b[1]), None)
        model.junk.set_value(model.b, 1.0)
        self.assertEqual(model.junk.get(model.b), None)
        self.assertEqual(model.junk.get(model.b[1]), 1.0)
        inst = pickle.loads(pickle.dumps(model))
        self.assertEqual(inst.junk.get(model.b[1]), None)
        self.assertEqual(inst.junk.get(inst.b[1]), 1.0)

    def test_pickle_BlockData(self):
        model = ConcreteModel()
        model.b = Block([1, 2, 3])
        model.junk = Suffix()
        self.assertEqual(model.junk.get(model.b[1]), None)
        model.junk.set_value(model.b[1], 1.0)
        self.assertEqual(model.junk.get(model.b[1]), 1.0)
        inst = pickle.loads(pickle.dumps(model))
        self.assertEqual(inst.junk.get(model.b[1]), None)
        self.assertEqual(inst.junk.get(inst.b[1]), 1.0)

    def test_pickle_model(self):
        model = ConcreteModel()
        model.junk = Suffix()
        self.assertEqual(model.junk.get(model), None)
        model.junk.set_value(model, 1.0)
        self.assertEqual(model.junk.get(model), 1.0)
        inst = pickle.loads(pickle.dumps(model))
        self.assertEqual(inst.junk.get(model), None)
        self.assertEqual(inst.junk.get(inst), 1.0)


class TestSuffixFinder(unittest.TestCase):
    def test_suffix_finder(self):
        # Build a dummy model
        m = ConcreteModel()
        m.v1 = Var()

        m.b1 = Block()
        m.b1.v2 = Var()

        m.b1.b2 = Block()
        m.b1.b2.v3 = Var([0])

        # Add Suffixes
        m.suffix = Suffix(direction=Suffix.EXPORT)
        # No suffix on b1 - make sure we can handle missing suffixes
        m.b1.b2.suffix = Suffix(direction=Suffix.EXPORT)

        _suffix_finder = SuffixFinder('suffix')
        _suffix_b1_finder = SuffixFinder('suffix', context=m.b1)
        _suffix_b2_finder = SuffixFinder('suffix', context=m.b1.b2)

        # Check for no suffix value
        self.assertEqual(_suffix_finder.find(m.b1.b2.v3[0]), None)
        self.assertEqual(_suffix_b1_finder.find(m.b1.b2.v3[0]), None)
        self.assertEqual(_suffix_b2_finder.find(m.b1.b2.v3[0]), None)

        # Check finding default values
        # Add a default at the top level
        m.suffix[None] = 1
        self.assertEqual(_suffix_finder.find(m.b1.b2.v3[0]), 1)
        self.assertEqual(_suffix_b1_finder.find(m.b1.b2.v3[0]), None)
        self.assertEqual(_suffix_b2_finder.find(m.b1.b2.v3[0]), None)

        # Add a default suffix at a lower level
        m.b1.b2.suffix[None] = 2
        self.assertEqual(_suffix_finder.find(m.b1.b2.v3[0]), 2)
        self.assertEqual(_suffix_b1_finder.find(m.b1.b2.v3[0]), 2)
        self.assertEqual(_suffix_b2_finder.find(m.b1.b2.v3[0]), 2)

        # Check for container at lowest level
        m.b1.b2.suffix[m.b1.b2.v3] = 3
        self.assertEqual(_suffix_finder.find(m.b1.b2.v3[0]), 3)
        self.assertEqual(_suffix_b1_finder.find(m.b1.b2.v3[0]), 3)
        self.assertEqual(_suffix_b2_finder.find(m.b1.b2.v3[0]), 3)

        # Check for container at top level
        m.suffix[m.b1.b2.v3] = 4
        self.assertEqual(_suffix_finder.find(m.b1.b2.v3[0]), 4)
        self.assertEqual(_suffix_b1_finder.find(m.b1.b2.v3[0]), 3)
        self.assertEqual(_suffix_b2_finder.find(m.b1.b2.v3[0]), 3)

        # Check for specific values at lowest level
        m.b1.b2.suffix[m.b1.b2.v3[0]] = 5
        self.assertEqual(_suffix_finder.find(m.b1.b2.v3[0]), 5)
        self.assertEqual(_suffix_b1_finder.find(m.b1.b2.v3[0]), 5)
        self.assertEqual(_suffix_b2_finder.find(m.b1.b2.v3[0]), 5)

        # Check for specific values at top level
        m.suffix[m.b1.b2.v3[0]] = 6
        self.assertEqual(_suffix_finder.find(m.b1.b2.v3[0]), 6)
        self.assertEqual(_suffix_b1_finder.find(m.b1.b2.v3[0]), 5)
        self.assertEqual(_suffix_b2_finder.find(m.b1.b2.v3[0]), 5)

        # Make sure we don't find default suffixes at lower levels
        self.assertEqual(_suffix_finder.find(m.b1.v2), 1)
        self.assertEqual(_suffix_b1_finder.find(m.b1.v2), None)
        self.assertEqual(_suffix_b2_finder.find(m.b1.v2), None)

        # Make sure we don't find specific suffixes at lower levels
        m.b1.b2.suffix[m.v1] = 5
        self.assertEqual(_suffix_finder.find(m.v1), 1)
        self.assertEqual(_suffix_b1_finder.find(m.v1), None)
        self.assertEqual(_suffix_b2_finder.find(m.v1), None)

        # Make sure we can look up Blocks and that they will match
        # suffixes that they hold
        self.assertEqual(_suffix_finder.find(m.b1.b2), 2)
        self.assertEqual(_suffix_b1_finder.find(m.b1.b2), 2)
        self.assertEqual(_suffix_b2_finder.find(m.b1.b2), 2)

        self.assertEqual(_suffix_finder.find(m.b1), 1)
        self.assertEqual(_suffix_b1_finder.find(m.b1), None)
        self.assertEqual(_suffix_b2_finder.find(m.b1), None)


if __name__ == "__main__":
    unittest.main()
