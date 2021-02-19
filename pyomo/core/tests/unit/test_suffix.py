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
# Unit Tests for Suffix
#

import os
import itertools
import pickle
from os.path import abspath, dirname
currdir = dirname(abspath(__file__))+os.sep

import pyutilib.th as unittest
from pyomo.core.base.suffix import \
    (active_export_suffix_generator,
     export_suffix_generator,
     active_import_suffix_generator,
     import_suffix_generator,
     active_local_suffix_generator,
     local_suffix_generator,
     active_suffix_generator,
     suffix_generator)
from pyomo.environ import ConcreteModel, Suffix, Var, Param, Set, Objective, Constraint, Block, sum_product

from six import StringIO

def simple_con_rule(model,i):
    return model.x[i] == 1
def simple_obj_rule(model,i):
    return model.x[i]


class TestSuffixMethods(unittest.TestCase):

    # test __init__
    def test_init(self):
        model = ConcreteModel()
        # no keywords
        model.junk = Suffix()
        model.del_component('junk')

        for direction,datatype in itertools.product(Suffix.SuffixDirections,Suffix.SuffixDatatypes):
            model.junk = Suffix(direction=direction,datatype=datatype)
            model.del_component('junk')

    # test import_enabled
    def test_import_enabled(self):
        model = ConcreteModel()
        model.test_local = Suffix(direction=Suffix.LOCAL)
        self.assertTrue(model.test_local.import_enabled() is False)

        model.test_out = Suffix(direction=Suffix.IMPORT)
        self.assertTrue(model.test_out.import_enabled() is True)

        model.test_in = Suffix(direction=Suffix.EXPORT)
        self.assertTrue(model.test_in.import_enabled() is False)

        model.test_inout = Suffix(direction=Suffix.IMPORT_EXPORT)
        self.assertTrue(model.test_inout.import_enabled() is True)

    # test export_enabled
    def test_export_enabled(self):
        model = ConcreteModel()

        model.test_local = Suffix(direction=Suffix.LOCAL)
        self.assertTrue(model.test_local.export_enabled() is False)

        model.test_out = Suffix(direction=Suffix.IMPORT)
        self.assertTrue(model.test_out.export_enabled() is False)

        model.test_in = Suffix(direction=Suffix.EXPORT)
        self.assertTrue(model.test_in.export_enabled() is True)

        model.test_inout = Suffix(direction=Suffix.IMPORT_EXPORT)
        self.assertTrue(model.test_inout.export_enabled() is True)

    # test set_value and getValue
    # and if Var arrays are correctly expanded
    def test_set_value_getValue_Var1(self):
        model = ConcreteModel()
        model.junk = Suffix()
        model.x = Var()
        model.X = Var([1,2,3], dense=True)

        model.junk.set_value(model.X,1.0)
        model.junk.set_value(model.X[1],2.0)

        self.assertEqual(model.junk.get(model.X), None)
        self.assertEqual(model.junk.get(model.X[1]), 2.0)
        self.assertEqual(model.junk.get(model.X[2]), 1.0)
        self.assertEqual(model.junk.get(model.x), None)

        model.junk.set_value(model.x,3.0)
        model.junk.set_value(model.X[2],3.0)

        self.assertEqual(model.junk.get(model.X), None)
        self.assertEqual(model.junk.get(model.X[1]), 2.0)
        self.assertEqual(model.junk.get(model.X[2]), 3.0)
        self.assertEqual(model.junk.get(model.x), 3.0)

        model.junk.set_value(model.X,1.0,expand=False)

        self.assertEqual(model.junk.get(model.X), 1.0)

    # test set_value and getValue
    # and if Var arrays are correctly expanded
    def test_set_value_getValue_Var2(self):
        model = ConcreteModel()
        model.junk = Suffix()
        model.x = Var()
        model.X = Var([1,2,3], dense=True)

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
        model.X = Var([1,2,3], dense=True)

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
        model.X = Var([1,2,3])
        model.c = Constraint(expr=model.x>=1)
        model.C = Constraint([1,2,3], rule=lambda model,i: model.X[i]>=1)

        model.junk.set_value(model.C,1.0)
        model.junk.set_value(model.C[1],2.0)

        self.assertEqual(model.junk.get(model.C), None)
        self.assertEqual(model.junk.get(model.C[1]), 2.0)
        self.assertEqual(model.junk.get(model.C[2]), 1.0)

        self.assertEqual(model.junk.get(model.c), None)

        model.junk.set_value(model.c,3.0)
        model.junk.set_value(model.C[2],3.0)

        self.assertEqual(model.junk.get(model.C), None)
        self.assertEqual(model.junk.get(model.C[1]), 2.0)
        self.assertEqual(model.junk.get(model.C[2]), 3.0)
        self.assertEqual(model.junk.get(model.c), 3.0)

        model.junk.set_value(model.C,1.0,expand=False)

        self.assertEqual(model.junk.get(model.C), 1.0)

    # test set_value and getValue
    # and if Constraint arrays are correctly expanded
    def test_set_value_getValue_Constraint2(self):
        model = ConcreteModel()
        model.junk = Suffix()
        model.x = Var()
        model.X = Var([1,2,3])
        model.c = Constraint(expr=model.x>=1)
        model.C = Constraint([1,2,3], rule=lambda model,i: model.X[i]>=1)

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
        model.X = Var([1,2,3])
        model.c = Constraint(expr=model.x>=1)
        model.C = Constraint([1,2,3], rule=lambda model,i: model.X[i]>=1)

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
        model.X = Var([1,2,3])
        model.obj = Objective(expr=sum_product(model.X)+model.x)
        model.OBJ = Objective([1,2,3], rule=lambda model,i: model.X[i])

        model.junk.set_value(model.OBJ,1.0)
        model.junk.set_value(model.OBJ[1],2.0)

        self.assertEqual(model.junk.get(model.OBJ), None)
        self.assertEqual(model.junk.get(model.OBJ[1]), 2.0)
        self.assertEqual(model.junk.get(model.OBJ[2]), 1.0)
        self.assertEqual(model.junk.get(model.obj), None)

        model.junk.set_value(model.obj,3.0)
        model.junk.set_value(model.OBJ[2],3.0)

        self.assertEqual(model.junk.get(model.OBJ), None)
        self.assertEqual(model.junk.get(model.OBJ[1]), 2.0)
        self.assertEqual(model.junk.get(model.OBJ[2]), 3.0)
        self.assertEqual(model.junk.get(model.obj), 3.0)

        model.junk.set_value(model.OBJ,1.0,expand=False)

        self.assertEqual(model.junk.get(model.OBJ), 1.0)

    # test set_value and getValue
    # and if Objective arrays are correctly expanded
    def test_set_value_getValue_Objective2(self):
        model = ConcreteModel()
        model.junk = Suffix()
        model.x = Var()
        model.X = Var([1,2,3])
        model.obj = Objective(expr=sum_product(model.X)+model.x)
        model.OBJ = Objective([1,2,3], rule=lambda model,i: model.X[i])

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
        model.X = Var([1,2,3])
        model.obj = Objective(expr=sum_product(model.X)+model.x)
        model.OBJ = Objective([1,2,3], rule=lambda model,i: model.X[i])

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
        model.X = Var([1,2,3])
        model.p = Param(initialize=1.0,mutable=True)
        model.P = Param([1,2,3],initialize=1.0,mutable=True)

        model.junk.set_value(model.P,1.0)
        model.junk.set_value(model.P[1],2.0)

        self.assertEqual(model.junk.get(model.P), None)
        self.assertEqual(model.junk.get(model.P[1]), 2.0)
        self.assertEqual(model.junk.get(model.P[2]), 1.0)
        self.assertEqual(model.junk.get(model.p), None)

        model.junk.set_value(model.p,3.0)
        model.junk.set_value(model.P[2],3.0)

        self.assertEqual(model.junk.get(model.P), None)
        self.assertEqual(model.junk.get(model.P[1]), 2.0)
        self.assertEqual(model.junk.get(model.P[2]), 3.0)
        self.assertEqual(model.junk.get(model.p), 3.0)

        model.junk.set_value(model.P,1.0,expand=False)

        self.assertEqual(model.junk.get(model.P), 1.0)

    # test set_value and getValue
    # and if mutable Param arrays are correctly expanded
    def test_set_value_getValue_mutableParam2(self):
        model = ConcreteModel()
        model.junk = Suffix()
        model.x = Var()
        model.X = Var([1,2,3])
        model.p = Param(initialize=1.0,mutable=True)
        model.P = Param([1,2,3],initialize=1.0,mutable=True)

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
        model.X = Var([1,2,3])
        model.p = Param(initialize=1.0,mutable=True)
        model.P = Param([1,2,3],initialize=1.0,mutable=True)

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
        model.X = Var([1,2,3])
        model.p = Param(initialize=1.0,mutable=False)
        model.P = Param([1,2,3],initialize=1.0,mutable=False)

        self.assertEqual(model.junk.get(model.P), None)

        model.junk.set_value(model.P,1.0,expand=False)

        self.assertEqual(model.junk.get(model.P), 1.0)

    # test set_value and getValue
    # and if immutable Param arrays are correctly expanded
    def test_set_value_getValue_immutableParam2(self):
        model = ConcreteModel()
        model.junk = Suffix()
        model.x = Var()
        model.X = Var([1,2,3])
        model.p = Param(initialize=1.0,mutable=False)
        model.P = Param([1,2,3],initialize=1.0,mutable=False)

        self.assertEqual(model.P.get_suffix_value('junk'), None)

        model.P.set_suffix_value('junk', 1.0, expand=False)

        self.assertEqual(model.P.get_suffix_value('junk'), 1.0)

    # test set_value and getValue
    # and if immutable Param arrays are correctly expanded
    def test_set_value_getValue_immutableParam3(self):
        model = ConcreteModel()
        model.junk = Suffix()
        model.x = Var()
        model.X = Var([1,2,3])
        model.p = Param(initialize=1.0,mutable=False)
        model.P = Param([1,2,3],initialize=1.0,mutable=False)

        self.assertEqual(model.P.get_suffix_value(model.junk), None)

        model.P.set_suffix_value(model.junk, 1.0, expand=False)

        self.assertEqual(model.P.get_suffix_value(model.junk), 1.0)

    # test set_value and getValue
    # and if Set arrays are correctly expanded
    def test_set_value_getValue_Set1(self):
        model = ConcreteModel()
        model.junk = Suffix()
        model.x = Var()
        model.X = Var([1,2,3])
        model.s = Set(initialize=[1,2,3])
        model.S = Set([1,2,3],initialize={1:[1,2,3],2:[1,2,3],3:[1,2,3]})

        model.junk.set_value(model.S,1.0)
        model.junk.set_value(model.S[1],2.0)

        self.assertEqual(model.junk.get(model.S), None)
        self.assertEqual(model.junk.get(model.S[1]), 2.0)
        self.assertEqual(model.junk.get(model.S[2]), 1.0)
        self.assertEqual(model.junk.get(model.s), None)

        model.junk.set_value(model.s,3.0)
        model.junk.set_value(model.S[2],3.0)

        self.assertEqual(model.junk.get(model.S), None)
        self.assertEqual(model.junk.get(model.S[1]), 2.0)
        self.assertEqual(model.junk.get(model.S[2]), 3.0)
        self.assertEqual(model.junk.get(model.s), 3.0)

        model.junk.set_value(model.S,1.0,expand=False)

        self.assertEqual(model.junk.get(model.S), 1.0)

    # test set_value and getValue
    # and if Set arrays are correctly expanded
    def test_set_value_getValue_Set2(self):
        model = ConcreteModel()
        model.junk = Suffix()
        model.x = Var()
        model.X = Var([1,2,3])
        model.s = Set(initialize=[1,2,3])
        model.S = Set([1,2,3],initialize={1:[1,2,3],2:[1,2,3],3:[1,2,3]})

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
        model.X = Var([1,2,3])
        model.s = Set(initialize=[1,2,3])
        model.S = Set([1,2,3],initialize={1:[1,2,3],2:[1,2,3],3:[1,2,3]})

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
        model.B = Block([1,2,3])

        # make sure each BlockData gets construced
        model.B[1].x = 1
        model.B[2].x = 2
        model.B[3].x = 3

        model.junk.set_value(model.B,1.0)
        model.junk.set_value(model.B[1],2.0)

        self.assertEqual(model.junk.get(model.B), None)
        self.assertEqual(model.junk.get(model.B[1]), 2.0)
        self.assertEqual(model.junk.get(model.B[2]), 1.0)
        self.assertEqual(model.junk.get(model.b), None)

        model.junk.set_value(model.b,3.0)
        model.junk.set_value(model.B[2],3.0)

        self.assertEqual(model.junk.get(model.B), None)
        self.assertEqual(model.junk.get(model.B[1]), 2.0)
        self.assertEqual(model.junk.get(model.B[2]), 3.0)
        self.assertEqual(model.junk.get(model.b), 3.0)

        model.junk.set_value(model.B,1.0,expand=False)

        self.assertEqual(model.junk.get(model.B), 1.0)

    # test set_value and getValue
    # and if Block arrays are correctly expanded
    def test_set_value_getValue_Block2(self):
        model = ConcreteModel()
        model.junk = Suffix()
        model.b = Block()
        model.B = Block([1,2,3])

        # make sure each BlockData gets construced
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
        model.B = Block([1,2,3])

        # make sure each BlockData gets construced
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
        model.y = Var([1,2,3], dense=True)
        model.z = Var([1,2,3], dense=True)

        model.junk.set_value(model.y[2],1.0)
        model.junk.set_value(model.z,2.0)

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
        model.y = Var([1,2,3], dense=True)
        model.z = Var([1,2,3], dense=True)

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
        model.y = Var([1,2,3], dense=True)
        model.z = Var([1,2,3], dense=True)

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
    def test_update_values1(self):
        model = ConcreteModel()
        model.junk = Suffix()
        model.x = Var()
        model.y = Var()
        model.z = Var()
        model.junk.set_value(model.x,0.0)
        self.assertEqual(model.junk.get(model.x),0.0)
        self.assertEqual(model.junk.get(model.y),None)
        self.assertEqual(model.junk.get(model.z),None)
        model.junk.update_values([(model.x,1.0),(model.y,2.0),(model.z,3.0)])
        self.assertEqual(model.junk.get(model.x),1.0)
        self.assertEqual(model.junk.get(model.y),2.0)
        self.assertEqual(model.junk.get(model.z),3.0)


    # test clear_value
    def test_clear_value(self):
        model = ConcreteModel()
        model.junk = Suffix()
        model.x = Var()
        model.y = Var([1,2,3], dense=True)
        model.z = Var([1,2,3], dense=True)

        model.junk.set_value(model.x,-1.0)
        model.junk.set_value(model.y,-2.0)
        model.junk.set_value(model.y[2],1.0)
        model.junk.set_value(model.z,2.0)
        model.junk.set_value(model.z[1],4.0)

        self.assertTrue(model.junk.get(model.x) == -1.0)
        self.assertTrue(model.junk.get(model.y) == None)
        self.assertTrue(model.junk.get(model.y[1]) == -2.0)
        self.assertEqual(model.junk.get(model.y[2]), 1.0)
        self.assertEqual(model.junk.get(model.z), None)
        self.assertEqual(model.junk.get(model.z[2]), 2.0)
        self.assertEqual(model.junk.get(model.z[1]), 4.0)

        model.junk.clear_value(model.y)
        model.junk.clear_value(model.x)
        model.junk.clear_value(model.z[1])

        self.assertTrue(model.junk.get(model.x) is None)
        self.assertTrue(model.junk.get(model.y) is None)
        self.assertTrue(model.junk.get(model.y[1]) is None)
        self.assertEqual(model.junk.get(model.y[2]), None)
        self.assertEqual(model.junk.get(model.z), None)
        self.assertEqual(model.junk.get(model.z[2]), 2.0)
        self.assertEqual(model.junk.get(model.z[1]), None)

    # test clear_value no args
    def test_clear_all_values(self):
        model = ConcreteModel()
        model.junk = Suffix()
        model.x = Var()
        model.y = Var([1,2,3], dense=True)
        model.z = Var([1,2,3], dense=True)

        model.junk.set_value(model.y[2],1.0)
        model.junk.set_value(model.z,2.0)

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
        self.assertTrue(model.junk.get_datatype() is Suffix.FLOAT)
        model.junk.set_datatype(Suffix.INT)
        self.assertTrue(model.junk.get_datatype() is Suffix.INT)
        model.junk.set_datatype(None)
        self.assertTrue(model.junk.get_datatype() is None)

    # test that calling set_datatype with a bad value fails
    def test_set_datatype_badvalue(self):
        model = ConcreteModel()
        model.junk = Suffix()
        try:
            model.junk.set_datatype(1.0)
        except ValueError:
            pass
        else:
            self.fail("Calling set_datatype with a bad type should fail.")

    # test set_direction and get_direction
    def test_set_direction_get_direction(self):
        model = ConcreteModel()
        model.junk = Suffix(direction=Suffix.LOCAL)
        self.assertTrue(model.junk.get_direction() is Suffix.LOCAL)
        model.junk.set_direction(Suffix.EXPORT)
        self.assertTrue(model.junk.get_direction() is Suffix.EXPORT)
        model.junk.set_direction(Suffix.IMPORT)
        self.assertTrue(model.junk.get_direction() is Suffix.IMPORT)
        model.junk.set_direction(Suffix.IMPORT_EXPORT)
        self.assertTrue(model.junk.get_direction() is Suffix.IMPORT_EXPORT)

    # test that calling set_direction with a bad value fails
    def test_set_direction_badvalue(self):
        model = ConcreteModel()
        model.junk = Suffix()
        try:
            model.junk.set_direction('a')
        except ValueError:
            pass
        else:
            self.fail("Calling set_datatype with a bad type should fail.")

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
        model.junk.set_direction(Suffix.IMPORT)
        model.junk.pprint(ostream=output)
        model.junk.set_direction(Suffix.LOCAL)
        model.junk.pprint(ostream=output)
        model.junk.set_direction(Suffix.IMPORT_EXPORT)
        model.junk.pprint(ostream=output)
        model.pprint(ostream=output)

    # test pprint(verbose=True)
    def test_pprint_verbose(self):
        model = ConcreteModel()
        model.junk = Suffix()
        model.s = Block()
        model.s.b = Block()
        model.s.B = Block([1,2,3])

        model.junk.set_value(model.s.B,1.0)
        model.junk.set_value(model.s.B[1],2.0)

        model.junk.set_value(model.s.b,3.0)
        model.junk.set_value(model.s.B[2],3.0)

        output = StringIO()
        model.junk.pprint(ostream=output,verbose=True)
        model.pprint(ostream=output,verbose=True)

    def test_active_export_suffix_generator(self):
        model = ConcreteModel()
        model.junk_EXPORT_int = Suffix(direction=Suffix.EXPORT,datatype=Suffix.INT)
        model.junk_EXPORT_float = Suffix(direction=Suffix.EXPORT,datatype=Suffix.FLOAT)
        model.junk_IMPORT_EXPORT_float = Suffix(direction=Suffix.IMPORT_EXPORT,datatype=Suffix.FLOAT)
        model.junk_IMPORT = Suffix(direction=Suffix.IMPORT,datatype=None)
        model.junk_LOCAL = Suffix(direction=Suffix.LOCAL,datatype=None)

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

        suffixes = dict(active_export_suffix_generator(model,datatype=Suffix.FLOAT))
        self.assertTrue('junk_EXPORT_int' not in suffixes)
        self.assertTrue('junk_EXPORT_float' in suffixes)
        self.assertTrue('junk_IMPORT_EXPORT_float' in suffixes)
        self.assertTrue('junk_IMPORT' not in suffixes)
        self.assertTrue('junk_LOCAL' not in suffixes)

        model.junk_EXPORT_float.deactivate()
        suffixes = dict(active_export_suffix_generator(model,datatype=Suffix.FLOAT))
        self.assertTrue('junk_EXPORT_int' not in suffixes)
        self.assertTrue('junk_EXPORT_float' not in suffixes)
        self.assertTrue('junk_IMPORT_EXPORT_float' in suffixes)
        self.assertTrue('junk_IMPORT' not in suffixes)
        self.assertTrue('junk_LOCAL' not in suffixes)

    def test_export_suffix_generator(self):
        model = ConcreteModel()
        model.junk_EXPORT_int = Suffix(direction=Suffix.EXPORT,datatype=Suffix.INT)
        model.junk_EXPORT_float = Suffix(direction=Suffix.EXPORT,datatype=Suffix.FLOAT)
        model.junk_IMPORT_EXPORT_float = Suffix(direction=Suffix.IMPORT_EXPORT,datatype=Suffix.FLOAT)
        model.junk_IMPORT = Suffix(direction=Suffix.IMPORT,datatype=None)
        model.junk_LOCAL = Suffix(direction=Suffix.LOCAL,datatype=None)

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

        suffixes = dict(export_suffix_generator(model,datatype=Suffix.FLOAT))
        self.assertTrue('junk_EXPORT_int' not in suffixes)
        self.assertTrue('junk_EXPORT_float' in suffixes)
        self.assertTrue('junk_IMPORT_EXPORT_float' in suffixes)
        self.assertTrue('junk_IMPORT' not in suffixes)
        self.assertTrue('junk_LOCAL' not in suffixes)

        model.junk_EXPORT_float.deactivate()
        suffixes = dict(export_suffix_generator(model,datatype=Suffix.FLOAT))
        self.assertTrue('junk_EXPORT_int' not in suffixes)
        self.assertTrue('junk_EXPORT_float' in suffixes)
        self.assertTrue('junk_IMPORT_EXPORT_float' in suffixes)
        self.assertTrue('junk_IMPORT' not in suffixes)
        self.assertTrue('junk_LOCAL' not in suffixes)

    def test_active_import_suffix_generator(self):
        model = ConcreteModel()
        model.junk_IMPORT_int = Suffix(direction=Suffix.IMPORT,datatype=Suffix.INT)
        model.junk_IMPORT_float = Suffix(direction=Suffix.IMPORT,datatype=Suffix.FLOAT)
        model.junk_IMPORT_EXPORT_float = Suffix(direction=Suffix.IMPORT_EXPORT,datatype=Suffix.FLOAT)
        model.junk_EXPORT = Suffix(direction=Suffix.EXPORT,datatype=None)
        model.junk_LOCAL = Suffix(direction=Suffix.LOCAL,datatype=None)

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

        suffixes = dict(active_import_suffix_generator(model,datatype=Suffix.FLOAT))
        self.assertTrue('junk_IMPORT_int' not in suffixes)
        self.assertTrue('junk_IMPORT_float' in suffixes)
        self.assertTrue('junk_IMPORT_EXPORT_float' in suffixes)
        self.assertTrue('junk_EXPORT' not in suffixes)
        self.assertTrue('junk_LOCAL' not in suffixes)

        model.junk_IMPORT_float.deactivate()
        suffixes = dict(active_import_suffix_generator(model,datatype=Suffix.FLOAT))
        self.assertTrue('junk_IMPORT_int' not in suffixes)
        self.assertTrue('junk_IMPORT_float' not in suffixes)
        self.assertTrue('junk_IMPORT_EXPORT_float' in suffixes)
        self.assertTrue('junk_EXPORT' not in suffixes)
        self.assertTrue('junk_LOCAL' not in suffixes)

    def test_import_suffix_generator(self):
        model = ConcreteModel()
        model.junk_IMPORT_int = Suffix(direction=Suffix.IMPORT,datatype=Suffix.INT)
        model.junk_IMPORT_float = Suffix(direction=Suffix.IMPORT,datatype=Suffix.FLOAT)
        model.junk_IMPORT_EXPORT_float = Suffix(direction=Suffix.IMPORT_EXPORT,datatype=Suffix.FLOAT)
        model.junk_EXPORT = Suffix(direction=Suffix.EXPORT,datatype=None)
        model.junk_LOCAL = Suffix(direction=Suffix.LOCAL,datatype=None)

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

        suffixes = dict(import_suffix_generator(model,datatype=Suffix.FLOAT))
        self.assertTrue('junk_IMPORT_int' not in suffixes)
        self.assertTrue('junk_IMPORT_float' in suffixes)
        self.assertTrue('junk_IMPORT_EXPORT_float' in suffixes)
        self.assertTrue('junk_EXPORT' not in suffixes)
        self.assertTrue('junk_LOCAL' not in suffixes)

        model.junk_IMPORT_float.deactivate()
        suffixes = dict(import_suffix_generator(model,datatype=Suffix.FLOAT))
        self.assertTrue('junk_IMPORT_int' not in suffixes)
        self.assertTrue('junk_IMPORT_float' in suffixes)
        self.assertTrue('junk_IMPORT_EXPORT_float' in suffixes)
        self.assertTrue('junk_EXPORT' not in suffixes)
        self.assertTrue('junk_LOCAL' not in suffixes)

    def test_active_local_suffix_generator(self):
        model = ConcreteModel()
        model.junk_LOCAL_int = Suffix(direction=Suffix.LOCAL,datatype=Suffix.INT)
        model.junk_LOCAL_float = Suffix(direction=Suffix.LOCAL,datatype=Suffix.FLOAT)
        model.junk_IMPORT_EXPORT = Suffix(direction=Suffix.IMPORT_EXPORT,datatype=None)
        model.junk_EXPORT = Suffix(direction=Suffix.EXPORT,datatype=None)
        model.junk_IMPORT = Suffix(direction=Suffix.IMPORT,datatype=None)

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

        suffixes = dict(active_local_suffix_generator(model,datatype=Suffix.FLOAT))
        self.assertTrue('junk_LOCAL_int' not in suffixes)
        self.assertTrue('junk_LOCAL_float' in suffixes)
        self.assertTrue('junk_IMPORT_EXPORT' not in suffixes)
        self.assertTrue('junk_EXPORT' not in suffixes)
        self.assertTrue('junk_IMPORT' not in suffixes)

        model.junk_LOCAL_float.deactivate()
        suffixes = dict(active_local_suffix_generator(model,datatype=Suffix.FLOAT))
        self.assertTrue('junk_LOCAL_int' not in suffixes)
        self.assertTrue('junk_LOCAL_float' not in suffixes)
        self.assertTrue('junk_IMPORT_EXPORT' not in suffixes)
        self.assertTrue('junk_EXPORT' not in suffixes)
        self.assertTrue('junk_IMPORT' not in suffixes)

    def test_local_suffix_generator(self):
        model = ConcreteModel()
        model.junk_LOCAL_int = Suffix(direction=Suffix.LOCAL,datatype=Suffix.INT)
        model.junk_LOCAL_float = Suffix(direction=Suffix.LOCAL,datatype=Suffix.FLOAT)
        model.junk_IMPORT_EXPORT = Suffix(direction=Suffix.IMPORT_EXPORT,datatype=None)
        model.junk_EXPORT = Suffix(direction=Suffix.EXPORT,datatype=None)
        model.junk_IMPORT = Suffix(direction=Suffix.IMPORT,datatype=None)

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

        suffixes = dict(local_suffix_generator(model,datatype=Suffix.FLOAT))
        self.assertTrue('junk_LOCAL_int' not in suffixes)
        self.assertTrue('junk_LOCAL_float' in suffixes)
        self.assertTrue('junk_IMPORT_EXPORT' not in suffixes)
        self.assertTrue('junk_EXPORT' not in suffixes)
        self.assertTrue('junk_IMPORT' not in suffixes)

        model.junk_LOCAL_float.deactivate()
        suffixes = dict(local_suffix_generator(model,datatype=Suffix.FLOAT))
        self.assertTrue('junk_LOCAL_int' not in suffixes)
        self.assertTrue('junk_LOCAL_float' in suffixes)
        self.assertTrue('junk_IMPORT_EXPORT' not in suffixes)
        self.assertTrue('junk_EXPORT' not in suffixes)
        self.assertTrue('junk_IMPORT' not in suffixes)

    def test_active_suffix_generator(self):
        model = ConcreteModel()
        model.junk_LOCAL_int = Suffix(direction=Suffix.LOCAL,datatype=Suffix.INT)
        model.junk_LOCAL_float = Suffix(direction=Suffix.LOCAL,datatype=Suffix.FLOAT)
        model.junk_IMPORT_EXPORT = Suffix(direction=Suffix.IMPORT_EXPORT,datatype=None)
        model.junk_EXPORT = Suffix(direction=Suffix.EXPORT,datatype=None)
        model.junk_IMPORT = Suffix(direction=Suffix.IMPORT,datatype=None)

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

        suffixes = dict(active_suffix_generator(model,datatype=Suffix.FLOAT))
        self.assertTrue('junk_LOCAL_int' not in suffixes)
        self.assertTrue('junk_LOCAL_float' in suffixes)
        self.assertTrue('junk_IMPORT_EXPORT' not in suffixes)
        self.assertTrue('junk_EXPORT' not in suffixes)
        self.assertTrue('junk_IMPORT' not in suffixes)

        model.junk_LOCAL_float.deactivate()
        suffixes = dict(active_suffix_generator(model,datatype=Suffix.FLOAT))
        self.assertTrue('junk_LOCAL_int' not in suffixes)
        self.assertTrue('junk_LOCAL_float' not in suffixes)
        self.assertTrue('junk_IMPORT_EXPORT' not in suffixes)
        self.assertTrue('junk_EXPORT' not in suffixes)
        self.assertTrue('junk_IMPORT' not in suffixes)

    def test_suffix_generator(self):
        model = ConcreteModel()
        model.junk_LOCAL_int = Suffix(direction=Suffix.LOCAL,datatype=Suffix.INT)
        model.junk_LOCAL_float = Suffix(direction=Suffix.LOCAL,datatype=Suffix.FLOAT)
        model.junk_IMPORT_EXPORT = Suffix(direction=Suffix.IMPORT_EXPORT,datatype=None)
        model.junk_EXPORT = Suffix(direction=Suffix.EXPORT,datatype=None)
        model.junk_IMPORT = Suffix(direction=Suffix.IMPORT,datatype=None)

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

        suffixes = dict(suffix_generator(model,datatype=Suffix.FLOAT))
        self.assertTrue('junk_LOCAL_int' not in suffixes)
        self.assertTrue('junk_LOCAL_float' in suffixes)
        self.assertTrue('junk_IMPORT_EXPORT' not in suffixes)
        self.assertTrue('junk_EXPORT' not in suffixes)
        self.assertTrue('junk_IMPORT' not in suffixes)

        model.junk_LOCAL_float.deactivate()
        suffixes = dict(suffix_generator(model,datatype=Suffix.FLOAT))
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
        self.assertEqual(model.junk.get(model.x),None)
        model.junk.set_value(model.x,1.0)
        self.assertEqual(model.junk.get(model.x),1.0)
        inst = model.clone()
        self.assertEqual(inst.junk.get(model.x),None)
        self.assertEqual(inst.junk.get(inst.x),1.0)

    def test_clone_VarArray(self):
        model = ConcreteModel()
        model.x = Var([1,2,3], dense=True)
        model.junk = Suffix()
        self.assertEqual(model.junk.get(model.x),None)
        self.assertEqual(model.junk.get(model.x[1]),None)
        model.junk.set_value(model.x,1.0)
        self.assertEqual(model.junk.get(model.x),None)
        self.assertEqual(model.junk.get(model.x[1]),1.0)
        inst = model.clone()
        self.assertEqual(inst.junk.get(model.x[1]),None)
        self.assertEqual(inst.junk.get(inst.x[1]),1.0)

    def test_clone_VarData(self):
        model = ConcreteModel()
        model.x = Var([1,2,3], dense=True)
        model.junk = Suffix()
        self.assertEqual(model.junk.get(model.x[1]),None)
        model.junk.set_value(model.x[1],1.0)
        self.assertEqual(model.junk.get(model.x[1]),1.0)
        inst = model.clone()
        self.assertEqual(inst.junk.get(model.x[1]),None)
        self.assertEqual(inst.junk.get(inst.x[1]),1.0)

    def test_clone_ConstraintElement(self):
        model = ConcreteModel()
        model.x = Var()
        model.c = Constraint(expr=model.x == 1.0)
        model.junk = Suffix()
        self.assertEqual(model.junk.get(model.c),None)
        model.junk.set_value(model.c,1.0)
        self.assertEqual(model.junk.get(model.c),1.0)
        inst = model.clone()
        self.assertEqual(inst.junk.get(model.c),None)
        self.assertEqual(inst.junk.get(inst.c),1.0)

    def test_clone_ConstraintArray(self):
        model = ConcreteModel()
        model.x = Var([1,2,3], dense=True)
        model.c = Constraint([1,2,3],rule=lambda model,i: model.x[i] == 1.0)
        model.junk = Suffix()
        self.assertEqual(model.junk.get(model.c),None)
        self.assertEqual(model.junk.get(model.c[1]),None)
        model.junk.set_value(model.c,1.0)
        self.assertEqual(model.junk.get(model.c),None)
        self.assertEqual(model.junk.get(model.c[1]),1.0)
        inst = model.clone()
        self.assertEqual(inst.junk.get(model.c[1]),None)
        self.assertEqual(inst.junk.get(inst.c[1]),1.0)

    def test_clone_ConstraintData(self):
        model = ConcreteModel()
        model.x = Var([1,2,3], dense=True)
        model.c = Constraint([1,2,3],rule=lambda model,i: model.x[i] == 1.0)
        model.junk = Suffix()
        self.assertEqual(model.junk.get(model.c[1]),None)
        model.junk.set_value(model.c[1],1.0)
        self.assertEqual(model.junk.get(model.c[1]),1.0)
        inst = model.clone()
        self.assertEqual(inst.junk.get(model.c[1]),None)
        self.assertEqual(inst.junk.get(inst.c[1]),1.0)

    def test_clone_ObjectiveElement(self):
        model = ConcreteModel()
        model.x = Var()
        model.obj = Objective(expr=model.x)
        model.junk = Suffix()
        self.assertEqual(model.junk.get(model.obj),None)
        model.junk.set_value(model.obj,1.0)
        self.assertEqual(model.junk.get(model.obj),1.0)
        inst = model.clone()
        self.assertEqual(inst.junk.get(model.obj),None)
        self.assertEqual(inst.junk.get(inst.obj),1.0)

    def test_clone_ObjectiveArray(self):
        model = ConcreteModel()
        model.x = Var([1,2,3], dense=True)
        model.obj = Objective([1,2,3], rule=lambda model,i: model.x[i])
        model.junk = Suffix()
        self.assertEqual(model.junk.get(model.obj),None)
        self.assertEqual(model.junk.get(model.obj[1]),None)
        model.junk.set_value(model.obj,1.0)
        self.assertEqual(model.junk.get(model.obj),None)
        self.assertEqual(model.junk.get(model.obj[1]),1.0)
        inst = model.clone()
        self.assertEqual(inst.junk.get(model.obj[1]),None)
        self.assertEqual(inst.junk.get(inst.obj[1]),1.0)

    def test_clone_ObjectiveData(self):
        model = ConcreteModel()
        model.x = Var([1,2,3], dense=True)
        model.obj = Objective([1,2,3], rule=lambda model,i: model.x[i])
        model.junk = Suffix()
        self.assertEqual(model.junk.get(model.obj[1]),None)
        model.junk.set_value(model.obj[1],1.0)
        self.assertEqual(model.junk.get(model.obj[1]),1.0)
        inst = model.clone()
        self.assertEqual(inst.junk.get(model.obj[1]),None)
        self.assertEqual(inst.junk.get(inst.obj[1]),1.0)

    def test_clone_SimpleBlock(self):
        model = ConcreteModel()
        model.b = Block()
        model.junk = Suffix()
        self.assertEqual(model.junk.get(model.b),None)
        model.junk.set_value(model.b,1.0)
        self.assertEqual(model.junk.get(model.b),1.0)
        inst = model.clone()
        self.assertEqual(inst.junk.get(model.b),None)
        self.assertEqual(inst.junk.get(inst.b),1.0)

    def test_clone_IndexedBlock(self):
        model = ConcreteModel()
        model.b = Block([1,2,3])
        model.junk = Suffix()
        self.assertEqual(model.junk.get(model.b),None)
        self.assertEqual(model.junk.get(model.b[1]),None)
        model.junk.set_value(model.b,1.0)
        self.assertEqual(model.junk.get(model.b),None)
        self.assertEqual(model.junk.get(model.b[1]),1.0)
        inst = model.clone()
        self.assertEqual(inst.junk.get(model.b[1]),None)
        self.assertEqual(inst.junk.get(inst.b[1]),1.0)

    def test_clone_BlockData(self):
        model = ConcreteModel()
        model.b = Block([1,2,3])
        model.junk = Suffix()
        self.assertEqual(model.junk.get(model.b[1]),None)
        model.junk.set_value(model.b[1],1.0)
        self.assertEqual(model.junk.get(model.b[1]),1.0)
        inst = model.clone()
        self.assertEqual(inst.junk.get(model.b[1]),None)
        self.assertEqual(inst.junk.get(inst.b[1]),1.0)

    def test_clone_model(self):
        model = ConcreteModel()
        model.junk = Suffix()
        self.assertEqual(model.junk.get(model),None)
        model.junk.set_value(model,1.0)
        self.assertEqual(model.junk.get(model),1.0)
        inst = model.clone()
        self.assertEqual(inst.junk.get(model),None)
        self.assertEqual(inst.junk.get(inst),1.0)


class TestSuffixPickleUsage(unittest.TestCase):

    def test_pickle_VarElement(self):
        model = ConcreteModel()
        model.x = Var()
        model.junk = Suffix()
        self.assertEqual(model.junk.get(model.x),None)
        model.junk.set_value(model.x,1.0)
        self.assertEqual(model.junk.get(model.x),1.0)
        inst = pickle.loads(pickle.dumps(model))
        self.assertEqual(inst.junk.get(model.x),None)
        self.assertEqual(inst.junk.get(inst.x),1.0)

    def test_pickle_VarArray(self):
        model = ConcreteModel()
        model.x = Var([1,2,3], dense=True)
        model.junk = Suffix()
        self.assertEqual(model.junk.get(model.x),None)
        self.assertEqual(model.junk.get(model.x[1]),None)
        model.junk.set_value(model.x,1.0)
        self.assertEqual(model.junk.get(model.x),None)
        self.assertEqual(model.junk.get(model.x[1]),1.0)
        inst = pickle.loads(pickle.dumps(model))
        self.assertEqual(inst.junk.get(model.x[1]),None)
        self.assertEqual(inst.junk.get(inst.x[1]),1.0)

    def test_pickle_VarData(self):
        model = ConcreteModel()
        model.x = Var([1,2,3], dense=True)
        model.junk = Suffix()
        self.assertEqual(model.junk.get(model.x[1]),None)
        model.junk.set_value(model.x[1],1.0)
        self.assertEqual(model.junk.get(model.x[1]),1.0)
        inst = pickle.loads(pickle.dumps(model))
        self.assertEqual(inst.junk.get(model.x[1]),None)
        self.assertEqual(inst.junk.get(inst.x[1]),1.0)

    def test_pickle_ConstraintElement(self):
        model = ConcreteModel()
        model.x = Var()
        model.c = Constraint(expr=model.x == 1.0)
        model.junk = Suffix()
        self.assertEqual(model.junk.get(model.c),None)
        model.junk.set_value(model.c,1.0)
        self.assertEqual(model.junk.get(model.c),1.0)
        inst = pickle.loads(pickle.dumps(model))
        self.assertEqual(inst.junk.get(model.c),None)
        self.assertEqual(inst.junk.get(inst.c),1.0)

    def test_pickle_ConstraintArray(self):
        model = ConcreteModel()
        model.x = Var([1,2,3], dense=True)
        model.c = Constraint([1,2,3],rule=simple_con_rule)
        model.junk = Suffix()
        self.assertEqual(model.junk.get(model.c),None)
        self.assertEqual(model.junk.get(model.c[1]),None)
        model.junk.set_value(model.c,1.0)
        self.assertEqual(model.junk.get(model.c),None)
        self.assertEqual(model.junk.get(model.c[1]),1.0)
        inst = pickle.loads(pickle.dumps(model))
        self.assertEqual(inst.junk.get(model.c[1]),None)
        self.assertEqual(inst.junk.get(inst.c[1]),1.0)

    def test_pickle_ConstraintData(self):
        model = ConcreteModel()
        model.x = Var([1,2,3], dense=True)
        model.c = Constraint([1,2,3],rule=simple_con_rule)
        model.junk = Suffix()
        self.assertEqual(model.junk.get(model.c[1]),None)
        model.junk.set_value(model.c[1],1.0)
        self.assertEqual(model.junk.get(model.c[1]),1.0)
        inst = pickle.loads(pickle.dumps(model))
        self.assertEqual(inst.junk.get(model.c[1]),None)
        self.assertEqual(inst.junk.get(inst.c[1]),1.0)

    def test_pickle_ObjectiveElement(self):
        model = ConcreteModel()
        model.x = Var()
        model.obj = Objective(expr=model.x)
        model.junk = Suffix()
        self.assertEqual(model.junk.get(model.obj),None)
        model.junk.set_value(model.obj,1.0)
        self.assertEqual(model.junk.get(model.obj),1.0)
        inst = pickle.loads(pickle.dumps(model))
        self.assertEqual(inst.junk.get(model.obj),None)
        self.assertEqual(inst.junk.get(inst.obj),1.0)

    def test_pickle_ObjectiveArray(self):
        model = ConcreteModel()
        model.x = Var([1,2,3], dense=True)
        model.obj = Objective([1,2,3],rule=simple_obj_rule)
        model.junk = Suffix()
        self.assertEqual(model.junk.get(model.obj),None)
        self.assertEqual(model.junk.get(model.obj[1]),None)
        model.junk.set_value(model.obj,1.0)
        self.assertEqual(model.junk.get(model.obj),None)
        self.assertEqual(model.junk.get(model.obj[1]),1.0)
        inst = pickle.loads(pickle.dumps(model))
        self.assertEqual(inst.junk.get(model.obj[1]),None)
        self.assertEqual(inst.junk.get(inst.obj[1]),1.0)

    def test_pickle_ObjectiveData(self):
        model = ConcreteModel()
        model.x = Var([1,2,3], dense=True)
        model.obj = Objective([1,2,3],rule=simple_obj_rule)
        model.junk = Suffix()
        self.assertEqual(model.junk.get(model.obj[1]),None)
        model.junk.set_value(model.obj[1],1.0)
        self.assertEqual(model.junk.get(model.obj[1]),1.0)
        inst = pickle.loads(pickle.dumps(model))
        self.assertEqual(inst.junk.get(model.obj[1]),None)
        self.assertEqual(inst.junk.get(inst.obj[1]),1.0)

    def test_pickle_SimpleBlock(self):
        model = ConcreteModel()
        model.b = Block()
        model.junk = Suffix()
        self.assertEqual(model.junk.get(model.b),None)
        model.junk.set_value(model.b,1.0)
        self.assertEqual(model.junk.get(model.b),1.0)
        inst = pickle.loads(pickle.dumps(model))
        self.assertEqual(inst.junk.get(model.b),None)
        self.assertEqual(inst.junk.get(inst.b),1.0)

    def test_pickle_IndexedBlock(self):
        model = ConcreteModel()
        model.b = Block([1,2,3])
        model.junk = Suffix()
        self.assertEqual(model.junk.get(model.b),None)
        self.assertEqual(model.junk.get(model.b[1]),None)
        model.junk.set_value(model.b,1.0)
        self.assertEqual(model.junk.get(model.b),None)
        self.assertEqual(model.junk.get(model.b[1]),1.0)
        inst = pickle.loads(pickle.dumps(model))
        self.assertEqual(inst.junk.get(model.b[1]),None)
        self.assertEqual(inst.junk.get(inst.b[1]),1.0)

    def test_pickle_BlockData(self):
        model = ConcreteModel()
        model.b = Block([1,2,3])
        model.junk = Suffix()
        self.assertEqual(model.junk.get(model.b[1]),None)
        model.junk.set_value(model.b[1],1.0)
        self.assertEqual(model.junk.get(model.b[1]),1.0)
        inst = pickle.loads(pickle.dumps(model))
        self.assertEqual(inst.junk.get(model.b[1]),None)
        self.assertEqual(inst.junk.get(inst.b[1]),1.0)

    def test_pickle_model(self):
        model = ConcreteModel()
        model.junk = Suffix()
        self.assertEqual(model.junk.get(model),None)
        model.junk.set_value(model,1.0)
        self.assertEqual(model.junk.get(model),1.0)
        inst = pickle.loads(pickle.dumps(model))
        self.assertEqual(inst.junk.get(model),None)
        self.assertEqual(inst.junk.get(inst),1.0)

if __name__ == "__main__":
    unittest.main()
