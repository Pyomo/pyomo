#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________
#
# Unit Tests for Suffix
#

import os
import sys
import copy
import itertools
import pickle
from os.path import abspath, dirname
currdir = dirname(abspath(__file__))+os.sep

from pyomo.environ import *
from pyomo.core.base.suffix import active_export_suffix_generator, \
                                    export_suffix_generator, \
                                    active_import_suffix_generator, \
                                    import_suffix_generator, \
                                    active_local_suffix_generator, \
                                    local_suffix_generator, \
                                    active_suffix_generator, \
                                    suffix_generator
import pyutilib.th as unittest

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
            
    # test importEnabled
    def test_importEnabled(self):
        model = ConcreteModel()
        model.test_local = Suffix(direction=Suffix.LOCAL)
        self.assertTrue(model.test_local.importEnabled() is False)

        model.test_out = Suffix(direction=Suffix.IMPORT)
        self.assertTrue(model.test_out.importEnabled() is True)

        model.test_in = Suffix(direction=Suffix.EXPORT)
        self.assertTrue(model.test_in.importEnabled() is False)

        model.test_inout = Suffix(direction=Suffix.IMPORT_EXPORT)        
        self.assertTrue(model.test_inout.importEnabled() is True)

    # test exportEnabled
    def test_exportEnabled(self):
        model = ConcreteModel()

        model.test_local = Suffix(direction=Suffix.LOCAL)        
        self.assertTrue(model.test_local.exportEnabled() is False)

        model.test_out = Suffix(direction=Suffix.IMPORT)
        self.assertTrue(model.test_out.exportEnabled() is False)

        model.test_in = Suffix(direction=Suffix.EXPORT)
        self.assertTrue(model.test_in.exportEnabled() is True)

        model.test_inout = Suffix(direction=Suffix.IMPORT_EXPORT)
        self.assertTrue(model.test_inout.exportEnabled() is True)

    # test setValue and getValue
    # and if Var arrays are correctly expanded
    def test_setValue_getValue_Var(self):
        model = ConcreteModel()
        model.junk = Suffix()
        model.x = Var()
        model.X = Var([1,2,3])

        model.junk.setValue(model.X,1.0)
        model.junk.setValue(model.X[1],2.0)

        self.assertEqual(model.junk.get(model.X), None)
        self.assertEqual(model.junk.get(model.X[1]), 2.0)
        self.assertEqual(model.junk.get(model.X[2]), 1.0)

        self.assertEqual(model.junk.get(model.x), None)

        model.junk.setValue(model.x,3.0)
        model.junk.setValue(model.X[2],3.0)

        self.assertEqual(model.junk.get(model.X), None)
        self.assertEqual(model.junk.get(model.X[1]), 2.0)
        self.assertEqual(model.junk.get(model.X[2]), 3.0)
        self.assertEqual(model.junk.get(model.x), 3.0)

        model.junk.setValue(model.X,1.0,expand=False)

        self.assertEqual(model.junk.get(model.X), 1.0)

    # test setValue and getValue
    # and if Constraint arrays are correctly expanded
    def test_setValue_getValue_Constraint(self):
        model = ConcreteModel()
        model.junk = Suffix()
        model.x = Var()
        model.X = Var([1,2,3])
        model.c = Constraint(expr=model.x>=1)
        model.C = Constraint([1,2,3], rule=lambda model,i: model.X[i]>=1)

        model.junk.setValue(model.C,1.0)
        model.junk.setValue(model.C[1],2.0)

        self.assertEqual(model.junk.get(model.C), None)
        self.assertEqual(model.junk.get(model.C[1]), 2.0)
        self.assertEqual(model.junk.get(model.C[2]), 1.0)

        self.assertEqual(model.junk.get(model.c), None)

        model.junk.setValue(model.c,3.0)
        model.junk.setValue(model.C[2],3.0)

        self.assertEqual(model.junk.get(model.C), None)
        self.assertEqual(model.junk.get(model.C[1]), 2.0)
        self.assertEqual(model.junk.get(model.C[2]), 3.0)
        self.assertEqual(model.junk.get(model.c), 3.0)

        model.junk.setValue(model.C,1.0,expand=False)

        self.assertEqual(model.junk.get(model.C), 1.0)

    # test setValue and getValue
    # and if Objective arrays are correctly expanded
    def test_setValue_getValue_Objective(self):
        model = ConcreteModel()
        model.junk = Suffix()
        model.x = Var()
        model.X = Var([1,2,3])
        model.obj = Objective(expr=summation(model.X)+model.x)
        model.OBJ = Objective([1,2,3], rule=lambda model,i: model.X[i])

        model.junk.setValue(model.OBJ,1.0)
        model.junk.setValue(model.OBJ[1],2.0)

        self.assertEqual(model.junk.get(model.OBJ), None)
        self.assertEqual(model.junk.get(model.OBJ[1]), 2.0)
        self.assertEqual(model.junk.get(model.OBJ[2]), 1.0)

        self.assertEqual(model.junk.get(model.obj), None)

        model.junk.setValue(model.obj,3.0)
        model.junk.setValue(model.OBJ[2],3.0)

        self.assertEqual(model.junk.get(model.OBJ), None)
        self.assertEqual(model.junk.get(model.OBJ[1]), 2.0)
        self.assertEqual(model.junk.get(model.OBJ[2]), 3.0)
        self.assertEqual(model.junk.get(model.obj), 3.0)

        model.junk.setValue(model.OBJ,1.0,expand=False)

        self.assertEqual(model.junk.get(model.OBJ), 1.0)

    # test setValue and getValue
    # and if mutable Param arrays are correctly expanded
    def test_setValue_getValue_mutableParam(self):
        model = ConcreteModel()
        model.junk = Suffix()
        model.x = Var()
        model.X = Var([1,2,3])
        model.p = Param(initialize=1.0,mutable=True)
        model.P = Param([1,2,3],initialize=1.0,mutable=True)

        model.junk.setValue(model.P,1.0)
        model.junk.setValue(model.P[1],2.0)

        self.assertEqual(model.junk.get(model.P), None)
        self.assertEqual(model.junk.get(model.P[1]), 2.0)
        self.assertEqual(model.junk.get(model.P[2]), 1.0)

        self.assertEqual(model.junk.get(model.p), None)

        model.junk.setValue(model.p,3.0)
        model.junk.setValue(model.P[2],3.0)

        self.assertEqual(model.junk.get(model.P), None)
        self.assertEqual(model.junk.get(model.P[1]), 2.0)
        self.assertEqual(model.junk.get(model.P[2]), 3.0)
        self.assertEqual(model.junk.get(model.p), 3.0)

        model.junk.setValue(model.P,1.0,expand=False)

        self.assertEqual(model.junk.get(model.P), 1.0)

    # test setValue and getValue
    # and if immutable Param arrays are correctly expanded
    def test_setValue_getValue_immutableParam(self):
        model = ConcreteModel()
        model.junk = Suffix()
        model.x = Var()
        model.X = Var([1,2,3])
        model.p = Param(initialize=1.0,mutable=False)
        model.P = Param([1,2,3],initialize=1.0,mutable=False)

        self.assertEqual(model.junk.get(model.P), None)

        model.junk.setValue(model.P,1.0,expand=False)

        self.assertEqual(model.junk.get(model.P), 1.0)

    # test setValue and getValue
    # and if Set arrays are correctly expanded
    def test_setValue_getValue_Set(self):
        model = ConcreteModel()
        model.junk = Suffix()
        model.x = Var()
        model.X = Var([1,2,3])
        model.s = Set(initialize=[1,2,3])
        model.S = Set([1,2,3],initialize={1:[1,2,3],2:[1,2,3],3:[1,2,3]})

        model.junk.setValue(model.S,1.0)
        model.junk.setValue(model.S[1],2.0)

        self.assertEqual(model.junk.get(model.S), None)
        self.assertEqual(model.junk.get(model.S[1]), 2.0)
        self.assertEqual(model.junk.get(model.S[2]), 1.0)

        self.assertEqual(model.junk.get(model.s), None)

        model.junk.setValue(model.s,3.0)
        model.junk.setValue(model.S[2],3.0)

        self.assertEqual(model.junk.get(model.S), None)
        self.assertEqual(model.junk.get(model.S[1]), 2.0)
        self.assertEqual(model.junk.get(model.S[2]), 3.0)
        self.assertEqual(model.junk.get(model.s), 3.0)

        model.junk.setValue(model.S,1.0,expand=False)

        self.assertEqual(model.junk.get(model.S), 1.0)

    # test setValue and getValue
    # and if Block arrays are correctly expanded
    def test_setValue_getValue_Block(self):
        model = ConcreteModel()
        model.junk = Suffix()
        model.b = Block()
        model.B = Block([1,2,3])

        # make sure each BlockData gets construced
        model.B[1].x = 1
        model.B[2].x = 2
        model.B[3].x = 3

        model.junk.setValue(model.B,1.0)
        model.junk.setValue(model.B[1],2.0)

        self.assertEqual(model.junk.get(model.B), None)
        self.assertEqual(model.junk.get(model.B[1]), 2.0)
        self.assertEqual(model.junk.get(model.B[2]), 1.0)

        self.assertEqual(model.junk.get(model.b), None)

        model.junk.setValue(model.b,3.0)
        model.junk.setValue(model.B[2],3.0)

        self.assertEqual(model.junk.get(model.B), None)
        self.assertEqual(model.junk.get(model.B[1]), 2.0)
        self.assertEqual(model.junk.get(model.B[2]), 3.0)
        self.assertEqual(model.junk.get(model.b), 3.0)

        model.junk.setValue(model.B,1.0,expand=False)

        self.assertEqual(model.junk.get(model.B), 1.0)

    # test setValue with no component argument
    def test_setAllValues(self):
        model = ConcreteModel()
        model.junk = Suffix()
        model.x = Var()
        model.y = Var([1,2,3])
        model.z = Var([1,2,3])

        model.junk.setValue(model.y[2],1.0)
        model.junk.setValue(model.z,2.0)
        
        self.assertTrue(model.junk.get(model.x) is None)
        self.assertTrue(model.junk.get(model.y) is None)
        self.assertTrue(model.junk.get(model.y[1]) is None)
        self.assertEqual(model.junk.get(model.y[2]), 1.0)
        self.assertEqual(model.junk.get(model.z), None)
        self.assertEqual(model.junk.get(model.z[1]), 2.0)
        
        model.junk.setAllValues(3.0)

        self.assertTrue(model.junk.get(model.x) is None)
        self.assertTrue(model.junk.get(model.y) is None)
        self.assertTrue(model.junk.get(model.y[1]) is None)
        self.assertEqual(model.junk.get(model.y[2]), 3.0)
        self.assertEqual(model.junk.get(model.z), None)
        self.assertEqual(model.junk.get(model.z[1]), 3.0)

    # test updateValues
    def test_updateValues(self):
        model = ConcreteModel()
        model.junk = Suffix()
        model.x = Var()
        model.y = Var()
        model.z = Var()
        model.junk.setValue(model.x,0.0)
        self.assertEqual(model.junk.get(model.x),0.0)
        self.assertEqual(model.junk.get(model.y),None)
        self.assertEqual(model.junk.get(model.z),None)
        model.junk.updateValues([(model.x,1.0),(model.y,2.0),(model.z,3.0)])
        self.assertEqual(model.junk.get(model.x),1.0)
        self.assertEqual(model.junk.get(model.y),2.0)
        self.assertEqual(model.junk.get(model.z),3.0)


    # test clearValue 
    def test_clearValue(self):
        model = ConcreteModel()
        model.junk = Suffix()
        model.x = Var()
        model.y = Var([1,2,3])
        model.z = Var([1,2,3])

        model.junk.setValue(model.x,-1.0)
        model.junk.setValue(model.y,-2.0)
        model.junk.setValue(model.y[2],1.0)
        model.junk.setValue(model.z,2.0)
        model.junk.setValue(model.z[1],4.0)
        
        self.assertTrue(model.junk.get(model.x) == -1.0)
        self.assertTrue(model.junk.get(model.y) == None)
        self.assertTrue(model.junk.get(model.y[1]) == -2.0)
        self.assertEqual(model.junk.get(model.y[2]), 1.0)
        self.assertEqual(model.junk.get(model.z), None)
        self.assertEqual(model.junk.get(model.z[2]), 2.0)
        self.assertEqual(model.junk.get(model.z[1]), 4.0)

        model.junk.clearValue(model.y)
        model.junk.clearValue(model.x)
        model.junk.clearValue(model.z[1])

        self.assertTrue(model.junk.get(model.x) is None)
        self.assertTrue(model.junk.get(model.y) is None)
        self.assertTrue(model.junk.get(model.y[1]) is None)
        self.assertEqual(model.junk.get(model.y[2]), None)
        self.assertEqual(model.junk.get(model.z), None)
        self.assertEqual(model.junk.get(model.z[2]), 2.0)
        self.assertEqual(model.junk.get(model.z[1]), None)

    # test clearValue no args
    def test_clearAllValues(self):
        model = ConcreteModel()
        model.junk = Suffix()
        model.x = Var()
        model.y = Var([1,2,3])
        model.z = Var([1,2,3])

        model.junk.setValue(model.y[2],1.0)
        model.junk.setValue(model.z,2.0)

        self.assertTrue(model.junk.get(model.x) is None)
        self.assertTrue(model.junk.get(model.y) is None)
        self.assertTrue(model.junk.get(model.y[1]) is None)
        self.assertEqual(model.junk.get(model.y[2]), 1.0)
        self.assertEqual(model.junk.get(model.z), None)
        self.assertEqual(model.junk.get(model.z[1]), 2.0)

        model.junk.clearAllValues()

        self.assertTrue(model.junk.get(model.x) is None)
        self.assertTrue(model.junk.get(model.y) is None)
        self.assertTrue(model.junk.get(model.y[1]) is None)
        self.assertTrue(model.junk.get(model.y[2]) is None)
        self.assertTrue(model.junk.get(model.z) is None)
        self.assertTrue(model.junk.get(model.z[1]) is None)

    # test setDatatype and getDatatype
    def test_setDatatype_getDatatype(self):
        model = ConcreteModel()
        model.junk = Suffix(datatype=Suffix.FLOAT)
        self.assertTrue(model.junk.getDatatype() is Suffix.FLOAT)
        model.junk.setDatatype(Suffix.INT)
        self.assertTrue(model.junk.getDatatype() is Suffix.INT)
        model.junk.setDatatype(None)
        self.assertTrue(model.junk.getDatatype() is None)

    # test that calling setDatatype with a bad value fails
    def test_setDatatype_badvalue(self):
        model = ConcreteModel()
        model.junk = Suffix()
        try:
            model.junk.setDatatype(1.0)
        except ValueError:
            pass
        else:
            self.fail("Calling setDatatype with a bad type should fail.")

    # test setDirection and getDirection
    def test_setDirection_getDirection(self):
        model = ConcreteModel()
        model.junk = Suffix(direction=Suffix.LOCAL)
        self.assertTrue(model.junk.getDirection() is Suffix.LOCAL)
        model.junk.setDirection(Suffix.EXPORT)
        self.assertTrue(model.junk.getDirection() is Suffix.EXPORT)
        model.junk.setDirection(Suffix.IMPORT)
        self.assertTrue(model.junk.getDirection() is Suffix.IMPORT)
        model.junk.setDirection(Suffix.IMPORT_EXPORT)
        self.assertTrue(model.junk.getDirection() is Suffix.IMPORT_EXPORT)
        
    # test that calling setDirection with a bad value fails
    def test_setDirection_badvalue(self):
        model = ConcreteModel()
        model.junk = Suffix()
        try:
            model.junk.setDirection('a')
        except ValueError:
            pass
        else:
            self.fail("Calling setDatatype with a bad type should fail.")    

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
        model.junk.setDirection(Suffix.IMPORT)
        model.junk.pprint(ostream=output)
        model.junk.setDirection(Suffix.LOCAL)
        model.junk.pprint(ostream=output)
        model.junk.setDirection(Suffix.IMPORT_EXPORT)
        model.junk.pprint(ostream=output)
        model.pprint(ostream=output)

    # test pprint(verbose=True)
    def test_pprint_verbose(self):
        model = ConcreteModel()
        model.junk = Suffix()
        model.s = Block()
        model.s.b = Block()
        model.s.B = Block([1,2,3])

        model.junk.setValue(model.s.B,1.0)
        model.junk.setValue(model.s.B[1],2.0)

        model.junk.setValue(model.s.b,3.0)
        model.junk.setValue(model.s.B[2],3.0)

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

    def test_reset(self):
        model = ConcreteModel()
        model.x = Var()
        model.y = Var()
        model.junk_no_rule = Suffix()
        self.assertEqual(model.junk_no_rule.get(model.x),None)
        self.assertEqual(model.junk_no_rule.get(model.y),None)
        model.junk_no_rule.setValue(model.x,1)
        model.junk_no_rule.setValue(model.y,2)
        self.assertEqual(model.junk_no_rule.get(model.x),1)
        self.assertEqual(model.junk_no_rule.get(model.y),2)
        model.junk_no_rule.reset()
        self.assertEqual(model.junk_no_rule.get(model.x),None)
        self.assertEqual(model.junk_no_rule.get(model.y),None)
        
        model.del_component('junk_no_rule')

        def _junk_rule(model):
            return [(model.x,1)]
        model.junk_rule = Suffix(rule=_junk_rule)
        self.assertEqual(model.junk_rule.get(model.x),1)
        self.assertEqual(model.junk_rule.get(model.y),None)
        model.junk_rule.setValue(model.y,2)
        self.assertEqual(model.junk_rule.get(model.x),1)
        self.assertEqual(model.junk_rule.get(model.y),2)
        model.junk_rule.reset()
        self.assertEqual(model.junk_rule.get(model.x),1)
        self.assertEqual(model.junk_rule.get(model.y),None)

class TestSuffixCloneUsage(unittest.TestCase):

    def test_clone_VarElement(self):
        model = ConcreteModel()
        model.x = Var()
        model.junk = Suffix()
        self.assertEqual(model.junk.get(model.x),None)
        model.junk.setValue(model.x,1.0)
        self.assertEqual(model.junk.get(model.x),1.0)
        inst = model.clone()
        self.assertEqual(inst.junk.get(model.x),None)
        self.assertEqual(inst.junk.get(inst.x),1.0)

    def test_clone_VarArray(self):
        model = ConcreteModel()
        model.x = Var([1,2,3])
        model.junk = Suffix()
        self.assertEqual(model.junk.get(model.x),None)
        self.assertEqual(model.junk.get(model.x[1]),None)
        model.junk.setValue(model.x,1.0)
        self.assertEqual(model.junk.get(model.x),None)
        self.assertEqual(model.junk.get(model.x[1]),1.0)
        inst = model.clone()
        self.assertEqual(inst.junk.get(model.x[1]),None)
        self.assertEqual(inst.junk.get(inst.x[1]),1.0)

    def test_clone_VarData(self):
        model = ConcreteModel()
        model.x = Var([1,2,3])
        model.junk = Suffix()
        self.assertEqual(model.junk.get(model.x[1]),None)
        model.junk.setValue(model.x[1],1.0)
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
        model.junk.setValue(model.c,1.0)
        self.assertEqual(model.junk.get(model.c),1.0)
        inst = model.clone()
        self.assertEqual(inst.junk.get(model.c),None)
        self.assertEqual(inst.junk.get(inst.c),1.0)

    def test_clone_ConstraintArray(self):
        model = ConcreteModel()
        model.x = Var([1,2,3])
        model.c = Constraint([1,2,3],rule=lambda model,i: model.x[i] == 1.0)
        model.junk = Suffix()
        self.assertEqual(model.junk.get(model.c),None)
        self.assertEqual(model.junk.get(model.c[1]),None)
        model.junk.setValue(model.c,1.0)
        self.assertEqual(model.junk.get(model.c),None)
        self.assertEqual(model.junk.get(model.c[1]),1.0)
        inst = model.clone()
        self.assertEqual(inst.junk.get(model.c[1]),None)
        self.assertEqual(inst.junk.get(inst.c[1]),1.0)

    def test_clone_ConstraintData(self):
        model = ConcreteModel()
        model.x = Var([1,2,3])
        model.c = Constraint([1,2,3],rule=lambda model,i: model.x[i] == 1.0)
        model.junk = Suffix()
        self.assertEqual(model.junk.get(model.c[1]),None)
        model.junk.setValue(model.c[1],1.0)
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
        model.junk.setValue(model.obj,1.0)
        self.assertEqual(model.junk.get(model.obj),1.0)
        inst = model.clone()
        self.assertEqual(inst.junk.get(model.obj),None)
        self.assertEqual(inst.junk.get(inst.obj),1.0)

    def test_clone_ObjectiveArray(self):
        model = ConcreteModel()
        model.x = Var([1,2,3])
        model.obj = Objective([1,2,3], rule=lambda model,i: model.x[i])
        model.junk = Suffix()
        self.assertEqual(model.junk.get(model.obj),None)
        self.assertEqual(model.junk.get(model.obj[1]),None)
        model.junk.setValue(model.obj,1.0)
        self.assertEqual(model.junk.get(model.obj),None)
        self.assertEqual(model.junk.get(model.obj[1]),1.0)
        inst = model.clone()
        self.assertEqual(inst.junk.get(model.obj[1]),None)
        self.assertEqual(inst.junk.get(inst.obj[1]),1.0)

    def test_clone_ObjectiveData(self):
        model = ConcreteModel()
        model.x = Var([1,2,3])
        model.obj = Objective([1,2,3], rule=lambda model,i: model.x[i])
        model.junk = Suffix()
        self.assertEqual(model.junk.get(model.obj[1]),None)
        model.junk.setValue(model.obj[1],1.0)
        self.assertEqual(model.junk.get(model.obj[1]),1.0)
        inst = model.clone()
        self.assertEqual(inst.junk.get(model.obj[1]),None)
        self.assertEqual(inst.junk.get(inst.obj[1]),1.0)

    def test_clone_SimpleBlock(self):
        model = ConcreteModel()
        model.b = Block()
        model.junk = Suffix()
        self.assertEqual(model.junk.get(model.b),None)
        model.junk.setValue(model.b,1.0)
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
        model.junk.setValue(model.b,1.0)
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
        model.junk.setValue(model.b[1],1.0)
        self.assertEqual(model.junk.get(model.b[1]),1.0)
        inst = model.clone()
        self.assertEqual(inst.junk.get(model.b[1]),None)
        self.assertEqual(inst.junk.get(inst.b[1]),1.0)

    def test_clone_model(self):
        model = ConcreteModel()
        model.junk = Suffix()
        self.assertEqual(model.junk.get(model),None)
        model.junk.setValue(model,1.0)
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
        model.junk.setValue(model.x,1.0)
        self.assertEqual(model.junk.get(model.x),1.0)
        inst = pickle.loads(pickle.dumps(model))
        self.assertEqual(inst.junk.get(model.x),None)
        self.assertEqual(inst.junk.get(inst.x),1.0)

    def test_pickle_VarArray(self):
        model = ConcreteModel()
        model.x = Var([1,2,3])
        model.junk = Suffix()
        self.assertEqual(model.junk.get(model.x),None)
        self.assertEqual(model.junk.get(model.x[1]),None)
        model.junk.setValue(model.x,1.0)
        self.assertEqual(model.junk.get(model.x),None)
        self.assertEqual(model.junk.get(model.x[1]),1.0)
        inst = pickle.loads(pickle.dumps(model))
        self.assertEqual(inst.junk.get(model.x[1]),None)
        self.assertEqual(inst.junk.get(inst.x[1]),1.0)

    def test_pickle_VarData(self):
        model = ConcreteModel()
        model.x = Var([1,2,3])
        model.junk = Suffix()
        self.assertEqual(model.junk.get(model.x[1]),None)
        model.junk.setValue(model.x[1],1.0)
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
        model.junk.setValue(model.c,1.0)
        self.assertEqual(model.junk.get(model.c),1.0)
        inst = pickle.loads(pickle.dumps(model))
        self.assertEqual(inst.junk.get(model.c),None)
        self.assertEqual(inst.junk.get(inst.c),1.0)

    def test_pickle_ConstraintArray(self):
        model = ConcreteModel()
        model.x = Var([1,2,3])
        model.c = Constraint([1,2,3],rule=simple_con_rule)
        model.junk = Suffix()
        self.assertEqual(model.junk.get(model.c),None)
        self.assertEqual(model.junk.get(model.c[1]),None)
        model.junk.setValue(model.c,1.0)
        self.assertEqual(model.junk.get(model.c),None)
        self.assertEqual(model.junk.get(model.c[1]),1.0)
        inst = pickle.loads(pickle.dumps(model))
        self.assertEqual(inst.junk.get(model.c[1]),None)
        self.assertEqual(inst.junk.get(inst.c[1]),1.0)

    def test_pickle_ConstraintData(self):
        model = ConcreteModel()
        model.x = Var([1,2,3])
        model.c = Constraint([1,2,3],rule=simple_con_rule)
        model.junk = Suffix()
        self.assertEqual(model.junk.get(model.c[1]),None)
        model.junk.setValue(model.c[1],1.0)
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
        model.junk.setValue(model.obj,1.0)
        self.assertEqual(model.junk.get(model.obj),1.0)
        inst = pickle.loads(pickle.dumps(model))
        self.assertEqual(inst.junk.get(model.obj),None)
        self.assertEqual(inst.junk.get(inst.obj),1.0)

    def test_pickle_ObjectiveArray(self):
        model = ConcreteModel()
        model.x = Var([1,2,3])
        model.obj = Objective([1,2,3],rule=simple_obj_rule)
        model.junk = Suffix()
        self.assertEqual(model.junk.get(model.obj),None)
        self.assertEqual(model.junk.get(model.obj[1]),None)
        model.junk.setValue(model.obj,1.0)
        self.assertEqual(model.junk.get(model.obj),None)
        self.assertEqual(model.junk.get(model.obj[1]),1.0)
        inst = pickle.loads(pickle.dumps(model))
        self.assertEqual(inst.junk.get(model.obj[1]),None)
        self.assertEqual(inst.junk.get(inst.obj[1]),1.0)

    def test_pickle_ObjectiveData(self):
        model = ConcreteModel()
        model.x = Var([1,2,3])
        model.obj = Objective([1,2,3],rule=simple_obj_rule)
        model.junk = Suffix()
        self.assertEqual(model.junk.get(model.obj[1]),None)
        model.junk.setValue(model.obj[1],1.0)
        self.assertEqual(model.junk.get(model.obj[1]),1.0)
        inst = pickle.loads(pickle.dumps(model))
        self.assertEqual(inst.junk.get(model.obj[1]),None)
        self.assertEqual(inst.junk.get(inst.obj[1]),1.0)

    def test_pickle_SimpleBlock(self):
        model = ConcreteModel()
        model.b = Block()
        model.junk = Suffix()
        self.assertEqual(model.junk.get(model.b),None)
        model.junk.setValue(model.b,1.0)
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
        model.junk.setValue(model.b,1.0)
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
        model.junk.setValue(model.b[1],1.0)
        self.assertEqual(model.junk.get(model.b[1]),1.0)
        inst = pickle.loads(pickle.dumps(model))
        self.assertEqual(inst.junk.get(model.b[1]),None)
        self.assertEqual(inst.junk.get(inst.b[1]),1.0)

    def test_pickle_model(self):
        model = ConcreteModel()
        model.junk = Suffix()
        self.assertEqual(model.junk.get(model),None)
        model.junk.setValue(model,1.0)
        self.assertEqual(model.junk.get(model),1.0)
        inst = pickle.loads(pickle.dumps(model))
        self.assertEqual(inst.junk.get(model),None)
        self.assertEqual(inst.junk.get(inst),1.0)

if __name__ == "__main__":
    unittest.main()
