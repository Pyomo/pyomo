#
# Unit Tests for the Differential() Object
#

import os
import sys
from os.path import abspath, dirname
currdir = dirname(abspath(__file__))+os.sep

import coopr.environ
from coopr.pyomo import *
from coopr.dae import *
import pyutilib.th as unittest

class TestDifferential(unittest.TestCase):
    
    def setUp(self):
        self.model = ConcreteModel()
        m = self.model
        m.t1 = DifferentialSet(bounds=(0,1))
        m.t2 = DifferentialSet(initialize=[2,4,6])
        m.s = Set(initialize=[1,2,3])
        m.s2 = Set(initialize=[(1,1),(2,2),(3,3)])

        m.v1 = Var(m.t1)
        m.v2 = Var(m.s,m.t1)
        m.v3 = Var(m.t1,m.s,m.t2)
        m.v4 = Var(m.s2,m.t1,m.s)

    # test __init__
    def test_init(self):
        m = self.model
        def _vdot1(model,ti):
            return model.v1[ti]
        def _vdot2(model,si,ti):
            return model.v2[si,ti]
        def _vdot3(model,t1i,si,t2i):
            return model.v3[t1i,si,t2i]
        def _vdot4(model,i,j,ti,si):
            return model.v4[i,j,ti,si]

        def _vdot2_bounds(model,si,ti):
            return (0,si)

        m.vdot1 = Differential(dvar=m.v1,expr=_vdot1)
        m.vdot2 = Differential(dv=m.v2,rule=_vdot2)
        m.vdot3 = Differential(dv=m.v3,rule=_vdot3,ds=m.t2)
        m.vdot4 = Differential(dv=m.v4,rule=_vdot4)
        
        del m.vdot1
        del m.vdot2
        del m.vdot3

        m.vdot1 = Differential(dv=m.v1,rule=_vdot1, bounds=(0,10), initialize=2)
        m.vdot2 = Differential(dv=m.v2,ds=m.t1, \
                                       bounds=_vdot2_bounds,rule=_vdot2)
        m.vdot3 = Differential(dvar=m.v3,expr=_vdot3,\
                                       bounds=(-10,10),dset=m.t1)
        

    # test bad keyword arguments
    def test_bad_kwds(self):
        m = self.model
                
        def _vdot(model,ti):
            return model.v1[ti]
        
        try:
            m.vdot = Differential(dv=m.v1,rule=_vdot,expr=_vdot)
            self.fail("Expected TypeError because both rule and expr are given as keyword arguments")
        except TypeError:
            pass

        try:
            m.vdot = Differential(dv=m.v1,dvar=m.v1,rule=_vdot)
            self.fail("Expected TypeError because both dv and dvar are given as keyword arguments")
        except TypeError:
            pass

        try:
            m.vdot = Differential(dv=m.v1,rule=_vdot,ds=m.t1,dset=m.t1)
            self.fail("Expected TypeError because both ds and dset are given as keyword arguments")
        except TypeError:
            pass
        

    # test valid declarations
    def test_valid_declaration(self):
        m = self.model
        m.bound_vdot2 = Param(m.s,m.t1,default=5)
        m.init_vdot2 = Param(m.t1,default=3)

        def _vdot1(model,ti):
            return model.v1[ti]
        def _vdot2(model,si,ti):
            return model.v2[si,ti]
        def _vdot3(model,t1i,si,t2i):
            return model.v3[t1i,si,t2i]
        def _vdot4(model,i,j,ti,si):
            return model.v4[i,j,ti,si]

        def _vdot2_bounds(model,si,ti):
            return (0,model.bound_vdot2[si,ti])             

        def _vdot2_init(model,si,ti):
            return model.init_vdot2[ti]

        m.vdot1 = Differential(dv=m.v1,rule=_vdot1, bounds=(0,10),initialize=2)
        m.vdot2 = Differential(dv=m.v2,ds=m.t1, \
                                       bounds=_vdot2_bounds,rule=_vdot2, initialize=_vdot2_init)
        m.vdot3 = Differential(dvar=m.v3,expr=_vdot3,\
                                       bounds=(-10,10),dset=m.t1)
        m.vdot4 = Differential(dv=m.v4,rule=_vdot4)

        self.assertTrue(None in m.vdot1._non_ds)
        self.assertTrue(m.s in m.vdot2._non_ds)
        self.assertTrue(m.t2 in m.vdot3._non_ds)
        self.assertTrue(m.s in m.vdot3._non_ds)

        self.assertTrue(m.t1 == m.vdot1._ds)
        self.assertTrue(m.t1 == m.vdot2._ds)
        self.assertTrue(m.t1 == m.vdot3._ds)
        
        self.assertTrue(m.v1 == m.vdot1._dv)
        self.assertTrue(m.v2 == m.vdot2._dv)
        self.assertTrue(m.v3 == m.vdot3._dv)

        self.assertTrue(m.vdot1._ds_argindex == 0)
        self.assertTrue(m.vdot2._ds_argindex == 1)
        self.assertTrue(m.vdot3._ds_argindex == 0)
        self.assertTrue(m.vdot4._ds_argindex == 2)

        self.assertTrue(str(m.vdot1) is 'vdot1')
        self.assertTrue(m.vdot1.name is 'vdot1')

        self.assertEqual(m.vdot1[1].value, 2)
        self.assertEqual(m.vdot2[1,0].value, 3)
        self.assertEqual(m.vdot3[0,1,2].value,None)
        
    # test invalid declarations
    def test_invalid_declaration(self):
        m = self.model
        m.tmp = Var(m.s)
        m.tmp2 = Var(m.s,m.s)
        
        def _tmpdot(model,si):
            return model.tmp[si]
        def _tmp2dot(model,i,j):
            return model.tmp2[i,j]
        def _vdot1(model,ti):
            return model.v1[ti]
        def _vdot2(model,si,ti):
            return model.v2[si,ti]
        def _vdot3(model,t1i,si,t2i):
            return model.v3[t1i,si,t2i]

        try:
            m.vdot2 = Differential(m.s,dv=m.v1,rule=_vdot1)
            self.fail("Expected ValueError because a positional argument was specified")
        except ValueError:
            pass

        try:
            m.vdot3 = Differential(rule=_vdot3)
            self.fail("Expected TypeError because no differential variable specified")
        except TypeError:
            pass

        try:
            m.vdot2 = Differential(dv=m.v2)
            self.fail("Expected TypeError because no rule was specified")
        except TypeError:
            pass

        try:
            m.tmpdot = Differential(dv=m.tmp,rule=_tmpdot)
            self.fail("Expected IndexError because the diffvar isn't indexed by a differentialset")
        except IndexError:
            pass

        try:
            m.tmp2dot = Differential(dv=m.tmp2,rule=_tmp2dot)
            self.fail("Expected IndexError because the diffvar isn't indexed by a differentialset")
        except IndexError:
            pass

        try:
            m.vdot2 = Differential(dv=m.v2,ds=m.t2,rule=_vdot2)
            self.fail("Expected TypeError because the specified differentialset is not"\
                          " an indexing set of the diffvar")
        except TypeError:
            pass

        try:
            m.vdot = Differential(dv=m.v1,ds=m.t2,rule=_vdot1)
            self.fail("Expected TypeError because the specified differentialset is not"\
                          " an indexing set of the diffvar")
        except TypeError:
            pass

        try:
            m.vdot3 = Differential(dv=m.v3,rule=_vdot3)
            self.fail("Expected TypeError because a differentialset was not specified")
        except TypeError:
            pass

    # test when keyword arguments with the wrong type are specified
    def test_badtype_kwds(self):
        m = self.model

        def _vdot1(model,ti):
            return model.v1[ti]
        def _vdot2(model,si,ti):
            return model.v2[si,ti]
        def _vdot3(model,t1i,si,t2i):
            return model.v3[t1i,si,t2i]

        try:
            m.vdot2 = Differential(dv=m.v2,rule=m.v2)
            self.fail("Expected TypeError becasue the rule is not a function")
        except TypeError:
            pass

        try:
            m.vdot2 = Differential(dv=m.t2,rule=_vdot2)
            self.fail("Expected TypeError because the specified differential" \
                          " variable is not a variable")
        except TypeError:
            pass

        try:
            m.vdot2 = Differential(dv=m.v2,rule=_vdot2,ds=m.s)
            self.fail("Expected TypeErrpr because the specified differential "\
                          "set is not a DifferentialSet component")
        except TypeError:
            pass

    # test __str__
    def test_str(self):
        m = self.model
        def _vdot1(model,ti):
            return model.v1[ti]
        m.vdot1 = Differential(dv=m.v1,rule=_vdot1, bounds=(0,10))
        self.assertTrue(m.vdot1.__str__() == "vdot1")

    # test pprint()
    def test_pprint(self):
        m = self.model
        def _vdot1(model,ti):
            return model.v1[ti]
        def _vdot2(model,si,ti):
            return model.v2[si,ti]
        def _vdot3(model,t1i,si,t2i):
            return model.v3[t1i,si,t2i]
        def _vdot4(model,i,j,ti,si):
            return model.v4[i,j,ti,si]

        def _vdot2_bounds(model,si,ti):
            return (0,si)             

        m.vdot1 = Differential(dv=m.v1,rule=_vdot1, bounds=(0,10))
        m.vdot2 = Differential(dv=m.v2,ds=m.t1, \
                                       bounds=_vdot2_bounds,rule=_vdot2)
        m.vdot3 = Differential(dvar=m.v3,expr=_vdot3,\
                                       bounds=(-10,10),dset=m.t1)
        m.vdot4 = Differential(dv=m.v4,rule=_vdot4)

        from six.moves import StringIO
        output = StringIO()
        m.vdot1.pprint(ostream=output)
        m.vdot2.pprint(ostream=output)
        m.vdot3.pprint(ostream=output)
        m.vdot4.pprint(ostream=output)

    # test construct
    def test_construct(self):
        mod = AbstractModel()
        mod.s = Set(initialize=[1,2,3])
        mod.t = DifferentialSet(bounds=(0,1))
        mod.v = Var(mod.s,mod.t)
        def _vdot(model,i,j):
            return model.v[i,j]

        mod.vdot = Differential(dv=mod.v,rule=_vdot)
        mod.vdot.pprint()
        mod.s.construct()
        mod.t.construct()
        mod.v.construct()
        self.assertFalse(mod.vdot._constructed)
        mod.vdot.construct()
        self.assertTrue(mod.vdot._constructed)
        mod.vdot.construct()

    # test __getitem__
    def test_getitem(self):
        m = self.model
        def _vdot1(model,ti):
            return model.v1[ti]
        def _vdot2(model,si,ti):
            return model.v2[si,ti]

        m.vdot1 = Differential(dv=m.v1,rule=_vdot1, bounds=(0,10))
        m.vdot2 = Differential(dv=m.v2,ds=m.t1,rule=_vdot2)

        for i in m.v1.keys():
            try:
                tmp = m.vdot1[i]
            except KeyError:
                self.fail("A valid index should not throw a KeyError")

        for i in m.v2.keys():
            try:
                tmp = m.vdot2[i]
            except KeyError:
                self.fail("A valid index shoulf not throw a KeyError")
        
        try:
            tmp = m.vdot1[100]
            self.fail("Expected KeyError because the supplied index is not a vlid index")
        except KeyError:
            pass
                
if __name__ == "__main__":
    unittest.main()
