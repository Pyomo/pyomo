#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

#
# Unit Tests for pyomo.dae.misc
#

import os
from os.path import abspath, dirname
currdir = dirname(abspath(__file__))+os.sep

import pyutilib.th as unittest

from pyomo.environ import *
from pyomo.dae import *
from pyomo.dae.misc import *

class TestDaeMisc(unittest.TestCase):
    
    # test generate_finite_elements method
    def test_generate_finite_elements(self):
        m = ConcreteModel()
        m.t = ContinuousSet(bounds=(0,10))
        m.t2 = ContinuousSet(bounds=(0,10))
        m.t3 = ContinuousSet(bounds=(0,1))

        oldt = sorted(m.t)
        generate_finite_elements(m.t,1)
        self.assertTrue(oldt == sorted(m.t))
        self.assertFalse(m.t.get_changed())
        generate_finite_elements(m.t,2)
        self.assertFalse(oldt == sorted(m.t))
        self.assertTrue(m.t.get_changed())
        self.assertTrue([0,5.0,10] == sorted(m.t))
        generate_finite_elements(m.t,3)
        self.assertTrue([0,2.5,5.0,10] == sorted(m.t))
        generate_finite_elements(m.t,5)
        self.assertTrue([0,1.25,2.5,5.0,7.5,10] == sorted(m.t))

        generate_finite_elements(m.t2,10)
        self.assertTrue(len(m.t2) == 11)
        self.assertTrue([0,1,2,3,4,5,6,7,8,9,10] == sorted(m.t2))

        generate_finite_elements(m.t3,7)
        self.assertTrue(len(m.t3) == 8)
        t = sorted(m.t3)
        print(t[1])
        self.assertTrue(t[1] == 0.142857)
      
    # test generate_collocation_points method
    def test_generate_collocation_points(self):
        m = ConcreteModel()
        m.t = ContinuousSet(initialize=[0,1])
        m.t2 = ContinuousSet(initialize=[0,2,4,6])
        
        tau1 = [1]
        oldt = sorted(m.t)
        generate_colloc_points(m.t,tau1)
        self.assertTrue(oldt == sorted(m.t))
        self.assertFalse(m.t.get_changed())

        tau1 = [0.5]
        oldt = sorted(m.t)
        generate_colloc_points(m.t,tau1)
        self.assertFalse(oldt == sorted(m.t))
        self.assertTrue(m.t.get_changed())
        self.assertTrue([0,0.5,1] == sorted(m.t))

        tau2 = [0.2,0.3,0.7,0.8,1]
        generate_colloc_points(m.t,tau2)
        self.assertTrue(len(m.t) == 11)
        self.assertTrue([0,0.1,0.15,0.35,0.4,0.5,0.6,0.65,0.85,0.9,1] == sorted(m.t))

        generate_colloc_points(m.t2,tau2)
        self.assertTrue(len(m.t2) == 16)
        self.assertTrue(m.t2.get_changed())
        t = sorted(m.t2)
        self.assertTrue(t[1] == 0.4)
        self.assertTrue(t[13] == 5.4)

    # test Params indexed only by a differentialset after discretizing
    def test_discretized_params_single(self):
        m = ConcreteModel()
        m.t = ContinuousSet(bounds=(0,10))
        m.s1 = Set(initialize=[1,2,3])
        m.s2 = Set(initialize=[(1,1),(2,2)])
        m.p1 = Param(m.t, initialize=1)
        m.p2 = Param(m.t, default=2)
        m.p3 = Param(m.t, initialize=1, default=2)
        def _rule1(m,i):
            return i**2
        def _rule2(m,i):
            return 2*i
        m.p4 = Param(m.t, initialize={0:5,10:5}, default=_rule1)
        m.p5 = Param(m.t, initialize=_rule1, default=_rule2)

        generate_finite_elements(m.t,5)
        
        try:
            for i in m.t:
                m.p1[i]
            self.fail("Expected ValueError because no default value "\
                          "was specified")
        except ValueError:
            pass

        for i in m.t:
            self.assertEqual(m.p2[i], 2)

            if i == 0 or i == 10:
                self.assertEqual(m.p3[i],1)
                self.assertEqual(m.p4[i],5)
                self.assertEqual(m.p5[i],i**2)
            else:
                self.assertEqual(m.p3[i],2)
                self.assertEqual(m.p4[i],i**2)
                self.assertEqual(m.p5[i], 2*i)

    # test Params with multiple indexing sets after discretizing
    def test_discretized_params_multiple(self):
        m = ConcreteModel()
        m.t = ContinuousSet(bounds=(0,10))
        m.s1 = Set(initialize=[1,2,3])
        m.s2 = Set(initialize=[(1,1),(2,2)])
        def _rule1(m,i):
            return i**2
        m.p1 = Param(m.s1,m.t,initialize=2,default=_rule1)
        m.p2 = Param(m.t,m.s1,default=5)
        def _rule2(m,i,j):
            return i+j
        m.p3 = Param(m.s1,m.t,initialize=2,default=_rule2)
        def _rule3(m,i,j,k):
            return i+j+k
        m.p4 = Param(m.s2,m.t,default=_rule3)

        generate_finite_elements(m.t,5)

        try:
            for i in m.p1:
                m.p1[i]
            self.fail("Expected TypeError because a function with the "\
                          "wrong number of arguments was specified as "\
                          "the default")
        except TypeError:
            pass

        for i in m.p2:
            self.assertEqual(m.p2[i],5)

        for i in m.t:
            for j in m.s1:
                if i==0 or i==10:
                    self.assertEqual(m.p3[j,i],2)
                else:
                    self.assertEqual(m.p3[j,i],i+j)

        for i in m.t:
            for j in m.s2:
                self.assertEqual(m.p4[j,i],sum(j,i))
        
        
    # test update_contset_indexed_component method for Vars with 
    # single index of the differentialset
    def test_update_contset_indexed_component_vars_single(self):
        m = ConcreteModel()
        m.t = ContinuousSet(bounds=(0,10))
        m.t2 = ContinuousSet(initialize=[1,2,3])
        m.s = Set(initialize=[1,2,3])
        m.v1 = Var(m.t,initialize=3)
        m.v2 = Var(m.t,bounds=(4,10),initialize={0:2,10:12})
        def _init(m,i):
            return i
        m.v3 = Var(m.t,bounds=(-5,5),initialize=_init)
        m.v4 = Var(m.s,initialize=7, dense=True)
        m.v5 = Var(m.t2, dense=True)

        generate_finite_elements(m.t,5)
        update_contset_indexed_component(m.v1)
        update_contset_indexed_component(m.v2)
        update_contset_indexed_component(m.v3)
        update_contset_indexed_component(m.v4)
        update_contset_indexed_component(m.v5)

        self.assertTrue(len(m.v1)==6)
        self.assertTrue(len(m.v2)==6)
        self.assertTrue(len(m.v3)==6)
        self.assertTrue(len(m.v4)==3)
        self.assertTrue(len(m.v5)==3)

        self.assertTrue(value(m.v1[2])==3)
        self.assertTrue(m.v1[4].ub is None)
        self.assertTrue(m.v1[6].lb is None)
        
        self.assertTrue(m.v2[2].value is None)
        self.assertTrue(m.v2[4].lb == 4)
        self.assertTrue(m.v2[8].ub == 10)
        self.assertTrue(value(m.v2[0])==2)

        self.assertTrue(value(m.v3[2]) == 2)
        self.assertTrue(m.v3[4].lb == -5)
        self.assertTrue(m.v3[6].ub == 5)
        self.assertTrue(value(m.v3[8]) == 8)

    # test update_contset_indexed_component method for Vars with 
    # multiple indices
    def test_update_contset_indexed_component_vars_multiple(self):
        m = ConcreteModel()
        m.t = ContinuousSet(bounds=(0,10))
        m.t2 = ContinuousSet(initialize=[1,2,3])
        m.s = Set(initialize=[1,2,3])
        m.s2 = Set(initialize=[(1,1),(2,2)])
        m.v1 = Var(m.s,m.t,initialize=3)
        m.v2 = Var(m.s,m.t,m.t2,bounds=(4,10),initialize={(1,0,1):22,(2,10,2):22})
        def _init(m,i,j,k):
            return i
        m.v3 = Var(m.t,m.s2,bounds=(-5,5),initialize=_init)
        m.v4 = Var(m.s,m.t2,initialize=7, dense=True)

        generate_finite_elements(m.t,5)
        update_contset_indexed_component(m.v1)
        update_contset_indexed_component(m.v2)
        update_contset_indexed_component(m.v3)
        update_contset_indexed_component(m.v4)

        self.assertTrue(len(m.v1)==18)
        self.assertTrue(len(m.v2)==54)
        self.assertTrue(len(m.v3)==12)
        self.assertTrue(len(m.v4)==9)

        self.assertTrue(value(m.v1[1,4])==3)
        self.assertTrue(m.v1[2,2].ub is None)
        self.assertTrue(m.v1[3,8].lb is None)
        
        self.assertTrue(value(m.v2[1,0,1]) == 22)
        self.assertTrue(m.v2[1,2,1].value is None)
        self.assertTrue(m.v2[2,4,3].lb == 4)
        self.assertTrue(m.v2[3,8,1].ub == 10)

        self.assertTrue(value(m.v3[2,2,2])==2)
        self.assertTrue(m.v3[4,1,1].lb == -5)
        self.assertTrue(m.v3[8,2,2].ub == 5)
        self.assertTrue(value(m.v3[6,1,1]) == 6)

    # test update_contset_indexed_component method for Constraints with
    # single index of the differentialset
    def test_update_contset_indexed_component_constraints_single(self):
        m = ConcreteModel()
        m.t = ContinuousSet(bounds=(0,10))
        m.p = Param(m.t,default=3)
        m.v = Var(m.t, initialize=5)
        
        def _con1(m,i):
            return m.p[i]*m.v[i] <= 20
        m.con1 = Constraint(m.t, rule=_con1)
        
        # Rules that iterate over a differentialset implicitly are not updated
        # after the discretization
        def _con2(m):
            return sum(m.v[i] for i in m.t) >= 0
        m.con2 = Constraint(rule=_con2)

        generate_finite_elements(m.t,5)
        update_contset_indexed_component(m.v)
        update_contset_indexed_component(m.p)
        update_contset_indexed_component(m.con1)
        update_contset_indexed_component(m.con2)

        self.assertTrue(len(m.con1)==6)
        self.assertEqual(m.con1[2](), 15)
        self.assertEqual(m.con1[8](), 15)
        self.assertEqual(m.con2(), 10)

    # test update_contset_indexed_component method for Constraints with
    # multiple indices
    def test_update_contset_indexed_component_constraints_multiple(self):
        m = ConcreteModel()
        m.t = ContinuousSet(bounds=(0,10))
        m.t2 = ContinuousSet(initialize=[1,2,3])
        m.s1 = Set(initialize=[1,2,3])
        m.s2 = Set(initialize=[(1,1),(2,2)])
        def _init(m,i,j):
            return j+i
        m.p1 = Param(m.s1,m.t,default=_init)
        m.v1 = Var(m.s1, m.t, initialize=5)
        m.v2 = Var(m.s2,m.t,initialize=2)
        m.v3 = Var(m.t2,m.s2, initialize=1)

        def _con1(m,si,ti):
            return m.v1[si,ti]*m.p1[si,ti] >= 0
        m.con1 = Constraint(m.s1,m.t,rule=_con1)

        def _con2(m,i,j,ti):
            return m.v2[i,j,ti]+m.p1[1,ti] == 10
        m.con2 = Constraint(m.s2,m.t,rule=_con2)

        def _con3(m,i,ti,ti2,j,k):
            return m.v1[i,ti]-m.v3[ti2,j,k]*m.p1[i,ti] <= 20
        m.con3 = Constraint(m.s1,m.t,m.t2,m.s2,rule=_con3)

        generate_finite_elements(m.t,5)
        update_contset_indexed_component(m.p1)
        update_contset_indexed_component(m.v1)
        update_contset_indexed_component(m.v2)
        update_contset_indexed_component(m.v3)
        update_contset_indexed_component(m.con1)
        update_contset_indexed_component(m.con2)
        update_contset_indexed_component(m.con3)
        
        self.assertTrue(len(m.con1) == 18)
        self.assertTrue(len(m.con2) == 12)
        self.assertTrue(len(m.con3) == 108)

        self.assertEqual(m.con1[1,4](), 25)
        self.assertEqual(m.con1[2,6](), 40)
        self.assertEqual(m.con1[3,8](), 55)
        self.assertTrue(value(m.con1[2,4].lower) == 0)
        self.assertTrue(value(m.con1[1,8].upper) == None)

        self.assertEqual(m.con2[1,1,2](), 5)
        self.assertEqual(m.con2[2,2,4](), 7)
        self.assertEqual(m.con2[1,1,8](), 11)
        self.assertTrue(value(m.con2[2,2,6].lower) == 10)
        self.assertTrue(value(m.con2[1,1,10].upper) == 10)
        
        self.assertEqual(m.con3[1,2,1,1,1](), 2)
        self.assertEqual(m.con3[1,4,1,2,2](), 0)
        self.assertEqual(m.con3[2,6,3,1,1](), -3)
        self.assertEqual(m.con3[3,8,2,2,2](), -6)
        self.assertTrue(value(m.con3[2,0,2,1,1].lower) == None)
        self.assertTrue(value(m.con3[3,2,3,2,2].upper) == 20)                   
 
    # test update_contset_indexed_component method for Constraints with
    # single index of the differentialset
    def test_update_contset_indexed_component_expressions_single(self):
        m = ConcreteModel()
        m.t = ContinuousSet(bounds=(0,10))
        m.p = Param(m.t,default=3)
        m.v = Var(m.t, initialize=5)

        def _con1(m,i):
            return m.p[i]*m.v[i]
        m.con1 = Expression(m.t, rule=_con1)

        # Rules that iterate over a differentialset implicitly are not updated
        # after the discretization
        def _con2(m):
            return sum(m.v[i] for i in m.t)
        m.con2 = Expression(rule=_con2)

        generate_finite_elements(m.t,5)
        update_contset_indexed_component(m.v)
        update_contset_indexed_component(m.p)
        update_contset_indexed_component(m.con1)
        update_contset_indexed_component(m.con2)

        self.assertTrue(len(m.con1)==6)
        self.assertEqual(m.con1[2](), 15)
        self.assertEqual(m.con1[8](), 15)
        self.assertEqual(m.con2(), 10)

    # test update_contset_indexed_component method for Constraints with
    # multiple indices
    def test_update_contset_indexed_component_expressions_multiple(self):
        m = ConcreteModel()
        m.t = ContinuousSet(bounds=(0,10))
        m.t2 = ContinuousSet(initialize=[1,2,3])
        m.s1 = Set(initialize=[1,2,3])
        m.s2 = Set(initialize=[(1,1),(2,2)])
        def _init(m,i,j):
            return j+i
        m.p1 = Param(m.s1,m.t,default=_init)
        m.v1 = Var(m.s1, m.t, initialize=5)
        m.v2 = Var(m.s2,m.t,initialize=2)
        m.v3 = Var(m.t2,m.s2, initialize=1)

        def _con1(m,si,ti):
            return m.v1[si,ti]*m.p1[si,ti]
        m.con1 = Expression(m.s1,m.t,rule=_con1)

        def _con2(m,i,j,ti):
            return m.v2[i,j,ti]+m.p1[1,ti]
        m.con2 = Expression(m.s2,m.t,rule=_con2)

        def _con3(m,i,ti,ti2,j,k):
            return m.v1[i,ti]-m.v3[ti2,j,k]*m.p1[i,ti]
        m.con3 = Expression(m.s1,m.t,m.t2,m.s2,rule=_con3)

        generate_finite_elements(m.t,5)
        update_contset_indexed_component(m.p1)
        update_contset_indexed_component(m.v1)
        update_contset_indexed_component(m.v2)
        update_contset_indexed_component(m.v3)
        update_contset_indexed_component(m.con1)
        update_contset_indexed_component(m.con2)
        update_contset_indexed_component(m.con3)

        self.assertTrue(len(m.con1) == 18)
        self.assertTrue(len(m.con2) == 12)
        self.assertTrue(len(m.con3) == 108)

        self.assertEqual(m.con1[1,4](), 25)
        self.assertEqual(m.con1[2,6](), 40)
        self.assertEqual(m.con1[3,8](), 55)

        self.assertEqual(m.con2[1,1,2](), 5)
        self.assertEqual(m.con2[2,2,4](), 7)
        self.assertEqual(m.con2[1,1,8](), 11)

        self.assertEqual(m.con3[1,2,1,1,1](), 2)
        self.assertEqual(m.con3[1,4,1,2,2](), 0)
        self.assertEqual(m.con3[2,6,3,1,1](), -3)
        self.assertEqual(m.con3[3,8,2,2,2](), -6)

    # test update_contset_indexed_component method for other components
    def test_update_contset_indexed_component_other(self):
        m = ConcreteModel()
        m.t = ContinuousSet(bounds=(0,10))
        m.junk = Suffix()
        m.s = Set(initialize=[1,2,3])
        m.v = Var(m.s)
        def _obj(m):
            return sum(m.v[i] for i in m.s)
        m.obj = Objective(rule=_obj)

        generate_finite_elements(m.t,5)
        update_contset_indexed_component(m.junk)
        update_contset_indexed_component(m.s)
        update_contset_indexed_component(m.obj)
        
    # test add_equality_constraints method
    # def test_add_equality_constraints(self):
    #     m = ConcreteModel()
    #     m.t = DifferentialSet(initialize=[1,2,3,4,5])
    #     m.s = Set(initialize=[1,2])
    #     m.v = Var(m.t)
    #     m.v2 = Var(m.s,m.t,initialize=2)
        
    #     def _vdot(m,i):
    #         return m.v[i]**2
    #     m.vdot = Differential(dv=m.v,rule=_vdot)

    #     def _vdot2(m,i,j):
    #         return m.v2[i,j]**2
    #     m.vdot2 = Differential(dv=m.v2,rule=_vdot2)

    #     self.assertTrue(len(m.vdot._cons)==0)
    #     add_equality_constraints(m.vdot)
    #     self.assertTrue(len(m.vdot._cons)==4)
    #     m.vdot._lhs_var[2] = 3
    #     m.v[2] = 3
    #     self.assertEqual(m.vdot._cons[1](),-6)

    #     add_equality_constraints(m.vdot2)
    #     self.assertTrue(len(m.vdot2._cons)==8)
    #     m.vdot2._lhs_var[1,2] = 3
    #     self.assertEqual(m.vdot2._cons[1](),-1)

    # test add_equality_constraints method with invalid 
    # Differential rules
    # def test_bad_add_equality_constraint(self):
    #     m = ConcreteModel()
    #     m.t = DifferentialSet(initialize=[1,2,3])
    #     m.v = Var(m.t)

    #     def _vdot1(m,i,j):
    #         return m.v[i,j]**2
    #     m.vdot1 = Differential(dv=m.v,rule=_vdot1)

    #     def _vdot2(m,i):
    #         return m.v[i] == 3
    #     m.vdot2 = Differential(dv=m.v,rule=_vdot2)

    #     try:
    #         add_equality_constraints(m.vdot1)
    #         self.fail("Expected TypeError because the rule supplied to the differential "\
    #                       "had the wrong number of arguments")
    #     except TypeError:
    #         pass

    #     try:
    #         add_equality_constraints(m.vdot2)
    #         self.fail("Expected TypeError because the rule supplied to the differential "\
    #                       "returns an invalid expression")
    #     except TypeError:
    #         pass

if __name__ == "__main__":
    unittest.main()
