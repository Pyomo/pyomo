# ____________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and
# Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
# rights in this software.
# This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________

"""
Unit Tests for interfacing with sIPOPT 
"""

import pyutilib.th as unittest

from pyomo.environ import ConcreteModel, Param, Var, Block,  Suffix, value
from pyomo.opt import SolverFactory
from pyomo.dae import ContinuousSet
from pyomo.dae.simulator import scipy_available
from pyomo.core.expr.current import identify_variables
from pyomo.contrib.sensitivity_toolbox.sens import sipopt
import pyomo.contrib.sensitivity_toolbox.examples.feedbackController as fc
import pyomo.contrib.sensitivity_toolbox.examples.rangeInequality as ri
import pyomo.contrib.sensitivity_toolbox.examples.HIV_Transmission as hiv

opt = SolverFactory('ipopt_sens', solver_io='nl')

class TestSensitivityToolbox(unittest.TestCase):

    #test arguments
    @unittest.skipIf(not opt.available(False), "ipopt_sens is not available")
    def test_bad_arg(self):
        m = ConcreteModel()
        m.t = ContinuousSet(bounds=(0,1))

        m.a = Param(initialize=1, mutable=True)
        m.b = Param(initialize=2, mutable=True)
        m.c = Param(initialize=3, mutable=False)

        m.x = Var(m.t)

        list_one = [m.a,m.b]
        list_two = [m.a,m.b,m.c]
        list_three = [m.a, m.x]
        list_four = [m.a,m.c]

        #verify ValueError thrown when param and perturb list are different
        #  lengths
        try:
            Result = sipopt(m,list_one,list_two)
            self.fail("Expected ValueError: for different length lists")
        except ValueError:
            pass

        #verify ValueError thrown when param list has a Var in it
        try:
            Result = sipopt(m,list_three,list_two)
            self.fail("Expected ValueError: variable sent through paramSubList")
        except ValueError:
            pass

        #verify ValueError thrown when perturb list has Var in it
        try:
            Result = sipopt(m,list_one,list_three)
            self.fail("Expected ValueError: variable sent through perturbList")
        except ValueError:
            pass

        #verify ValueError thrown when param list has an unmutable param
        try:
            Result = sipopt(m,list_four,list_one)
            self.fail("Expected ValueError:" 
                       "unmutable param sent through paramSubList")
        except ValueError:
            pass



    #test feedbackController Solution when the model gets cloned
    @unittest.skipIf(not scipy_available, "scipy is required for this test")
    @unittest.skipIf(not opt.available(False), "ipopt_sens is not available")
    def test_clonedModel_soln(self):

        m_orig = fc.create_model()
        fc.initialize_model(m_orig,100)

        m_orig.perturbed_a = Param(initialize=-0.25)
        m_orig.perturbed_H = Param(initialize=0.55)

        m_sipopt = sipopt(m_orig,[m_orig.a,m_orig.H],
                            [m_orig.perturbed_a,m_orig.perturbed_H],
                            cloneModel=True)        
  
        #verify cloned model has _sipopt_data block
        #    and original model is untouched
        self.assertFalse(m_sipopt == m_orig)

        self.assertTrue(hasattr(m_sipopt,'_sipopt_data') and
                        m_sipopt._sipopt_data.ctype is Block)

        self.assertFalse(hasattr(m_orig,'_sipopt_data'))
        self.assertFalse(hasattr(m_orig,'b'))

        #verify variable declaration
        self.assertTrue(hasattr(m_sipopt._sipopt_data,'a') and 
                        m_sipopt._sipopt_data.a.ctype is Var)
        self.assertTrue(hasattr(m_sipopt._sipopt_data,'H') and 
                        m_sipopt._sipopt_data.H.ctype is Var)
   
        #verify suffixes
        self.assertTrue(hasattr(m_sipopt,'sens_state_0') and
                        m_sipopt.sens_state_0.ctype is Suffix and
                        m_sipopt.sens_state_0[m_sipopt._sipopt_data.H]==2 and
                        m_sipopt.sens_state_0[m_sipopt._sipopt_data.a]==1)
  
        self.assertTrue(hasattr(m_sipopt,'sens_state_1') and
                        m_sipopt.sens_state_1.ctype is Suffix and
                        m_sipopt.sens_state_1[m_sipopt._sipopt_data.H]==2 and
                        m_sipopt.sens_state_1[m_sipopt._sipopt_data.a]==1)  

        self.assertTrue(hasattr(m_sipopt,'sens_state_value_1') and
                        m_sipopt.sens_state_value_1.ctype is Suffix and
                        m_sipopt.sens_state_value_1[
                                        m_sipopt._sipopt_data.H]==0.55 and
                        m_sipopt.sens_state_value_1[
                                        m_sipopt._sipopt_data.a]==-0.25)
  
        self.assertTrue(hasattr(m_sipopt,'sens_init_constr') and
                        m_sipopt.sens_init_constr.ctype is Suffix and
                        m_sipopt.sens_init_constr[
                                     m_sipopt._sipopt_data.paramConst[1]]==1 and
                        m_sipopt.sens_init_constr[
                                     m_sipopt._sipopt_data.paramConst[2]]==2)

        self.assertTrue(hasattr(m_sipopt,'sens_sol_state_1') and
                        m_sipopt.sens_sol_state_1.ctype is Suffix)
        self.assertAlmostEqual(
                        m_sipopt.sens_sol_state_1[
                           m_sipopt.F[15]],-0.00102016765,8)

        self.assertTrue(hasattr(m_sipopt,'sens_sol_state_1_z_L') and
                        m_sipopt.sens_sol_state_1_z_L.ctype is Suffix)
        self.assertAlmostEqual(
                        m_sipopt.sens_sol_state_1_z_L[
                           m_sipopt.u[15]],-2.181712e-09,13)

        self.assertTrue(hasattr(m_sipopt,'sens_sol_state_1_z_U') and
                        m_sipopt.sens_sol_state_1_z_U.ctype is Suffix)
        self.assertAlmostEqual(
                        m_sipopt.sens_sol_state_1_z_U[
                           m_sipopt.u[15]],6.580899e-09,13)

        #verify deactivated constraints for cloned model
        self.assertFalse(m_sipopt.FDiffCon[0].active and
                         m_sipopt.FDiffCon[7.5].active and
                         m_sipopt.FDiffCon[15].active )

        self.assertFalse(m_sipopt.x_dot[0].active and
                         m_sipopt.x_dot[7.5].active and
                         m_sipopt.x_dot[15].active )

        #verify constraints on original model are still active
        self.assertTrue(m_orig.FDiffCon[0].active and
                        m_orig.FDiffCon[7.5].active and
                        m_orig.FDiffCon[15].active )

        self.assertTrue(m_orig.x_dot[0].active and
                        m_orig.x_dot[7.5].active and
                        m_orig.x_dot[15].active )

        #verify solution
        self.assertAlmostEqual(value(m_sipopt.J),0.0048956783,8)
         

    @unittest.skipIf(not scipy_available, "scipy is required for this test")
    @unittest.skipIf(not opt.available(False), "ipopt_sens is not available")
    def test_noClone_soln(self):

        m_orig = fc.create_model()
        fc.initialize_model(m_orig,100)

        m_orig.perturbed_a = Param(initialize=-0.25)
        m_orig.perturbed_H = Param(initialize=0.55)

        m_sipopt = sipopt(m_orig,[m_orig.a,m_orig.H],
                            [m_orig.perturbed_a,m_orig.perturbed_H],
                            cloneModel=False)

        self.assertTrue(m_sipopt == m_orig)

        #test _sipopt_data block exists
        self.assertTrue(hasattr(m_orig,'_sipopt_data') and
                        m_orig._sipopt_data.ctype is Block)
        
        #test variable declaration
        self.assertTrue(hasattr(m_sipopt._sipopt_data,'a') and 
                        m_sipopt._sipopt_data.a.ctype is Var)
        self.assertTrue(hasattr(m_sipopt._sipopt_data,'H') and 
                        m_sipopt._sipopt_data.H.ctype is Var)

        #test for suffixes
        self.assertTrue(hasattr(m_sipopt,'sens_state_0') and
                        m_sipopt.sens_state_0.ctype is Suffix and
                        m_sipopt.sens_state_0[m_sipopt._sipopt_data.H]==2 and  
                        m_sipopt.sens_state_0[m_sipopt._sipopt_data.a]==1)
  
        self.assertTrue(hasattr(m_sipopt,'sens_state_1') and
                        m_sipopt.sens_state_1.ctype is Suffix and
                        m_sipopt.sens_state_1[m_sipopt._sipopt_data.H]==2 and
                        m_sipopt.sens_state_1[m_sipopt._sipopt_data.a]==1)  

        self.assertTrue(hasattr(m_sipopt,'sens_state_value_1') and
                        m_sipopt.sens_state_value_1.ctype is Suffix and
                        m_sipopt.sens_state_value_1[
                                        m_sipopt._sipopt_data.H]==0.55 and
                        m_sipopt.sens_state_value_1[
                                        m_sipopt._sipopt_data.a]==-0.25)
  
        self.assertTrue(hasattr(m_sipopt,'sens_init_constr') and
                        m_sipopt.sens_init_constr.ctype is Suffix and
                        m_sipopt.sens_init_constr[
                                     m_sipopt._sipopt_data.paramConst[1]]==1 and
                        m_sipopt.sens_init_constr[
                                     m_sipopt._sipopt_data.paramConst[2]]==2)

        self.assertTrue(hasattr(m_sipopt,'sens_sol_state_1') and
                        m_sipopt.sens_sol_state_1.ctype is Suffix)
        self.assertAlmostEqual(
                        m_sipopt.sens_sol_state_1[
                           m_sipopt.F[15]],-0.00102016765,8)

        self.assertTrue(hasattr(m_sipopt,'sens_sol_state_1_z_L') and
                        m_sipopt.sens_sol_state_1_z_L.ctype is Suffix)
        self.assertAlmostEqual(
                        m_sipopt.sens_sol_state_1_z_L[
                           m_sipopt.u[15]],-2.181712e-09,13)

        self.assertTrue(hasattr(m_sipopt,'sens_sol_state_1_z_U') and
                        m_sipopt.sens_sol_state_1_z_U.ctype is Suffix)
        self.assertAlmostEqual(
                        m_sipopt.sens_sol_state_1_z_U[
                           m_sipopt.u[15]],6.580899e-09,13)

        #verify deactivated constraints on model
        self.assertFalse(m_sipopt.FDiffCon[0].active and
                         m_sipopt.FDiffCon[7.5].active and
                         m_sipopt.FDiffCon[15].active )

        self.assertFalse(m_sipopt.x_dot[0].active and
                         m_sipopt.x_dot[7.5].active and
                         m_sipopt.x_dot[15].active )

        #test model solution
        self.assertAlmostEqual(value(m_sipopt.J),0.0048956783,8)




    #test indexed param mapping to var and perturbed values
    @unittest.skipIf(not scipy_available, "scipy is required for this test")
    @unittest.skipIf(not opt.available(False), "ipopt_sens is not available")
    def test_indexedParamsMapping(self):

        m = hiv.create_model()
        hiv.initialize_model(m,10,5,1)

        m.epsDelta = Param(initialize = 0.75001)

        q_del = {}
        q_del[(0,0)] = 1.001
        q_del[(0,1)] = 1.002
        q_del[(1,0)] = 1.003
        q_del[(1,1)] = 1.004
        q_del[(2,0)] = 0.83001
        q_del[(2,1)] = 0.83002
        q_del[(3,0)] = 0.42001
        q_del[(4,0)] = 0.17001
        m.qqDelta = Param(m.ij, initialize = q_del)

        m.aaDelta = Param(initialize =0.0001001)

        m_sipopt = sipopt(m, [m.eps,m.qq,m.aa],
                             [m.epsDelta,m.qqDelta,m.aaDelta])

        #param to var data
        self.assertEqual(
            m_sipopt._sipopt_data.paramConst[1].lower.local_name, 'eps')
        self.assertEqual(
            m_sipopt._sipopt_data.paramConst[1].body.local_name, 'eps')
        self.assertEqual(
            m_sipopt._sipopt_data.paramConst[1].upper.local_name, 'eps')
        self.assertEqual(
            m_sipopt._sipopt_data.paramConst[6].lower.local_name, 'qq[2,0]')
        self.assertEqual(
            m_sipopt._sipopt_data.paramConst[6].body.local_name, 'qq[2,0]')
        self.assertEqual(
            m_sipopt._sipopt_data.paramConst[6].upper.local_name, 'qq[2,0]')
        self.assertEqual(
            m_sipopt._sipopt_data.paramConst[10].lower.local_name, 'aa')
        self.assertEqual(
            m_sipopt._sipopt_data.paramConst[10].body.local_name, 'aa')
        self.assertEqual(
            m_sipopt._sipopt_data.paramConst[10].upper.local_name, 'aa')
    


    #test Constraint substitution
    @unittest.skipIf(not opt.available(False), "ipopt_sens is not available")
    def test_constraintSub(self):
        
        m = ri.create_model()

        m.pert_a = Param(initialize=0.01)
        m.pert_b = Param(initialize=1.01)

        m_sipopt = sipopt(m,[m.a,m.b], [m.pert_a,m.pert_b])

        #verify substitutions in equality constraint
        self.assertTrue(m_sipopt.C_equal.lower.ctype is Param and
                        m_sipopt.C_equal.upper.ctype is Param)
        self.assertFalse(m_sipopt.C_equal.active)

        self.assertTrue(m_sipopt._sipopt_data.constList[3].lower == 0.0 and
                       m_sipopt._sipopt_data.constList[3].upper == 0.0 and
                       len(list(identify_variables(
                                m_sipopt._sipopt_data.constList[3].body))) == 2)

        #verify substitutions in one-sided bounded constraint
        self.assertTrue(m_sipopt.C_singleBnd.lower is None and
                        m_sipopt.C_singleBnd.upper.ctype is Param)
        self.assertFalse(m_sipopt.C_singleBnd.active)

        self.assertTrue(m_sipopt._sipopt_data.constList[4].lower is None and
                        m_sipopt._sipopt_data.constList[4].upper == 0.0 and
                        len(list(identify_variables(
                                 m_sipopt._sipopt_data.constList[4].body))) == 2)
       
        #verify substitutions in ranged inequality constraint
        self.assertTrue(m_sipopt.C_rangedIn.lower.ctype is Param and
                        m_sipopt.C_rangedIn.upper.ctype is Param)
        self.assertFalse(m_sipopt.C_rangedIn.active)

        self.assertTrue(m_sipopt._sipopt_data.constList[1].lower is None and
                       m_sipopt._sipopt_data.constList[1].upper == 0.0 and
                       len(list(identify_variables(
                                m_sipopt._sipopt_data.constList[1].body))) == 2)

        self.assertTrue(m_sipopt._sipopt_data.constList[2].lower is None and
                       m_sipopt._sipopt_data.constList[2].upper == 0.0 and
                       len(list(identify_variables(
                                m_sipopt._sipopt_data.constList[2].body))) == 2)

    # Test example `parameter.py`
    @unittest.skipIf(not opt.available(False), "ipopt_sens is not available")
    def test_parameter_example(self):
    
        from pyomo.contrib.sensitivity_toolbox.examples.parameter import run_example
        d = run_example()
        
        d_correct = {'eta1':4.5, 'eta2':1.0, 'x1_init':0.15, 'x2_init':0.15, 'x3_init':0.0,
            'cost_sln':0.5, 'x1_sln':0.5, 'x2_sln':0.5, 'x3_sln':0.0, 'eta1_pert':4.0,
            'eta2_pert':1.0, 'x1_pert':0.3333333,'x2_pert':0.6666667,'x3_pert':0.0,
            'cost_pert':0.55555556}
        
        for k in d_correct.keys():
            # Check each element of the 'correct' dictionary against the returned 
            # dictionary to 3 decimal places
            self.assertAlmostEqual(d[k],d_correct[k],3)

if __name__=="__main__":
    unittest.main()
