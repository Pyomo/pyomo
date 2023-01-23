#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#
#  Pyomo.DoE was produced under the Department of Energy Carbon Capture Simulation 
#  Initiative (CCSI), and is copyright (c) 2022 by the software owners: 
#  TRIAD National Security, LLC., Lawrence Livermore National Security, LLC., 
#  Lawrence Berkeley National Laboratory, Pacific Northwest National Laboratory,  
#  Battelle Memorial Institute, University of Notre Dame,
#  The University of Pittsburgh, The University of Texas at Austin, 
#  University of Toledo, West Virginia University, et al. All rights reserved.
# 
#  NOTICE. This Software was developed under funding from the 
#  U.S. Department of Energy and the U.S. Government consequently retains 
#  certain rights. As such, the U.S. Government has been granted for itself
#  and others acting on its behalf a paid-up, nonexclusive, irrevocable, 
#  worldwide license in the Software to reproduce, distribute copies to the 
#  public, prepare derivative works, and perform publicly and display
#  publicly, and to permit other to do so.
#  ___________________________________________________________________________


# import libraries
from pyomo.common.dependencies import (
    numpy as np, numpy_available,
    pandas as pd, pandas_available,
)

import pyomo.common.unittest as unittest
from pyomo.contrib.doe import DesignOfExperiments, Measurements
from pyomo.environ import value

from pyomo.opt import SolverFactory
ipopt_available = SolverFactory('ipopt').available()

class Test_doe_object(unittest.TestCase):
    """ Test the kinetics example with both the sequential_finite mode and the direct_kaug mode
    """
    @unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
    def test_setUP(self):
        from pyomo.contrib.doe.example import reactor_kinetics as reactor
        
        # define create model function 
        createmod = reactor.create_model
        
        # discretizer 
        disc = reactor.disc_for_measure
        
        # design variable and its control time set
        t_control = [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1]
        dv_pass = {'CA0': [0],'T': t_control}

        # Define measurement time points
        t_measure = [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1]
        measure_pass = {'C':{'CA': t_measure, 'CB': t_measure, 'CC': t_measure}}
        measure_class =  Measurements(measure_pass)
        
        # Define parameter nominal value 
        parameter_dict = {'A1': 84.79085853498033, 'A2': 371.71773413976416, 'E1': 7.777032028026428, 'E2': 15.047135137500822}

        def generate_exp(t_set, CA0, T):  
            """Generate experiments. 
            t_set: time control set for T.
            CA0: CA0 value
            T: A list of T 
            """
            assert(len(t_set)==len(T)), 'T should have the same length as t_set'

            T_con_initial = {}
            for t, tim in enumerate(t_set):
                T_con_initial[tim] = T[t]

            dv_dict_overall = {'CA0': {0: CA0},'T': T_con_initial}
            return dv_dict_overall
        
        # empty prior
        prior_all = np.zeros((4,4))

        prior_pass=np.asarray(prior_all)
        
        ### Test sequential_finite mode
        exp1 = generate_exp(t_control, 5, [300, 300, 300, 300, 300, 300, 300, 300, 300])

        doe_object = DesignOfExperiments(parameter_dict, dv_pass,
                             measure_class, createmod,
                            prior_FIM=prior_pass, discretize_model=disc, args=[True])

    
        result = doe_object.compute_FIM(exp1,mode='sequential_finite', FIM_store_name = 'dynamic.csv', 
                                        store_output = 'store_output', read_output=None,
                                        scale_nominal_param_value=True, formula='central')


        result.calculate_FIM(doe_object.design_values)
        
        self.assertAlmostEqual(np.log10(result.trace), 2.96, places=2)
        self.assertAlmostEqual(result.FIM[0][1], 1.84, places=2)
        self.assertAlmostEqual(result.FIM[0][2], -70.238, places=2)
        
        ### Test direct_kaug mode
        exp2 = generate_exp(t_control, 5, [570, 300, 300, 300, 300, 300, 300, 300, 300])
        
        doe_object2 = DesignOfExperiments(parameter_dict, dv_pass,
                             measure_class, createmod,
                            prior_FIM=prior_pass, discretize_model=disc, args=[False])
        result2 = doe_object2.compute_FIM(exp2,mode='direct_kaug', FIM_store_name = 'dynamic.csv', 
                                        store_output = 'store_output', read_output=None,
                                        scale_nominal_param_value=True, formula='central')
        
        result2.calculate_FIM(doe_object2.design_values)
        
        self.assertAlmostEqual(np.log10(result2.trace), 2.788587, places=2)
        self.assertAlmostEqual(np.log10(result2.det), 2.821840, places=2)
        self.assertAlmostEqual(np.log10(result2.min_eig), -1.012346, places=2)
        
        ### Test stochastic_program mode
        
        # prior
        exp1 = generate_exp(t_control, 3, [500, 300, 300, 300, 300, 300, 300, 300, 300])

        # add a prior information (scaled FIM with T=500 and T=300 experiments)
        prior = np.asarray([[  28.67892806 ,   5.41249739 , -81.73674601 , -24.02377324],
              [   5.41249739 ,  26.40935036 , -12.41816477 , -139.23992532],
              [ -81.73674601 , -12.41816477 , 240.46276004 ,  58.76422806],
              [ -24.02377324 , -139.23992532 ,  58.76422806 , 767.25584508]])


        doe_object3 = DesignOfExperiments(parameter_dict, dv_pass,
                                     measure_class, createmod,
                                    prior_FIM=prior, discretize_model=disc, args=[True])

        square_result, optimize_result= doe_object3.stochastic_program(exp1, if_optimize=True, if_Cholesky=True, 
                                                             scale_nominal_param_value=True, objective_option='det', 
                                                             L_initial=np.linalg.cholesky(prior))
        
        self.assertAlmostEqual(value(optimize_result.model.T[0]), 579.212348, places=2)
        self.assertAlmostEqual(value(optimize_result.model.T[1]), 300.000450, places=2)
        self.assertAlmostEqual(np.log10(optimize_result.trace), 3.214432, places=2)
        self.assertAlmostEqual(np.log10(optimize_result.det), 6.214118, places=2)
        
        

if __name__ == '__main__':
    unittest.main()
