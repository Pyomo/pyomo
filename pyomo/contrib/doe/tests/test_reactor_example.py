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
    pandas_available
)

import pyomo.common.unittest as unittest
from pyomo.contrib.doe import DesignOfExperiments, Measurements, mode_lib, DesignVariables, formula_lib
from pyomo.environ import value

from pyomo.opt import SolverFactory
ipopt_available = SolverFactory('ipopt').available()

class Test_doe_object(unittest.TestCase):
    """ Test the kinetics example with both the sequential_finite mode and the direct_kaug mode
    """
    @unittest.skipIf(not ipopt_available, "The 'ipopt' solver is not available")
    @unittest.skipIf(not numpy_available, "Numpy is not available")
    @unittest.skipIf(not pandas_available, "Pandas is not available")
    def test_setUP(self):
        from pyomo.contrib.doe.example import create_model, disc_for_measure
        
        ### Define inputs
        # Control time set [h]
        t_control = [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1]
        # Define parameter nominal value 
        parameter_dict = {'A1': 84.79, 'A2': 371.72, 'E1': 7.78, 'E2': 15.05}

        # measurement object 
        total_name = ["C"]
        extra_index = [[["CA", "CB", "CC"]]]
        time_index = [t_control] 

        measure_class = Measurements()
        measure_class.add_elements(total_name, extra_index=extra_index, time_index = time_index)

        # design object 
        total_name = ["CA0", "T"]
        dtime_index = [[0], t_control] 
        exp1 = [5, 570, 300, 300, 300, 300, 300, 300, 300, 300]
        upper_bound = [5, 700, 700, 700, 700, 700, 700, 700, 700, 700]
        lower_bound = [1, 300, 300, 300, 300, 300, 300, 300, 300, 300]

        design_gen = DesignVariables()
        design_gen.add_elements(total_name, time_index = dtime_index, values=exp1)
        design_gen.add_bounds(upper_bound=upper_bound, lower_bound=lower_bound)
        
        # empty prior
        prior_pass = np.zeros((4,4))
        
        ### Test sequential_finite mode
        sensi_opt = mode_lib.sequential_finite

        doe_object = DesignOfExperiments(parameter_dict, design_gen,
                                 measure_class, create_model,
                                prior_FIM=prior_pass, discretize_model=disc_for_measure)


        result = doe_object.compute_FIM(design_gen, mode=sensi_opt,  
                                        scale_nominal_param_value=True,
                                    formula = formula_lib.central)


        result.calculate_FIM(doe_object.design_values)

        self.assertAlmostEqual(np.log10(result.trace), 2.7885, places=2)
        self.assertAlmostEqual(np.log10(result.det), 2.8218, places=2)
        self.assertAlmostEqual(np.log10(result.min_eig), -1.0123, places=2)
        
        ### Test direct_kaug mode
        sensi_opt = mode_lib.direct_kaug
        # Define a new experiment
        exp1 = [5, 570, 400, 300, 300, 300, 300, 300, 300, 300]
        design_gen.update_values(exp1)

        result = doe_object.compute_FIM(design_gen, mode=sensi_opt,  
                                        scale_nominal_param_value=True,
                                    formula = formula_lib.central)
        
        result.calculate_FIM(doe_object.design_values)
        
        self.assertAlmostEqual(np.log10(result.trace), 2.7211, places=2)
        self.assertAlmostEqual(np.log10(result.det), 2.0845, places=2)
        self.assertAlmostEqual(np.log10(result.min_eig), -1.3510, places=2)


        ### check subset feature 
        sub_name = ["C"]
        sub_extra_index = [[["CB", "CC"]]]
        sub_time_index = [[0.125, 0.25, 0.5, 0.75, 0.875]] 

        measure_subset = Measurements()
        measure_subset.add_elements(sub_name, extra_index=sub_extra_index, time_index=sub_time_index)

        sub_result = result.subset(measure_subset)
        sub_result.calculate_FIM(doe_object.design_values)
        
        self.assertAlmostEqual(np.log10(result.trace), 2.4975, places=2)
        self.assertAlmostEqual(np.log10(result.det), 0.5973, places=2)
        self.assertAlmostEqual(np.log10(result.min_eig), -1.9669, places=2)

        ### Test stochastic_program mode

        exp1 = [5, 570, 300, 300, 300, 300, 300, 300, 300, 300]
        design_gen.update_values(exp1)

        # add a prior information (scaled FIM with T=500 and T=300 experiments)
        prior = np.asarray([[  28.67892806 ,   5.41249739 , -81.73674601 , -24.02377324],
        [   5.41249739 ,  26.40935036 , -12.41816477 , -139.23992532],
        [ -81.73674601 , -12.41816477 , 240.46276004 ,  58.76422806],
        [ -24.02377324 , -139.23992532 ,  58.76422806 , 767.25584508]])

        doe_object2 = DesignOfExperiments(parameter_dict, design_gen,
                                 measure_class, create_model,
                                prior_FIM=prior_pass, discretize_model=disc_for_measure)

        square_result, optimize_result= doe_object2.stochastic_program(design_gen, if_optimize=True, if_Cholesky=True, 
                                                                scale_nominal_param_value=True, objective_option='det', 
                                                                L_initial=np.linalg.cholesky(prior))
        
        self.assertAlmostEqual(value(optimize_result.model.CA0[0]), 5.0, places=2)
        self.assertAlmostEqual(value(optimize_result.model.T[0.5]), 300, places=2)
        self.assertAlmostEqual(np.log10(optimize_result.trace), 2.9822, places=2)
        self.assertAlmostEqual(np.log10(optimize_result.det), 3.3049, places=2)
        

if __name__ == '__main__':
    unittest.main()
