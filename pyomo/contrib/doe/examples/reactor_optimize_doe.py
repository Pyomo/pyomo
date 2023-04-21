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

import numpy as np
import pyomo.common.unittest as unittest
from pyomo.contrib.doe.examples.reactor_kinetics import create_model, disc_for_measure
from pyomo.contrib.doe import DesignOfExperiments, MeasurementVariables, DesignVariables, objective_lib

def main():
    ### Define inputs
    # Control time set [h]
    t_control = [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1]
    # Define parameter nominal value 
    parameter_dict = {'A1': 84.79, 'A2': 371.72, 'E1': 7.78, 'E2': 15.05}

    # measurement object 
    variable_name = "C"
    indices = {0:['CA', 'CB', 'CC'], 1: t_control}

    measurements = MeasurementVariables()
    measurements.add_variables(variable_name, indices=indices, time_index_position = 1)

    # design object 
    exp_design = DesignVariables()
    
    # add CAO as design variable
    var_C = 'CA0'
    indices_C = {0:[0]}
    exp1_C = [5]
    exp_design.add_variables(var_C, indices = indices_C, time_index_position=0,
                            values=exp1_C,lower_bounds=1, upper_bounds=5)

    # add T as design variable
    var_T = 'T'
    indices_T = {0:t_control}
    exp1_T = [470, 300, 300, 300, 300, 300, 300, 300, 300]

    exp_design.add_variables(var_T, indices = indices_T, time_index_position=0,
                            values=exp1_T,lower_bounds=300, upper_bounds=700)

    
    exp1 = [5, 570, 300, 300, 300, 300, 300, 300, 300, 300]
    exp_design.update_values(exp1)

    # add a prior information (scaled FIM with T=500 and T=300 experiments)
    prior = np.asarray([[  28.67892806 ,   5.41249739 , -81.73674601 , -24.02377324],
    [   5.41249739 ,  26.40935036 , -12.41816477 , -139.23992532],
    [ -81.73674601 , -12.41816477 , 240.46276004 ,  58.76422806],
    [ -24.02377324 , -139.23992532 ,  58.76422806 , 767.25584508]])

    doe_object2 = DesignOfExperiments(parameter_dict, 
                                      exp_design,
                                    measurements, 
                                    create_model,
                                    prior_FIM=prior, 
                                    discretize_model=disc_for_measure)

    square_result, optimize_result= doe_object2.stochastic_program(if_optimize=True, 
                                                                   if_Cholesky=True, 
                                                                    scale_nominal_param_value=True, 
                                                                    objective_option=objective_lib.det, 
                                                                    L_initial=np.linalg.cholesky(prior))
    
if __name__ == "__main__":
    main()
