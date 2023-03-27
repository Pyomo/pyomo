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
from pyomo.contrib.doe.example.reactor_kinetics import create_model, disc_for_measure
from pyomo.contrib.doe import DesignOfExperiments, Measurements, calculation_mode, DesignVariables, finite_difference_lib

def main():
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
    design_gen.add_elements(total_name, time_index = dtime_index, values=exp1, upper_bound=upper_bound, lower_bound=lower_bound)
    
    # empty prior
    prior_pass = np.zeros((4,4))
    
    ### Test sequential_finite mode
    sensi_opt = calculation_mode.sequential_finite

    doe_object = DesignOfExperiments(parameter_dict, design_gen,
                                measure_class, create_model,
                            prior_FIM=prior_pass, discretize_model=disc_for_measure)


    result = doe_object.compute_FIM(mode=sensi_opt,  
                                    scale_nominal_param_value=True,
                                formula = finite_difference_lib.central)


    result.calculate_FIM(doe_object.design_values)
    
    # test result 
    relative_error = abs(np.log10(result.trace) - 2.7885)
    assert relative_error < 0.01

    relative_error = abs(np.log10(result.det) - 2.8218)
    assert relative_error < 0.01
    
if __name__ == "__main__":
    main()
    
