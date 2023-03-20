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
from pyomo.contrib.doe import DesignOfExperiments, Measurements, DesignVariables, mode_lib


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
    design_gen.add_elements(total_name, time_index = dtime_index, values=exp1)
    design_gen.add_bounds(upper_bound=upper_bound, lower_bound=lower_bound)
    
    # Design variable ranges as lists 
    design_ranges = [list(np.linspace(1,5,3)), list(np.linspace(300,700,3))]

    # Design variable names 
    dv_apply_name = ['CA0[0]',['T[0]','T[0.125]','T[0.25]','T[0.375]','T[0.5]','T[0.625]','T[0.75]','T[0.875]','T[1]']]

    ## choose from 'sequential_finite', 'direct_kaug'
    #sensi_opt = 'sequential_finite'
    sensi_opt = mode_lib.direct_kaug

    prior_pass = np.zeros((4,4))
        
    doe_object = DesignOfExperiments(parameter_dict, design_gen,
                                 measure_class, create_model,
                                prior_FIM=prior_pass, discretize_model=disc_for_measure)

    all_fim = doe_object.run_grid_search(design_gen, design_ranges, dv_apply_name, 
                                        mode=sensi_opt)
    
    all_fim.extract_criteria()

    ### 3 design variable example
    # Define design ranges
    design_ranges = [list(np.linspace(1,5,2)),  list(np.linspace(300,700,2)), [300,500]]

    # Design variable names 
    dv_apply_name = ['CA0[0]','T[0]',['T[0.125]','T[0.25]','T[0.375]','T[0.5]','T[0.625]','T[0.75]','T[0.875]','T[1]']]

    # Define experiments
    exp1 = [5, 300, 300, 300, 300, 300, 300, 300, 300, 300]

    sensi_opt = mode_lib.direct_kaug

    doe_object = DesignOfExperiments(parameter_dict, design_gen,
                                 measure_class, create_model,
                                prior_FIM=prior_pass, discretize_model=disc_for_measure)

    all_fim = doe_object.run_grid_search(design_gen, design_ranges, dv_apply_name, 
                                        mode=sensi_opt)
    
    test = all_fim.extract_criteria()

if __name__ == "__main__":
    main()
