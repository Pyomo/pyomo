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
#  Pyomo.DOE was produced under the DOE Carbon Capture Simulation 
#  Initiative (CCSI), and is copyright (c) 2022 by the software owners: 
#  TRIAD, LLNS, BERKELEY LAB, PNNL, UT-Battelle, LLC, NOTRE
#  DAME, PITT, UT Austin, TOLEDO, WVU, et al. All rights reserved.
# 
#  NOTICE. This Software was developed under funding from the 
#  U.S. Department of Energy and the U.S. Government consequently retains 
#  certain rights. As such, the U.S. Government has been granted for itself
#  and others acting on its behalf a paid-up, nonexclusive, irrevocable, 
#  worldwide license in the Software to reproduce, distribute copies to the 
#  public, prepare derivative works, and perform publicly and display
#  publicly, and to permit other to do so.
#  ___________________________________________________________________________

import pyomo.contrib.doe.fim_doe as doe
import numpy as np
import pyomo.common.unittest as unittest
from pyomo.contrib.doe.example.reactor_kinetics import create_model, disc_for_measure



def main():
    # Create model function
    createmod = create_model

    # discretization by Pyomo.DAE
    disc = disc_for_measure

    # Control time set [h]
    t_control = [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1]

    # Measurement time points [h]
    t_measure = [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1]

    # design variable and its control time set
    dv_pass = {'CA0': [0],'T': t_control}

    # Create measurement object
    measure_pass = {'C':{'CA': t_measure, 'CB': t_measure, 'CC': t_measure}}
    measure_class =  doe.Measurements(measure_pass)

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
    
    # Design variable ranges as lists 
    design_ranges = [list(np.linspace(1,5,5)), list(np.linspace(300,700,5))]

    # Design variable names 
    dv_apply_name = ['CA0','T']

    # Design variable should be fixed at these time points
    dv_apply_time = [[0],t_control]

    # Define experiments. This is a starting point of which the value does not matter
    exp1 = generate_exp(t_control, 5, [300, 300, 300, 300, 300, 300, 300, 300, 300])

    ## choose from 'sequential_finite', 'direct_kaug'
    #sensi_opt = 'sequential_finite'
    sensi_opt = 'direct_kaug'

    # model option
    if sensi_opt == 'direct_kaug':
        args_ = [False]
    else:
        args_ = [True]
        
    # add prior information
    prior_all = np.zeros((4,4))

    prior_pass=np.asarray(prior_all)
    
    doe_object = doe.DesignOfExperiments(parameter_dict, dv_pass,
                                 measure_class, createmod,
                                prior_FIM=prior_pass, discretize_model=disc, args=args_)

    all_fim = doe_object.run_grid_search(exp1, design_ranges, dv_apply_name, dv_apply_time, 
                                     mode=sensi_opt)
    
    test = all_fim.extract_criteria()


    ### 3 design variable example
    # Define design ranges
    design_ranges = [list(np.linspace(1,5,2)),  list(np.linspace(300,700,2)), [300,500]]

    # Define design variable 
    # Here the two T are for different controlling time subsets
    dv_apply_name = ['CA0', 'T', 'T']
    dv_apply_time = [[0], [0], [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875,1]]

    # Define experiments
    exp1 = generate_exp(t_control, 5, [300, 300, 300, 300, 300, 300, 300, 300, 300])

    ## choose from 'sequential_finite', 'direct_kaug'
    #sensi_opt = 'sequential_finite'
    sensi_opt = 'direct_kaug'

    # model option
    if sensi_opt == 'direct_kaug':
        args_ = [False]
    else:
        args_ = [True]
        
    doe_object = doe.DesignOfExperiments(parameter_dict, dv_pass,
                                 measure_class, createmod,
                                prior_FIM=prior_pass, discretize_model=disc, args=args_)

    all_fim = doe_object.run_grid_search(exp1, design_ranges, dv_apply_name, dv_apply_time, 
                                     mode=sensi_opt)
    
    test = all_fim.extract_criteria()
    
    
if __name__ == "__main__":
    main()
