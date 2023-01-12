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
from pyomo.contrib.doe import Measurements, DesignOfExperiments


def main():
    # Create model function
    ## TODO: use create_model directly
    createmod = create_model

    # discretization by Pyomo.DAE
    ## TODO: directly use 
    disc = disc_for_measure

    # Control time set [h]
    t_control = [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1]

    # Measurement time points [h]
    t_measure = [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1]

    # design variable and its control time set
    dv_pass = {'CA0': [0],'T': t_control}

    # Create measurement object
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
    
    # choose from 'sequential_finite', 'direct_kaug'
    # 'sequential_sipopt', 'sequential_kaug' is also available
    #sensi_opt = 'sequential_finite'
    sensi_opt = 'direct_kaug'

    # model option
    if sensi_opt == 'direct_kaug':
        args_ = [False]
    else:
        args_ = [True]


    # Define experiments
    exp1 = generate_exp(t_control, 5, [570, 300, 300, 300, 300, 300, 300, 300, 300])
    
    doe_object = DesignOfExperiments(parameter_dict, dv_pass,
                                 measure_class, createmod,
                                prior_FIM=prior_pass, discretize_model=disc, args=args_)


    result = doe_object.compute_FIM(exp1, mode=sensi_opt,
                                    scale_nominal_param_value=True, 
                                    formula='central')


    result.calculate_FIM(doe_object.design_values)
    
    # test result 
    relative_error = abs(np.log10(result.trace) - 2.7885870986653556)
    assert relative_error < 0.01

    relative_error = abs(np.log10(result.det) - 2.82184091661587)
    assert relative_error < 0.01
    
if __name__ == "__main__":
    main()
    
