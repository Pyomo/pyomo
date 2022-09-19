#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
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
    
    # empty prior
    prior_all = np.zeros((4,4))
    prior_pass=np.asarray(prior_all)


    # Define experiments
    exp1 = generate_exp(t_control, 3, [300, 300, 300, 300, 300, 300, 300, 300, 300])
    
    doe_object = doe.DesignOfExperiments(parameter_dict, dv_pass,
                                 measure_class, createmod,
                                prior_FIM=prior_pass, discretize_model=disc, args=[True])

    square_result, optimize_result= doe_object.optimize_doe(exp1, if_optimize=True, if_Cholesky=True, 
                                                         scale_nominal_param_value=True, objective_option='det', 
                                                         L_initial=None)
    
    
if __name__ == "__main__":
    main()
