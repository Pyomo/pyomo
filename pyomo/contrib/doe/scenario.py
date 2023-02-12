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

import pickle

class ScenarioGenerator:
    def __init__(self, para_dict, formula='central', step=0.001, store=False):
        """Generate scenarios.
        DoE library first calls this function to generate scenarios.

        Parameters
        -----------
        para_dict:
            a ``dict`` of parameter, keys are names of ''string'', values are their nominal value of ''float''.
            for e.g., {'A1': 84.79, 'A2': 371.72, 'E1': 7.78, 'E2': 15.05}
        formula:
            choose from 'central', 'forward', 'backward', None.
        step:
            Sensitivity perturbation step size, a fraction between [0,1]. default is 0.001
        store:
            if True, store results.
        """

        if formula not in ['central', 'forward', 'backward', None]:
            raise ValueError('Undefined formula. Available formulas: central, forward, backward, none.')

        # get info from parameter dictionary
        self.para_dict = para_dict
        self.para_names = list(para_dict.keys())
        self.no_para = len(self.para_names)
        self.formula = formula
        self.step = step
        self.store = store
        self.scenario_nominal = [para_dict[d] for d in self.para_names]

    def generate_scenario(self):
        """
        Generate scenario dict

        Returns:
        -------
        scenario_dict: a dictionary containing scenarios dictionaries.
        scenario_dict['sceanrio']: a list of dictionaries, each dictionary contains a perturbed scenario
        scenario_dict['eps-abs']: keys are parameter name, values are the step it is perturbed
        scena_overall['scena-num']: a dict of scenario number related to one parameter

        For e.g., if a dict {'P':100, 'D':20} is given, step=0.1, formula='central', it will return:
            scena_dict = {'eps-abs': {'P': 2.0, 'D': 0.4}, 
            'scenario': [{'P':101, 'D':20}, {'P':99, 'D':20}, {'P':100, 'D':20.2}, {'P':100, 'D':19.8}],
            'scena_num': {'P':[0,1], 'D':[2,3]}}
        if formula ='forward', it will return:
            scena_dict = {'eps-abs':  {'P': 2.0, 'D': 0.4}, 
            'scenario': [{'P':101, 'D':20}, {'P':100, 'D':20.2}, {'P':100, 'D':20}],
            'scena_num': {'P':[0,2], 'D':[1,2]}}
        """

        # overall dict to return
        scenario_dict = {}
        # dict for parameter perturbation step size
        eps_abs = {}
        # scenario dict for block
        scenario = []
        # number of scenario 
        scena_num = {}

        # loop over parameter name
        for p, para in enumerate(self.para_names):

            ## get scenario dictionary
            if self.formula == "central":
                scena_num[para] = [2*p, 2*p+1]
                scena_dict_up, scena_dict_lo = self.para_dict.copy(), self.para_dict.copy()
                # corresponding parameter dictionary for the scenario
                scena_dict_up[para] *= (1+self.step)
                scena_dict_lo[para] *= (1-self.step)

                scenario.append(scena_dict_up)
                scenario.append(scena_dict_lo)

            elif self.formula in ["forward", "backward"]:
                # the base case is added as the last one
                scena_num[para] = [p,len(self.param_names)]
                scena_dict_up, scena_dict_lo = self.para_dict.copy(), self.para_dict.copy()
                if self.formula=="forward":
                    scena_dict_up[para] *= (1+self.step)
                
                elif self.formula=="backward":
                    scena_dict_lo[para] *= (1-self.step)

                scenario.append(scena_dict_up)
                scenario.append(scena_dict_lo)

            ## get perturbation sizes
            # for central difference scheme, perturbation size is two times the step size
            if self.formula == 'central':
                eps_abs[para] = 2 * self.step * self.para_dict[para]
            else:
                eps_abs[para] = self.step * self.para_dict[para]

        scenario_dict['eps-abs'] = eps_abs
        scenario_dict['scenario'] = scenario 
        scenario_dict['scena_num'] = scena_num


        # store scenario
        if self.store:
            with open('scenario_simultaneous.pickle', 'wb') as f:
                pickle.dump(scenario_dict, f)

        return scenario_dict
