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

class Scenario_generator:
    def __init__(self, para_dict, formula='central', step=0.001, store=False):
        """Generate scenarios.
        DoE library first calls this function to generate scenarios.
        For sequential and simultaneous models, call different functions in this class.

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

    def simultaneous_scenario(self):
        """
        Generate scenario dict for simultaneous models

        Returns:
        -------
        scena_overall: a dictionary containing scenarios dictionaries.
        scena_overall[name of parameter]: a dict, keys are the scenario name(numeric integer starting from 0), values are parameter value in this scenario
        scena_overall['jac-index']: keys are parameter name, values are the scenario names perturbing this parameter.
        scena_overall['eps-abs']: keys are parameter name, values are the step it is perturbed
        scena_overall['scena-name']: a list of scenario names

        For e.g., if a dict {'P':100, 'D':20} is given, step=0.1, formula='central', it will return:
        scena_overall = {'P': {0: 101.0, 1: 100, 2: 99.0, 3: 100}, 'D': {0: 20, 1: 20.2, 2: 20, 3: 19.8}, 'jac-index': {'P': [0, 2], 'D': [1, 3]}, 'eps-abs': {'P': 2.0, 'D': 0.4}, 'scena-name': [0, 1, 2, 3]}
        if formula ='forward', it will return:
        scena_overall = {'P':{'0':110, '1':100, '2':100}, 'D':{'0':20, '1':22, '2':20}, 'jac-index':{'P':[0,2], 'D':[1,2]}, 'eps-abs':{'P':10,'D':2}, 'scena-name': [0,1,2]}
        """
        # generate scenarios
        scena_keys, scena = self._scena_generate(self.scenario_nominal, self.formula)
        self.scena_keys = scena_keys
        self.scena = scena

        # call scenario class and method
        scenario_object = Scenario_data(self.para_dict, self.scena_keys, self.scena, self.formula, self.step)
        scenario_overall = scenario_object.create_scenario()

        # store scenario
        if self.store:
            with open('scenario_simultaneous.pickle', 'wb') as f:
                pickle.dump(scenario_overall, f)

        return scenario_overall


    def next_sequential_scenario(self, count):
        """
        Generate a single scenario class for one of the sequential models

        Parameters:
        ----------
        count: the No. of the sequential models

        Returns:
        -------
        scenario_next: scenario dict for this sequential model
        """
        scena_keys, scena = self._scena_generate(list(self.scena[count].values()), None)

        # each model is basically a 'none' case of an invasive model
        scenario_object = Scenario_data(self.scena[count], scena_keys, scena, None, self.step)
        scenario_next = scenario_object.create_scenario()

        return scenario_next

    def generate_sequential_para(self):
        """
        Generate object and some 'parameters' for sequential models

        Returns (added to self object):
        -------
        self.scena_keys: scenario name, a list of numbers
        self.scena: a list of parameter dictionaries for all sequential models
        self.scenario_para: a list of two No. of models involved in calculating one parameter sensitivity
        self.eps_abs: keys are parameter name, values are the step it is perturbed
        """

        scena_keys, scena = self._scena_generate(self.scenario_nominal, self.formula)
        self.scena_keys = scena_keys
        self.scena = scena

        # record the number of scenarios involved in calculating a certain parameter sensitivities
        scenario_para = {}
        for p, para in enumerate(self.para_names):
            # the scenario involved in Jacobian calculation
            if self.formula == 'central':
                scenario_para[para] = [p, p + self.no_para]
            elif self.formula == None:
                raise ValueError('Finite difference scheme should be chosen.')
            else:
                scenario_para[para] = [p, self.no_para]

        self.scenario_para = scenario_para

        # calculate the perturbation size of every parameter
        eps_abs = {}
        for para in self.para_names:
            # for central difference scheme, perturbation size is two times the step size
            eps_abs[para] = self.step * self.para_dict[para]
            if self.formula == 'central':
                eps_abs[para] *= 2 
                

        self.eps_abs = eps_abs

    def _scena_generate(self, para_nominal, formula):
        """
        Generate scenario logics

        Returns: (store in self object)
        --------
        self.scena_keys: a list of scenario names
        self.scena: a dict, keys are scenario names, values are a list of parameter values
        """
        # generate scenario names
        if formula == 'central':
            scena_keys = list(range(2 * self.no_para))
        elif formula == None:
            scena_keys = [0]
        else:
            scena_keys = list(range(self.no_para + 1))

        # generate all parameter dict needed for creating a scenario
        scena = {}
        # generate a dict, keys are scenario number, values are a list of parameter values in this scenario
        for i, name in enumerate(scena_keys):
            scenario = para_nominal.copy()

            if formula == 'central':
                # scenario 0 to #_of_para-1 are forward perturbed
                if i < self.no_para:
                    scenario[i] *= (1 + self.step)
                # scenario #_of_para to 2*#_of_para-1 are backward perturbed
                else:
                    scenario[i - self.no_para] *= (1 - self.step)

            elif formula == 'forward':
                # scenario 0 to #_of_para-1 are forward perturbed
                if i < self.no_para:
                    scenario[i] *= (1 + self.step)

            elif formula == 'backward':
                # scenario 0 to #_of_para-1 are backward perturbed
                if i < self.no_para:
                    scenario[i] *= (1 - self.step)

            scenario_dict = {}
            for n, pname in enumerate(self.para_names):
                scenario_dict[pname] = scenario[n]

            scena[name] = scenario_dict

        return scena_keys, scena

        # TODO: need to consider how to store both hyperparameter and scenario classes in pickle
        # if self.store:
        #    f = open('scenario_combine','wb')
        #    pickle.dump(scenario_comp, f)
        #    f.close()


class Scenario_data:
    def __init__(self, parameter_dict, scena_keys, scena, form, step):
        """
        Generate scenario for a simultaneous model

        parameter_dict: parameter dictionaries
        scena_keys: scenario name, a list of numbers
        scena: a list of parameter dictionaries for all sequential models
        form: choose from 'central', 'forward', 'backward', 'none'.
        step: stepsize of a fraction, such as 0.01

        """
        # get info from parameter dictionary
        self.para_dict = parameter_dict
        self.para_names = list(parameter_dict.keys())

        self.scena = scena
        self.scena_keys = scena_keys
        self.no_para = len(self.para_names)
        self.formula = form
        self.step = step

        # This is the parameter nominal values
        self.scenario_nominal = []
        for d in self.para_names:
            self.scenario_nominal.append(parameter_dict[d])

    def create_scenario(self):
        """
        Returns:
        --------
        scena_dict: a dictionary containing scenarios dictionaries.
            scena_dict[name of parameter]: a dict, keys are the scenario name(numeric integer starting from 0), 
            values are parameter value in this scenario
            scena_dict['jac-index']: keys are parameter name, values are the scenario names perturbing this parameter.
            scena_dict['eps-abs']: keys are parameter name, values are the step it is perturbed
            scena_dict['scena-name']: a list of scenario names

            For e.g., if a dict {'P':100, 'D':20} is given, step=0.1, formula='central', it will return:
            scena_dict = {'P': {0: 101.0, 1: 100, 2: 99.0, 3: 100}, 'D': {0: 20, 1: 20.2, 2: 20, 3: 19.8}, 
            'jac-index': {'P': [0, 2], 'D': [1, 3]}, 'eps-abs': {'P': 2.0, 'D': 0.4}, 'scena-name': [0, 1, 2, 3]}
            if formula ='forward', it will return:
            scena_dict = {'P':{'0':110, '1':100, '2':100}, 'D':{'0':20, '1':22, '2':20}, 
            'jac-index':{'P':[0,2], 'D':[1,2]}, 'eps-abs':{'P':10,'D':2}, 'scena-name': [0,1,2]}

            V2 added:
            'scenario': [{'P':101, 'D':20}, {'P':99, 'D':20}, 
            {'P':100, 'D':20.2}, {'P':100, 'D':19.8}]
            'scena_num': {'P':[0,1], 'D':[2,3]}

        """
        # overall dict to return
        scenario_dict = {}
        # dict for scenario position
        jac_index = {}
        # dict for parameter perturbation step size
        eps_abs = {}
        # scenario dict for block
        scenario = []
        # number of scenario 
        scena_num = {}

        # loop over parameter name
        for p, para in enumerate(self.para_names):

            if self.formula == "central":
                scena_num[para] = [2*p, 2*p+1]
                scena_dict_up, scena_dict_lo = self.para_dict.copy(), self.para_dict.copy()
                scena_dict_up[para] *= (1+self.step)
                scena_dict_lo[para] *= (1-self.step)

                scenario.append(scena_dict_up)
                scenario.append(scena_dict_lo)

            elif self.formula == "forward":
                # TODO: add Forward formulation 
                print("Todo")

            scena_p = {}
            for n in self.scena_keys:
                scena_p[n] = self.scena[n][para]

            # a dictionary of scenarios and its corresponding parameter values
            scenario_dict[para] = scena_p

            # for central difference scheme, perturbation size is two times the step size
            if self.formula == 'central':
                eps_abs[para] = 2 * self.step * self.para_dict[para]
            else:
                eps_abs[para] = self.step * self.para_dict[para]

            # the scenario involved in Jacobian calculation
            if self.formula == 'central':
                jac_index[para] = [p, p + self.no_para]
            elif self.formula == None:
                jac_index[para] = [0]
            else:
                jac_index[para] = [p, self.no_para]

        scenario_dict['jac-index'] = jac_index
        scenario_dict['eps-abs'] = eps_abs
        scenario_dict['scena-name'] = self.scena_keys
        scenario_dict['scenario'] = scenario 
        scenario_dict['scena_num'] = scena_num


        return scenario_dict
