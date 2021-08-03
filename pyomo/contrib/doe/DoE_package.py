import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from scipy.stats import chi2
from pyomo.environ import *
from pyomo.dae import *
import pandas as pd
import time
import os
import pickle
import csv
#import idaes
#from pyomo.contrib.sensitivity_toolbox.sens import sipopt
from pyomo.contrib.sensitivity_toolbox.sens import sensitivity_calculation


class DesignOfExperiments: 
    def __init__(self, param_init, design_variable_timepoints, measurement_variables, measurement_timeset, create_model, solver=None,
                 prior_FIM=None, discretize_model=None, verbose=True):
        '''
        This package solves DOE by optimization, and enumeration.
        Optimization feature can solve dynamic or static models, while enumeration only solves static models where design variables are constant throughout the experiment.
        They both support integrating with Pyomo.DAE.
        
        param_init: a dictionary of parameter names and values. If they are an indexed variable, put the variable name and index, such as 'theta["A1"]'. Note: if sIPOPT is used, parameter shouldn't be indexed. 
        design_variable_timepoints: a dictionary, keys are design variable names, values are its control time points.
                if this design var is independent of time (constant), set the time to [0]
        measurement_variables: the variable name of the model output, for e.g., ['CA', 'CB', 'CC'].
        measurement_timeset: a list of measurement time points. can be different from control time points
        create_model: a function that returns the model, where:
                      - parameter and design variables are defined as variables
                      - define every state variables dependent on parameters with a scenario index
                      - take scenarios as the first argument of this function
                      - define time index as 't'.
                      - design variables are defined with and only with a time index.
        solver: User specified solver, default=None. If not specified, default solver is IPOPT MA57.
        prior_FIM: Fisher information matrix (FIM) for prior experiments, default=None
        discretize_model: A user-specified function that deiscretizes the model. Only use with Pyomo.DAE, default=None
        verbose: if print statements are made
        '''  
        
        # parameters
        self.param_init = param_init
        self.param = list(param_init.keys())
        self.param_v = list(param_init.values())
        # design variable name
        self.design_timeset = design_variable_timepoints
        self.dv_name = list(self.design_timeset.keys())
        # the control time point for each design variable
        self.dv_time = list(self.design_timeset.values())
        # model output (measurement) name
        self.measurement_variables = measurement_variables
        # model measurement time
        self.measurement_timeset = measurement_timeset
        # create_model()
        self.create_model = create_model
        # check if user-defined solver is given
        if solver is not None:
            self.solver = solver
        # if not given, use default solver
        else:
            self.solver = self.__solve_with_default_ipopt()

        # check if discretization is needed
        self.discretize_model = discretize_model

        # check if there is prior info
        self.prior_FIM = prior_FIM

        # if print statements
        self.verbose = verbose
        
    def __check_inputs(self, check_mode=False, check_dimension_dv=False):
        '''Check if inputs are consistent

        Parameters
        ----------
        check_mode: check FIM calculation mode
        check_dimension_dv: if the number of design variable is checked for heatmap
        '''
        if self.obj_opt not in ['det', 'trace', 'zero']:
            raise ValueError('Error: Objective function should be chosen from "det", "zero" and "trace"')

        if self.formula not in ['central', 'forward', 'backward', None]:
            raise ValueError('Error: Finite difference scheme should be chosen from "central", "forward", "backward" and "none".')

        if self.prior_FIM is not None:
            assert (np.shape(self.prior_FIM)[0] == np.shape(self.prior_FIM)[1]), 'Prior information should be a n*n matrix.'

        if self.scale_nominal_param_value:
            print('Sensitivity information is scaled by its corresponding parameter nominal value.')

        if (self.scale_constant_value != 1):
            print('Sensitivity information is scaled by constant ', self.scale_constant_value, ' times itself.')

        if check_mode:
            # finite or sipopt needs to be chosen
            if self.mode not in ['simultaneous_finite', 'sequential_finite', 'sequential_sipopt', 'sequential_kaug']:
                print('Wrong mode. Choose from "simultaneous_finite", "sequential_finite", "0sequential_sipopt", "sequential_kaug"')

        if check_dimension_dv:
            # input check, only for heatmap
            # check if there are two design variables and two ranges
            if not len(self.dv_ranges.keys()) == 2:
                raise ValueError('Deign variable should be 2')



    def optimize_doe(self,  design_values, if_optimize=True, obj_opt='det',
                     scale_nominal_param_value=False, scale_constant_value=1, if_cho=False, L_LB = 1E-10, L_initial=None,
                     formula='central', step=0.001, check=True):
        '''
        Optimize DOE problem with design variables being the decisions.
        The DOE model is formed invasively and all scenarios are computed simultaneously.
        The function will first fun a square problem with design variable being fixed at
        the given initial points, and then unfix the design variable and do the
        optimization.

        Args:
            design_values: initial point for optimization, a dict whose keys are design variable names, values are a dict whose keys are time point and values are the design variable value at that time point
            if_optimize: if True, continue to do optimization. else, just run square problem with given design variable values
            obj_opt: choose from 'det' and 'trace'
            scale_nominal_param_value: if scale Jacobian by the corresponding parameter nominal value
            scale_constant_value: how many order of magnitudes the Jacobian value is magnified by. Use when the Jac or FIM value is too small
            if_cho: if true, cholesky decomposition is used for Objective function (to optimize determinant).
                L_LB: if FIM is P.D., the diagonal element should be positive, so we can set a LB like 1E-10
                L_initial: initialize the L
            formula: Finite difference formula, choose from 'central', 'forward', 'backward', None
            step: Finite difference sensitivity perturbation step size, a fraction between [0,1]. default is 0.001
            check: if True check input toggles consistency to be checked multiple times.

        Returns:
            analysis_square: result summary of the square problem solved at the initial point
            analysis_optimize: result summary of the optimization problem solved
        '''

        # store inputs in object
        self.design_values = design_values
        self.optimize = if_optimize
        self.obj_opt = obj_opt
        self.scale_nominal_param_value = scale_nominal_param_value
        self.scale_constant_value = scale_constant_value
        self.cho_opt = if_cho
        self.L_LB = L_LB
        self.L_initial = L_initial
        self.formula = formula
        self.step = step
        self.tee_opt = True

        # calculate how much the FIM element is magnified
        # FIM = Q.T@Q, the FIM is scaled by squared value the Jacobian is scaled
        self.fim_scale_constant_value = self.scale_constant_value **2

        # check if inputs are valid
        # simultaneous mode does not need to check mode and dimension of design variables
        if check:
            self.__check_inputs(check_mode=False, check_dimension_dv=False)

        # build the large DOE pyomo model
        m = self.__create_doe_model()

        # solve model, achieve results for square problem, and results for optimization problem

        # Solve square problem first
        # result_square: solver result
        result_square = self.__solve_doe(m, fix=True)

        analysis_square = FIM_result(self.param, prior_FIM=self.prior_FIM, scale_constant_value=self.scale_constant_value)
        analysis_square.extract_FIM(m, self.design_timeset, result_square, obj=obj_opt)

        if self.optimize:
            # solve problem with DOF then
            result_doe = self.__solve_doe(m, fix=False)

            analysis_optimize = FIM_result(self.param, prior_FIM=self.prior_FIM)
            analysis_optimize.extract_FIM(m, self.design_timeset, result_doe, obj=obj_opt)

            return analysis_square, analysis_optimize

        else:
            return analysis_square


    def compute_FIM(self, design_values, mode='sequential_finite', FIM_store_name=None, specified_prior=None,
                    tee_opt=True, scale_nominal_param_value=False, scale_constant_value=1, formula='central', step=0.01,
                    if_Cho=False, L_LB=1E-10, L_initial=None):
        '''
        This function solves a square Pyomo model with fixed design variables to compute the FIM.
        The problem is structured in one of the four following modes:
        1, simultaneous_finite: Calculate a multiple scenario model. Sensitivity info estimated by finite difference
        2, sequential_finite: Calculates a one scenario model multiple times for
        multiple scenarios. Sensitivity info estimated by finite difference
        3, sequential_sipopt: calculate sensitivity by sIPOPT.
        4, sequential_kaug: calculate sensitivity by k_aug

        Argument:
        General:
            design_values: a dict whose keys are design variable names, values are a dict whose keys are time point and values are the design variable value at that time point
            mode: use Mode='sequential_finite', 'simultaneous_finite', 'sequential_sipopt', 'sequential_kaug'
            FIM_store_name: if storing the FIM in a .csv, give the file name here as a string, '**.csv' or '**.txt'.
            specified_prior: if user needs a different prior, replace this toggle without creating a new object
            tee_opt: if IPOPT console output is printed
            scale_nominal_param_value: if True, the parameters are scaled by its own nominal value in param_init
            scale_constant_value: how many order of magnitudes the Jacobian value is magnified by. Use when the Jac or FIM value is too small

        Only effective when finite=True:
            formula: choose from 'central', 'forward', 'backward', None
            step: Sensitivity perturbation step size, a fraction between [0,1]. default is 0.001

        Cholesky option:
        if_Cho: if true, Cholesky decomposition is used for Objective function (to optimize determinant).
                L_LB: if FIM is positive definite, the diagonal element should be positive, so we can set a LB like 1E-10
                L_initial: initialize the L

        Return:
            FIM_analysis: result summary object of this solve
        '''
        # save inputs in object
        self.design_values = design_values
        self.mode = mode
        self.scale_nominal_param_value = scale_nominal_param_value
        self.scale_constant_value = scale_constant_value
        self.formula = formula
        self.step = step

        # This method only solves square problem
        self.optimize = False
        # Set the OF to 0 helps solve square problem quickly
        self.obj_opt = 'zero'
        self.tee_opt = tee_opt

        self.cho_opt = if_Cho
        self.L_LB = L_LB
        self.L_initial = L_initial

        # calculate how much the FIM element is magnified
        # As FIM~Q.T@Q, FIM is scaled twice the number the Q is scaled
        self.fim_scale_constant_value = self.scale_constant_value ** 2

        # check inputs valid
        self.__check_inputs(check_mode=True, check_dimension_dv=False)

        # if using simultaneous model
        if (self.mode == 'simultaneous_finite'):
            m = self.__create_doe_model()

            # solve model, achieve results for square problem, and results for optimization problem
            square_result = self.__solve_doe(m, fix=True)

            # analyze results
            if specified_prior is None:
                prior_in_use = self.prior_FIM
            else:
                prior_in_use = specified_prior

            FIM_analysis = FIM_result(self.param, prior_FIM=prior_in_use, store_FIM=FIM_store_name, scale_constant_value=self.scale_constant_value)

            # add the formed simultaneous model to the object so that users can have access
            self.m = m
            self.square_result = square_result

            return FIM_analysis

        elif self.mode=='sequential_finite':
            no_para = len(self.param)

            # if using sequential model
            # call generator function to get scenario dictionary
            scena_gen = Scenario_generator(self.param_init, formula=self.formula, step=self.step)
            scena_gen.generate_sequential_para()

            # dict for storing model outputs
            output_record = {}
            # dict for storing Jacobian
            jac = {}

            # loop over each scenario
            for no_s in (scena_gen.scena_keys):

                scenario_iter = scena_gen.next_sequential_scenario(no_s)
                # create the model
                # TODO:(long term) add options to create model once and then update. only try this after the
                # package is completed and unitest is finished
                mod = self.create_model(scenario_iter)

                # discretize if needed
                if self.discretize_model is not None:
                    mod = self.discretize_model(mod)

                # extract (discretized) time
                time_set = []
                for t in mod.t:
                    time_set.append(value(t))

                # solve model
                square_result = self.__solve_doe(mod, fix=True)

                # loop over measurement item and time to store model measurements
                output_combine = []
                for j in self.measurement_variables:
                    for t in self.measurement_timeset:
                        C_value = eval('mod.' + j + '[0,' + str(t) + ']')
                        output_combine.append(value(C_value))
                output_record[no_s] = output_combine

            # After collecting outputs from all scenarios, calculate sensitivity
            for para in self.param:
                # extract involved scenario No. for each parameter from scenario class
                involved_s = scena_gen.scenario_para[para]
                # each parameter has two involved scenarios
                s1 = involved_s[0]
                s2 = involved_s[1]
                list_jac = []
                for i in range(len(output_record[s1])):
                    if self.scale_nominal_param_value:
                        sensi = (output_record[s1][i] - output_record[s2][i]) / scena_gen.eps_abs[para] *self.param_init[para] *self.scale_constant_value
                    else:
                        sensi = (output_record[s1][i] - output_record[s2][i]) / scena_gen.eps_abs[para]*self.scale_constant_value
                    list_jac.append(sensi)
                # get Jacobian dict, keys are parameter name, values are sensitivity info
                jac[para] = list_jac


            # analyze results
            if specified_prior is None:
                prior_in_use = self.prior_FIM
            else:
                prior_in_use = specified_prior

            FIM_analysis = FIM_result(self.param, prior_FIM=prior_in_use, store_FIM=FIM_store_name, scale_constant_value=self.scale_constant_value)

            # add jacobian info to the object so that users can have access
            self.jac = jac

            return FIM_analysis


        elif self.mode in ['sequential_sipopt', 'sequential_kaug']:
            # create scenario class for a base case
            scena_gen = Scenario_generator(self.param_init, formula=None, step=self.step)
            scenario_all = scena_gen.simultaneous_scenario()

            # sipopt only uses backward difference scheme
            # store measurements for scenarios
            all_perturb_measure = []
            all_base_measure = []
            # store jacobian info
            jac={}

            # loop over parameters
            for pa in range(len(self.param)):
                perturb_mea = []
                base_mea = []

                # create model
                mod = self.create_model(scenario_all)

                # discretize if needed
                if self.discretize_model is not None:
                    mod = self.discretize_model(mod)

                # fix model DOF
                mod = self.__fix_design(mod, self.design_values, fix_opt=True)

                # extract (discretized) time
                time_set = []
                for t in mod.t:
                    time_set.append(value(t))

                # add sIPOPT perturbation parameters
                mod = self.__add_para(mod, perturb=pa)

                # parameter name lists for sipopt
                list_original = []
                list_perturb = []
                for ele in self.param:
                    list_original.append(eval('mod.'+ele+'[0]'))
                for elem in self.perturb_names:
                    list_perturb.append(eval('mod.'+elem+'[0]'))

                # solve model
                if self.mode =='sequential_sipopt':
                    m_sipopt = sensitivity_calculation('sipopt', mod, list_original, list_perturb, tee=self.tee_opt)
                # TODO: add k_aug solver
                else:
                    m_sipopt = sensitivity_calculation('k_aug', mod, list_original, list_perturb, tee=True)

                # extract sipopt result
                for j in self.measurement_variables:
                    for t in self.measurement_timeset:
                        # fetch the measurement variable
                        measure_var = getattr(m_sipopt,j)
                        # check if this variable is fixed
                        if (measure_var[0,t].fixed == True):
                            perturb_value = value(measure_var[0,t])
                            #if self.verbose:
                            #    print(measure_var[0, t], ' is fixed')

                        else:
                            # if it is not fixed, record its perturbed value
                            perturb_value = eval('m_sipopt.sens_sol_state_1[m_sipopt.' + j + '[0,' + str(t) + ']]')
                        perturb_mea.append(perturb_value)

                        # base case values
                        base_value = eval('m_sipopt.'+j+'[0,' + str(t) + '].value')
                        base_mea.append(base_value)
                # store extracted measurements
                all_perturb_measure.append(perturb_mea)
                all_base_measure.append(base_mea)


            # After collecting outputs from all scenarios, calculate sensitivity
            for count, para in enumerate(self.param):
                list_jac = []
                for i in range(len(all_perturb_measure[0])):
                    if self.scale_nominal_param_value:
                        sensi = -(all_perturb_measure[count][i] - all_base_measure[count][i]) / self.step * self.scale_constant_value
                    else:
                        sensi = -(all_perturb_measure[count][i] - all_base_measure[count][i]) / self.step /self.param_init[para] * self.scale_constant_value
                    list_jac.append(sensi)
                # get Jacobian dict, keys are parameter name, values are sensitivity info
                jac[para] = list_jac

            # analyze results
            if specified_prior is None:
                prior_in_use = self.prior_FIM
            else:
                prior_in_use = specified_prior

            # analyze results
            FIM_analysis = FIM_result(self.param, prior_FIM=prior_in_use, store_FIM=FIM_store_name, scale_constant_value=self.scale_constant_value)

            self.jac = jac
            #self.square_result = None

            return FIM_analysis


    def sequential_exp(self, design_values_set, mode='sequential_finite', tee_option=False,
                       scale_nominal_param_value=False, scale_constant_value=1,
                       formula='central', step=0.001):
        '''
        Run a series of experiments sequentially, and use the FIM from one experiment as the prior information for the next experiment
        Args:
            design_values_set: a list of experiments, each element is one design_values dictionary
            mode: use Mode='sequential_finite', 'simultaneous_finite', 'sequential_sipopt', 'sequential_kaug'
            tee_option: if IPOPT console output is printed
            scale_nominal_param_value: if True, the parameters are scaled by its own nominal value in param_init
            scale_constant_value: how many order of magnitudes the Jacobian value is magnified by. Use when the Jac or FIM value is too small
            formula: choose from 'central', 'forward', 'backward', None
            step: Sensitivity perturbation step size, a fraction between [0,1]. default is 0.001

        Returns:
            result_obj_list: a list of the result summary objects of every experiment
            fim_list: a list of the FIM of every experiment
        '''
        # how many exps to run in a row
        self.no_exp = len(design_values_set)
        self.design_values_set = design_values_set
        self.formula = formula
        self.mode = mode
        self.scale_nominal_param_value = scale_nominal_param_value
        self.scale_constant_value = scale_constant_value
        self.obj_opt = 'det'

        # calculate how much the FIM element is magnified
        self.fim_scale_constant_value = self.scale_constant_value ** 2

        self.__check_inputs(check_mode=True, check_dimension_dv=False)

        # store all results object list
        result_object_list = []
        # store all FIM
        fim_list = []
        # loop over experiments
        for i in range(self.no_exp):
            print('========This is the No.', i, ' experiment.========')
            if self.verbose:
                print('Design variables:', self.design_values_set[i])

            # call compute_FIM to get FIM
            if i==0:
                prior_in_use = self.prior_FIM
            else:
                prior_in_use = fim_list[i-1]

            # run the experiment with compute_FIM
            result_iter = self.compute_FIM(self.design_values_set[i], mode=self.mode, specified_prior=prior_in_use,
                                           tee_opt=tee_option,
                                           scale_nominal_param_value=self.scale_nominal_param_value,
                                           formula=formula, step=step)

            if (self.mode == 'simultaneous_finite'):
                result_iter.extract_FIM(self.m, self.design_timeset, self.square_result, self.obj_opt, add_fim=True)

            elif (self.mode in ['sequential_finite', 'sequential_sipopt']):
                result_iter.calculate_FIM(self.jac, self.design_values)

            # attach these results to the store list
            result_object_list.append(result_iter)
            fim_list.append(result_iter.FIM)

        return result_object_list, fim_list

    def run_grid_search(self, design_values, dv_ranges, dv_apply_time, mode='sequential_finite',
                        tee_option=False, scale_nominal_param_value=False, scale_constant_value=1,
                        filename=None, formula='central', step=0.001):
        '''
        Enumerate through full factorial grid search for two design variables;
        solve square problems sequentially to compute FIMs.
        It calculates FIM with sensitivity information from four ways:
        1, Simultaneous: Calculate a multiple scenario model. Sensitivity info estimated by finite difference
        2, Sequential_ipopt: Calculates a one scenario model multiple times for
        multiple scenarios. Sensitivity info estimated by finite difference
        3, Sequential_sipopt: calculate sensitivity by sIPOPT.
        4, Sequential_kaug: calculate sensitivity by k_aug

        Argument:
        General:
            design_values: a dict whose keys are design variable names, values are a dict whose keys are time point and values are the design variable value at that time point
            dv_ranges: a dict whose keys are design variable names, values are a list of design variable values to go over
            dv_apply_time: a dict whose keys are design variable names, values are a list of control time points that
            should be fixed to the values in dv_ranges
            mode: use Mode='sequential_finite', 'simultaneous_finite', 'sequential_sipopt', 'sequential_kaug'
            tee_option: if IPOPT console output is made
            scale_nominal_param_value: if True, the parameters are scaled by its own nominal value in param_init
            scale_constant_value: how many order of magnitudes the Jacobian value is magnified by. Use when the Jac or FIM value is too small
            filename: if True, grid search results stored with this file name

        Only effective when finite=True:
            formula: choose from 'central', 'forward', 'backward', None
            step: Sensitivity perturbation step size, a fraction between [0,1]. default is 0.001

        Return:
            result_combine: a list of dictionaries, which stores the FIM info from every grid searched.
        '''
        self.dv_ranges = dv_ranges
        self.dv_apply_time = dv_apply_time
        self.formula = formula
        self.mode = mode
        # it can be det or trace as it is not optimizing now. put it here for the check_inputs()
        self.scale_nominal_param_value = scale_nominal_param_value
        self.scale_constant_value = scale_constant_value
        self.obj_opt='det'
        self.filename = filename

        # calculate how much the FIM element is magnified
        self.fim_scale_constant_value = self.scale_constant_value ** 2

        self.__check_inputs(check_mode=True, check_dimension_dv=True)

        # when defining design space, design variable values are defined as in design_values argument
        # the design var value defined in dv_ranges only applies to control time points given in dv_apply_time
        self.dv_control_times = list(dv_apply_time.values())

        # to store all FIM results
        result_combine = []

        # time 0
        t_begin = time.time()

        # iteration 0
        count = 0
        # how many sets of design variables will be run
        total_count = len(dv_ranges[self.dv_name[0]]) * len(dv_ranges[self.dv_name[1]])

        # loop over design variables
        for i, value_0 in enumerate(dv_ranges[self.dv_name[0]]):
            for j, value_1 in enumerate(dv_ranges[self.dv_name[1]]):
                # generate the design variable dictionary needed for running compute_FIM
                design_iter = {}

                # first copy value from design_Values
                dv_iter1 = design_values[self.dv_name[0]].copy()
                dv_iter2 = design_values[self.dv_name[1]].copy()

                # for timepoints given in dv_apply_times, change its value to be the ones in dv_ranges to go over
                for t, tim in enumerate(self.dv_control_times[0]):
                    dv_iter1[tim] = value_0
                design_iter[self.dv_name[0]] = dv_iter1

                for t, tim in enumerate(self.dv_control_times[1]):
                    dv_iter2[tim] = value_1
                design_iter[self.dv_name[1]] = dv_iter2

                print('=======This is the ', count+1, 'th iteration=======')
                print('Design variable values of this iteration:', design_iter)

                t_each_begin = time.time()

                # call compute_FIM to get FIM
                result_iter = self.compute_FIM(design_iter, mode=self.mode,
                                               tee_opt=tee_option,
                                               scale_nominal_param_value=self.scale_nominal_param_value,
                                               formula=formula, step=step)

                if (self.mode=='simultaneous_finite'):
                    result_iter.extract_FIM(self.m, self.design_timeset, self.square_result, self.obj_opt)

                elif (self.mode == 'sequential_finite'):
                    result_iter.calculate_FIM(self.jac, self.design_values)

                elif (self.mode == 'sequential_sipopt'):
                    result_iter.calculate_FIM(self.jac, self.design_values)

                t_now = time.time()

                if self.verbose:
                    # give run information at each iteration
                    print('This is the ', count+1, ' run out of ', total_count, 'run.')
                    print('The code has run %.04f seconds.'% (t_now-t_begin))
                    print('Estimated remaining time: %.4f seconds' % ((t_now-t_begin)/(count+1)*(total_count-count-1)))
                count += 1

                result_combine.append(result_iter)

        t_end = time.time()
        print('The whole run takes ', t_end - t_begin, ' s.')

        # store results
        if self.filename is not None:
            f = open(filename, 'wb')
            pickle.dump(result_combine, f)
            f.close()

        return result_combine

    def sensitivity_analysis_1D(self, design_values, design_var_range, sensitivity_step=0.1, compare_opt='D',
                                mode='sequential_finite', tee_option=False,
                                scale_nominal_param_value=False, scale_constant_value=1,
                                formula='central', step=0.001):
        '''
        This method is used for 1D sensitivity analysis.
        TODO: Let's decide if we are gonna keep this. This seems not necessary to this package.
        Args:
            design_values: a dict whose keys are design variable names,
                            values are a dict whose keys are time point and values are the design variable value at that time point
            design_var_range: a dict, the key is the design variable name to conduct analysis,
                            the value is a list [Lower bound, Upper bound] for this sensitivity analysis
            sensitivity_step: the interval of the design range. For e.g., 0.2 in [2,3], then design variable is computed at [2,2.2,2.4,2.6,2.8]
            compare_opt: which design criteria is to compare. Choose from 'A', 'D', 'E', 'ME'
            mode: use Mode='sequential_finite', 'simultaneous_finite', 'sequential_sipopt', 'sequential_kaug'
            tee_option: if IPOPT console output is printed
            scale_nominal_param_value: if True, the parameters are scaled by its own nominal value in param_init
            scale_constant_value: how many order of magnitudes the Jacobian value is magnified by. Use when the Jac or FIM value is too small
            formula: choose from 'central', 'forward', 'backward', None
            step: Sensitivity perturbation step size, a fraction between [0,1]. default is 0.001

        Returns:
            result_list: a list of the specified optimality value
        '''

        self.design_values = design_values
        self.formula = formula
        self.mode = mode
        self.scale_nominal_param_value = scale_nominal_param_value
        self.scale_constant_value = scale_constant_value
        self.obj_opt = 'zero'

        # calculate how much the FIM element is magnified
        self.fim_scale_constant_value = self.scale_constant_value ** 2

        self.__check_inputs(check_mode=True, check_dimension_dv=False)

        # get the name and the range of the design variable to conduct analysis
        vary_dv_name = list(design_var_range.keys())
        vary_range_list = list(design_var_range.values())
        vary_range = np.arange(vary_range_list[0][0], vary_range_list[0][1], step=sensitivity_step)
        if self.verbose:
            print('Sensitivity analysis is for ', vary_dv_name, ' at the range ', vary_range_list)
            print('At the points:', vary_range)

        dv_time_set = list(design_values[vary_dv_name[0]].keys())

        # store the criteria value of every run
        result_list= []
        # loop over the test points
        for i in range(len(vary_range)):
            if self.verbose:
                print('This is the ',i,'-th iteration')
            design_value_list = design_values.copy()
            for t in dv_time_set:
                design_value_list[vary_dv_name[0]][t] = vary_range[i]
            if self.verbose:
                print('Design variables for this run:', design_value_list)

            # compute square problems
            result_iter = self.compute_FIM(design_value_list, mode=self.mode,
                                           tee_opt=tee_option,
                                           scale_nominal_param_value=self.scale_nominal_param_value,
                                           formula=formula, step=step)

            if (self.mode == 'simultaneous_finite'):
                result_iter.extract_FIM(self.m, self.design_timeset, self.square_result, self.obj_opt)

            elif (self.mode in ['sequential_finite','sequential_sipopt']):
                result_iter.calculate_FIM(self.jac, self.design_values)

            # decide which design criteria is the user asked for
            if compare_opt == 'A':
                result_list.append(result_iter.trace)

            elif compare_opt =='D':
                result_list.append(result_iter.det)

            elif compare_opt =='E':
                result_list.append(result_iter.min_eig)

            elif compare_opt =='ME':
                result_list.append(result_iter.cond)

        sensi_result_dict = {}
        for i in range(len(result_list)):
            sensi_result_dict[vary_range[i]] = result_list[i]

        if self.verbose:
            print('The result list:', result_list)
        return sensi_result_dict


    def __create_doe_model(self):
        '''
        Add features for DOE.

        Information needed from self object:
            self.measurement_variables: the variable name of the model output, for e.g., ['CA', 'CB', 'CC'].
            self.measurement_timeset: a list of measurement time points. can be different from control time points
            self.design_values: a dict of dictionaries, keys are the name of design variables, values are a dict where keys are the time points, values are the design variable value at that time point

        DoE options:
            self.optimize: if True, solve the problem unfixing the design variables. if False, solve the problem as a
            square problem
            self.obj_opt: choose from 'det' or 'trace'. Optimization problem maximizes determinant or trace.
            self.scale_nominal_param_value: if True, scale FIM but not scale Jacobian. This toggle can be opened for better performance when the
            problem is poorly scaled.
            self.tee_opt: if True, print IPOPT console output
            self.cho_opt: if true, cholesky decomposition is used for Objective function (to optimize determinant). If true, determinant will not be calculated.
                self.L_LB: if FIM is P.D., the diagonal element should be positive, so we can set a LB like 1E-10
                self.L_initial: initialize the L
            self.formula: choose from 'central', 'forward', 'backward', None
            self.step: Sensitivity perturbation step size, a fraction between [0,1]. default is 0.001

        Return:
            m: the solved DOE model
        '''
        # call generator function to get scenario dictionary
        scena_gen = Scenario_generator(self.param_init, formula=self.formula, step=self.step, store=True)
        scenario_all = scena_gen.simultaneous_scenario()
        
        # create model
        m = self.create_model(scenario_all)
        # discretize if discretization function is provided
        if self.discretize_model is not None:
            m = self.discretize_model(m)
        
        # extract (discretized) time 
        time_set=[]
        for t in m.t:
            time_set.append(value(t))

        # create parameter, measurement, time and measurement time set
        m.para_set = Set(initialize=self.param)
        param_name = self.param
        m.y_set = Set(initialize=self.measurement_variables)
        m.t_set = Set(initialize=time_set)
        m.tmea_set = Set(initialize=self.measurement_timeset)

        # we can be sure about the name of scenarios, because they are generated by our function
        m.scenario = Set(initialize=scenario_all['scena-name'])
        m.optimize = self.optimize

        # check if measurement time points are in the time set
        for t in m.tmea_set:
            if not (t in m.t):
                raise ValueError('Warning: Measure timepoints should be in the time list.')

        # check if control time points are in the time set
        for d in range(len(self.dv_name)):
            for t in self.dv_time[d]:
                if not (t in m.t):
                    raise ValueError('Warning: Control timepoints should be in the time list.')

        ### Define variables
        # Elements in Jacobian matrix
        m.jac = Var(m.y_set, m.para_set, m.tmea_set, initialize=1E-20)

        # Initialize Hessian with an identity matrix
        def identity_matrix(m,j,d):
            if j==d:
                return 1 
            else: 
                return 0
        m.FIM = Var(m.para_set, m.para_set, initialize=identity_matrix)

        if self.obj_opt=='trace':
            # Trace of FIM
            m.trace = Var(initialize=1, within=NonNegativeReals)
        elif self.obj_opt=='det':
            # Determinant of FIM
            m.det = Var(initialize=0.5, within=NonNegativeReals)
        elif (self.obj_opt != 'zero'):
            raise ValueError('Undefined objective function type. Available options are "trace" and "det".')


        if self.L_initial is not None:
            dict_cho={}
            for i, bu in enumerate(m.para_set):
                for j, un in enumerate(m.para_set):
                    dict_cho[(bu,un)] = self.L_initial[i][j]

        def init_cho(m,i,j):
            return dict_cho[(i,j)]

        if self.cho_opt:
            # Define elements of Cholesky decomposition matrix as Pyomo variables and either
            # Initialize with L in L_initial
            if self.L_initial is not None:
                m.L_ele = Var(m.para_set, m.para_set, initialize=init_cho)
            # or initialize with the identity matrix
            else:
                m.L_ele = Var(m.para_set, m.para_set, initialize=identity_matrix)

            # loop over parameter name
            for c in m.para_set:
                for d in m.para_set:
                    # fix the 0 half of L matrix to be 0.0
                    if (param_name.index(c) < param_name.index(d)):
                        m.L_ele[c,d].fix(0.0)
                    # Give LB to the diagonal entries 
                    if self.L_LB is not None:
                        if c==d:
                            m.L_ele[c,d].setlb(self.L_LB)


        def jac_numerical(m,j,p,t):
            '''
            Calculate the Jacobian
            j: model responses
            p: model parameters
            t: timepoints
            '''
            # A better way to do this: 
            # https://github.com/IDAES/idaes-pse/blob/274e58bef55f2f969f0df97cbb1fb7d99342388e/idaes/apps/uncertainty_propagation/sens.py#L296

            up_C = eval('m.'+j+'['+str(scenario_all['jac-index'][p][0])+','+str(t)+']')
            lo_C = eval('m.'+j+'['+str(scenario_all['jac-index'][p][1])+','+str(t)+']')

            return m.jac[j,p,t] == (up_C - lo_C)/scenario_all['eps-abs'][p] *self.scale_constant_value

        #A constraint to calculate elements in Hessian matrix
        # transfer prior FIM to be Expressions
        dict_fele={}
        for i, bu in enumerate(m.para_set):
            for j, un in enumerate(m.para_set):
                dict_fele[(bu,un)] = self.prior_FIM[i][j]

        def ele_todict(m,i,j):
            return dict_fele[(i,j)]
        m.refele = Expression(m.para_set, m.para_set, rule=ele_todict)

        def calc_FIM(m,j,d):
            '''
            Calculate FIM elements
            '''
            # check if scale
            if self.scale_nominal_param_value:
                return m.FIM[j,d] == sum(sum(m.jac[z,j,i]*self.param_init[j]*self.param_init[d]*m.jac[z,d,i] for z in m.y_set) for i in m.tmea_set) + m.refele[j, d]*self.fim_scale_constant_value
            else:
                return m.FIM[j,d] == sum(sum(m.jac[z,j,i]*m.jac[z,d,i] for z in m.y_set) for i in m.tmea_set) + m.refele[j, d]*self.fim_scale_constant_value

        def trace_calc(m):
            '''
            Calculate FIM elements. Can magnify each element with 1000 for performance
            '''
            sum_x = 0  
            for j in m.para_set:
                for d in m.para_set:
                    if d==j:
                        sum_x += m.FIM[j,d]
            return m.trace == sum_x 

        def det_general(m):
            '''Calculate determinant. Can be applied to FIM of any size. 
            det(A) = sum_{\sigma \in \S_n} (sgn(\sigma) * \Prod_{i=1}^n a_{i,\sigma_i})
            Use permutation() to get permutations, sgn() to get signature
            '''
            r_list = list(range(len(m.para_set)))
            # get all permutations 
            object_p = permutations(r_list)
            list_p = list(object_p)

            # generate a name_order to iterate \sigma_i
            det_perm = 0
            for i in range(len(list_p)):
                name_order = []
                x_order = list_p[i]
                # sigma_i is the value in the i-th position after the reordering \sigma
                for x in range(len(x_order)):
                    for y, element in enumerate(m.para_set):
                        if x_order[x] == y:
                            name_order.append(element)

            # det(A) = sum_{\sigma \in \S_n} (sgn(\sigma) * \Prod_{i=1}^n a_{i,\sigma_i})
            det_perm = sum( sgn(list_p[d])*sum(m.FIM[each, name_order[b]] for b, each in enumerate(m.para_set)) for d in range(len(list_p)))
            return m.det == det_perm


        def cholesky_imp(m,c,d):
            '''
            Calculate Cholesky L matrix using algebraic constraints
            '''
            # If it is the left bottom half of L
            if (param_name.index(c)>=param_name.index(d)):
                return m.FIM[c,d] ==  sum(m.L_ele[c, param_name[k]]*m.L_ele[d, param_name[k]] for k in range(param_name.index(d)+1))  
            else:
                # This is the empty half of L above the diagonal
                return Constraint.Skip


        ### Constraints and OF
        m.dC_value = Constraint(m.y_set, m.para_set, m.tmea_set, rule=jac_numerical)
        m.ele_rule = Constraint(m.para_set, m.para_set, rule=calc_FIM)  

        # Only giving OF when there's DOF. Make OBJ=0 when it's a square problem, which helps converge. 
        if self.optimize:
            # if cholesky, calculating L and evaluate the OBJ with Cholesky decomposition
            if self.cho_opt:
                m.cholesky_cons = Constraint(m.para_set, m.para_set, rule=cholesky_imp)
                m.obj = Objective(expr=2*sum(log(m.L_ele[j,j]) for j in m.para_set), sense=maximize)
            # if not cholesky but determinant, calculating det and evaluate the OBJ with det 
            elif (self.obj_opt=='det'):
                m.det_rule =  Constraint(rule=det_general)
                m.obj = Objective(expr=log(m.det), sense=maximize)
            # if not determinant or cholesky, calculating the OBJ with trace
            elif (self.obj_opt=='trace'):
                m.trace_rule = Constraint(rule=trace_calc)
                m.obj = Objective(expr=log(m.trace), sense=maximize)
            elif (self.obj_opt=='zero'):
                m.obj = Objective(expr=0)
        else:
            m.obj = Objective(expr=0)

        return m


    def __fix_design(self, m, design_val, fix_opt=True):
        ''' Fix design variable

        Args:
            m: model
            design_val: design variable values dict
            fix_opt: if True, fix. Else, unfix

        Returns:
            m: model
        '''
        # loop over the design variables and time index and to fix values specified in design_val
        for d, dname in enumerate(self.dv_name):
            for t, time in enumerate(self.dv_time[d]):
                newvar = eval('m.' + dname + '[' + str(time) + ']')
                fix_v = design_val[dname][time]
                if fix_opt:
                    newvar.fix(fix_v)
                else:
                    newvar.unfix()
        return m

    def __solve_with_default_ipopt(self):
        ''' Default solver
        '''
        solver = SolverFactory('ipopt')
        solver.options['linear_solver'] = 'ma57'
        solver.options['halt_on_ampl_error'] = 'yes'
        return solver

    def __solve_doe(self, m, fix=False):
        '''Solve DOE model.
        If it's a square problem, fix design variable and solve.
        Else, fix design variable and solve square problem firstly, then unfix them and solve the optimization problem

        m:model
        fix: if True, solve two times (square first). Else, just solve the square problem

        Return:
            solver_results: solver results
        '''
        ### Solve square problem
        mod = self.__fix_design(m, self.design_values, fix_opt=fix)

        # if user gives solver, use this solver. if not, use default IPOPT solver
        solver_result = self.solver.solve(mod,tee=self.tee_opt)

        return solver_result

    def __add_para(self, m, perturb=0):
        '''
        For sIPOPT: add parameter perturbation set
        m: model name
        self.param_names: perturbation parameter names
        perturb: which parameter to perturb
        '''
        # model parameters perturbation, backward disturb
        param_do = self.param_v.copy()
        # perturb parameter
        param_do[perturb] *= (1-self.step)

        # generate sIPOPT perturbed parameter names
        param_perturb_names = self.param.copy()
        for x, xname in enumerate(self.param):
            param_perturb_names[x] = xname+'_pert'
        if self.verbose:
            print('perturb names are:', param_perturb_names)

        self.perturb_names = param_perturb_names
        if self.verbose:
            print('Perturbation parameters are set:')
        for change in range(len(self.perturb_names)):
            setattr(m, self.perturb_names[change], Param(m.scena, initialize=param_do[change]))
            if self.verbose:
                print(self.perturb_names[change], ': ', value(eval('m.'+self.perturb_names[change]+'[0]')))
        return m


class Scenario_generator:
    def __init__(self, para_dict, formula='central', step=0.001, store=False):
        '''Generate scenarios.
        DoE library first calls this function to generate scenarios.
        For sequential and simultaneous models, call different functions in this class.

        Args:
        para_dict: a Dict of parameter, keys are names, values are their nominal value. for e.g.,
                        {'A1': 84.79085853498033, 'A2': 371.71773413976416, 'E1': 7.777032028026428, 'E2': 15.047135137500822}
        formula: choose from 'central', 'forward', 'backward', None.
        step: Sensitivity perturbation step size, a fraction between [0,1]. default is 0.001
        store: if store results
        '''

        if formula not in ['central', 'forward', 'backward', None]:
            raise ValueError('Error: undefined formula. Available formulas: central, forward, backward, none.')

        # get info from parameter dictionary
        self.para_dict = para_dict
        self.para_names = list(para_dict.keys())
        self.no_para = len(self.para_names)
        self.formula = formula
        self.step = step
        self.store = store
        # This is the parameter nominal values
        self.scenario_nominal = []
        for d in self.para_names:
            self.scenario_nominal.append(para_dict[d])

    def simultaneous_scenario(self):
        '''
        Generate scenario dict for simultaneous models

        Return:
            scena_overall: a dictionary containing scenarios dictionaries.
            scena_overall[name of parameter]: a dict, keys are the scenario name(numeric integer starting from 0), values are parameter value in this scenario
            scena_overall['jac-index']: keys are parameter name, values are the scenario names perturbing this parameter.
            scena_overall['eps-abs']: keys are parameter name, values are the step it is perturbed
            scena_overall['scena-name']: a list of scenario names

            For e.g., if a dict {'P':100, 'D':20} is given, step=0.1, formula='central', it will return:
            scena_overall = {'P': {0: 101.0, 1: 100, 2: 99.0, 3: 100}, 'D': {0: 20, 1: 20.2, 2: 20, 3: 19.8}, 'jac-index': {'P': [0, 2], 'D': [1, 3]}, 'eps-abs': {'P': 2.0, 'D': 0.4}, 'scena-name': [0, 1, 2, 3]}
            if formula ='forward', it will return:
            scena_overall = {'P':{'0':110, '1':100, '2':100}, 'D':{'0':20, '1':22, '2':20}, 'jac-index':{'P':[0,2], 'D':[1,2]}, 'eps-abs':{'P':10,'D':2}, 'scena-name': [0,1,2]}
        '''
        # generate scenarios
        scena_keys, scena = self.__scena_generate(self.scenario_nominal, self.formula)
        self.scena_keys = scena_keys
        self.scena = scena

        # call scneario class and method
        scenario_object = Scenario_data(self.para_dict, self.scena_keys, self.scena, self.formula, self.step)
        scenario_overall = scenario_object.create_scenario()

        # store scenario
        if self.store:
            f = open('scenario_simultaneous', 'wb')
            pickle.dump(scenario_overall, f)
            f.close()

        return scenario_overall

    def next_sequential_scenario(self, count):
        '''
        Generate a single scenario class for one of the sequential models

        Parameters
        ----------
        count: the No. of the sequential models

        Returns
        -------
        scenario_next: scenario dict for this sequential model
        '''
        scena_keys, scena = self.__scena_generate(list(self.scena[count].values()), None)

        # each model is basically a 'none' case of an invasive model
        scenario_object = Scenario_data(self.scena[count], scena_keys, scena, None, self.step)
        scenario_next = scenario_object.create_scenario()

        return scenario_next

    def generate_sequential_para(self):
        '''
        Generate object and some 'parameters' for sequential models

        Returns (added to self object)
        -------
        self.scena_keys: scenario name, a list of numbers
        self.scena: a list of parameter dictionaries for all sequential models
        self.scenario_para: a list of two No. of models involved in calculating one parameter sensitivity
        self.eps_abs: keys are parameter name, values are the step it is perturbed
        '''

        scena_keys, scena = self.__scena_generate(self.scenario_nominal, self.formula)
        self.scena_keys = scena_keys
        self.scena = scena

        # record the number of scenarios involved in calculating a certain parameter sensitivities
        scenario_para = {}
        for p, para in enumerate(self.para_names):
            # the scenario involved in Jacobian calculation
            if self.formula == 'central':
                scenario_para[para] = [p, p + self.no_para]
            elif self.formula == None:
                raise ValueError('Error: finite difference scheme should be chosen.')
            else:
                scenario_para[para] = [p, self.no_para]

        self.scenario_para = scenario_para

        # calculate the perturbation size of every parameter
        eps_abs = {}
        for para in self.para_names:
            # for central difference scheme, perturbation size is two times the step size
            if self.formula == 'central':
                eps_abs[para] = 2 * self.step * self.para_dict[para]
            else:
                eps_abs[para] = self.step * self.para_dict[para]

        self.eps_abs = eps_abs

    def __scena_generate(self, para_nominal, formula):
        '''
        Generate scenario logics

        Return: (store in self object)
        self.scena_keys: a list of scenario names
        self.scena: a dict, keys are scenario names, values are a list of parameter values
        '''
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
        '''
        Generate scenario for a simultaneous model

        parameter_dict: parameter dictionaries
        scena_keys: scenario name, a list of numbers
        scena: a list of parameter dictionaries for all sequential models
        form: choose from 'central', 'forward', 'backward', 'none'.
        step: stepsize of a fraction, such as 0.01

        Return:
            scena_dict: a dictionary containing scenarios dictionaries.
            scena_dict[name of parameter]: a dict, keys are the scenario name(numeric integer starting from 0), values are parameter value in this scenario
            scena_dict['jac-index']: keys are parameter name, values are the scenario names perturbing this parameter.
            scena_dict['eps-abs']: keys are parameter name, values are the step it is perturbed
            scena_dict['scena-name']: a list of scenario names

            For e.g., if a dict {'P':100, 'D':20} is given, step=0.1, formula='central', it will return:
            scena_dict = {'P': {0: 101.0, 1: 100, 2: 99.0, 3: 100}, 'D': {0: 20, 1: 20.2, 2: 20, 3: 19.8}, 'jac-index': {'P': [0, 2], 'D': [1, 3]}, 'eps-abs': {'P': 2.0, 'D': 0.4}, 'scena-name': [0, 1, 2, 3]}
            if formula ='forward', it will return:
            scena_dict = {'P':{'0':110, '1':100, '2':100}, 'D':{'0':20, '1':22, '2':20}, 'jac-index':{'P':[0,2], 'D':[1,2]}, 'eps-abs':{'P':10,'D':2}, 'scena-name': [0,1,2]}
        '''
        # get info from parameter dictionary
        self.para_dict = parameter_dict
        self.para_names = list(parameter_dict.keys())

        self.scena = scena
        self.scena_keys = scena_keys
        # print('scena:', scena)
        # print('scena keys:', scena_keys)
        self.no_para = len(self.para_names)
        self.formula = form
        self.step = step

        # This is the parameter nominal values
        self.scenario_nominal = []
        for d in self.para_names:
            self.scenario_nominal.append(parameter_dict[d])

    def create_scenario(self):
        # overall dict to return
        scenario_dict = {}
        # dict for scenario position
        jac_index = {}
        # dict for parameter perturbaion step size
        eps_abs = {}

        # loop over parameter name
        for p, para in enumerate(self.para_names):
            scena_p = {}
            for n in self.scena_keys:
                # print(self.scena[n][para])
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

        # print('Return scenario dict as:', scenario_dict)
        return scenario_dict


class FIM_result:
    def __init__(self, para_name, prior_FIM=None, store_FIM=None, scale_constant_value=1, max_condition_number=1.0E12,
                 verbose=True):
        '''
        Analyze the FIM result for a single run

        Args:
            para_name: parameter names
            prior_FIM: if there's prior FIM to be added
            store_FIM: if storing the FIM in a .csv, give the file name here as a string, '**.csv' or '**.txt'.
            scale_constant_value: the constant value used to scale the sensitivity
            max_condition_number: max condition number
            verbose: if print statements are used
        '''
        self.para_name = para_name
        self.prior_FIM = prior_FIM
        self.store_FIM = store_FIM
        self.scale_constant_value = scale_constant_value
        self.fim_scale_constant_value = scale_constant_value ** 2
        self.max_condition_number = max_condition_number
        self.verbose = verbose

    def calculate_FIM(self, jaco_info, dv_values, result=None):
        '''
        Calculate FIM from Jacobian information. This is for grid search (combined models) results

        Args:
            jaco_info: jacobian dictionary
            dv_values: design variable value dictionary
            sq_result: solver status returned by IPOPT

        Return:
            fim_info: a FIM dictionary
                ~['FIM']: FIM itself
                ~[design variable name]: design variable values at each time point
                ~['Trace']: Trace
                ~['Determinant']: determinant
                ~['Condition number:']: condition number
                ~['Minimal eigen value:']: minimal eigen value
                ~['Eigen values:']: all eigen values
                ~['Eigen vectors:']: all eigen vectors
            solver_info: a solver infomation dictionary
                ~['square']: square result solver status
        '''
        self.result = result
        self.doe_result = None
        # create a dict for FIM. It has the same keys as the Jacobian dict.

        no_param = len(self.para_name)
        ### calculate the FIM
        fim = np.zeros((no_param, no_param))
        # loop over parameters
        for row, para_n in enumerate(self.para_name):
            for col, para_m in enumerate(self.para_name):
                jaco_n = np.asarray(jaco_info[para_n])
                jaco_m = np.asarray(jaco_info[para_m])
                #fim[row][col] = jaco_info[para_n].T@jaco_info[para_m]
                fim[row][col] = jaco_n.T@jaco_m

        # add prior information
        if (self.prior_FIM is not None):
            try:
                fim = fim + self.prior_FIM
                print('Existed information has been added.')
            except:
                raise ValueError('Check the shape of prior FIM')

        if np.linalg.cond(fim) > self.max_condition_number:
            print("Warning: FIM is near singular.")

        # call private methods
        self.__print_FIM_info(fim, dv_set=dv_values)
        if self.result is not None:
            self.__solver_info()

        # if given store file name, store the FIM
        if (self.store_FIM is not None):
            self.__store_FIM()

    def extract_FIM(self, m, dv_set, result, obj=None, add_fim=False):
        '''
        Extract FIM from an invasive model

        Args:
            m: model
            dv_set: design variable value dictionary
            result: problem solver status by IPOPT
            obj: chosen from 'det' and 'trace'
            add_fim: if the given FIM needs to be added. Do not add for optimize_doe().

        Return:
            fim_info: a FIM dictionary
                ~['FIM']: FIM itself
                ~[design variable name]: design variable values at each time point
                ~['Trace']: Trace
                ~['Determinant']: determinant
                ~['Condition number:']: condition number
                ~['Minimal eigen value:']: minimal eigen value
                ~['Eigen values:']: all eigen values
                ~['Eigen vectors:']: all eigen vectors
            model_info: model solutions dictionary
                ~['obj']: objective function value
                ~['det']: determinant calculated by the model (different from FIM_info['det'] which
                is calculated by numpy)
                -['trace']: trace calculated by the model
                -[design variable name]: design variable solution
            solver_status: a solver infomation dictionary
                ~['square']: square result solver status
                -['doe']: doe result solver status
        '''
        self.result = result
        self.obj = obj
        no_para = len(self.para_name)

        # Extract FIM infomation
        FIM = np.ones((no_para, no_para))
        for n1, name1 in enumerate(self.para_name):
            for n2, name2 in enumerate(self.para_name):
                FIM[n1, n2] = value(m.FIM[name1, name2]) / self.fim_scale_constant_value

        # add prior information
        if add_fim:
            if (self.prior_FIM is not None):
                try:
                    FIM = FIM + self.prior_FIM
                    print('Existed information has been added.')
                except:
                    raise ValueError('Check the shape of prior FIM')

        # call private methods
        self.__print_FIM_info(FIM, dv_set=dv_set)
        self.__solution_info(m, dv_set)
        self.__solver_info()

        # if given store file name, store the FIM
        if (self.store_FIM is not None):
            self.__store_FIM()

    def __print_FIM_info(self, FIM, dv_set=None):
        '''
        using a dictionary to store all FIM information

        Args:
            FIM: the Fisher Information Matrix, needs to be P.D. and symmetric
            dv_set: design variable dictionary

        Return:
            self attributes:
                ~['FIM']: FIM itself
                ~[design variable name]: design variable values at each time point
                ~['Trace']: Trace
                ~['Determinant']: determinant
                ~['Condition number:']: condition number
                ~['Minimal eigen value:']: minimal eigen value
                ~['Eigen values:']: all eigen values
                ~['Eigen vectors:']: all eigen vectors
        '''
        eig = np.linalg.eigvals(FIM)
        self.FIM = FIM
        self.trace = np.trace(FIM)
        self.det = np.linalg.det(FIM)
        self.min_eig = min(eig)
        self.cond = max(eig) / min(eig)
        self.eig_vals = eig
        self.eig_vecs = np.linalg.eig(FIM)[1]

        dv_names = list(dv_set.keys())

        FIM_dv_info = {}
        FIM_dv_info[dv_names[0]] = dv_set[dv_names[0]]
        FIM_dv_info[dv_names[1]] = dv_set[dv_names[1]]

        self.dv_info = FIM_dv_info

        if self.verbose:
            print('FIM:', self.FIM)

            print('Trace:', self.trace)
            print('Determinant:', self.det)
            print('Condition number:', self.cond)
            print('Minimal eigen value:', self.min_eig)
            print('Eigen values:', self.eig_vals)
            print('Eigen vectors:', self.eig_vecs)

    def __solution_info(self, m, dv_set):
        '''
        Solution information. Only for optimization problem

        Args:
            m: model
            dv_set: design variable dictionary

        Return:
            self attributes: model solved information dictionary
                ~['obj']: objective function value
                ~['det']: determinant calculated by the model (different from FIM_info['det'] which
                is calculated by numpy)
                -['trace']: trace calculated by the model
                -[design variable name]: design variable solution
        '''
        self.obj_value = value(m.obj)
        print('Model objective:', self.obj_value)

        if self.obj == 'det':
            self.obj_det = np.exp(value(m.obj)) / (self.fim_scale_constant_value) ** (len(self.para_name))
            print('Objective(determinant) is:', self.obj_det)
        elif self.obj == 'trace':
            self.obj_trace = np.exp(value(m.obj)) / (self.fim_scale_constant_value)
            print('Objective(trace) is:', self.obj_trace)

        dv_names = list(dv_set.keys())
        dv_times = list(dv_set.values())

        solution = {}
        for d, dname in enumerate(dv_names):
            sol = []
            for t, time in enumerate(dv_times[d]):
                newvar = eval('m.' + dname + '[' + str(time) + ']')
                sol.append(value(newvar))
            solution[dname] = sol
            if self.verbose:
                print('Solution of ', dname, ' :', solution[dname])
        self.solution = solution

    def __store_FIM(self):
        # if given store file name, store the FIM
        store_dict = {}
        for i, name in enumerate(self.para_name):
            store_dict[name] = self.FIM[i]
        FIM_store = pd.DataFrame(store_dict)
        FIM_store.to_csv(self.store_FIM, index=False)

    def __solver_info(self):
        '''
        Solver information dictionary

        Return:
            self.status: solver status
        '''
        print('======problem solver output======')

        if (self.result.solver.status == SolverStatus.ok) and (
                self.result.solver.termination_condition == TerminationCondition.optimal):
            self.status = 'converged'
            print('converged')
        elif (self.result.solver.termination_condition == TerminationCondition.infeasible):
            self.status = 'infeasible'
            print('infeasible solution')
        else:
            self.status = self.result.solver.status
            print('solver status:', self.result.solver.status)


class Grid_Search_Result:
    def __init__(self, dv_ranges, FIM_result_list, store_optimality_name=None, verbose=True):
        '''
        This class deals with the FIM results from grid search,
        turns them into heatmaps.

        Args:
            dv_ranges: a dict whose keys are design variable names, values are a list of design variable values to go over
            FIM_result_list: FIM results list from grid search functions
            store_optimality_name: a csv file name containing all four optimalities value
            verbose: if print statements

        Return:
            heatmap
        '''
        # design variables
        self.dv_names = list(dv_ranges.keys())
        self.dv_ranges = dv_ranges
        self.FIM_result_list = FIM_result_list
        self.len_range1 = len(dv_ranges[self.dv_names[0]])
        self.len_range2 = len(dv_ranges[self.dv_names[1]])
        self.store_optimality_name = store_optimality_name
        self.verbose = verbose

    def __extract_criteria(self):
        '''
        Extract criteria values from each FIM info class
        '''
        # initialize the resulted matrix
        # A-opt results in numpy array
        cri_a = np.zeros((self.len_range1, self.len_range2))
        # D-opt
        cri_d = np.zeros((self.len_range1, self.len_range2))
        # E-opt
        cri_e = np.zeros((self.len_range1, self.len_range2))
        # Modified E-opt
        cri_e_cond = np.zeros((self.len_range1, self.len_range2))

        # a list store all results
        store_all_results = []

        # loop over design space
        for no_i, i in enumerate(self.dv_ranges[self.dv_names[0]]):
            for no_j, j in enumerate(self.dv_ranges[self.dv_names[1]]):
                if self.verbose:
                    print('At ', self.dv_names[0], '=', i, ', ', self.dv_names[1], '= ', j, ':')

                # map the FIM info class to the overall list
                fim_result_no = no_j + no_i * self.len_range2
                fim_iter = self.FIM_result_list[fim_result_no]

                if self.verbose:
                    print('We found the class where ', self.dv_names[0], '=', fim_iter.dv_info[self.dv_names[0]], ',',
                          self.dv_names[1], '=', fim_iter.dv_info[self.dv_names[1]])

                cri_a[no_i][no_j] = fim_iter.trace
                cri_d[no_i][no_j] = fim_iter.det
                cri_e[no_i][no_j] = fim_iter.min_eig
                cri_e_cond[no_i][no_j] = fim_iter.cond

                # store results
                store_iteration_result = [i, j, fim_iter.trace, fim_iter.det, fim_iter.min_eig, fim_iter.cond]
                store_all_results.append(store_iteration_result)

                if self.verbose:
                    print('A-optimal result is', cri_a[no_i][no_j], 'D-optimal result is', cri_d[no_i][no_j],
                          'E-optimal(minimal eigenvalue) result is', cri_e[no_i][no_j],
                          'Modified E-optimal (condition number) result is', cri_e_cond[no_i][no_j])

        self.cri_a = cri_a
        self.cri_d = cri_d
        self.cri_e = cri_e
        self.cri_e_cond = cri_e_cond
        # give user access to all results
        self.all_result = store_all_results

        # store optimality values
        if self.store_optimality_name is not None:
            column_names = [self.dv_names[0], self.dv_names[1], 'A', 'D', 'E', 'ME']
            store_df = pd.DataFrame(store_all_results, columns=column_names)
            store_df.to_csv(self.store_optimality_name, index=False)

    def heatmap(self, title_text, xlabel_text, ylabel_text, font_axes=16, font_tick=14, log_scale=True):
        '''
        draw heatmaps of the three criteria

        title_text: name of the heatmap, a string
        xlabel_text: x label title, a string, should be the second design varialbe in the dv_ranges
        ylabel_text: y label title, a string, should be the first design variable in the dv_ranges
        font_axes: axis font size
        font_tick: axis tick font size
        log_scale: if True, the result matrix will be scaled by log10
        '''
        # Get meshgrids of design variables and results for plotting

        self.__extract_criteria()

        if log_scale:
            hes_a = np.log10(self.cri_a)
            hes_e = np.log10(self.cri_e)
            hes_d = np.log10(self.cri_d)
            hes_e2 = np.log10(self.cri_e_cond)
        else:
            hes_a = self.cri_a
            hes_e = self.cri_e
            hes_d = self.cri_d
            hes_e2 = self.cri_e_cond

        xLabel = self.dv_ranges[self.dv_names[0]]
        yLabel = self.dv_ranges[self.dv_names[1]]

        fig = plt.figure()
        plt.rc('axes', titlesize=font_axes)
        plt.rc('axes', labelsize=font_axes)
        plt.rc('xtick', labelsize=font_tick)
        plt.rc('ytick', labelsize=font_tick)
        ax = fig.add_subplot(111)
        params = {'mathtext.default': 'regular'}
        plt.rcParams.update(params)
        ax.set_yticks(range(len(yLabel)))
        ax.set_yticklabels(yLabel)
        ax.set_ylabel(ylabel_text)
        ax.set_xticks(range(len(xLabel)))
        ax.set_xticklabels(xLabel)
        ax.set_xlabel(xlabel_text)
        im = ax.imshow(hes_a.T, cmap=plt.cm.hot_r)
        ba = plt.colorbar(im)
        ba.set_label('log10(trace(FIM))')
        plt.title(title_text + ' - A optimality')
        plt.show()

        fig = plt.figure()
        plt.rc('axes', titlesize=font_axes)
        plt.rc('axes', labelsize=font_axes)
        plt.rc('xtick', labelsize=font_tick)
        plt.rc('ytick', labelsize=font_tick)
        ax = fig.add_subplot(111)
        params = {'mathtext.default': 'regular'}
        plt.rcParams.update(params)
        ax.set_yticks(range(len(yLabel)))
        ax.set_yticklabels(yLabel)
        ax.set_ylabel(ylabel_text)
        ax.set_xticks(range(len(xLabel)))
        ax.set_xticklabels(xLabel)
        ax.set_xlabel(xlabel_text)
        im = ax.imshow(hes_d.T, cmap=plt.cm.hot_r)
        ba = plt.colorbar(im)
        ba.set_label('log10(det(FIM))')
        plt.title(title_text + ' - D optimality')
        plt.show()

        fig = plt.figure()
        plt.rc('axes', titlesize=font_axes)
        plt.rc('axes', labelsize=font_axes)
        plt.rc('xtick', labelsize=font_tick)
        plt.rc('ytick', labelsize=font_tick)
        ax = fig.add_subplot(111)
        params = {'mathtext.default': 'regular'}
        plt.rcParams.update(params)
        ax.set_yticks(range(len(yLabel)))
        ax.set_yticklabels(yLabel)
        ax.set_ylabel(ylabel_text)
        ax.set_xticks(range(len(xLabel)))
        ax.set_xticklabels(xLabel)
        ax.set_xlabel(xlabel_text)
        im = ax.imshow(hes_e.T, cmap=plt.cm.hot_r)
        ba = plt.colorbar(im)
        ba.set_label('log10(minimal eig(FIM))')
        plt.title(title_text + ' - E optimality')
        plt.show()

        fig = plt.figure()
        plt.rc('axes', titlesize=font_axes)
        plt.rc('axes', labelsize=font_axes)
        plt.rc('xtick', labelsize=font_tick)
        plt.rc('ytick', labelsize=font_tick)
        ax = fig.add_subplot(111)
        params = {'mathtext.default': 'regular'}
        plt.rcParams.update(params)
        ax.set_yticks(range(len(yLabel)))
        ax.set_yticklabels(yLabel)
        ax.set_ylabel(ylabel_text)
        ax.set_xticks(range(len(xLabel)))
        ax.set_xticklabels(xLabel)
        ax.set_xlabel(xlabel_text)
        im = ax.imshow(hes_e2.T, cmap=plt.cm.hot_r)
        ba = plt.colorbar(im)
        ba.set_label('log10(cond(FIM))')
        plt.title(title_text + ' - Modified E-optimality')
        plt.show()


def permutations(lst, r=None):
    '''Function for calculating determinant
    It generates unrepeatable permutations reordering elements in the given list
    
    Argument:
        lst: a list for permutation 
        
    Return:
        All permutations of this list
    
    Adapted from https://stackoverflow.com/questions/58924650/finding-determinant-of-a-nxn-matrix-using-leibniz-formula-on-python
    '''
    if len(lst) <= 1:
        return [lst]
    templst = []
    for i in range(len(lst)):
        part = lst[:i] + lst[i+1:]
        for j in permutations(part):
            templst.append(lst[i:i+1] + j)
    return templst
    
def sgn(p):
    '''
    Give the signature of a permutation
    
    Argument:
        p: the permutation (a list)
        
    Return: 
        1 if the number of exchange is an even number 
        -1 if the number is an odd number 
    '''

    if len(p) == 1:
        return True

    trans = 0

    for i in range(0,len(p)):
        j = i + 1

        for j in range(j, len(p)):
            if p[i] > p[j]: 
                trans = trans + 1
                
    if (trans % 2) == 0:
        return 1
    else:
        return -1  
    
def simulate_discretize_model(m,NFE,collo=True,initialize=True):
    ''' Simulation, discretize, and initialize the Pyomo model
    
    Arguments:
        m: Pyomo model
        NFE: number of finite elements to consider (integer)
        initialize: if True, initialize the discretized model with the 
             integrator solution (boolean)
    
    Returns:
        sim: Simulator object from Pyomo.DAE
        tsim: Timesteps returned from simulator
        profiles: Results returned from simulator
    '''
    # Simulate the model using casadi
    sim = Simulator(m, package='casadi')
    tsim, profiles = sim.simulate(integrator='idas', varying_inputs=m.var_input)
    
    if not collo:
        discretizer = TransformationFactory('dae.finite_difference')
        discretizer.apply_to(m, nfe=NFE, scheme='BACKWARD', wrt=m.t)
    else:
        # Discretize model using Orthogonal Collocation
        discretizer = TransformationFactory('dae.collocation')
        #discretizer.apply_to(m, nfe=NFE, scheme='LAGRANGE-RADAU', ncp=3, wrt=m.t)
        discretizer.apply_to(m, nfe=NFE, scheme='LAGRANGE-LEGENDRE', ncp=3, wrt=m.t)

    if initialize:
    # Initialize the discretized model using the simulator profiles
        sim.initialize_model()
    
    return sim, tsim, profiles

    
    
    