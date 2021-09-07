import numpy as np
import matplotlib.pyplot as plt
from pyomo.environ import *
from pyomo.dae import *
import pandas as pd
import time
import pickle
from itertools import permutations, product
from sens import get_dsdp
from pyomo.contrib.sensitivity_toolbox.sens import sipopt, sensitivity_calculation
from idaes.apps.uncertainty_propagation.sens import get_dsdp


class DesignOfExperiments: 
    def __init__(self, param_init, design_variable_timepoints, measurement_variables, measurement_timeset, create_model, solver=None,
                 prior_FIM=None, discretize_model=None, verbose=True):
        '''
        This package enables model-based design of experiments analysis with Pyomo. Both direct optimization and enumeration modes are supported.
        NLP sensitivity tools, e.g.,  sipopt and k_aug, are supported to accelerate analysis via enumeration.
        It can be applied to dynamic models, where design variables are controlled throughout the experiment.

        Parameters:
        -----------
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
        self.param_name = list(param_init.keys())
        self.param_value = list(param_init.values())
        # design variable name
        self.design_timeset = design_variable_timepoints
        self.design_name = list(self.design_timeset.keys())
        # the control time point for each design variable
        self.design_time = list(self.design_timeset.values())
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
        
    def __check_inputs(self, check_mode=False):
        '''Check if inputs are consistent

        Parameters
        ----------
        check_mode: check FIM calculation mode
        '''
        if self.objective_option not in ['det', 'trace', 'zero']:
            raise ValueError('Error: Objective function should be chosen from "det", "zero" and "trace"')

        if self.formula not in ['central', 'forward', 'backward', None]:
            raise ValueError('Error: Finite difference scheme should be chosen from "central", "forward", "backward" and "none".')

        if self.prior_FIM is not None:
            assert (np.shape(self.prior_FIM)[0] == np.shape(self.prior_FIM)[1]), \
                'Expect prior information matrix shape: ['+str(len(self.param_name))+','+str(len(self.param_name)) +']'

        if self.scale_nominal_param_value:
            print('Sensitivity information is scaled by its corresponding parameter nominal value.')

        if (self.scale_constant_value != 1):
            print('Sensitivity information is scaled by constant ', self.scale_constant_value, ' times itself.')

        if check_mode:
            if self.mode not in ['simultaneous_finite', 'sequential_finite', 'sequential_sipopt', 'sequential_kaug', 'direct_kaug']:
                print('Wrong mode. Choose from "simultaneous_finite", "sequential_finite", "0sequential_sipopt", "sequential_kaug"')



    def optimize_doe(self,  design_values, if_optimize=True, objective_option='det',
                     scale_nominal_param_value=False, scale_constant_value=1, if_Cholesky=False, L_LB = 1E-10, L_initial=None,
                     formula='central', step=0.001, check=True):
        '''
        Optimize DOE problem with design variables being the decisions.
        The DOE model is formed invasively and all scenarios are computed simultaneously.
        The function will first fun a square problem with design variable being fixed at
        the given initial points, and then unfix the design variable and do the
        optimization.

        Parameters:
        -----------
        design_values: initial point for optimization, a dict whose keys are design variable names, values are a dict whose keys are time point and values are the design variable value at that time point
        if_optimize: if true, continue to do optimization. else, just run square problem with given design variable values
        objective_option: supporting maximizing the 'det' determinant or the 'trace' trace of the FIM
        scale_nominal_param_value: if scale Jacobian by the corresponding parameter nominal value
        scale_constant_value: how many order of magnitudes the Jacobian value is scaled by. Use when the Jac or FIM value is too small
        if_Cholesky: if true, cholesky decomposition is used for Objective function (to optimize determinant).
            L_LB: if FIM is P.D., the diagonal element should be positive, so we can set a LB like 1E-10
            L_initial: initialize the L
        formula: Finite difference formula, choose from 'central', 'forward', 'backward', None
        step: Finite difference sensitivity perturbation step size, a fraction between [0,1]. default is 0.001
        check: if True check input toggles consistency to be checked multiple times.

        Returns:
        --------
        analysis_square: result summary of the square problem solved at the initial point
        analysis_optimize: result summary of the optimization problem solved

        Steps:
        ------
        1. Build two-stage stochastic programming optimization model where scenarios correspond to
        finite difference approximations for the Jacobian of the response variables with respect to calibrated model parameters
        2. Fix the experiment design decisions and solve a square (i.e., zero degrees of freedom) instance of the two-stage DOE problem.
        This step is for initialization.
        3. Unfix the experiment design decisions and solve the two-stage DOE problem.
        '''
        time0 = time.time()

        # store inputs in object
        self.design_values = design_values
        self.optimize = if_optimize
        self.objective_option = objective_option
        self.scale_nominal_param_value = scale_nominal_param_value
        self.scale_constant_value = scale_constant_value
        self.Cholesky_option = if_Cholesky
        self.L_LB = L_LB
        self.L_initial = L_initial
        self.formula = formula
        self.step = step
        self.tee_opt = True

        # calculate how much the FIM element is scaled by a constant number
        # FIM = Jacobian.T@Jacobian, the FIM is scaled by squared value the Jacobian is scaled
        self.fim_scale_constant_value = self.scale_constant_value **2

        # check if inputs are valid
        # simultaneous mode does not need to check mode and dimension of design variables
        if check:
            self.__check_inputs(check_mode=False)

        # build the large DOE pyomo model
        m = self.__create_doe_model()

        # solve model, achieve results for square problem, and results for optimization problem

        # Solve square problem first
        # result_square: solver result
        time0_solve = time.time()
        result_square = self.__solve_doe(m, fix=True)
        time1_solve = time.time()

        time_solve1 = time1_solve-time0_solve

        analysis_square = FIM_result(self.param_name, prior_FIM=self.prior_FIM, scale_constant_value=self.scale_constant_value)
        analysis_square.extract_FIM(m, self.design_timeset, result_square, obj=objective_option)


        if self.optimize:
            # solve problem with DOF then
            time0_solve2 = time.time()
            result_doe = self.__solve_doe(m, fix=False)
            time1_solve2 = time.time()

            time_solve2 = time1_solve2 - time0_solve2

            analysis_optimize = FIM_result(self.param_name, prior_FIM=self.prior_FIM)
            analysis_optimize.extract_FIM(m, self.design_timeset, result_doe, obj=objective_option)
            analysis_optimize.model = m


            time1 = time.time()
            if self.verbose:
                print('Total solve time with simultaneous_finite mode (Wall clock) [s]:', time_solve1 + time_solve2)
                print('Total wall clock time [s]:', time1-time0)

            return analysis_square, analysis_optimize

        else:
            analysis_optimize.model = m

            time1 = time.time()
            if self.verbose:
                print('Total solve time with simultaneous_finite mode (Wall clock) [s]:', time_solve1)
                print('Total wall clock time [s]:', time1 - time0)

            return analysis_square


    def compute_FIM(self, design_values, mode='sequential_finite', FIM_store_name=None, specified_prior=None,
                    tee_opt=True, scale_nominal_param_value=False, scale_constant_value=1, formula='central', step=0.01,
                    if_Cholesky=False, L_LB=1E-10, L_initial=None):
        '''
        This function solves a square Pyomo model with fixed design variables to compute the FIM.
        The problem is structured in one of the four following modes:
        1. simultaneous_finite: Calculate a multiple scenario model. Sensitivity info estimated by finite difference. Instead of 1,
        2. sequential_finite: Calculates a one scenario model multiple times for
        multiple scenarios. Sensitivity info estimated by finite difference
        3. sequential_sipopt: calculate sensitivity by sIPOPT.
        4. sequential_kaug: calculate sensitivity by k_aug
        5, direct_kaug: calculate sensitivity by k_aug with direct sensitivity. **In construction**

        Parameters:
        -----------
        design_values: a dict whose keys are design variable names, values are a dict whose keys are time point and values are the design variable value at that time point
        mode: use mode='sequential_finite', 'simultaneous_finite', 'sequential_sipopt', 'sequential_kaug'
        FIM_store_name: if storing the FIM in a .csv, give the file name here as a string, '**.csv' or '**.txt'.
        specified_prior: if user needs a different prior, replace this toggle without creating a new object
        tee_opt: if IPOPT console output is printed
        scale_nominal_param_value: if True, the parameters are scaled by its own nominal value in param_init
        scale_constant_value: how many order of magnitudes the Jacobian value is scaled by. Use when the Jac or FIM value is too small

        Only effective when finite=True:
        formula: choose from 'central', 'forward', 'backward', None
        step: Sensitivity perturbation step size, a fraction between [0,1]. default is 0.001

        Cholesky option:
        if_Cholesky: if true, Cholesky decomposition is used for Objective function (to optimize determinant).
        L_LB: if FIM is positive definite, the diagonal element should be positive, so we can set a LB like 1E-10
        L_initial: initialize the L

        Return:
        -------
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
        # Set the Objective Function to 0 helps solve square problem quickly
        self.objective_option = 'zero'
        self.tee_opt = tee_opt

        self.Cholesky_option = if_Cholesky
        self.L_LB = L_LB
        self.L_initial = L_initial

        # calculate how much the FIM element is scaled by a constant number
        # As FIM~Jacobian.T@Jacobian, FIM is scaled twice the number the Q is scaled
        self.fim_scale_constant_value = self.scale_constant_value ** 2

        # check inputs valid
        self.__check_inputs(check_mode=True)

        # if using simultaneous model
        if (self.mode == 'simultaneous_finite'):
            time0_build = time.time()
            m = self.__create_doe_model()
            time1_build = time.time()

            time0_solve = time.time()
            # solve model, achieve results for square problem, and results for optimization problem
            square_result = self.__solve_doe(m, fix=True)
            time1_solve = time.time()

            time_build = time1_build - time0_build
            time_solve = time1_solve - time0_solve
            if self.verbose:
                print('Build time with simultaneous_finite mode [s]:', time_build)
                print('Solve time with simultaneous_finite mode [s]:', time_solve)
            
            # analyze results
            if specified_prior is None:
                prior_in_use = self.prior_FIM
            else:
                prior_in_use = specified_prior

            FIM_analysis = FIM_result(self.param_name, prior_FIM=prior_in_use, store_FIM=FIM_store_name, scale_constant_value=self.scale_constant_value)

            # add the formed simultaneous model to the object so that users can have access
            self.m = m
            self.square_result = square_result
            FIM_analysis.build_time = time_build
            FIM_analysis.solve_time = time_solve

            return FIM_analysis

        elif self.mode=='sequential_finite':
            no_para = len(self.param_name)

            # if using sequential model
            # call generator function to get scenario dictionary
            scena_gen = Scenario_generator(self.param_init, formula=self.formula, step=self.step)
            scena_gen.generate_sequential_para()

            # dict for storing model outputs
            output_record = {}
            # dict for storing Jacobian
            jac = {}

            time_allbuild = []
            time_allsolve = []
            # loop over each scenario
            for no_s in (scena_gen.scena_keys):

                scenario_iter = scena_gen.next_sequential_scenario(no_s)
                # create the model
                # TODO:(long term) add options to create model once and then update. only try this after the
                # package is completed and unitest is finished
                time0_build = time.time()
                mod = self.create_model(scenario_iter)
                time1_build = time.time()
                time_allbuild.append(time1_build-time0_build)

                # discretize if needed
                if self.discretize_model is not None:
                    mod = self.discretize_model(mod)

                # extract (discretized) time
                time_set = []
                for t in mod.t:
                    time_set.append(value(t))

                # solve model
                time0_solve = time.time()
                square_result = self.__solve_doe(mod, fix=True)
                time1_solve = time.time()
                time_allsolve.append(time1_solve-time0_solve)

                # loop over measurement item and time to store model measurements
                output_combine = []
                for j in self.measurement_variables:
                    for t in self.measurement_timeset:
                        C_value = eval('mod.' + j + '[0,' + str(t) + ']')
                        output_combine.append(value(C_value))
                output_record[no_s] = output_combine


            # After collecting outputs from all scenarios, calculate sensitivity
            for para in self.param_name:
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

            if self.verbose:
                print('Build time with sequential_finite mode [s]:', sum(time_allbuild))
                print('Solve time with sequential_finite mode [s]:', sum(time_allsolve))

            # Assemble and analyze results
            if specified_prior is None:
                prior_in_use = self.prior_FIM
            else:
                prior_in_use = specified_prior

            FIM_analysis = FIM_result(self.param_name, prior_FIM=prior_in_use, store_FIM=FIM_store_name, scale_constant_value=self.scale_constant_value)

            # Store the Jacobian information for access by users
            self.jac = jac
            FIM_analysis.build_time = sum(time_allbuild)
            FIM_analysis.solve_time = sum(time_allsolve)

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

            # time building time and solving time store list
            time_allbuild = []
            time_allsolve = []
            # loop over parameters
            for pa in range(len(self.param_name)):
                perturb_mea = []
                base_mea = []

                # create model
                time0_build = time.time()
                mod = self.create_model(scenario_all)
                time1_build = time.time()
                time_allbuild.append(time1_build - time0_build)

                # discretize if needed
                if self.discretize_model is not None:
                    mod = self.discretize_model(mod)

                # For sIPOPT, fix model DOF
                if self.mode =='sequential_sipopt':
                    mod = self.__fix_design(mod, self.design_values, fix_opt=True)

                mod.obj = Objective(rule=0.0, sense=minimize)

                # extract (discretized) time
                time_set = []
                for t in mod.t:
                    time_set.append(value(t))

                # add sIPOPT perturbation parameters
                mod = self.__add_parameter(mod, perturb=pa)

                # solve the square problem with the original parameters for k_aug mode, since k_aug does not calculate these
                if self.mode == 'sequential_kaug':
                    self.__solve_doe(mod, fix=True)

                # parameter name lists for sipopt
                list_original = []
                list_perturb = []
                for ele in self.param_name:
                    list_original.append(eval('mod.'+ele+'[0]'))
                for elem in self.perturb_names:
                    list_perturb.append(eval('mod.'+elem+'[0]'))

                # solve model
                if self.mode =='sequential_sipopt':
                    time0_solve = time.time()
                    m_sipopt = sensitivity_calculation('sipopt', mod, list_original, list_perturb, tee=self.tee_opt, solver_options='ma57')
                else:
                    time0_solve = time.time()
                    m_sipopt = sensitivity_calculation('k_aug', mod, list_original, list_perturb, tee=self.tee_opt, solver_options='ma57')

                time1_solve = time.time()
                time_allsolve.append(time1_solve - time0_solve)

                # extract sipopt result
                for j in self.measurement_variables:
                    for t in self.measurement_timeset:
                        # fetch the measurement variable
                        measure_var = getattr(m_sipopt,j)
                        # check if this variable is fixed
                        if (measure_var[0,t].fixed == True):
                            perturb_value = value(measure_var[0,t])

                        else:
                            # if it is not fixed, record its perturbed value
                            if self.mode =='sequential_sipopt':
                                perturb_value = eval('m_sipopt.sens_sol_state_1[m_sipopt.' + j + '[0,' + str(t) + ']]')
                            else:
                                perturb_value = eval('m_sipopt.' + j + '[0,' + str(t) + ']()')
                        perturb_mea.append(perturb_value)

                        # base case values
                        if self.mode =='sequential_sipopt':
                            base_value = eval('m_sipopt.'+j+'[0,' + str(t) + '].value')
                        else:
                            base_value = value(eval('mod.' + j + '[0,' + str(t) + ']'))

                        base_mea.append(base_value)

                # store extracted measurements
                all_perturb_measure.append(perturb_mea)
                all_base_measure.append(base_mea)
                print(all_perturb_measure)
                print(all_base_measure)

            # After collecting outputs from all scenarios, calculate sensitivity
            for count, para in enumerate(self.param_name):
                list_jac = []
                for i in range(len(all_perturb_measure[0])):
                    if self.scale_nominal_param_value:
                        sensi = -(all_perturb_measure[count][i] - all_base_measure[count][i]) / self.step * self.scale_constant_value
                    else:
                        sensi = -(all_perturb_measure[count][i] - all_base_measure[count][i]) / self.step /self.param_init[para] * self.scale_constant_value
                    list_jac.append(sensi)
                # get Jacobian dict, keys are parameter name, values are sensitivity info
                jac[para] = list_jac

            # check if another prior experiment FIM is provided other than the user-specified one
            if specified_prior is None:
                prior_in_use = self.prior_FIM
            else:
                prior_in_use = specified_prior

            # Assemble and analyze results
            FIM_analysis = FIM_result(self.param_name, prior_FIM=prior_in_use, store_FIM=FIM_store_name, scale_constant_value=self.scale_constant_value)

            if self.verbose:
                print('Build time with sequential_sipopt or kaug mode [s]:', sum(time_allbuild))
                print('Solve time with sequential_sipopt or kaug mode [s]:', sum(time_allsolve))

            self.jac = jac
            FIM_analysis.build_time = sum(time_allbuild)
            FIM_analysis.solve_time = sum(time_allsolve)

            return FIM_analysis

        elif self.mode =='direct_kaug':
            print('===In construction===')

            # create scenario class for a base case
            scena_gen = Scenario_generator(self.param_init, formula=None, step=self.step)
            scenario_all = scena_gen.simultaneous_scenario()

            # create model
            time0_build = time.time()
            mod = self.create_model(scenario_all)
            time1_build = time.time()
            time_build = time1_build - time0_build

            # discretize if needed
            if self.discretize_model is not None:
                mod = self.discretize_model(mod)

            # get all time
            t_all = []
            for t in mod.t:
                t_all.append(t)

            measurement_accurate_time = []
            # check if measurement time points are in this time set
            for tt in self.measurement_timeset:
                if tt not in t_all:
                    print('A measurement time point not measured by this model: ', tt)
                # For e.g. if a measurement time point is 0.0 in the model but is given as 0, it is corrected here.
                measurement_accurate_time.append(t_all[t_all.index(tt)])

            # fix model DOF
            mod = self.__fix_design(mod, self.design_values, fix_opt=True)

            # set ub and lb to parameters
            for par in self.param_name:
                component = eval('mod.'+par+'[0]')
                component.setlb(self.param_init[par])
                component.setub(self.param_init[par])

            # generate parameter name list and value dictionary with index
            var_name = []
            var_dict = {}

            for name in self.param_name:
                var_name.append(name+'[0]')
                var_dict[name+'[0]'] = self.param_init[name]

            time0_solve = time.time()
            dsdp_re, col = get_dsdp(mod, var_name, var_dict, tee=self.tee_opt)
            time1_solve = time.time()
            time_solve = time1_solve - time0_solve
            print(col)

            # analyze result
            dsdp_array = dsdp_re.toarray().T
            # here for construction. Remove after finishing.
            dd = pd.DataFrame(dsdp_array)
            dd.to_csv('dsdp_test.csv')
            # store dsdp returned
            dsdp_extract = []
            # get right lines from results
            measurement_index = []
            measurement_names = []
            for mname in self.measurement_variables:
                for tim in measurement_accurate_time:
                    measure_name = mname+'[0,'+str(tim)+']'
                    measurement_names.append(measure_name)
                    # get right line number in kaug results
                    kaug_no = col.index(measure_name)
                    measurement_index.append(kaug_no)
                    # get right line of dsdp
                    dsdp_extract.append(dsdp_array[kaug_no])

            if self.verbose:
                print('Build time with direct kaug mode [s]:', time_build)
                print('Solve time with direct kaug mode [s]:', time_solve)

            #FIM_analysis.build_time = time_build
            #FIM_analysis.solve_time = time_solve

            return dsdp_extract


        else:
            raise ValueError('This is not a valid mode. Choose from "sequential_finite", "simultaneous_finite", "sequential_sipopt", "sequential_kaug"')


    def generate_sequential_experiments(self, design_values_set, mode='sequential_finite', tee_option=False,
                       scale_nominal_param_value=False, scale_constant_value=1,
                       formula='central', step=0.001):
        '''
        Run a series of experiments sequentially, and use the FIM from one experiment as the prior information for the next experiment
        Parameters:
        -----------
        design_values_set: a list of experiments, each element is one design_values dictionary
        mode: use mode='sequential_finite', 'simultaneous_finite', 'sequential_sipopt', 'sequential_kaug'
        tee_option: if IPOPT console output is printed
        scale_nominal_param_value: if True, the parameters are scaled by its own nominal value in param_init
        scale_constant_value: how many order of magnitudes the Jacobian value is scaled by. Use when the Jac or FIM value is too small
        formula: choose from 'central', 'forward', 'backward', None
        step: Sensitivity perturbation step size, a fraction between [0,1]. default is 0.001

        Returns:
        --------
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
        # Set the Objective Function to 0 helps solve square problem quickly
        self.objective_option = 'zero'

        # calculate how much the FIM element is scaled by a constant number
        self.fim_scale_constant_value = self.scale_constant_value ** 2

        self.__check_inputs(check_mode=True)

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
                result_iter.extract_FIM(self.m, self.design_timeset, self.square_result, self.objective_option, add_fim=True)

            elif (self.mode in ['sequential_finite', 'sequential_sipopt', 'sequential_kaug']):
                result_iter.calculate_FIM(self.jac, self.design_values)

            # attach these results to the store list
            result_object_list.append(result_iter)
            fim_list.append(result_iter.FIM)

        return result_object_list, fim_list

    def run_grid_search(self, design_values, design_ranges, design_dimension_names, design_control_time, mode='sequential_finite',
                        tee_option=False, scale_nominal_param_value=False, scale_constant_value=1,
                        filename=None, formula='central', step=0.001):
        '''
        Enumerate through full grid search for any number of design variables;
        solve square problems sequentially to compute FIMs.
        It calculates FIM with sensitivity information from four ways:
        1. Simultaneous: Calculate a multiple scenario model. Sensitivity info estimated by finite difference
        2. Sequential_ipopt: Calculates a one scenario model multiple times for
        multiple scenarios. Sensitivity info estimated by finite difference
        3. Sequential_sipopt: calculate sensitivity by sIPOPT.
        4. Sequential_kaug: calculate sensitivity by k_aug

        Parameters:
        -----------
        design_values: a dict whose keys are design variable names, values are a dict whose keys are time point and values are the design variable value at that time point
        design_ranges: a list of design variable values to go over
        design_dimension_names: a list of design variable names of each design range
        deisgn_control_time: a list of control time points that should be fixed to the values in dv_ranges
        mode: use mode='sequential_finite', 'simultaneous_finite', 'sequential_sipopt', 'sequential_kaug'
        tee_option: if IPOPT console output is made
        scale_nominal_param_value: if True, the parameters are scaled by its own nominal value in param_init
        scale_constant_value: how many order of magnitudes the Jacobian value is scaled by. Use when the Jac or FIM value is too small
        filename: if True, grid search results stored with this file name

        Only effective when finite=True:
        formula: choose from 'central', 'forward', 'backward', None
        step: Sensitivity perturbation step size, a fraction between [0,1]. default is 0.001

        Return:
        -------
        figure_draw_object: a combined result object of class Grid_search_result
        '''

        # time 0
        t_enumeration_begin = time.time()

        # Set the Objective Function to 0 helps solve square problem quickly
        self.objective_option='zero'
        self.filename = filename

        # calculate how much the FIM element is scaled
        self.fim_scale_constant_value = scale_constant_value ** 2

        # when defining design space, design variable values are defined as in design_values argument
        # the design var value defined in dv_ranges only applies to control time points given in dv_apply_time
        grid_dimension = len(design_ranges)

        # to store all FIM results
        result_combine = {}



        # iteration 0
        count = 0
        # how many sets of design variables will be run
        total_count = 1
        for i in range(grid_dimension):
            total_count *= len(design_ranges[i])
        print(total_count, ' design vectors will be searched.')

        # generate combinations of design variable values to go over
        search_design_set = product(*design_ranges)

        build_time_store=[]
        solve_time_store=[]

        # loop over deign value combinations
        for design_set_iter in search_design_set:
            # generate the design variable dictionary needed for running compute_FIM
            # first copy value from design_Values
            design_iter = design_values.copy()

            # update the controlled value of certain time points for certain design variables
            for i in range(grid_dimension):
                for v, value in enumerate(design_control_time[i]):
                    design_iter[design_dimension_names[i]][value] = list(design_set_iter)[i]

            print('=======This is the ', count+1, 'th iteration=======')
            print('Design variable values of this iteration:', design_iter)

            # call compute_FIM to get FIM
            result_iter = self.compute_FIM(design_iter, mode=mode,
                                           tee_opt=tee_option,
                                           scale_nominal_param_value=scale_nominal_param_value,
                                           scale_constant_value = scale_constant_value,
                                           formula=formula, step=step)
            build_time_store.append(result_iter.build_time)
            solve_time_store.append(result_iter.solve_time)

            if (mode=='simultaneous_finite'):
                result_iter.extract_FIM(self.m, self.design_timeset, self.square_result, self.objective_option)

            elif (mode in ['sequential_finite', 'sequential_sipopt', 'sequential_kaug']):
                result_iter.calculate_FIM(self.jac, self.design_values)

            t_now = time.time()

            if self.verbose:
                # give run information at each iteration
                print('This is the ', count+1, ' run out of ', total_count, 'run.')
                print('The code has run %.04f seconds.'% (t_now-t_enumeration_begin))
                print('Estimated remaining time: %.4f seconds' % ((t_now-t_enumeration_begin)/(count+1)*(total_count-count-1)))
            count += 1

            # the combined result object are organized as a dictionary, keys are a tuple of the design variable values, values are a result object
            result_combine[tuple(design_set_iter)] = result_iter

        # For user's access
        self.all_fim = result_combine

        # Create figure drawing object
        figure_draw_object = Grid_Search_Result(design_ranges, design_dimension_names, design_control_time, result_combine, store_optimality_name=filename)

        # store results
        #if self.filename is not None:
        #    f = open(filename, 'wb')
        #    pickle.dump(result_combine, f)
        #    f.close()

        t_enumeration_stop = time.time()
        if self.verbose:
            print('Overall model building time [s]:', sum(build_time_store))
            print('Overall model solve time [s]:', sum(solve_time_store))
            print('Overall wall clock time [s]:', t_enumeration_stop - t_enumeration_begin)

        return figure_draw_object


    def __create_doe_model(self):
        '''
        Add features for DOE.

        Parameters:
        -----------
        self.measurement_variables: the variable name of the model output, for e.g., ['CA', 'CB', 'CC'].
        self.measurement_timeset: a list of measurement time points. can be different from control time points
        self.design_values: a dict of dictionaries, keys are the name of design variables, values are a dict where keys are the time points, values are the design variable value at that time point

        self.optimize: if True, solve the problem unfixing the design variables. if False, solve the problem as a
        square problem
        self.objective_option: choose from 'det' or 'trace'. Optimization problem maximizes determinant or trace.
        self.scale_nominal_param_value: if True, scale FIM but not scale Jacobian. This toggle can be opened for better performance when the
        problem is poorly scaled.
        self.tee_opt: if True, print IPOPT console output
        self.Cholesky_option: if true, cholesky decomposition is used for Objective function (to optimize determinant). If true, determinant will not be calculated.
            self.L_LB: if FIM is P.D., the diagonal element should be positive, so we can set a LB like 1E-10
            self.L_initial: initialize the L
        self.formula: choose from 'central', 'forward', 'backward', None
        self.step: Sensitivity perturbation step size, a fraction between [0,1]. default is 0.001

        Return:
        -------
        m: the DOE model
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
        m.para_set = Set(initialize=self.param_name)
        param_name = self.param_name
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
        for d in range(len(self.design_name)):
            for t in self.design_time[d]:
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

        if self.objective_option=='trace':
            # Trace of FIM
            m.trace = Var(initialize=1, within=NonNegativeReals)
        elif self.objective_option=='det':
            # Determinant of FIM
            m.det = Var(initialize=0.5, within=NonNegativeReals)
        elif (self.objective_option != 'zero'):
            raise ValueError('Undefined objective function type. Available options are "trace" and "det".')

        # move the L matrix initial point to a dictionary
        if self.L_initial is not None:
            dict_cho={}
            for i, bu in enumerate(m.para_set):
                for j, un in enumerate(m.para_set):
                    dict_cho[(bu,un)] = self.L_initial[i][j]
        # use the L dictionary to initialize L matrix
        def init_cho(m,i,j):
            return dict_cho[(i,j)]

        if self.Cholesky_option:
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
            Calculate FIM elements. Can scale each element with 1000 for performance
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
            object_p = itertools.permutations(r_list)
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
            det_perm = sum( self.__sgn(list_p[d])*sum(m.FIM[each, name_order[b]] for b, each in enumerate(m.para_set)) for d in range(len(list_p)))
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


        ### Constraints and Objective function
        m.dC_value = Constraint(m.y_set, m.para_set, m.tmea_set, rule=jac_numerical)
        m.ele_rule = Constraint(m.para_set, m.para_set, rule=calc_FIM)  

        # Only giving the objective function when there's Degree of freedom. Make OBJ=0 when it's a square problem, which helps converge.
        if self.optimize:
            # if cholesky, calculating L and evaluate the OBJ with Cholesky decomposition
            if self.Cholesky_option:
                m.cholesky_cons = Constraint(m.para_set, m.para_set, rule=cholesky_imp)
                m.obj = Objective(expr=2*sum(log(m.L_ele[j,j]) for j in m.para_set), sense=maximize)
            # if not cholesky but determinant, calculating det and evaluate the OBJ with det 
            elif (self.objective_option=='det'):
                m.det_rule =  Constraint(rule=det_general)
                m.obj = Objective(expr=log(m.det), sense=maximize)
            # if not determinant or cholesky, calculating the OBJ with trace
            elif (self.objective_option=='trace'):
                m.trace_rule = Constraint(rule=trace_calc)
                m.obj = Objective(expr=log(m.trace), sense=maximize)
            elif (self.objective_option=='zero'):
                m.obj = Objective(expr=0)
        else:
            m.obj = Objective(expr=0)

        return m


    def __fix_design(self, m, design_val, fix_opt=True):
        ''' Fix design variable

        Parameters:
        -----------
        m: model
        design_val: design variable values dict
        fix_opt: if True, fix. Else, unfix

        Returns:
        --------
        m: model
        '''
        # loop over the design variables and time index and to fix values specified in design_val
        for d, dname in enumerate(self.design_name):
            for t, time in enumerate(self.design_time[d]):
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

        Parameters:
        -----------
        m:model
        fix: if true, solve two times (square first). Else, just solve the square problem

        Return:
        -------
        solver_results: solver results
        '''
        ### Solve square problem
        mod = self.__fix_design(m, self.design_values, fix_opt=fix)

        # if user gives solver, use this solver. if not, use default IPOPT solver
        solver_result = self.solver.solve(mod,tee=self.tee_opt)

        return solver_result

    def __add_parameter(self, m, perturb=0):
        '''
        For sIPOPT: add parameter perturbation set

        Parameters:
        -----------
        m: model name
        self.param_names: perturbation parameter names
        perturb: which parameter to perturb
        '''
        # model parameters perturbation, backward disturb
        param_backward = self.param_value.copy()
        # perturb parameter
        param_backward[perturb] *= (1-self.step)

        # generate sIPOPT perturbed parameter names
        param_perturb_names = self.param_name.copy()
        for x, xname in enumerate(self.param_name):
            param_perturb_names[x] = xname+'_pert'
        if self.verbose:
            print('perturb names are:', param_perturb_names)

        self.perturb_names = param_perturb_names
        if self.verbose:
            print('Perturbation parameters are set:')
        for change in range(len(self.perturb_names)):
            setattr(m, self.perturb_names[change], Param(m.scena, initialize=param_backward[change]))
            if self.verbose:
                print(self.perturb_names[change], ': ', value(eval('m.'+self.perturb_names[change]+'[0]')))
        return m

    def __sgn(self,p):
        '''
        Give the signature of a permutation

        Parameters:
        -----------
        p: the permutation (a list)

        Return:
        ------
        1 if the number of exchange is an even number
        -1 if the number is an odd number
        '''

        if len(p) == 1:
            return True

        trans = 0

        for i in range(0, len(p)):
            j = i + 1

            for j in range(j, len(p)):
                if p[i] > p[j]:
                    trans = trans + 1

        if (trans % 2) == 0:
            return 1
        else:
            return -1


class Scenario_generator:
    def __init__(self, para_dict, formula='central', step=0.001, store=False):
        '''Generate scenarios.
        DoE library first calls this function to generate scenarios.
        For sequential and simultaneous models, call different functions in this class.

        Parameters:
        -----------
        para_dict: a Dict of parameter, keys are names, values are their nominal value. for e.g.,
                        {'A1': 84.79, 'A2': 371.72, 'E1': 7.78, 'E2': 15.05}
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

        Parameters:
        ----------
        count: the No. of the sequential models

        Returns:
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

        Returns (added to self object):
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

        Returns: (store in self object)
        --------
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

        Returns:
        --------
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

        Parameters:
        -----------
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

        Parameters:
        -----------
        jaco_info: jacobian dictionary
        dv_values: design variable value dictionary
        sq_result: solver status returned by IPOPT

        Return:
        ------
        fim_info: a FIM dictionary containing the following key:value pairs
            ~['FIM']: a list containing FIM itself
            ~[design variable name]: a list of design variable values at each time point
            ~['Trace']: a scalar number of Trace
            ~['Determinant']: a scalar number of determinant
            ~['Condition number:']: a scalar number of condition number
            ~['Minimal eigen value:']: a scalar number of minimal eigen value
            ~['Eigen values:']: a list of all eigen values
            ~['Eigen vectors:']: a list of all eigen vectors
        solver_info: a solver infomation dictionary containing the following key:value pairs
            ~['square']: a string of square result solver status
        '''
        self.result = result
        self.doe_result = None
        # create a dict for FIM. It has the same keys as the Jacobian dict.

        # get number of parameters
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
            print('The condition number is:', np.linalg.cond(fim), ';')
            print('A condition number bigger than ', self.max_condition_number, ' is considered near singular.')

        # call private methods
        self.__print_FIM_info(fim, dv_set=dv_values)
        if self.result is not None:
            self.__get_solver_info()

        # if given store file name, store the FIM
        if (self.store_FIM is not None):
            self.__store_FIM()

    def extract_FIM(self, m, dv_set, result, obj=None, add_fim=False):
        '''
        Extract FIM from an invasive model

        Parameters:
        -----------
        m: model
        dv_set: design variable value dictionary
        result: problem solver status by IPOPT
        obj: chosen from 'det' and 'trace'
        add_fim: if the given FIM needs to be added. Do not add for optimize_doe().

        Return:
        ------
        fim_info: a FIM dictionary containing the following key:value pairs
            ~['FIM']: a list of FIM itself
            ~[design variable name]: a list of design variable values at each time point
            ~['Trace']: a scalar number of Trace
            ~['Determinant']: a scalar number of determinant
            ~['Condition number:']: a scalar number of condition number
            ~['Minimal eigen value:']: a scalar number of minimal eigen value
            ~['Eigen values:']: a list of all eigen values
            ~['Eigen vectors:']: a list of all eigen vectors
        model_info: model solutions dictionary containing the following key:value pairs
            ~['obj']: a scalar number of objective function value
            ~['det']: a scalar number of determinant calculated by the model (different from FIM_info['det'] which
            is calculated by numpy)
            -['trace']: a scalar number of trace calculated by the model
            -[design variable name]: a list of design variable solution
        solver_status: a solver infomation dictionary containing the following key:value pairs
            ~['square']: a string of square result solver status
            -['doe']: a string of doe result solver status
        '''
        self.result = result
        self.obj = obj
        no_para = len(self.para_name)

        # Extract FIM infomation
        FIM = np.ones((no_para, no_para))
        # loop over row
        for n1, name1 in enumerate(self.para_name):
            # loop over column
            for n2, name2 in enumerate(self.para_name):
                FIM[n1, n2] = value(m.FIM[name1, name2]) / self.fim_scale_constant_value

        # add prior information
        if add_fim:
            if (self.prior_FIM is not None):
                try:
                    FIM = FIM + self.prior_FIM
                    print('FIM prior has been added.')
                except:
                    raise ValueError('Prior FIM has shape ', np.shape(self.prior_FIM), ', but expecting shape ', np.shape(FIM))

        # call private methods
        self.__print_FIM_info(FIM, dv_set=dv_set)
        self.__solution_info(m, dv_set)
        self.__get_solver_info()

        # if given store file name, store the FIM
        if (self.store_FIM is not None):
            self.__store_FIM()

    def __print_FIM_info(self, FIM, dv_set=None):
        '''
        using a dictionary to store all FIM information

        Parameters:
        -----------
        FIM: the Fisher Information Matrix, needs to be P.D. and symmetric
        dv_set: design variable dictionary

        Return:
        ------
        fim_info: a FIM dictionary containing the following key:value pairs
            ~['FIM']: a list of FIM itself
            ~[design variable name]: a list of design variable values at each time point
            ~['Trace']: a scalar number of Trace
            ~['Determinant']: a scalar number of determinant
            ~['Condition number:']: a scalar number of condition number
            ~['Minimal eigen value:']: a scalar number of minimal eigen value
            ~['Eigen values:']: a list of all eigen values
            ~['Eigen vectors:']: a list of all eigen vectors
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

        Parameters:
        -----------
        m: model
        dv_set: design variable dictionary

        Return:
        ------
        model_info: model solutions dictionary containing the following key:value pairs
            ~['obj']: a scalar number of objective function value
            ~['det']: a scalar number of determinant calculated by the model (different from FIM_info['det'] which
            is calculated by numpy)
            -['trace']: a scalar number of trace calculated by the model
            -[design variable name]: a list of design variable solution
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

    def __get_solver_info(self):
        '''
        Solver information dictionary

        Return:
        ------
        solver_status: a solver infomation dictionary containing the following key:value pairs
            ~['square']: a string of square result solver status
            -['doe']: a string of doe result solver status
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
    def __init__(self, design_ranges, design_dimension_names, design_control_time, FIM_result_list, store_optimality_name=None, verbose=True):
        '''
        This class deals with the FIM results from grid search, providing A, D, E, ME-criteria results for each design variable.
        Can choose to draw 1D sensitivity curves and 2D heatmaps.

        Parameters:
        -----------
        design_ranges: a dict whose keys are design variable names, values are a list of design variable values to go over
        design_dimension_names: a list of design variables names
        design_control_time: a list of design control timesets
        FIM_result_list: a dictionary containing FIM results, keys are a tuple of design variable values, values are FIM result objects
        store_optimality_name: a csv file name containing all four optimalities value
        verbose: if print statements
        '''
        # design variables
        self.design_names = design_dimension_names
        self.design_ranges = design_ranges
        self.design_control_time = design_control_time
        self.FIM_result_list = FIM_result_list

        self.store_optimality_name = store_optimality_name
        self.verbose = verbose

    def extract_criteria(self):
        '''
        Extract design criteria values for every 'grid' (design variable combination) searched.

        Returns:
        -------
        self.store_all_results_dataframe: a pandas dataframe with columns as design variable names and A, D, E, ME-criteria names.
            Each row contains the design variable value for this 'grid', and the 4 design criteria value for this 'grid'.
        '''

        # a list store all results
        store_all_results = []

        # generate combinations of design variable values to go over
        search_design_set = product(*self.design_ranges)

        # loop over deign value combinations
        for design_set_iter in search_design_set:
            if self.verbose:
                print('Design variable: ', self.design_names)
                print('Value          : ', design_set_iter)

            # locate this grid in the dictionary of combined results
            result_object_asdict = {k:v for k,v in self.FIM_result_list.items() if k==design_set_iter}
            # an result object is identified by a tuple of the design variable value it uses
            result_object_iter = result_object_asdict[design_set_iter]

            # store results as a row in the dataframe
            store_iteration_result = list(design_set_iter)
            store_iteration_result.append(result_object_iter.trace)
            store_iteration_result.append(result_object_iter.det)
            store_iteration_result.append(result_object_iter.min_eig)
            store_iteration_result.append(result_object_iter.cond)

            # add this row to the dataframe
            store_all_results.append(store_iteration_result)

        # generate column names for the dataframe
        column_names = []
        # this count is for repeated design variable names which can happen in dynamic problems
        count = 0
        for i in self.design_names:
            # if a name is in the design variable name list more than once, name them as name_itself, name_itself2, ...
            # this is because it can be errornous when we extract values from a dataframe with two columns having the same name
            if i in column_names:
                count += 1
                i_original = i
                i = i+str(count+1)
                print('Reminder: the ', count+1, 'th design variable ', i_original, ' is renamed as ', i, '.')
            column_names.append(i)

        # Each design criteria has a column to store values
        column_names.append('A')
        column_names.append('D')
        column_names.append('E')
        column_names.append('ME')
        # generate the dataframe
        self.store_all_results_dataframe = pd.DataFrame(store_all_results, columns=column_names)
        # if needs to store the values
        if self.store_optimality_name is not None:
            self.store_all_results_dataframe.to_csv(self.store_optimality_name, index=False)


    def figure_drawing(self, fixed_design_dimensions, sensitivity_dimension, title_text, xlabel_text, ylabel_text, font_axes=16, font_tick=14, log_scale=True):
        '''
        Extract results needed for drawing figures from the overall result dataframe.
        Draw 1D sensitivity curve or 2D heatmap.
        It can be applied to results of any dimensions, but requires design variable values in other dimensions be fixed.

        Parameters:
        ----------
        fixed_design_dimensions: a dictionary, keys are the design variable names to be fixed, values are the value of it to be fixed.
        sensitivity_dimension: a list of design variable names to draw figures.
            If only one name is given, a 1D sensitivity curve is drawn
            if two names are given, a 2D heatmap is drawn.
        title_text: name of the figure, a string
        xlabel_text: x label title, a string.
            In a 1D sensitivity curve, it is the design variable by which the curve is drawn.
            In a 2D heatmap, it should be the second design varialbe in the design_ranges
        ylabel_text: y label title, a string.
            A 1D sensitivity cuve does not need it. In a 2D heatmap, it should be the first design variable in the dv_ranges
        font_axes: axes label font size
        font_tick: tick label font size
        log_scale: if True, the result matrix will be scaled by log10

        Returns:
        --------
        None
        '''
        self.fixed_design_names = list(fixed_design_dimensions.keys())
        self.fixed_design_values = list(fixed_design_dimensions.values())
        self.sensitivity_dimension = sensitivity_dimension

        assert (len(self.fixed_design_names)+len(self.sensitivity_dimension)==len(self.design_names)), \
            'Error: All dimensions except for those the figures are drawn by should be fixed.'

        assert (len(self.sensitivity_dimension) in [1,2]), 'Error: Either 1D or 2D figures can be drawn.'

        # generate a combination of logic sentences to filter the results of the DOF needed.
        if len(self.fixed_design_names) != 0:
            if self.verbose:
                print( self.fixed_design_names, 'is/are fixed.')
            filter = ''
            for i in range(len(self.fixed_design_names)):
                filter += '(self.store_all_results_dataframe['
                filter += str(self.fixed_design_names[i])
                filter += ']=='
                filter += str(self.fixed_design_values[i])
                filter += ')'
                if i != (len(self.fixed_design_names)-1):
                    filter += '&'
            # extract results with other dimensions fixed
            figure_result_data = self.store_all_results_dataframe.loc[eval(filter)]
        # if there is no other fixed dimensions
        else:
            figure_result_data = self.store_all_results_dataframe

        # add results for figures
        self.figure_result_data = figure_result_data

        # if one design variable name is given as DOF, draw 1D sensitivity curve
        if (len(sensitivity_dimension) == 1):
            if self.verbose:
                print('1D sensitivity curve is plotted with ', self.sensitivity_dimension[0], '.')
            self.__curve1D(title_text, xlabel_text, font_axes=16, font_tick=14, log_scale=True)
        # if two design variable names are given as DOF, draw 2D heatmaps
        elif (len(sensitivity_dimension) == 2):
            if self.verbose:
                print('2D heatmap is plotted with ', self.sensitivity_dimension, '.')
            self.__heatmap(title_text, xlabel_text, ylabel_text, font_axes=16, font_tick=14, log_scale=True)


    def __curve1D(self, title_text, xlabel_text, font_axes=16, font_tick=14, log_scale=True):
        '''
        Draw 1D sensitivity curves for all design criteria

        Parameters:
        ----------
        title_text: name of the figure, a string
        xlabel_text: x label title, a string.
            In a 1D sensitivity curve, it is the design variable by which the curve is drawn.
        font_axes: axes label font size
        font_tick: tick label font size
        log_scale: if True, the result matrix will be scaled by log10

        Returns:
        --------
        4 Figures of 1D sensitivity curves for each criteria
        '''

        # extract the range of the DOF design variable
        x_range = self.figure_result_data[self.sensitivity_dimension[0]].values.tolist()

        # decide if the results are log scaled
        if log_scale:
            y_range_A = np.log10(self.figure_result_data['A'].values.tolist())
            y_range_D = np.log10(self.figure_result_data['D'].values.tolist())
            y_range_E = np.log10(self.figure_result_data['E'].values.tolist())
            y_range_ME = np.log10(self.figure_result_data['ME'].values.tolist())
        else:
            y_range_A = self.figure_result_data['A'].values.tolist()
            y_range_D = self.figure_result_data['D'].values.tolist()
            y_range_E = self.figure_result_data['E'].values.tolist()
            y_range_ME = self.figure_result_data['ME'].values.tolist()

        # Draw A-optimality
        fig = plt.figure()
        plt.rc('axes', titlesize=font_axes)
        plt.rc('axes', labelsize=font_axes)
        plt.rc('xtick', labelsize=font_tick)
        plt.rc('ytick', labelsize=font_tick)
        ax = fig.add_subplot(111)
        params = {'mathtext.default': 'regular'}
        #plt.rcParams.update(params)
        ax.plot(x_range, y_range_A)
        ax.scatter(x_range, y_range_A)
        ax.set_ylabel('$log_{10}$ Trace')
        ax.set_xlabel(xlabel_text)
        plt.title(title_text + ' - A optimality')
        plt.show()

        # Draw D-optimality
        fig = plt.figure()
        plt.rc('axes', titlesize=font_axes)
        plt.rc('axes', labelsize=font_axes)
        plt.rc('xtick', labelsize=font_tick)
        plt.rc('ytick', labelsize=font_tick)
        ax = fig.add_subplot(111)
        params = {'mathtext.default': 'regular'}
        # plt.rcParams.update(params)
        ax.plot(x_range, y_range_D)
        ax.scatter(x_range, y_range_D)
        ax.set_ylabel('$log_{10}$ Determinant')
        ax.set_xlabel(xlabel_text)
        plt.title(title_text + ' - D optimality')
        plt.show()

        # Draw E-optimality
        fig = plt.figure()
        plt.rc('axes', titlesize=font_axes)
        plt.rc('axes', labelsize=font_axes)
        plt.rc('xtick', labelsize=font_tick)
        plt.rc('ytick', labelsize=font_tick)
        ax = fig.add_subplot(111)
        params = {'mathtext.default': 'regular'}
        # plt.rcParams.update(params)
        ax.plot(x_range, y_range_E)
        ax.scatter(x_range, y_range_E)
        ax.set_ylabel('$log_{10}$ Minimal eigenvalue')
        ax.set_xlabel(xlabel_text)
        plt.title(title_text + ' - E optimality')
        plt.show()

        # Draw Modified E-optimality
        fig = plt.figure()
        plt.rc('axes', titlesize=font_axes)
        plt.rc('axes', labelsize=font_axes)
        plt.rc('xtick', labelsize=font_tick)
        plt.rc('ytick', labelsize=font_tick)
        ax = fig.add_subplot(111)
        params = {'mathtext.default': 'regular'}
        # plt.rcParams.update(params)
        ax.plot(x_range, y_range_ME)
        ax.scatter(x_range, y_range_ME)
        ax.set_ylabel('$log_{10}$ Condition number')
        ax.set_xlabel(xlabel_text)
        plt.title(title_text + ' - Modified E optimality')
        plt.show()

    def __heatmap(self, title_text, xlabel_text, ylabel_text, font_axes=16, font_tick=14, log_scale=True):
        '''
        Draw 2D heatmaps for all design criteria

        Parameters:
        ----------
        title_text: name of the figure, a string
        xlabel_text: x label title, a string.
            In a 2D heatmap, it should be the second design varialbe in the design_ranges
        ylabel_text: y label title, a string.
            In a 2D heatmap, it should be the first design variable in the dv_ranges
        font_axes: axes label font size
        font_tick: tick label font size
        log_scale: if True, the result matrix will be scaled by log10

        Returns:
        --------
        4 Figures of 2D heatmap for each criteria
        '''

        # achieve the design variable ranges this figure needs
        x_range = list(set(self.figure_result_data[self.sensitivity_dimension[0]].values.tolist()))
        y_range = list(set(self.figure_result_data[self.sensitivity_dimension[1]].values.tolist()))

        # extract the design criteria values
        A_range = self.figure_result_data['A'].values.tolist()
        D_range = self.figure_result_data['D'].values.tolist()
        E_range = self.figure_result_data['E'].values.tolist()
        ME_range = self.figure_result_data['ME'].values.tolist()

        # reshape the design criteria values for heatmaps
        cri_a = np.asarray(A_range).reshape(len(x_range), len(y_range))
        cri_d = np.asarray(D_range).reshape(len(x_range), len(y_range))
        cri_e = np.asarray(E_range).reshape(len(x_range), len(y_range))
        cri_e_cond = np.asarray(ME_range).reshape(len(x_range), len(y_range))

        self.cri_a = cri_a
        self.cri_d = cri_d
        self.cri_e = cri_e
        self.cri_e_cond = cri_e_cond

        # decide if log scaled
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

        # set heatmap x,y ranges
        xLabel = x_range
        yLabel = y_range

        # A-optimality
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

        # D-optimality
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

        # E-optimality
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

        # modified E-optimality
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

    

    
def simulate_discretize_model(m,NFE,collo=True,initialize=True):
    ''' Simulation, discretize, and initialize the Pyomo model.
    This is only used with Pyomo.DAE models.
    
    Parameters:
    -----------
    m: Pyomo model
    NFE: number of finite elements to consider (integer)
    initialize: if True, initialize the discretized model with the
         integrator solution (boolean)
    
    Returns:
    -------
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

    
    
    