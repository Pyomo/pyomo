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


from pyomo.common.dependencies import (
    numpy as np, numpy_available
)

import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import time
import pickle
from itertools import permutations, product
import logging
from pyomo.contrib.sensitivity_toolbox.sens import sensitivity_calculation, get_dsdp
#from pyomo.contrib.doe.scenario import Scenario_generator
#from pyomo.contrib.doe.result import FisherResults, GridSearchResult
from scenario import Scenario_generator
from result import FisherResults, GridSearchResult

class DesignOfExperiments:
    def __init__(self, param_init, design_variable_timepoints, measurement_object, create_model, solver=None,
                 time_set_name = "t", prior_FIM=None, discretize_model=None, args=None):
        """
        This package enables model-based design of experiments analysis with Pyomo. 
        Both direct optimization and enumeration modes are supported.
        NLP sensitivity tools, e.g.,  sipopt and k_aug, are supported to accelerate analysis via enumeration.
        It can be applied to dynamic models, where design variables are controlled throughout the experiment.

        Parameters
        ----------
        param_init:
            A  ``dictionary`` of parameter names and values. 
            If they defined as indexed Pyomo variable, put the variable name and index, such as 'theta["A1"]'.
            Note: if sIPOPT is used, parameter shouldn't be indexed.
        design_variable_timepoints:
            A ``dictionary`` where keys are design variable names, values are its control time points.
            If this design var is independent of time (constant), set the time to [0]
        measurement_object:
            A measurement ``object``.
        create_model:
            A  ``function`` that returns the model
        solver:
            A ``solver`` object that User specified, default=None. 
            If not specified, default solver is IPOPT MA57.
        time_set_name:
            A ``string`` of the name of the time set in the model. Default is "t".
        prior_FIM:
            A ``list`` of lists containing Fisher information matrix (FIM) for prior experiments.
        discretize_model:
            A user-specified ``function`` that discretizes the model. Only use with Pyomo.DAE, default=None
        args:
            Additional arguments for the create_model function.
        """
        
        # parameters
        self.param = param_init
        # design variable name
        self.design_timeset = design_variable_timepoints
        self.design_name = list(self.design_timeset.keys())
        # the control time point for each design variable
        self.design_time = list(self.design_timeset.values())
        self.create_model = create_model
        self.args = args

        # create the measurement information object
        self.measure = measurement_object
        self.measure_name = self.measure.measurement_name
        #self.flatten_measure_name = self.measure.flatten_measure_name
        #self.flatten_variance = self.measure.flatten_variance
        #self.flatten_measure_timeset = self.measure.flatten_measure_timeset

        # check if user-defined solver is given
        if solver:
            self.solver = solver
        # if not given, use default solver
        else:
            self.solver = self._get_default_ipopt_solver()

        # time set name 
        self.t = time_set_name

        # check if discretization is needed
        self.discretize_model = discretize_model

        # check if there is prior info
        self.prior_FIM = prior_FIM

        # if print statements
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(level=logging.WARN)
        
    def _check_inputs(self, check_mode=False):
        """
        Check if inputs are consistent

        Parameters
        ----------
        check_mode: check FIM calculation mode
        """
        if self.objective_option not in ['det', 'trace', 'zero']:
            raise ValueError('Objective function should be chosen from "det", "zero" and "trace" while receiving {}'.format(self.objective_option))

        if self.formula not in ['central', 'forward', 'backward', None]:
            raise ValueError('Finite difference scheme should be chosen from "central", "forward", "backward" and None while receiving {}.'.formate(self.formula))

        if type(self.prior_FIM)!=type(None):
            if np.shape(self.prior_FIM)[0] != np.shape(self.prior_FIM)[1]:
                raise ValueError('Found wrong prior information matrix shape.')

        if check_mode:
            curr_available_mode = ['sequential_finite','direct_kaug']
            if self.mode not in curr_available_mode:
                raise ValueError('Wrong mode.')

    def stochastic_program(self,  design_values, if_optimize=True, objective_option='det',
                     jac_involved_measurement=None,
                     scale_nominal_param_value=False, scale_constant_value=1, optimize_opt=None, if_Cholesky=False, L_LB = 1E-7, L_initial=None,
                     jac_initial=None, fim_initial=None,
                     formula='central', step=0.001, check=True, tee_opt=True):
        """
        Optimize DOE problem with design variables being the decisions.
        The DOE model is formed invasively and all scenarios are computed simultaneously.
        The function will first run a square problem with design variable being fixed at
        the given initial points (Objective function being 0), then a square problem with
        design variables being fixed at the given initial points (Objective function being Design optimality),
        and then unfix the design variable and do the optimization.

        Parameters
        -----------
        design_values:
            a ``dict`` where keys are design variable names, values are a dict whose keys are time point
            and values are the design variable value at that time point
        if_optimize:
            if true, continue to do optimization. else, just run square problem with given design variable values
        objective_option:
            supporting maximizing the 'det' determinant or the 'trace' trace of the FIM
        jac_involved_measurement:
            the measurement class involved in calculation. If None, take the overall measurement class
        scale_nominal_param_value:
            if True, the parameters are scaled by its own nominal value in param_init
        scale_constant_value:
            scale all elements in Jacobian matrix, default is 1.
        optimize_opt:
            A dictionary, keys are design variables, values are True or False deciding if this design variable will be optimized as DOF or not
        if_Cholesky:
            if True, Cholesky decomposition is used for Objective function for D-optimality.
        L_LB:
            L is the Cholesky decomposition matrix for FIM, i.e. FIM = L*L.T. 
            L_LB is the lower bound for every element in L.
            if FIM is positive definite, the diagonal element should be positive, so we can set a LB like 1E-10
        L_initial:
            initialize the L
        jac_initial:
            a matrix used to initialize jacobian matrix
        fim_initial:
            a matrix used to initialize FIM matrix
        formula:
            choose from 'central', 'forward', 'backward', None. This option is only used for 'sequential_finite' mode.
        step:
            Sensitivity perturbation step size, a fraction between [0,1]. default is 0.001
        check:
            if True, inputs are checked for consistency, default is True.

        Returns
        --------
        analysis_square: result summary of the square problem solved at the initial point
        analysis_optimize: result summary of the optimization problem solved

        """
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
        self.jac_initial = jac_initial
        self.fim_initial = fim_initial
        self.formula = formula
        self.step = step
        self.tee_opt = tee_opt

        # calculate how much the FIM element is scaled by a constant number
        # FIM = Jacobian.T@Jacobian, the FIM is scaled by squared value the Jacobian is scaled
        self.fim_scale_constant_value = self.scale_constant_value **2

        # identify measurements involved in calculation
        if jac_involved_measurement:
            self.jac_involved_name = jac_involved_measurement.flatten_measure_name.copy()
            self.timepoint_overall_set = jac_involved_measurement.timepoint_overall_set.copy()
        else:
            self.jac_involved_name = self.flatten_measure_name.copy()
            self.timepoint_overall_set = self.measure.timepoint_overall_set.copy()
            

        # check if inputs are valid
        # simultaneous mode does not need to check mode and dimension of design variables
        if check:
            self._check_inputs(check_mode=False)

        # build the large DOE pyomo model
        m = self._create_doe_model(no_obj=True)

        # solve model, achieve results for square problem, and results for optimization problem
        m, analysis_square = self._compute_stochastic_program(m, optimize_opt)

        if self.optimize:
            analysis_optimize = self._optimize_stochastic_program(m)

            time1 = time.time()
            analysis_optimize.total_time = time1-time0
            self.logger.info('Total wall clock time [s]: %s', time1-time0)
            return analysis_square, analysis_optimize
            
        else:
            time1 = time.time()
            # record square problem time
            analysis_square.total_time = time1-time0
            self.logger.info('Total wall clock time [s]: %s', time1 - time0)

            return analysis_square
            

    def _compute_stochastic_program(self, m, optimize_option):
        """
        Solve the stochastic program problem as a square problem. 
        """

        # Solve square problem first
        # result_square: solver result
        time0_solve = time.time()
        result_square = self._solve_doe(m, fix=True, opt_option=optimize_option)
        time1_solve = time.time()

        time_solve1 = time1_solve-time0_solve

        # extract Jac
        jac_square = self._extract_jac(m)

        # create result object
        analysis_square = FisherResults(list(self.param.keys()), self.measure, jacobian_info=None, all_jacobian_info=jac_square,
                                     prior_FIM=self.prior_FIM, scale_constant_value=self.scale_constant_value)
        # for simultaneous mode, FIM and Jacobian are extracted with extract_FIM()
        analysis_square.calculate_FIM(self.design_timeset, result=result_square)

        analysis_square.model = m

        self.analysis_square = analysis_square
        analysis_square.solve_time = time_solve1
        self.logger.info('Total solve time with simultaneous_finite mode (Wall clock) [s]:  %s', time_solve1)
        
        return m, analysis_square

    def _optimize_stochastic_program(self, m):
        """
        Solve the stochastic program problem with degrees of freedom. 
        """
        
        m = self._add_objective(m)

        self.logger.info('Solve with given objective:')
        time0_solve2 = time.time()
        result_doe = self._solve_doe(m, fix=False)
        time1_solve2 = time.time()
        time_solve2 = time1_solve2 - time0_solve2

        # extract Jac
        jac_optimize = self._extract_jac(m)

        # create result object
        analysis_optimize = FisherResults(list(self.param.keys()), self.measure, jacobian_info=None, all_jacobian_info=jac_optimize,
                                        prior_FIM=self.prior_FIM)
        # for simultaneous mode, FIM and Jacobian are extracted with extract_FIM()
        analysis_optimize.calculate_FIM(self.design_timeset, result=result_doe)
        analysis_optimize.model = m

        # record optimization time
        analysis_optimize.solve_time = time_solve2

        return analysis_optimize



    def compute_FIM(self, design_values, mode='sequential_finite', FIM_store_name=None, specified_prior=None,
                    tee_opt=True, scale_nominal_param_value=False, scale_constant_value=1,
                    store_output = None, read_output=None, extract_single_model=None,
                    formula='central', step=0.001,
                    objective_option='det'):
        """
        This function solves a square Pyomo model with fixed design variables to compute the FIM.
        It calculates FIM with sensitivity information from four modes:
            1.  sequential_finite: Calculates a one scenario model multiple times for multiple scenarios. 
            Sensitivity info estimated by finite difference
            2.  sequential_sipopt: calculate sensitivity by sIPOPT [Experimental]
            3.  sequential_kaug: calculate sensitivity by k_aug [Experimental]
            4.  direct_kaug: calculate sensitivity by k_aug with direct sensitivity

        "Simultaneous_finite" mode is not included in this function.

        Parameters
        -----------
        design_values:
            a ``dict`` where keys are design variable names, 
            values are a dict whose keys are time point and values are the design variable value at that time point
        mode:
            use mode='sequential_finite', 'sequential_sipopt', 'sequential_kaug', 'direct_kaug'
        FIM_store_name:
            if storing the FIM in a .csv or .txt, give the file name here as a string.
        specified_prior:
            provide alternate prior matrix, default is no prior.
        tee_opt:
            if True, IPOPT console output is printed
        scale_nominal_param_value:
            if True, the parameters are scaled by its own nominal value in param_init
        scale_constant_value:
            scale all elements in Jacobian matrix, default is 1.
        store_output:
            if storing the output (value stored in Var 'output_record') as a pickle file, give the file name here as a string.
        read_output:
            if reading the output (value for Var 'output_record') as a pickle file, give the file name here as a string.
        extract_single_model:
            if True, the solved model outputs for each scenario are all recorded as a .csv file. 
            The output file uses the name AB.csv, where string A is store_output input, B is the index of scenario.  
            scenario index is the number of the scenario outputs which is stored.
        formula:
            choose from 'central', 'forward', 'backward', None. This option is only used for 'sequential_finite' mode.
        step:
            Sensitivity perturbation step size, a fraction between [0,1]. default is 0.001
        objective_option: 
            choose from 'det' or 'trace' or 'zero'. Optimization problem maximizes determinant or trace or using 0 as objective function.

        Return
        ------
        FIM_analysis: result summary object of this solve
        """
        
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

        self.FIM_store_name = FIM_store_name
        self.specified_prior = specified_prior

        # calculate how much the FIM element is scaled by a constant number
        # As FIM~Jacobian.T@Jacobian, FIM is scaled twice the number the Q is scaled
        self.fim_scale_constant_value = self.scale_constant_value ** 2

        # check inputs valid
        self._check_inputs(check_mode=True)

        if self.mode=='sequential_finite':
            FIM_analysis = self._sequential_finite(read_output, extract_single_model, store_output)
            return FIM_analysis

        elif self.mode =='direct_kaug':
            FIM_analysis = self._direct_kaug()
            return FIM_analysis
            
        else:
            raise ValueError(self.mode+' is not a valid mode. Choose from "sequential_finite" and "direct_kaug".')

    def _sequential_finite(self, read_output, extract_single_model, store_output):

        # if measurements are provided
        if read_output:
            with open(read_output, 'rb') as f:
                output_record = pickle.load(f)
                f.close()
            jac = self._finite_calculation(output_record, scena_gen)

        # if measurements are not provided
        else:
            scena_object = Scenario_generator(self.param, formula=self.formula, step=self.step)
            scena_gen = scena_object.simultaneous_scenario()
            print(scena_gen)
            self.scenario_list = scena_gen["scenario"]
            self.scenario_num = scena_gen["scena_num"]
            # dict for storing model outputs
            output_record = {}

            mod = pyo.ConcreteModel()

            mod.scena = pyo.Set(initialize=list(range(len(self.scenario_list))))

            # Allow user to self-define complex design variables
            #mod.t0 = pyo.Set(initialize=[0])
            #mod.t_con = pyo.Set(initialize=[0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1])
            #mod.CA0 = pyo.Var(mod.t0, bounds=[1,5], within=pyo.NonNegativeReals)
            #mod.T = pyo.Var(mod.t_con, bounds=[300, 700], within=pyo.NonNegativeReals)

            mod.add_component('t0', pyo.Set(initialize=[0]))
            mod.add_component('t_con', pyo.Set(initialize=[0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1]))
            mod.add_component('CA0', pyo.Var(mod.t0, bounds=[1,5], within=pyo.NonNegativeReals))
            mod.add_component('T', pyo.Var(mod.t_con, bounds=[300, 700], within=pyo.NonNegativeReals))

            def block_build(b,s):
                self.create_model(m=b)
                
                for par in self.param:
                    par_strname = eval('b.'+str(par))
                    par_strname.fix(scena_gen["scenario"][s][par])

            mod.lsb = pyo.Block(mod.scena, rule=block_build)

            # discretize the model        
            if self.discretize_model:
                mod = self.discretize_model(mod)

            # force all design variables in blocks be the same as global design variables
            def fix_design1(m,s):
                return m.lsb[s].CA0[0]  == m.CA0[0]
            
            def fix_design2(m,s,t):
                return m.lsb[s].T[t] == m.T[t]
            
            mod.fix_con1 = pyo.Constraint(mod.scena, rule=fix_design1)
            mod.fix_con2 = pyo.Constraint(mod.scena, mod.t_con, rule=fix_design2)

            # solve model
            square_result = self._solve_doe(mod, fix=True)

            if extract_single_model:
                mod_name = store_output + '.csv'
                dataframe = extract_single_model(mod, square_result)
                dataframe.to_csv(mod_name)

            # loop over blocks for results
            for s in range(len(self.scenario_list)):
                # loop over measurement item and time to store model measurements
                output_iter = []

                for r in self.measure_name:
                    cuid = pyo.ComponentUID(r)
                    var_up = cuid.find_component_on(mod.lsb[s])
                    output_iter.append(pyo.value(var_up))

                output_record[s] = output_iter

                output_record['design'] = self.design_values
                
                if store_output:
                    f = open(store_output, 'wb')
                    pickle.dump(output_record, f)
                    f.close()
            # calculate jacobian
            jac = self._finite_calculation(output_record, scena_gen)

            # return all models formed
            self.model = mod

            # Store the Jacobian information for access by users, not necessarily call result object to achieve jacobian information
            # It is the overall set of Jacobian information, 
            # while in the result object the jacobian can be cut to achieve part of the FIM information
            self.jac = jac

        # Assemble and analyze results
        if self.specified_prior is None:
            prior_in_use = self.prior_FIM
        else:
            prior_in_use = self.specified_prior

        FIM_analysis = FisherResults(list(self.param.keys()), self.measure, jacobian_info=None, all_jacobian_info=jac,
                                    prior_FIM=prior_in_use, store_FIM=self.FIM_store_name, scale_constant_value=self.scale_constant_value)

        

        return FIM_analysis

    def _direct_kaug(self):
        time00 = time.time()
        # create scenario class for a base case
        scena_gen = Scenario_generator(self.param, formula=None, step=self.step)
        scenario_all = scena_gen.simultaneous_scenario()

        # create model
        time0_build = time.time()
        mod = self.create_model(scenario_all, args=self.args)
        time1_build = time.time()
        time_build = time1_build - time0_build

        # discretize if needed
        if self.discretize_model:
            mod = self.discretize_model(mod)

        time_set_attr = getattr(mod, self.t)
        # get all time
        t_all = list(time_set_attr)

        # add objective function
        mod.Obj = pyo.Objective(expr=0, sense=pyo.minimize)

        # Check if measurement time points are in this time set
        # Also correct the measurement time points
        # For e.g. if a measurement time point is 0.0 in the model but is given as 0, it is corrected here
        measurement_accurate_time = self.flatten_measure_timeset.copy()

        for j in self.flatten_measure_name:
            for no_t, tt in enumerate(self.flatten_measure_timeset[j]):
                if tt not in t_all:
                    self.logger.warning('A measurement time point not measured by this model:  %s', tt)
                else:
                    measurement_accurate_time[j][no_t] = t_all[t_all.index(tt)]

        # set ub and lb to parameters
        for par in list(self.param.keys()):
            component = getattr(mod, par)[0]
            component.setlb(self.param[par])
            component.setub(self.param[par])

        # generate parameter name list and value dictionary with index
        var_name = []
        var_dict = {}
        for name in list(self.param.keys()):
            # [0] is the scenario index
            var_name.append(name+'[0]')
            var_dict[name+'[0]'] = self.param[name]

        # call k_aug get_dsdp function
        time0_solve = time.time()
        square_result = self._solve_doe(mod, fix=True)
        dsdp_re, col = get_dsdp(mod, var_name, var_dict, tee=self.tee_opt)
        time1_solve = time.time()
        time_solve = time1_solve - time0_solve

        # analyze result
        dsdp_array = dsdp_re.toarray().T
        self.dsdp = dsdp_array
        self.dsdp = col
        # store dsdp returned
        dsdp_extract = []
        # get right lines from results
        measurement_index = []
        # produce the sensitivity for fixed variables
        zero_sens = np.zeros(len(self.param))

        # loop over measurement variables and their time points
        for measurement_name in self.measure.model_measure_name:
            # get right line number in kaug results
            if self.discretize_model:
                # for DAE model, some variables are fixed
                try:
                    kaug_no = col.index(measurement_name)
                    measurement_index.append(kaug_no)
                    # get right line of dsdp
                    dsdp_extract.append(dsdp_array[kaug_no])
                except:
                    self.logger.debug('The variable is fixed:  %s', measurement_name)
                    # for fixed variables, the sensitivity are a zero vector
                    dsdp_extract.append(zero_sens)
            else:
                kaug_no = col.index(measurement_name)
                measurement_index.append(kaug_no)
                # get right line of dsdp
                dsdp_extract.append(dsdp_array[kaug_no])

        # Extract and calculate sensitivity if scaled by constants or parameters.
        # Convert sensitivity to a dictionary
        jac = {}
        for par in list(self.param.keys()):
            jac[par] = []

        for d in range(len(dsdp_extract)):
            for p, par in enumerate(list(self.param.keys())):
                # if scaled by parameter value or constant value
                sensi = dsdp_extract[d][p]*self.scale_constant_value
                if self.scale_nominal_param_value:
                    sensi *= self.param[par]
                jac[par].append(sensi)

        time11 = time.time()
        self.logger.info('Build time with direct kaug mode [s]:  %s', time_build)
        self.logger.info('Solve time with direct kaug mode [s]:  %s', time_solve)
        self.logger.info('Total wall clock time [s]:  %s', time11-time00)
            
        # check if another prior experiment FIM is provided other than the user-specified one
        if self.specified_prior is None:
            prior_in_use = self.prior_FIM
        else:
            prior_in_use = self.specified_prior

        # Assemble and analyze results
        FIM_analysis = FisherResults(list(self.param.keys()),self.measure, jacobian_info=None, all_jacobian_info=jac,
                                    prior_FIM=prior_in_use, store_FIM=self.FIM_store_name,
                                    scale_constant_value=self.scale_constant_value)
        
        self.jac = jac
        FIM_analysis.build_time = time_build
        FIM_analysis.solve_time = time_solve
        

        return FIM_analysis


    def _finite_calculation(self, output_record, scena_gen):
        """
        Calculate Jacobian for sequential_finite mode

        Parameters
        ----------
        output_record: a dict of outputs, keys are scenario names, values are a list of measurements values
        scena_gen: an object generated by Scenario_creator class

        Returns
        --------
        jac: Jacobian matrix, a dictionary, keys are parameter names, values are a list of jacobian values with respect to this parameter
        """
        # dictionary form of jacobian
        jac = {}

        # After collecting outputs from all scenarios, calculate sensitivity
        for para in list(self.param.keys()):
            # extract involved scenario No. for each parameter from scenario class
            involved_s = scena_gen['scena_num'][para]

            # each parameter has two involved scenarios
            s1 = involved_s[0] # upper perturbation
            s2 = involved_s[1] # loweer perturbation
            list_jac = []
            for i in range(len(output_record[s1])):
                sensi = (output_record[s1][i] - output_record[s2][i]) / scena_gen['eps-abs'][para] * self.scale_constant_value
                if self.scale_nominal_param_value:
                    sensi *= self.param[para]
                list_jac.append(sensi)
            # get Jacobian dict, keys are parameter name, values are sensitivity info
            jac[para] = list_jac

        return jac

    def _extract_jac(self, m):
        """
        Extract jacobian from simultaneous mode
        Arguments
        ---------
        m: solved simultaneous model
        Returns
        ------
        JAC: the overall jacobian as a dictionary
        """
        # dictionary form of jacobian
        jac = {}
        # loop over parameters
        for p in list(self.param.keys()): 
            jac_para = []
            for name1 in self.jac_involved_name:
                for tim in self.timepoint_overall_set:
                    jac_para.append(pyo.value(m.jac[name1, p, tim]))
            jac[p] = jac_para
        return jac

    def run_grid_search(self, design_values, design_ranges, design_dimension_names, 
                    design_control_time, mode='sequential_finite', tee_option=False, 
                    scale_nominal_param_value=False, scale_constant_value=1, store_name= None, read_name=None,
                        filename=None, formula='central', step=0.001):
        """
        Enumerate through full grid search for any number of design variables;
        solve square problems sequentially to compute FIMs.
        It calculates FIM with sensitivity information from four modes:
            1.  sequential_finite: Calculates a one scenario model multiple times for multiple scenarios. 
            Sensitivity info estimated by finite difference
            2.  sequential_sipopt: calculate sensitivity by sIPOPT [Experimental]
            3.  sequential_kaug: calculate sensitivity by k_aug [Experimental]
            4.  direct_kaug: calculate sensitivity by k_aug with direct sensitivity

        Parameters
        -----------
        design_values:
            a ``dict`` where keys are design variable names, values are a dict whose keys are time point and values are the design variable value at that time point
        design_ranges:
            a ``list`` of design variable values to go over
        design_dimension_names:
            a ``list`` of design variable names of each design range
        design_control_time:
            a ``list`` of control time points that should be fixed to the values in dv_ranges
        mode:
            use mode='sequential_finite', 'sequential_sipopt', 'sequential_kaug', 'direct_kaug'
        tee_option:
            if solver console output is made
        scale_nominal_param_value:
            if True, the parameters are scaled by its own nominal value in param_init
        scale_constant_value:
            scale all elements in Jacobian matrix, default is 1.
        store_name:
            a string of file name. If not None, store results with this name.
            Since there are maultiple experiments, results are numbered with a scalar number, 
            and the result for one grid is 'store_name(count).csv' (count is the number of count).
        read_name: 
            a string of file name. If not None, read result files. 
            Since there are multiple experiments, this string should be the common part of all files;
            Real name of the file is "read_name(count)", where count is the number of the experiment. 
        filename:
            if True, grid search results stored with this file name
        formula:
            choose from 'central', 'forward', 'backward', None. This option is only used for 'sequential_finite' mode.
        step:
            Sensitivity perturbation step size, a fraction between [0,1]. default is 0.001

        Return
        -------
        figure_draw_object: a combined result object of class Grid_search_result
        """
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
        failed_count = 0
        # how many sets of design variables will be run
        total_count = 1
        for rng in design_ranges:
            total_count *= len(rng)

        # generate combinations of design variable values to go over
        search_design_set = product(*design_ranges)

        build_time_store=[]
        solve_time_store=[]

        # loop over design value combinations
        for design_set_iter in search_design_set:
            # generate the design variable dictionary needed for running compute_FIM
            # first copy value from design_values
            design_iter = design_values.copy()

            # update the controlled value of certain time points for certain design variables
            for i in range(grid_dimension):
                for v, value in enumerate(design_control_time[i]):
                    design_iter[design_dimension_names[i]][value] = list(design_set_iter)[i]

            self.logger.info('=======Iteration Number: %s =====', count+1)
            self.logger.debug('Design variable values of this iteration: %s', design_iter)

            # generate store name
            if store_name is None:
                store_output_name = None
            else:
                store_output_name = store_name + str(count)

            if read_name:
                read_input_name = read_name+str(count)
            else:
                read_input_name = None

            # call compute_FIM to get FIM
            try:
                result_iter = self.compute_FIM(design_iter, mode=mode,
                                                tee_opt=tee_option,
                                                scale_nominal_param_value=scale_nominal_param_value,
                                                scale_constant_value = scale_constant_value,
                                                store_output=store_output_name, read_output=read_input_name,
                                                formula=formula, step=step)
                if read_input_name is None:
                    build_time_store.append(result_iter.build_time)
                    solve_time_store.append(result_iter.solve_time)

                count += 1

                result_iter.calculate_FIM(self.design_values)

                t_now = time.time()

                # give run information at each iteration
                self.logger.info('This is the  %s run out of  %s run.', count+1, total_count)
                self.logger.info('The code has run  %s seconds.', t_now-t_enumeration_begin)
                self.logger.info('Estimated remaining time:  %s seconds', (t_now-t_enumeration_begin)/(count+1)*(total_count-count-1))

                # the combined result object are organized as a dictionary, keys are a tuple of the design variable values, values are a result object
                result_combine[tuple(design_set_iter)] = result_iter

            except:
                self.logger.warning(':::::::::::Warning: Cannot converge this run.::::::::::::')
                count += 1
                failed_count += 1
                self.logger.warning('failed count:', failed_count)
                result_combine[tuple(design_set_iter)] = None

        # For user's access
        self.all_fim = result_combine

        # Create figure drawing object
        figure_draw_object = GridSearchResult(design_ranges, design_dimension_names, design_control_time, result_combine, store_optimality_name=filename)

        t_enumeration_stop = time.time()
        self.logger.info('Overall model building time [s]:  %s', sum(build_time_store))
        self.logger.info('Overall model solve time [s]:  %s', sum(solve_time_store))
        self.logger.info('Overall wall clock time [s]:  %s', t_enumeration_stop - t_enumeration_begin)

        return figure_draw_object


    def _create_doe_model(self, no_obj=True):
        """
        Add equations to compute sensitivities, FIM, and objective. 

        Parameters:
        -----------
        no_obj: if True, objective function is 0.
        self.design_values: a dict of dictionaries, keys are the name of design variables, 
        values are a dict where keys are the time points, values are the design variable value at that time point
        self.optimize: if True, solve the problem unfixing the design variables. if False, solve the problem as a
        square problem
        self.objective_option: choose from 'det' or 'trace'. Optimization problem maximizes determinant or trace.
        self.scale_nominal_param_value: if True, scale FIM but not scale Jacobian. This toggle can be opened for better performance when the
        problem is poorly scaled.
        self.tee_opt: if True, print IPOPT console output
        self.Cholesky_option: if true, cholesky decomposition is used for Objective function (to optimize determinant). 
            If true, determinant will not be calculated.
            self.L_LB: if FIM is P.D., the diagonal element should be positive, so we can set a LB like 1E-10
            self.L_initial: initialize the L
        self.formula: choose from 'central', 'forward', 'backward', None
        self.step: Sensitivity perturbation step size, a fraction between [0,1]. default is 0.001

        Return:
        -------
        m: the DOE model
        """
        # call generator function to get scenario dictionary
        scena_gen = Scenario_generator(self.param, formula=self.formula, step=self.step, store=True)
        scenario_all = scena_gen.simultaneous_scenario()
        
        # create model
        m = self.create_model(scenario_all, args= self.args)
        # discretize if discretization function is provided
        if self.discretize_model:
            m = self.discretize_model(m)

        # get time set 
        time_set_attr = getattr(m, self.t)
        
        # extract (discretized) time 
        time_set=list(time_set_attr)
        self.time_set = time_set

        # create parameter, measurement, time and measurement time set
        m.para_set = pyo.Set(initialize=list(self.param.keys()))
        param_name = list(self.param.keys())
        m.y_set = pyo.Set(initialize=self.jac_involved_name)
        m.t_set = pyo.Set(initialize=time_set)

        m.tmea_set = pyo.Set(initialize=self.timepoint_overall_set)

        # we can be sure about the name of scenarios, because they are generated by our function
        m.scenario = pyo.Set(initialize=scenario_all['scena-name'])
        m.optimize = self.optimize

        # check if measurement time points are in the time set
        for j in m.y_set:
            for t in m.tmea_set:
                if not (t in time_set_attr):
                    raise ValueError('Measure timepoints should be in the time list.')

        # check if control time points are in the time set
        for d in range(len(self.design_name)):
            if self.design_time[d]:
                for t in self.design_time[d]:
                    if not (t in time_set_attr):
                        raise ValueError('Control timepoints should be in the time list.')

        ### Define variables
        # Elements in Jacobian matrix
        if self.jac_initial:
            dict_jac = {}
            for i, bu in enumerate(m.y_set):
                for j, un in enumerate(m.para_set):
                    for t, tim in enumerate(m.tmea_set):
                        dict_jac[(bu,un,tim)] = self.jac_initial[i,j,t]

            def jac_initialize(m,i,j,t):
                return dict_jac[(bu,un,tim)]

            m.jac = pyo.Var(m.y_set, m.para_set, m.tmea_set, initialize=jac_initialize)

        else:
            m.jac = pyo.Var(m.y_set, m.para_set, m.tmea_set, initialize=1E-20)

        # Initialize Hessian with an identity matrix
        def identity_matrix(m,j,d):
            if j==d:
                return 1 
            else: 
                return 0

        # initialize FIM
        if self.fim_initial:
            dict_fim = {}
            for i, bu in enumerate(m.para_set):
                for j, un in enumerate(m.para_set):
                    dict_fim[(bu, un)] = self.fim_initial[i][j]

            def initialize_fim(m, j, d):
                return dict_fim[(j,d)]
            m.FIM = pyo.Var(m.para_set, m.para_set, initialize=initialize_fim)
        else:
            m.FIM = pyo.Var(m.para_set, m.para_set, initialize=identity_matrix)

        # move the L matrix initial point to a dictionary
        if type(self.L_initial) != type(None):
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
            if type(self.L_initial) != type(None):
                m.L_ele = pyo.Var(m.para_set, m.para_set, initialize=init_cho)
            # or initialize with the identity matrix
            else:
                m.L_ele = pyo.Var(m.para_set, m.para_set, initialize=identity_matrix)

            # loop over parameter name
            for c in m.para_set:
                for d in m.para_set:
                    # fix the 0 half of L matrix to be 0.0
                    if (param_name.index(c) < param_name.index(d)):
                        m.L_ele[c,d].fix(0.0)
                    # Give LB to the diagonal entries 
                    if self.L_LB:
                        if c==d:
                            m.L_ele[c,d].setlb(self.L_LB)


        def jac_numerical(m,j,p,t):
            """
            Calculate the Jacobian
            j: model responses
            p: model parameters
            t: timepoints
            """
            # A better way to do this: 
            # https://github.com/IDAES/idaes-pse/blob/274e58bef55f2f969f0df97cbb1fb7d99342388e/idaes/apps/uncertainty_propagation/sens.py#L296
            # check if j is a measurement with extra index by checking if there is '_index_' in its name
            up_C_name, lo_C_name, legal_t_option = self.measure.SP_measure_name(j,t,scenario_all=scenario_all, mode='simultaneous_finite', p=p)
            if legal_t_option:
                up_C = eval(up_C_name)
                lo_C = eval(lo_C_name)
                if self.scale_nominal_param_value:
                    return m.jac[j, p, t] == (up_C - lo_C) / scenario_all['eps-abs'][p] * self.param[p] * self.scale_constant_value
                else:
                    return m.jac[j, p, t] == (up_C - lo_C) / scenario_all['eps-abs'][p] * self.scale_constant_value
                # if t is not measured, let the value be 0
            else:
                return m.jac[j, p, t] == 0

        #A constraint to calculate elements in Hessian matrix
        # transfer prior FIM to be Expressions
        dict_fele={}
        for i, bu in enumerate(m.para_set):
            for j, un in enumerate(m.para_set):
                dict_fele[(bu,un)] = self.prior_FIM[i][j]

        def ele_todict(m,i,j):
            return dict_fele[(i,j)]
        m.refele = pyo.Expression(m.para_set, m.para_set, rule=ele_todict)

        def calc_FIM(m,j,d):
            """
            Calculate FIM elements
            """
            return m.FIM[j,d] == sum(sum(m.jac[z,j,i]*m.jac[z,d,i] for z in m.y_set) for i in m.tmea_set) + m.refele[j, d]*self.fim_scale_constant_value

        ### Constraints and Objective function
        m.dC_value = pyo.Constraint(m.y_set, m.para_set, m.tmea_set, rule=jac_numerical)
        m.ele_rule = pyo.Constraint(m.para_set, m.para_set, rule=calc_FIM)

        return m

    def _add_objective(self, m):

        def cholesky_imp(m, c, d):
            """
            Calculate Cholesky L matrix using algebraic constraints
            """
        # If it is the left bottom half of L
            if (list(self.param.keys()).index(c) >= list(self.param.keys()).index(d)):
                return m.FIM[c, d] == sum(
                    m.L_ele[c, list(self.param.keys())[k]] * m.L_ele[d, list(self.param.keys())[k]] for k in range(list(self.param.keys()).index(d) + 1))
            else:
        # This is the empty half of L above the diagonal
                return pyo.Constraint.Skip

        def trace_calc(m):
            """
            Calculate FIM elements. Can scale each element with 1000 for performance
            """
            return m.trace == sum(m.FIM[j,j] for j in m.para_set)

        def det_general(m):
            """Calculate determinant. Can be applied to FIM of any size.
            det(A) = sum_{\sigma \in \S_n} (sgn(\sigma) * \Prod_{i=1}^n a_{i,\sigma_i})
            Use permutation() to get permutations, sgn() to get signature
            """
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
            det_perm = sum( self._sgn(list_p[d])*sum(m.FIM[each, name_order[b]] for b, each in enumerate(m.para_set)) for d in range(len(list_p)))
            return m.det == det_perm

        if self.Cholesky_option:
            m.cholesky_cons = pyo.Constraint(m.para_set, m.para_set, rule=cholesky_imp)
            m.Obj = pyo.Objective(expr=2 * sum(pyo.log(m.L_ele[j, j]) for j in m.para_set), sense=pyo.maximize)
        # if not cholesky but determinant, calculating det and evaluate the OBJ with det
        elif (self.objective_option == 'det'):
            m.det_rule = pyo.Constraint(rule=det_general)
            m.Obj = pyo.Objective(expr=pyo.log(m.det), sense=pyo.maximize)
        # if not determinant or cholesky, calculating the OBJ with trace
        elif (self.objective_option == 'trace'):
            m.trace_rule = pyo.Constraint(rule=trace_calc)
            m.Obj = pyo.Objective(expr=pyo.log(m.trace), sense=pyo.maximize)
        elif (self.objective_option == 'zero'):
            m.Obj = pyo.Objective(expr=0)

        return m

    def _fix_design(self, m, design_val, fix_opt=True, optimize_option=None):
        """
        Fix design variable

        Parameters:
        -----------
        m: model
        design_val: design variable values dict
        fix_opt: if True, fix. Else, unfix
        optimize: a dictionary, keys are design variable name, values are True or False, deciding if this design variable is optimized as DOF this time

        Returns:
        --------
        m: model
        """
        # loop over the design variables and time index and to fix values specified in design_val
        for d, dname in enumerate(self.design_name):
            # if design variables are indexed by time
            if self.design_time[d]:
                for time in self.design_time[d]:
                    fix_v = design_val[dname][time]

                    if fix_opt:
                        getattr(m, dname)[time].fix(fix_v)
                    else:
                        if optimize_option is None:
                            getattr(m, dname)[time].unfix()
                        else:
                            if optimize_option[dname]:
                                getattr(m, dname)[time].unfix()
            else:
                fix_v = design_val[dname][0]

                if fix_opt:
                    getattr(m, dname).fix(fix_v)
                else:
                    getattr(m, dname).unfix()
        return m

    def _get_default_ipopt_solver(self):
        """Default solver
        """
        solver = SolverFactory('ipopt')
        solver.options['linear_solver'] = 'ma57'
        solver.options['halt_on_ampl_error'] = 'yes'
        solver.options['max_iter'] = 3000
        return solver

    def _solve_doe(self, m, fix=False, opt_option=None):
        """Solve DOE model.
        If it's a square problem, fix design variable and solve.
        Else, fix design variable and solve square problem firstly, then unfix them and solve the optimization problem

        Parameters:
        -----------
        m:model
        fix: if true, solve two times (square first). Else, just solve the square problem
        opt_option: a dictionary, keys are design variable name, values are True or False, 
            deciding if this design variable is optimized as DOF this time.
            If None, all design variables are optimized as DOF this time.
            
        Return:
        -------
        solver_results: solver results
        """
        ### Solve square problem
        mod = self._fix_design(m, self.design_values, fix_opt=fix, optimize_option=opt_option)

        # if user gives solver, use this solver. if not, use default IPOPT solver
        solver_result = self.solver.solve(mod,tee=self.tee_opt)

        return solver_result

    def _add_parameter(self, m, perturb=0):
        """
        For sIPOPT: add parameter perturbation set

        Parameters:
        -----------
        m: model name
        perturb: which parameter to perturb
        """
        # model parameters perturbation, backward disturb
        param_backward = self.param_value.copy()
        # perturb parameter
        param_backward[perturb] *= (1-self.step)

        # generate sIPOPT perturbed parameter names
        param_perturb_names = list(self.param.keys()).copy()
        for x, xname in enumerate(list(self.param.keys())):
            param_perturb_names[x] = xname+'_pert'

        self.perturb_names = param_perturb_names

        for change in range(len(self.perturb_names)):
            setattr(m, self.perturb_names[change], Param(m.scena, initialize=param_backward[change]))
        return m

    def _sgn(self,p):
        """
        This is a helper function for stochastic_program function to compute the determinant formula.
        Give the signature of a permutation

        Parameters:
        -----------
        p: the permutation (a list)

        Return:
        ------
        1 if the number of exchange is an even number
        -1 if the number is an odd number
        """

        if len(p) == 1:
            return 1

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




