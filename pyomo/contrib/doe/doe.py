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
from pyomo.common.timing import TicTocTimer
from pyomo.contrib.sensitivity_toolbox.sens import sensitivity_calculation, get_dsdp
#from pyomo.contrib.doe.scenario import Scenario_generator
#from pyomo.contrib.doe.result import FisherResults, GridSearchResult
from scenario import ScenarioGenerator
from result import FisherResults, GridSearchResult

class DesignOfExperiments:
    def __init__(self, param_init, design_object, measurement_object, create_model, solver=None,
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
        design_names:
            A designvariable ``object``
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
        self.design_name = design_object.design_name
        self.create_model = create_model
        self.args = args

        # create the measurement information object
        self.measure = measurement_object
        self.measure_name = self.measure.measurement_name

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
        self.logger.setLevel(level=logging.INFO)
        
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

    def stochastic_program(self,  design_object, if_optimize=True, objective_option='det',
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
            designVariable object
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
        # store inputs in object
        self.design_values = design_object.special_set_value
        self.design_object = design_object
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
    
        # check if inputs are valid
        # simultaneous mode does not need to check mode and dimension of design variables
        if check:
            self._check_inputs(check_mode=False)

        sp_timer = TicTocTimer()
        sp_timer.tic()

        # build the large DOE pyomo model
        m = self._create_doe_model(no_obj=True)

        # solve model, achieve results for square problem, and results for optimization problem
        m, analysis_square = self._compute_stochastic_program(m, optimize_opt)

        if self.optimize:
            analysis_optimize = self._optimize_stochastic_program(m)
            dT = sp_timer.toc()
            self.logger.info("elapsed time: %0.1f"%dT)
            return analysis_square, analysis_optimize
            
        else:
            dT = sp_timer.toc(msg=None)
            self.logger.info("elapsed time: %0.1f"%dT)
            return analysis_square
            

    def _compute_stochastic_program(self, m, optimize_option):
        """
        Solve the stochastic program problem as a square problem. 
        """

        # Solve square problem first
        # result_square: solver result
        result_square = self._solve_doe(m, fix=True, opt_option=optimize_option)

        # extract Jac
        jac_square = self._extract_jac(m)

        # create result object
        analysis_square = FisherResults(list(self.param.keys()), self.measure, jacobian_info=None, all_jacobian_info=jac_square,
                                    prior_FIM=self.prior_FIM, scale_constant_value=self.scale_constant_value)
        # for simultaneous mode, FIM and Jacobian are extracted with extract_FIM()
        analysis_square.calculate_FIM(self.design_values, result=result_square)

        analysis_square.model = m

        self.analysis_square = analysis_square
        return m, analysis_square

    def _optimize_stochastic_program(self, m):
        """
        Solve the stochastic program problem with degrees of freedom. 
        """
        
        m = self._add_objective(m)

        result_doe = self._solve_doe(m, fix=False)

        # extract Jac
        jac_optimize = self._extract_jac(m)

        # create result object
        analysis_optimize = FisherResults(list(self.param.keys()), self.measure, jacobian_info=None, all_jacobian_info=jac_optimize,
                                        prior_FIM=self.prior_FIM)
        # for simultaneous mode, FIM and Jacobian are extracted with extract_FIM()
        analysis_optimize.calculate_FIM(self.design_name, result=result_doe)
        analysis_optimize.model = m

        return analysis_optimize



    def compute_FIM(self, design_object, mode='sequential_finite', FIM_store_name=None, specified_prior=None,
                    tee_opt=True, scale_nominal_param_value=False, scale_constant_value=1,
                    store_output = None, read_output=None, extract_single_model=None,
                    formula='central', step=0.001):
        """
        This function solves a square Pyomo model with fixed design variables to compute the FIM.
        It calculates FIM with sensitivity information from four modes:
            1.  sequential_finite: use finite difference scheme to evaluate sensitivity
            2.  direct_kaug: use k_aug to evaluate sensitivity

        Parameters
        -----------
        design_values:
            designVariable object 
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

        Return
        ------
        FIM_analysis: result summary object of this solve
        """
        
        # save inputs in object
        self.design_values = design_object.special_set_value
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

        square_timer = TicTocTimer()
        square_timer.tic(msg=None)
        if self.mode=='sequential_finite':
            FIM_analysis = self._sequential_finite(read_output, extract_single_model, store_output)

        elif self.mode =='direct_kaug':
            FIM_analysis = self._direct_kaug()
            
        else:
            raise ValueError(self.mode+' is not a valid mode. Choose from "sequential_finite" and "direct_kaug".')
        
        dT = square_timer.toc(msg=None)
        self.logger.info("elapsed time: %0.1f"%dT)
        
        return FIM_analysis

    def _sequential_finite(self, read_output, extract_single_model, store_output):
        """ Sequential_finite mode uses Pyomo Block to evaluate the sensitivity information.
        """

        # if measurements are provided
        if read_output:
            with open(read_output, 'rb') as f:
                output_record = pickle.load(f)
                f.close()
            jac = self._finite_calculation(output_record, scena_gen)

        # if measurements are not provided
        else:
            mod = self._create_block()

            # dict for storing model outputs
            output_record = {}

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

                # extract variable values 
                for r in self.measure_name:
                    cuid = pyo.ComponentUID(r)
                    var_up = cuid.find_component_on(mod.block[s])
                    output_iter.append(pyo.value(var_up))

                output_record[s] = output_iter

                output_record['design'] = self.design_values
                
                if store_output:
                    f = open(store_output, 'wb')
                    pickle.dump(output_record, f)
                    f.close()

            # calculate jacobian
            jac = self._finite_calculation(output_record, self.scena_gen)

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
        # create model
        mod = self.create_model(model_option="parmest")

        # discretize if needed
        if self.discretize_model:
            mod = self.discretize_model(mod, block=False)

        # add objective function
        mod.Obj = pyo.Objective(expr=0, sense=pyo.minimize)

        # set ub and lb to parameters
        for par in list(self.param.keys()):
            component = getattr(mod, par)
            component.setlb(self.param[par])
            component.setub(self.param[par])

        # generate parameter name list and value dictionary with index
        var_name = []
        for name in list(self.param.keys()):
            # [0] is the scenario index
            var_name.append(name)

        # call k_aug get_dsdp function
        square_result = self._solve_doe(mod, fix=True)
        dsdp_re, col = get_dsdp(mod, list(self.param.keys()), self.param, tee=self.tee_opt)

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
        for mname in self.measure_name:
            try: 
                kaug_no = col.index(mname)
                measurement_index.append(kaug_no)
                # get right line of dsdp
                dsdp_extract.append(dsdp_array[kaug_no])
            except: 
                # k_aug does not provide value for fixed variables
                self.logger.debug('The variable is fixed:  %s', mname)
                # for fixed variables, the sensitivity are a zero vector
                dsdp_extract.append(zero_sens)

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
        
        return FIM_analysis

    def _create_block(self):

        # create scenario information for block scenarios
        scena_object = ScenarioGenerator(self.param, formula=self.formula, step=self.step)
        scena_gen = scena_object.generate_scenario()
            
        # a list of dictionary, each one is a parameter dictionary with perturbed parameter values
        self.scenario_list = scena_gen["scenario"]
        # dictionary, keys are parameter name, values are a list of scenario index where this parameter is perturbed.
        self.scenario_num = scena_gen["scena_num"]
        # dictionary, keys are parameter name, values are the perturbation step 
        self.eps_abs = scena_gen["eps-abs"]
        self.scena_gen = scena_gen

        # Create a global model 
        mod = pyo.ConcreteModel()

        # Set for block/scenarios
        mod.scena = pyo.Set(initialize=list(range(len(self.scenario_list))))

        # Allow user to self-define complex design variables
        self.create_model(mod=mod, model_option="global")

        def block_build(b,s):
            # create block scenarios
            self.create_model(mod=b, model_option="block")
            
            # fix parameter values to perturbed values
            for par in self.param:
                par_strname = eval('b.'+str(par))
                par_strname.fix(scena_gen["scenario"][s][par])

        mod.block = pyo.Block(mod.scena, rule=block_build)

        # discretize the model        
        if self.discretize_model:
            mod = self.discretize_model(mod)

        # force design variables in blocks to be equal to global design values
        for name in self.design_name:
            
            def fix1(mod, s):
                cuid = pyo.ComponentUID(name)
                design_var_global = cuid.find_component_on(mod)
                design_var_global = cuid.find_component_on(mod)
                design_var = cuid.find_component_on(mod.block[s])
                return design_var == design_var_global
            
            con_name = "con"+name
            mod.add_component(con_name, pyo.Constraint(mod.scena, expr=fix1)) 

        return mod 


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
        Extract jacobian from the stochastic program 
        Arguments
        ---------
        m: solved stochastic program model
        Returns
        ------
        JAC: the overall jacobian as a dictionary
        """
        # dictionary form of jacobian
        jac = {}
        # loop over parameters
        for p in list(self.param.keys()): 
            jac_para = []
            for res in m.res:
                jac_para.append(pyo.value(m.jac[p, res]))
            jac[p] = jac_para
        return jac

    def run_grid_search(self, design_object, design_ranges, design_dimension_names, 
                     mode='sequential_finite', tee_option=False, 
                    scale_nominal_param_value=False, scale_constant_value=1, store_name= None, read_name=None,
                        filename=None, formula='central', step=0.001):
        """
        Enumerate through full grid search for any number of design variables;
        solve square problems sequentially to compute FIMs.
        It calculates FIM with sensitivity information from two modes:
            1.  sequential_finite: Calculates a one scenario model multiple times for multiple scenarios. 
            Sensitivity info estimated by finite difference
            2.  direct_kaug: calculate sensitivity by k_aug with direct sensitivity

        Parameters
        -----------
        design_values:
            designvariable object
        design_ranges:
            a ``list`` of design variable values to go over
        design_dimension_names:
            a ``list`` of design variable names of each design range
        mode:
            use mode='sequential_finite', 'direct_kaug'
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
        # Set the Objective Function to 0 helps solve square problem quickly
        self.objective_option='zero'
        self.filename = filename

        # calculate how much the FIM element is scaled
        self.fim_scale_constant_value = scale_constant_value ** 2

        # to store all FIM results
        result_combine = {}

        # iteration 0
        count = 0
        failed_count = 0
        # how many sets of design variables will be run
        total_count = 1
        for rng in design_ranges:
            total_count *= len(rng)

        time_set = [] # record time for every iteration

        # generate combinations of design variable values to go over
        search_design_set = product(*design_ranges)

        # loop over design value combinations
        for design_set_iter in search_design_set:
            # generate the design variable dictionary needed for running compute_FIM
            # first copy value from design_values
            design_iter = design_object.special_set_value.copy()

            # update the controlled value of certain time points for certain design variables
            for i in range(len(design_dimension_names)):
                names = design_dimension_names[i]
                # if the element is a list, all design variables in this list share the same values
                if type(names) is list:
                    for n in names:
                        design_iter[n] = list(design_set_iter)[i] 
                else:
                    design_iter[names] = list(design_set_iter)[i]

            design_object.special_set_value = design_iter

            iter_timer = TicTocTimer()
            self.logger.info('=======Iteration Number: %s =====', count+1)
            self.logger.debug('Design variable values of this iteration: %s', design_iter)
            iter_timer.tic(msg=None)
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
                result_iter = self.compute_FIM(design_object, mode=mode,
                                                tee_opt=tee_option,
                                                scale_nominal_param_value=scale_nominal_param_value,
                                                scale_constant_value = scale_constant_value,
                                                store_output=store_output_name, read_output=read_input_name,
                                                formula=formula, step=step)

                count += 1

                result_iter.calculate_FIM(self.design_values)

                # iteration time
                iter_t = iter_timer.toc(msg=None)
                time_set.append(iter_t)

                # give run information at each iteration
                self.logger.info('This is the  %s run out of  %s run.', (count+1), total_count)
                self.logger.info('The code has run  %s seconds.', sum(time_set))
                self.logger.info('Estimated remaining time:  %s seconds', (sum(time_set)/(count+1)*(total_count-count-1)))

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
        figure_draw_object = GridSearchResult(design_ranges, design_dimension_names, result_combine, store_optimality_name=filename)
        
        self.logger.info('Overall wall clock time [s]:  %s', sum(time_set))

        return figure_draw_object


    def _create_doe_model(self, no_obj=True):
        """
        Add equations to compute sensitivities, FIM, and objective. 

        Parameters:
        -----------
        no_obj: if True, objective function is 0.

        Return:
        -------
        m: the DOE model
        """
        mod = self._create_block()

        # variables for jacobian and FIM
        mod.param = pyo.Set(initialize=list(self.param.keys()))
        mod.res = pyo.Set(initialize=self.measure_name)

        def identity_matrix(m,i,j):
            if i==j:
                return 1 
            else:
                return 0 
        
        mod.jac = pyo.Var(mod.param, mod.res, initialize=0.1)

        if self.fim_initial:
            dict_fim = {}
            for i, bu in enumerate(mod.param):
                for j, un in enumerate(mod.param):
                    dict_fim[(bu,un)]=self.fim_initial[i][j]

        def initialize_fim(m,j,d):
            return dict_fim[(j,d)]

        if self.fim_initial:
            mod.fim = pyo.Var(mod.param, mod.param, initialize=initialize_fim)
        else:
            mod.fim = pyo.Var(mod.param, mod.param, initialize=identity_matrix)

        # move the L matrix initial point to a dictionary
        if type(self.L_initial) != type(None):
            dict_cho={}
            for i, bu in enumerate(mod.param):
                for j, un in enumerate(mod.param):
                    dict_cho[(bu,un)] = self.L_initial[i][j]
        # use the L dictionary to initialize L matrix
        def init_cho(m,i,j):
            return dict_cho[(i,j)]

        # if cholesky, define L elements as variables
        if self.Cholesky_option:
            # Define elements of Cholesky decomposition matrix as Pyomo variables and either
            # Initialize with L in L_initial
            if type(self.L_initial) != type(None):
                mod.L_ele = pyo.Var(mod.param, mod.param, initialize=init_cho)
            # or initialize with the identity matrix
            else:
                mod.L_ele = pyo.Var(mod.param, mod.param, initialize=identity_matrix)

            # loop over parameter name
            for i, c in enumerate(mod.param):
                for j, d in enumerate(mod.param):
                    # fix the 0 half of L matrix to be 0.0
                    if i < j:
                        mod.L_ele[c,d].fix(0.0)
                    # Give LB to the diagonal entries 
                    if self.L_LB:
                        if c==d:
                            mod.L_ele[c,d].setlb(self.L_LB)

        # jacobian rule
        def jacobian_rule(m, p, n):
            """
            p: parameter 
            n: response
            """
            cuid = pyo.ComponentUID(n)
            var_up = cuid.find_component_on(m.block[self.scenario_num[p][0]])
            var_lo = cuid.find_component_on(m.block[self.scenario_num[p][1]])
            if self.scale_nominal_param_value:
                return m.jac[p,n] == (var_up-var_lo)/self.eps_abs[p]*self.param[p]*self.scale_constant_value
            else:
                return m.jac[p,n] == (var_up-var_lo)/self.eps_abs[p]*self.scale_constant_value
            
        #A constraint to calculate elements in Hessian matrix
        # transfer prior FIM to be Expressions
        dict_fele={}
        for i, bu in enumerate(mod.param):
            for j, un in enumerate(mod.param):
                dict_fele[(bu,un)] = self.prior_FIM[i][j]

        def read_prior(m,i,j):
            return dict_fele[(i,j)]
        mod.priorFIM = pyo.Expression(mod.param, mod.param, rule=read_prior)
        
        def fim_rule(m,p,q):
            """
            p: parameter 
            q: parameter
            """
            return m.fim[p,q] == sum(1/self.measure.variance[n]*m.jac[p,n]*m.jac[q,n] for n in mod.res) + m.priorFIM[p,q]*self.fim_scale_constant_value 
        
        mod.jac_const = pyo.Constraint(mod.param, mod.res, rule=jacobian_rule)
        mod.fim_const = pyo.Constraint(mod.param, mod.param, rule=fim_rule)

        return mod

    def _add_objective(self, m):

        def cholesky_imp(m, c, d):
            """
            Calculate Cholesky L matrix using algebraic constraints
            """
        # If it is the left bottom half of L
            if (list(self.param.keys()).index(c) >= list(self.param.keys()).index(d)):
                return m.fim[c, d] == sum(
                    m.L_ele[c, list(self.param.keys())[k]] * m.L_ele[d, list(self.param.keys())[k]] for k in range(list(self.param.keys()).index(d) + 1))
            else:
        # This is the empty half of L above the diagonal
                return pyo.Constraint.Skip

        def trace_calc(m):
            """
            Calculate FIM elements. Can scale each element with 1000 for performance
            """
            return m.trace == sum(m.fim[j,j] for j in m.para_set)

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
            det_perm = sum( self._sgn(list_p[d])*sum(m.fim[each, name_order[b]] for b, each in enumerate(m.param)) for d in range(len(list_p)))
            return m.det == det_perm

        if self.Cholesky_option:
            m.cholesky_cons = pyo.Constraint(m.param, m.param, rule=cholesky_imp)
            m.Obj = pyo.Objective(expr=2 * sum(pyo.log(m.L_ele[j, j]) for j in m.param), sense=pyo.maximize)
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
        for i, name in enumerate(self.design_name):
            cuid = pyo.ComponentUID(name)
            var = cuid.find_component_on(m)
            if fix_opt:
                #getattr(m, dname)[time].fix(fix_v)    
                var.fix(design_val[name])
            else:
                if optimize_option is None:
                    var.unfix()
                else:
                    if optimize_option[name]:
                        var.unfix()
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




