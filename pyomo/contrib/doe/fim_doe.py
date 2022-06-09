#################################################################################################################
# Copyright (c) 2022
# *** Copyright Notice ***
# Pyomo.DOE was produced under the DOE Carbon Capture Simulation Initiative (CCSI), and is
# copyright (c) 2022 by the software owners: TRIAD, LLNS, BERKELEY LAB, PNNL, UT-Battelle, LLC, NOTRE
# DAME, PITT, UT Austin, TOLEDO, WVU, et al. All rights reserved.
# 
# NOTICE. This Software was developed under funding from the U.S. Department of Energy and the U.S.
# Government consequently retains certain rights. As such, the U.S. Government has been granted for itself
# and others acting on its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the Software to
# reproduce, distribute copies to the public, prepare derivative works, and perform publicly and display
# publicly, and to permit other to do so.
# 
# *** License Agreement ***
# 
# Pyomo.DOE Copyright (c) 2022, by the software owners: TRIAD, LLNS, BERKELEY LAB, PNNL, UT-
# Battelle, LLC, NOTRE DAME, PITT, UT Austin, TOLEDO, WVU, et al. All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without modification, are permitted provided
# that the following conditions are met:
# (1) Redistributions of source code must retain the above copyright notice, this list of conditions and the
# following disclaimer.
# (2) Redistributions in binary form must reproduce the above copyright notice, this list of conditions and
# the following disclaimer in the documentation and/or other materials provided with the distribution.
# (3) Neither the name of the Carbon Capture Simulation for Industry Impact, TRIAD, LLNS, BERKELEY LAB,
# PNNL, UT-Battelle, LLC, ORNL, NOTRE DAME, PITT, UT Austin, TOLEDO, WVU, U.S. Dept. of Energy nor
# the names of its contributors may be used to endorse or promote products derived from this software
# without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL
# THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
# THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# 
# You are under no obligation whatsoever to provide any bug fixes, patches, or upgrades to the features,
# functionality or performance of the source code ("Enhancements") to anyone; however, if you choose to
# make your Enhancements available either publicly, or directly to Lawrence Berkeley National Laboratory,
# without imposing a separate written license agreement for such Enhancements, then you hereby grant
# the following license: a non-exclusive, royalty-free perpetual license to install, use, modify, prepare
# derivative works, incorporate into other computer software, distribute, and sublicense such
# enhancements or derivative works thereof, in binary and source code form.
#
# Lead Developers: Jialu Wang and Alexander Dowling, University of Notre Dame
#
#################################################################################################################

#import numpy as np
#import matplotlib.pyplot as plt
from pyomo.common.dependencies import (
    numpy as np, numpy_available,
    pandas as pd, pandas_available,
    scipy, scipy_available,
    matplotlib as plt, matplotlib_available,
)

from pyomo.environ import *
from pyomo.dae import *
#import pandas as pd
import time
import pickle
from itertools import permutations, product
from pyomo.contrib.sensitivity_toolbox.sens import sipopt, sensitivity_calculation, get_dsdp

class Measurements:
    def __init__(self, measurement_index_time, variance=None, ind_string='_index_'):
        '''This class stores measurements' information

        Parameters
        ----------
        measurement_index_time:
            a ``dict``, keys are measurement variable names, values are a dictionary, keys are its extra index, values are its measuring time points, values are a list of measuring time point if there is no extra index for this measurement.
            For e.g., for the kinetics illustrative example, it should be {'C':{'CA':[0,1,..], 'CB':[0,2,...]}, 'k':[0,4,..]},
            so the measurements are C[scenario, 'CA', 0]..., k[scenario, 0]....
        variance:
            a ``dict``, keys are measurement variable names, values are a dictionary, keys are its extra index, values are its variance (a scalar number), values are its variance if there is no extra index for this measurement.
            For e.g., for the kinetics illustrative example, it should be {'C':{'CA': 10, 'CB': 1, 'CC': 2}}.
            If given None, the default is {'C':{'CA': 1, 'CB': 1, 'CC': 1}}.
        ind_string:
            a ''string'', used to flatten the name of variables and extra index. Default is '_index_'.
            For e.g., for {'C':{'CA': 10, 'CB': 1, 'CC': 2}}, the reformulated name is 'C_index_CA'.
        '''
        self.measurement_all_info = measurement_index_time
        self.ind_string = ind_string
        # a list of measurement names
        self.measurement_name = list(measurement_index_time.keys())
        # begin flatten
        self.__name_and_index_generator(self.measurement_all_info)
        self.__generate_flatten_name(self.name_and_index)
        self.__generate_variance(self.flatten_measure_name, variance, self.name_and_index)
        self.__generate_flatten_timeset(self.measurement_all_info, self.flatten_measure_name, self.name_and_index)
        self.__model_measure_name()
        print('All measurements are flattened.')
        print('Flatten measurement name:', self.flatten_measure_name)
        #print('Flatten measurement timeset:', self.flatten_measure_timeset)

        # generate the overall measurement time points set, including the measurement time for all measurements
        flatten_timepoint = list(self.flatten_measure_timeset.values())
        overall_time = []
        for i in flatten_timepoint:
            overall_time += i
            timepoint_overall_set = list(set(overall_time))
        self.timepoint_overall_set = timepoint_overall_set


    def __name_and_index_generator(self, all_info):
        '''
        Generate a dictionary, keys are the variable names, values are the indexes of this variable.
        For e.g., name_and_index = {'C': ['CA', 'CB', 'CC']}
        Parameters
        ---------
        all_info: a dictionary, keys are measurement variable names,
                values are a dictionary, keys are its extra index, values are its measuring time points
                values are a list of measuring time point if there is no extra index for this measurement
            Note: all_info can be the self.measurement_all_info, but does not have to be it.
        '''
        measurement_name = list(all_info.keys())
        # a list of measurement extra indexes
        measurement_extra_index = []
        # a list of measurement names with extra indexes
        extra_measure_name = []
        # check if the measurement has extra indexes
        for i in measurement_name:
            if type(all_info[i]) is dict:
                index_list = list(all_info[i].keys())
                extra_measure_name.append(i)
                measurement_extra_index.append(index_list)
            elif type(all_info[i]) is list:
                measurement_extra_index.append(None)
        # a dictionary, keys are measurement names, values are a list of extra indexes
        name_and_index = {}
        for i, iname in enumerate(measurement_name):
            name_and_index[iname] = measurement_extra_index[i]

        self.name_and_index = name_and_index

    def __generate_flatten_name(self, measure_name_and_index):
        '''Generate measurement flattened names
        Parameters
        ----------
        measure_name_and_index: a dictionary, keys are measurement names, values are lists of extra indexes

        Returns
        ------
        jac_involved_name: a list of flattened measurement names
        '''
        flatten_names = []
        for j in list(measure_name_and_index.keys()):
            if measure_name_and_index[j] is not None: # if it has extra index
                for ind in measure_name_and_index[j]:
                    flatten_name = j + self.ind_string + str(ind)
                    flatten_names.append(flatten_name)
            else:
                flatten_names.append(j)

        self.flatten_measure_name = flatten_names

    def __generate_variance(self, flatten_measure_name, variance, name_and_index):
        '''Generate the variance dictionary
        Parameters
        ----------
        flatten_measure_name: flattened measurement names. For e.g., flattenning {'C':{'CA': 10, 'CB': 1, 'CC': 2}} will be 'C_index_CA', ..., 'C_index_CC'.
        variance:
            a ``dict``, keys are measurement variable names, values are a dictionary, keys are its extra index, values are its variance (a scalar number), values are its variance if there is no extra index for this measurement.
            For e.g., for the kinetics illustrative example, it should be {'C':{'CA': 10, 'CB': 1, 'CC': 2}}.
            If given None, the default is {'C':{'CA': 1, 'CB': 1, 'CC': 1}}.
        '''
        flatten_variance = {}
        for i in flatten_measure_name:
            if variance is None:
                flatten_variance[i] = 1
            else:
                # split the flattened name if needed
                if self.ind_string in i:
                    measure_name = i.split(self.ind_string)[0]
                    measure_index = i.split(self.ind_string)[1]
                    if type(name_and_index[measure_name][0]) is int:
                        measure_index = int(measure_index)
                    flatten_variance[i] = variance[measure_name][measure_index]
                else:
                    flatten_variance[i] = variance[i]
        self.flatten_variance = flatten_variance

    def __generate_flatten_timeset(self, all_info, flatten_measure_name,name_and_index):
        '''
        Generate flatten variables timeset. Return a dict where keys are the flattened variable names,
        values are a list of measurement time.

        '''
        flatten_measure_timeset = {}
        for i in flatten_measure_name:
            # split the flattened name if needed
            if self.ind_string in i:
                measure_name = i.split(self.ind_string)[0]
                measure_index = i.split(self.ind_string)[1]
                if type(name_and_index[measure_name][0]) is int:
                    measure_index = int(measure_index)
                flatten_measure_timeset[i] = all_info[measure_name][measure_index]
            else:
                flatten_measure_timeset[i] = all_info[i]
        self.flatten_measure_timeset = flatten_measure_timeset

    def __model_measure_name(self):
        '''Return pyomo string name 
        '''
        # store pyomo string name
        measurement_names = []
        # loop over measurement name
        for mname in self.flatten_measure_name:
            # check if there is extra index
            if self.ind_string in mname:
                measure_name = mname.split(self.ind_string)[0]
                measure_index = mname.split(self.ind_string)[1]
                for tim in self.flatten_measure_timeset[mname]:
                    # get the measurement name in the model
                    measurement_name = measure_name + '[0,' + measure_index + ',' + str(tim) + ']'
                    measurement_names.append(measurement_name)
            else:
                for tim in self.flatten_measure_timeset[mname]:
                    # get the measurement name in the model
                    measurement_name = mname + '[0,' + str(tim) + ']'
                    measurement_names.append(measurement_name)
        self.model_measure_name = measurement_names

    def SP_measure_name(self, j, t,scenario_all=None, p=None, mode=None, legal_t=True):
        '''Return pyomo string name for different modes
        Arguments
        ---------
        j: flatten measurement name
        t: time 
        scenario_all: all scenario object, only needed for simultaneous finite mode
        p: parameter, only needed for simultaneous finite mode
        mode: mode name, can be 'simultaneous_finite' or 'sequential_finite'
        legal_t: if the time point is legal for this measurement. default is True
        
        Return
        ------
        up_C, lo_C: two measurement pyomo string names for simultaneous mode
        legal_t: if the time point is legal for this measurement 
        string_name: one measurement pyomo string name for sequential 
        '''
        if mode=='simultaneous_finite':
            # check extra index
            if self.ind_string in j:
                measure_name = j.split(self.ind_string)[0]
                measure_index = j.split(self.ind_string)[1]
                if type(self.name_and_index[measure_name][0]) is str:
                    measure_index = '"' + measure_index + '"'
                if t in self.flatten_measure_timeset[j]:
                    up_C = 'm.' + measure_name + '[' + str(scenario_all['jac-index'][p][0]) + ',' + measure_index + ',' + str(t) + ']'
                    lo_C = 'm.' + measure_name + '[' + str(scenario_all['jac-index'][p][1]) + ',' + measure_index + ',' + str(t) + ']'
                else:
                    legal_t = False
            else:
                up_C = 'm.' + j + '[' + str(scenario_all['jac-index'][p][0]) + ',' + str(t) + ']'
                lo_C = 'm.' + j + '[' + str(scenario_all['jac-index'][p][1]) + ',' + str(t) + ']'

            return up_C, lo_C, legal_t
        
        elif mode == 'sequential_finite':
            if self.ind_string in j:
                measure_name = j.split(self.ind_string)[0]
                measure_index = j.split(self.ind_string)[1]
                if type(self.name_and_index[measure_name][0]) is str:
                    measure_index = '"' + measure_index + '"'
                if t in self.flatten_measure_timeset[j]:
                    string_name = 'mod.' + measure_name + '[0,' + str((measure_index)) + ',' + str(t) + ']'
            else:
                string_name = 'mod.' + j + '[0,' + str(t) + ']'

            return string_name


    def check_subset(self,subset, throw_error=True, valid_subset=True):
        '''
        Check if the subset is correctly defined with right name, index and time.

        subset:
            a ''dict'' where measurement name and index are involved in jacobian calculation
        throw_error:
            if the given subset is not a subset of the measurement set, throw error message
        '''
        flatten_subset = subset.flatten_measure_name
        flatten_timeset = subset.flatten_measure_timeset
        # loop over subset measurement names
        for i in flatten_subset:
            # check if subset measurement names are in the overall measurement names
            if i not in self.flatten_measure_name:
                valid_subset = False
                if throw_error:
                    raise ValueError('This is not a legal subset of the measurement overall set!')
            else:
                # check if subset measurement timepoints are in the overall measurement timepoints
                for t in flatten_timeset[i]:
                    if t not in self.flatten_measure_timeset[i]:
                        valid_subset = False
                        if throw_error:
                            raise ValueError('The time of ', t, ' is not included as measurements before.')
        return valid_subset

class DesignOfExperiments:
    def __init__(self, param_init, design_variable_timepoints, measurement_object, create_model, solver=None,
                 prior_FIM=None, discretize_model=None, verbose=True, args=None):
        """This package enables model-based design of experiments analysis with Pyomo. Both direct optimization and enumeration modes are supported.
        NLP sensitivity tools, e.g.,  sipopt and k_aug, are supported to accelerate analysis via enumeration.
        It can be applied to dynamic models, where design variables are controlled throughout the experiment.

        Parameters
        ----------
        param_init:
            A  ``dictionary`` of parameter names and values. If they are an indexed variable, put the variable name and index, such as 'theta["A1"]'. Note: if sIPOPT is used, parameter shouldn't be indexed.
        design_variable_timepoints:
            A ``dictionary`` where keys are design variable names, values are its control time points. If this design var is independent of time (constant), set the time to [0]
        measurement_object:
            A measurement ``object``.
        create_model:
            A  ``function`` that returns the model
        solver:
            A ``solver`` object that User specified, default=None. If not specified, default solver is IPOPT MA57.
        prior_FIM:
            The list of ``param`` represents for Fisher information matrix (FIM) for prior experiments, default=None
        discretize_model:
            A user-specified ``function`` that discretizes the model. Only use with Pyomo.DAE, default=None
        verbose:
            A ``var`` if print statements are made
        args:
            Other arguments as ``var`` of the create_model function, in a list
        """
        
        # parameters
        self.param_init = param_init
        self.param_name = list(param_init.keys())
        self.param_value = list(param_init.values())
        # design variable name
        self.design_timeset = design_variable_timepoints
        self.design_name = list(self.design_timeset.keys())
        # the control time point for each design variable
        self.design_time = list(self.design_timeset.values())
        # create_model()
        self.create_model = create_model
        self.args = args

        # create the measurement information object
        self.measure = measurement_object
        self.flatten_measure_name = self.measure.flatten_measure_name
        self.flatten_variance = self.measure.flatten_variance
        self.flatten_measure_timeset = self.measure.flatten_measure_timeset

        # check if user-defined solver is given
        if solver is not None:
            self.solver = solver
        # if not given, use default solver
        else:
            self.solver = self.__get_default_ipopt_solver()

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
            raise ValueError('Error: Finite difference scheme should be chosen from "central", "forward", "backward" and None.')

        if self.prior_FIM is not None:
            if not (np.shape(self.prior_FIM)[0] == np.shape(self.prior_FIM)[1]):
                raise ValueError('Found wrong prior information matrix shape.')

        if self.scale_nominal_param_value:
            print('Sensitivity information is scaled by its corresponding parameter nominal value.')

        if (self.scale_constant_value != 1):
            print('Sensitivity information is scaled by constant ', self.scale_constant_value, ' times itself.')

        if check_mode:
            if self.mode not in ['simultaneous_finite', 'sequential_finite', 'sequential_sipopt', 'sequential_kaug', 'direct_kaug']:
                raise ValueError('Wrong mode. Choose from "simultaneous_finite", "sequential_finite", "0sequential_sipopt", "sequential_kaug"')



    def optimize_doe(self,  design_values, if_optimize=True, objective_option='det',
                     jac_involved_measurement=None,
                     scale_nominal_param_value=False, scale_constant_value=1, optimize_opt=None, if_Cholesky=False, L_LB = 1E-10, L_initial=None,
                     jac_initial=None, fim_initial=None,
                     formula='central', step=0.001, check=True):
        '''Optimize DOE problem with design variables being the decisions.
        The DOE model is formed invasively and all scenarios are computed simultaneously.
        The function will first fun a square problem with design variable being fixed at
        the given initial points (Objective function being 0), then a square problem with
        design variables being fixed at the given initial points (Objective function being Design optimality),
        and then unfix the design variable and do the optimization.

        Parameters
        -----------
        design_values:
            a ``dict`` where keys are design variable names, values are a dict whose keys are time point and values are the design variable value at that time point
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
            if FIM is P.D., the diagonal element should be positive, so we can set a LB like 1E-10
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
        self.jac_initial = jac_initial
        self.fim_initial = fim_initial
        self.formula = formula
        self.step = step
        self.tee_opt = True

        # calculate how much the FIM element is scaled by a constant number
        # FIM = Jacobian.T@Jacobian, the FIM is scaled by squared value the Jacobian is scaled
        self.fim_scale_constant_value = self.scale_constant_value **2

        # identify measurements involved in calculation
        if jac_involved_measurement is not None:
            self.jac_involved_name = jac_involved_measurement.flatten_measure_name.copy()
            self.timepoint_overall_set = jac_involved_measurement.timepoint_overall_set.copy()
        else:
            self.jac_involved_name = self.flatten_measure_name.copy()
            self.timepoint_overall_set = self.measure.timepoint_overall_set.copy()
            

        # check if inputs are valid
        # simultaneous mode does not need to check mode and dimension of design variables
        if check:
            self.__check_inputs(check_mode=False)

        # build the large DOE pyomo model
        m = self.__create_doe_model(no_obj=True)

        # solve model, achieve results for square problem, and results for optimization problem

        # Solve square problem first
        # result_square: solver result
        time0_solve = time.time()
        result_square = self.__solve_doe(m, fix=True, opt_option=optimize_opt)
        time1_solve = time.time()

        time_solve1 = time1_solve-time0_solve

        # extract Jac
        jac_square = self.__extract_jac(m)

        # create result object
        analysis_square = FIM_result(self.param_name, self.measure, jacobian_info=None, all_jacobian_info=jac_square,
                                     prior_FIM=self.prior_FIM, scale_constant_value=self.scale_constant_value)
        # for simultaneous mode, FIM and Jacobian are extracted with extract_FIM()
        analysis_square.calculate_FIM(self.design_timeset, result=result_square)

        analysis_square.model = m

        self.analysis_square = analysis_square
        analysis_square.solve_time = time_solve1

        if self.optimize:

            m = self.__add_objective(m, deactive_obj=True)

            # solve problem with DOF then
            print('First solve with given objective:')
            result_doe1 = self.__solve_doe(m, fix=True)

            print('Second solve with given objective:')
            time0_solve2 = time.time()
            result_doe = self.__solve_doe(m, fix=False)
            time1_solve2 = time.time()
            time_solve2 = time1_solve2 - time0_solve2

            # extract Jac
            jac_optimize = self.__extract_jac(m)

            # create result object
            analysis_optimize = FIM_result(self.param_name, self.measure, jacobian_info=None, all_jacobian_info=jac_optimize,
                                           prior_FIM=self.prior_FIM)
            # for simultaneous mode, FIM and Jacobian are extracted with extract_FIM()
            analysis_optimize.calculate_FIM(self.design_timeset, result=result_doe)
            analysis_optimize.model = m

            time1 = time.time()
            # record optimization time
            analysis_optimize.solve_time = time_solve2
            analysis_optimize.total_time = time1-time0
            if self.verbose:
                print('Total solve time with simultaneous_finite mode (Wall clock) [s]:', time_solve1 + time_solve2)
                print('Total wall clock time [s]:', time1-time0)

            return analysis_square, analysis_optimize
            print('changed')

        else:

            time1 = time.time()
            # record square problem time
            analysis_square.total_time = time1-time0
            if self.verbose:
                print('Total solve time with simultaneous_finite mode (Wall clock) [s]:', time_solve1)
                print('Total wall clock time [s]:', time1 - time0)

            return analysis_square

    def compute_FIM(self, design_values, mode='sequential_finite', FIM_store_name=None, specified_prior=None,
                    tee_opt=True, scale_nominal_param_value=False, scale_constant_value=1,
                    store_output = None, read_output=None, extract_single_model=None,
                    formula='central', step=0.001,
                    objective_option='det'):
        '''This function solves a square Pyomo model with fixed design variables to compute the FIM.
        It calculates FIM with sensitivity information from four modes:
            1.  sequential_finite: Calculates a one scenario model multiple times for multiple scenarios. Sensitivity info estimated by finite difference
            2.  sequential_sipopt: calculate sensitivity by sIPOPT [Experimental]
            3.  sequential_kaug: calculate sensitivity by k_aug [Experimental]
            4.  direct_kaug: calculate sensitivity by k_aug with direct sensitivity

        Parameters
        -----------
        design_values:
            a ``dict`` where keys are design variable names, values are a dict whose keys are time point and values are the design variable value at that time point
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
        formula:
            choose from 'central', 'forward', 'backward', None. This option is only used for 'sequential_finite' mode.
        step:
            Sensitivity perturbation step size, a fraction between [0,1]. default is 0.001

        Return
        ------
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

        # calculate how much the FIM element is scaled by a constant number
        # As FIM~Jacobian.T@Jacobian, FIM is scaled twice the number the Q is scaled
        self.fim_scale_constant_value = self.scale_constant_value ** 2

        # check inputs valid
        self.__check_inputs(check_mode=True)

        if self.mode=='sequential_finite':
            time00 = time.time()
            no_para = len(self.param_name)

            # if using sequential model
            # call generator function to get scenario dictionary
            scena_gen = Scenario_generator(self.param_init, formula=self.formula, step=self.step)
            scena_gen.generate_sequential_para()

            # if measurements are provided
            if read_output is not None:
                with open(read_output, 'rb') as f:
                    output_record = pickle.load(f)
                    f.close()
                jac = self.__finite_calculation(output_record, scena_gen)

            # if measurements are not provided
            else:
                # dict for storing model outputs
                output_record = {}

                # dict for storing Jacobian
                models = []
                time_allbuild = []
                time_allsolve = []
                # loop over each scenario
                for no_s in (scena_gen.scena_keys):

                    scenario_iter = scena_gen.next_sequential_scenario(no_s)
                    #print('This scenario:', scenario_iter)
                    # create the model
                    # TODO:(long term) add options to create model once and then update. only try this after the
                    # package is completed and unitest is finished
                    time0_build = time.time()
                    mod = self.create_model(scenario_iter, args=self.args)
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
                    models.append(mod)

                    if extract_single_model is not None:
                        mod_name = store_output + str(no_s) + '.csv'
                        dataframe = extract_single_model(mod, square_result)
                        dataframe.to_csv(mod_name)

                    # loop over measurement item and time to store model measurements
                    output_iter = []

                    for j in self.flatten_measure_name:
                        for t in self.flatten_measure_timeset[j]:
                            measure_string_name = self.measure.SP_measure_name(j,t,mode='sequential_finite')
                            C_value = value(eval(measure_string_name))
                            output_iter.append(C_value)

                    output_record[no_s] = output_iter

                    #print('Output this time: ', output_record[no_s])

                output_record['design'] = design_values
                if store_output is not None:
                    f = open(store_output, 'wb')
                    pickle.dump(output_record, f)
                    f.close()

                # calculate jacobian
                jac = self.__finite_calculation(output_record, scena_gen)

                time11 = time.time()
                if self.verbose:
                    print('Build time with sequential_finite mode [s]:', sum(time_allbuild))
                    print('Solve time with sequential_finite mode [s]:', sum(time_allsolve))
                    print('Total wall clock time [s]:', time11-time00)

                # return all models formed
                self.models = models

            # Assemble and analyze results
            if specified_prior is None:
                prior_in_use = self.prior_FIM
            else:
                prior_in_use = specified_prior

            FIM_analysis = FIM_result(self.param_name, self.measure, jacobian_info=None, all_jacobian_info=jac,
                                      prior_FIM=prior_in_use, store_FIM=FIM_store_name, scale_constant_value=self.scale_constant_value)

            # Store the Jacobian information for access by users

            self.jac = jac

            if read_output is None:
                FIM_analysis.build_time = sum(time_allbuild)
                FIM_analysis.solve_time = sum(time_allsolve)

            return FIM_analysis


        elif self.mode in ['sequential_sipopt', 'sequential_kaug']:
            time00 = time.time()
            # create scenario class for a base case
            scena_gen = Scenario_generator(self.param_init, formula=None, step=self.step)
            scenario_all = scena_gen.simultaneous_scenario()

            # sipopt only uses backward difference scheme
            # store measurements for scenarios
            all_perturb_measure = []
            all_base_measure = []
            # store jacobian info
            jac={}

            # if measurements are provided
            # TODO: update this read_output toggle
            if read_output is not None:
                with open(read_output, 'rb') as f:
                    output_record = pickle.load(f)
                    f.close()
                jac = self.__finite_calculation(output_record, scena_gen)

            else:
                # time building time and solving time store list
                time_allbuild = []
                time_allsolve = []
                # loop over parameters
                for pa in range(len(self.param_name)):
                    perturb_mea = []
                    base_mea = []

                    # create model
                    time0_build = time.time()
                    mod = self.create_model(scenario_all, self.args)
                    time1_build = time.time()
                    time_allbuild.append(time1_build - time0_build)

                    # discretize if needed
                    if self.discretize_model is not None:
                        mod = self.discretize_model(mod)

                    # For sIPOPT, fix model DOF
                    if self.mode =='sequential_sipopt':
                        mod = self.__fix_design(mod, self.design_values, fix_opt=True)

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
                    for j in self.flatten_measure_name:
                        # check if this variable needs split name
                        if self.measure.ind_string in j:
                            measure_name = j.split(self.measure.ind_string)[0]
                            measure_index = j.split(self.measure.ind_string)[1]
                            # this is needed for using eval(). if the extra index is 'CA', it converts to "'CA'". only for the extra index as a string
                            if type(measure_index) is str:
                                measure_index_doublequotes = '"' + measure_index + '"'
                            for t in self.flatten_measure_timeset[j]:
                                measure_var = getattr(m_sipopt, measure_name)
                                # check if this variable is fixed
                                if (measure_var[0,measure_index,t].fixed == True):
                                    perturb_value = value(measure_var[0,measure_index,t])
                                else:
                                    # if it is not fixed, record its perturbed value
                                    if self.mode =='sequential_sipopt':
                                        perturb_value = eval('m_sipopt.sens_sol_state_1[m_sipopt.' + measure_name + '[0,'+str(measure_index_doublequotes)+',' + str(t) + ']]')
                                    else:
                                        perturb_value = eval('m_sipopt.' + measure_name + '[0,' +str(measure_index_doublequotes)+',' + str(t) + ']()')

                                # base case values
                                if self.mode == 'sequential_sipopt':
                                    base_value = eval('m_sipopt.' + measure_name + '[0,'+str(measure_index_doublequotes)+',' + str(t) + '].value')
                                else:
                                    base_value = value(eval('mod.' + measure_name + '[0,' +str(measure_index_doublequotes)+','+ str(t) + ']'))

                                perturb_mea.append(perturb_value)
                                base_mea.append(base_value)

                        else:
                            # fetch the measurement variable
                            measure_var = getattr(m_sipopt, j)
                            for t in self.flatten_measure_timeset[j]:
                                if (measure_var[0,t].fixed == True):
                                    perturb_value = value(measure_var[0, t])
                                else:
                                    # if it is not fixed, record its perturbed value
                                    if self.mode == 'sequential_sipopt':
                                        perturb_value = eval('m_sipopt.sens_sol_state_1[m_sipopt.' + j + '[0,' + str(t) + ']]')
                                    else:
                                        perturb_value = eval('m_sipopt.' + j + '[0,' + str(t) + ']()')

                                # base case values
                                if self.mode == 'sequential_sipopt':
                                    base_value = eval('m_sipopt.' + j + '[0,'+ str(t) + '].value')
                                else:
                                    base_value = value(eval('mod.' + j + '[0,'+ str(t) + ']'))

                                perturb_mea.append(perturb_value)
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
            FIM_analysis = FIM_result(self.param_name, self.measure, jacobian_info=None, all_jacobian_info=jac,
                                      prior_FIM=prior_in_use, store_FIM=FIM_store_name, scale_constant_value=self.scale_constant_value)

            time11 = time.time()
            if self.verbose:
                print('Build time with sequential_sipopt or kaug mode [s]:', sum(time_allbuild))
                print('Solve time with sequential_sipopt or kaug mode [s]:', sum(time_allsolve))
                print('Total wall clock time [s]:', time11-time00)

            self.jac = jac
            FIM_analysis.build_time = sum(time_allbuild)
            FIM_analysis.solve_time = sum(time_allsolve)

            return FIM_analysis

        elif self.mode =='direct_kaug':
            time00 = time.time()
            # create scenario class for a base case
            scena_gen = Scenario_generator(self.param_init, formula=None, step=self.step)
            scenario_all = scena_gen.simultaneous_scenario()

            # create model
            time0_build = time.time()
            mod = self.create_model(scenario_all, args=self.args)
            time1_build = time.time()
            time_build = time1_build - time0_build

            # discretize if needed
            if self.discretize_model is not None:
                mod = self.discretize_model(mod)

            # get all time
            t_all = []
            for t in mod.t:
                t_all.append(t)

            # add objective function
            mod.Obj = Objective(expr=0, sense=minimize)

            # Check if measurement time points are in this time set
            # Also correct the measurement time points
            # For e.g. if a measurement time point is 0.0 in the model but is given as 0, it is corrected here
            measurement_accurate_time = self.flatten_measure_timeset.copy()
            for j in self.flatten_measure_name:
                for no_t, tt in enumerate(self.flatten_measure_timeset[j]):
                    if tt not in t_all:
                        print('A measurement time point not measured by this model: ', tt)
                    else:
                        measurement_accurate_time[j][no_t] = t_all[t_all.index(tt)]

            print('After practice:', measurement_accurate_time)

            # fix model DOF
            #mod = self.__fix_design(mod, self.design_values, fix_opt=True)

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

            # call k_aug get_dsdp function
            time0_solve = time.time()
            square_result = self.__solve_doe(mod, fix=True)
            dsdp_re, col = get_dsdp(mod, var_name, var_dict, tee=self.tee_opt)
            time1_solve = time.time()
            time_solve = time1_solve - time0_solve

            # analyze result
            dsdp_array = dsdp_re.toarray().T
            # here for construction. Remove after finishing.
            #dd = pd.DataFrame(dsdp_array)
            #print(dd)
            #dd.to_csv('test_kaug.csv')
            # here for fixed bed
            self.dsdp = dsdp_array
            self.dsdp = col
            # store dsdp returned
            dsdp_extract = []
            # get right lines from results
            measurement_index = []
            measurement_names = []
            # produce the sensitivity for fixed variables
            zero_sens = np.zeros(len(self.param_name))

            # loop over measurement variables and their time points
            for measurement_name in self.measure.model_measure_name:
                # get right line number in kaug results
                if self.discretize_model is not None:
                    # for DAE model, some variables are fixed
                    try:
                        kaug_no = col.index(measurement_name)
                        measurement_index.append(kaug_no)
                        # get right line of dsdp
                        dsdp_extract.append(dsdp_array[kaug_no])
                    except:
                        if self.verbose:
                            print('The variable is fixed:', measurement_name)
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
            for par in self.param_name:
                jac[par] = []

            for d in range(len(dsdp_extract)):
                for p, par in enumerate(self.param_name):
                    # if scaled by parameter value or constant value
                    if self.scale_nominal_param_value:
                        jac[par].append(self.param_init[par]*dsdp_extract[d][p]*self.scale_constant_value)
                    else:
                        jac[par].append(dsdp_extract[d][p]*self.scale_constant_value)

            time11 = time.time()
            if self.verbose:
                print('Build time with direct kaug mode [s]:', time_build)
                print('Solve time with direct kaug mode [s]:', time_solve)
                print('Total wall clock time [s]:', time11-time00)
                
            # check if another prior experiment FIM is provided other than the user-specified one
            if specified_prior is None:
                prior_in_use = self.prior_FIM
            else:
                prior_in_use = specified_prior

            # Assemble and analyze results
            FIM_analysis = FIM_result(self.param_name,self.measure, jacobian_info=None, all_jacobian_info=jac,
                                      prior_FIM=prior_in_use, store_FIM=FIM_store_name,
                                      scale_constant_value=self.scale_constant_value)
            
            
            self.jac = jac
            FIM_analysis.build_time = time_build
            FIM_analysis.solve_time = time_solve
            

            return FIM_analysis


        else:
            raise ValueError('This is not a valid mode. Choose from "sequential_finite", "simultaneous_finite", "sequential_sipopt", "sequential_kaug"')

    def __finite_calculation(self, output_record, scena_gen):
        '''
        Calculate Jacobian for sequential_finite mode

        Parameters
        ----------
        output_record: output record
        scena_gen: scena_gen generated

        Returns
        --------
        jac: Jacobian matrix, a dictionary, keys are parameter names, values are a list of jacobian values with respect to this parameter
        '''
        # dictionary form of jacobian
        jac = {}

        # After collecting outputs from all scenarios, calculate sensitivity
        for no_p, para in enumerate(self.param_name):
            # extract involved scenario No. for each parameter from scenario class
            involved_s = scena_gen.scenario_para[para]

            # each parameter has two involved scenarios
            s1 = involved_s[0]
            s2 = involved_s[1]
            list_jac = []
            for i in range(len(output_record[s1])):
                if self.scale_nominal_param_value:
                    sensi = (output_record[s1][i] - output_record[s2][i]) / scena_gen.eps_abs[para] * self.param_init[para] * self.scale_constant_value
                else:
                    sensi = (output_record[s1][i] - output_record[s2][i]) / scena_gen.eps_abs[para] * self.scale_constant_value
                list_jac.append(sensi)
            # get Jacobian dict, keys are parameter name, values are sensitivity info
            jac[para] = list_jac

        return jac

    def __extract_jac(self, m):
        '''
        Extract jacobian from simultaneous mode
        Arguments
        ---------
        m: solved simultaneous model
        Returns
        ------
        JAC: the overall jacobian as a dictionary
        '''
        no_para = len(self.param_name)
        # dictionary form of jacobian
        jac = {}
        # loop over parameters
        for p in self.param_name: 
            jac_para = []
            for n1, name1 in enumerate(self.jac_involved_name):
                for t, tim in enumerate(self.timepoint_overall_set):
                    jac_para.append(value(m.jac[name1, p, tim]))
            jac[p] = jac_para
        
        return jac

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
                        tee_option=False, scale_nominal_param_value=False, scale_constant_value=1, store_name= None, read_name=None,
                        filename=None, formula='central', step=0.001):
        """Enumerate through full grid search for any number of design variables;
        solve square problems sequentially to compute FIMs.
        It calculates FIM with sensitivity information from four modes:
            1.  sequential_finite: Calculates a one scenario model multiple times for multiple scenarios. Sensitivity info estimated by finite difference
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
            if IPOPT console output is made
        scale_nominal_param_value:
            if True, the parameters are scaled by its own nominal value in param_init
        scale_constant_value:
            scale all elements in Jacobian matrix, default is 1.
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

            # generate store name
            if store_name is None:
                store_output_name = None
            else:
                store_output_name = store_name + str(count)

            if read_name is not None:
                read_input_name = read_name+str(count)+'_tend'
            else:
                read_input_name = None

            # call compute_FIM to get FIM
            try:
                result_iter = self.compute_FIM(design_iter, mode=mode,
                                               tee_opt=tee_option,
                                               scale_nominal_param_value=scale_nominal_param_value,
                                               scale_constant_value = scale_constant_value,
                                               store_output=store_output_name, read_output=read_input_name,
                                               #extract_single_model=extract3_v2,
                                               formula=formula, step=step)
                if read_input_name is None:
                    build_time_store.append(result_iter.build_time)
                    solve_time_store.append(result_iter.solve_time)

                count += 1

                result_iter.calculate_FIM(self.design_values)

                t_now = time.time()

                if self.verbose:
                    # give run information at each iteration
                    print('This is the ', count+1, ' run out of ', total_count, 'run.')
                    print('The code has run %.04f seconds.'% (t_now-t_enumeration_begin))
                    print('Estimated remaining time: %.4f seconds' % ((t_now-t_enumeration_begin)/(count+1)*(total_count-count-1)))


                # the combined result object are organized as a dictionary, keys are a tuple of the design variable values, values are a result object
                result_combine[tuple(design_set_iter)] = result_iter

            except:
                print(':::::::::::ERROR: Cannot converge this run.::::::::::::')
                count += 1
                failed_count += 1
                print('failed count:', failed_count)
                result_combine[tuple(design_set_iter)] = None

        # For user's access
        self.all_fim = result_combine

        #

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


    def __create_doe_model(self, no_obj=True):
        '''
        Add features for DOE.

        Parameters:
        -----------
        no_obj: if True, objective function is 0.
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
        m = self.create_model(scenario_all, args= self.args)
        # discretize if discretization function is provided
        if self.discretize_model is not None:
            m = self.discretize_model(m)
        
        # extract (discretized) time 
        time_set=[]
        for t in m.t:
            time_set.append(value(t))
        self.time_set = time_set

        # create parameter, measurement, time and measurement time set
        m.para_set = Set(initialize=self.param_name)
        param_name = self.param_name
        m.y_set = Set(initialize=self.jac_involved_name)
        m.t_set = Set(initialize=time_set)

        m.tmea_set = Set(initialize=self.timepoint_overall_set)

        # we can be sure about the name of scenarios, because they are generated by our function
        m.scenario = Set(initialize=scenario_all['scena-name'])
        m.optimize = self.optimize

        # check if measurement time points are in the time set
        for j in m.y_set:
            for t in m.tmea_set:
                if not (t in m.t):
                    raise ValueError('Warning: Measure timepoints should be in the time list.')

        # check if control time points are in the time set
        for d in range(len(self.design_name)):
            if self.design_time[d] is not None:
                for t in self.design_time[d]:
                    if not (t in m.t):
                        raise ValueError('Warning: Control timepoints should be in the time list.')

        ### Define variables
        # Elements in Jacobian matrix
        if self.jac_initial is not None:
            dict_jac = {}
            for i, bu in enumerate(m.y_set):
                for j, un in enumerate(m.para_set):
                    for t, tim in enumerate(m.tmea_set):
                        dict_jac[(bu,un,tim)] = self.jac_initial[i,j,t]

            def jac_initialize(m,i,j,t):
                return dict_jac[(bu,un,tim)]

            m.jac = Var(m.y_set, m.para_set, m.tmea_set, initialize=jac_initialize)

        else:
            m.jac = Var(m.y_set, m.para_set, m.tmea_set, initialize=1E-20)

        # Initialize Hessian with an identity matrix
        def identity_matrix(m,j,d):
            if j==d:
                return 1 
            else: 
                return 0

        # initialize FIM
        if self.fim_initial is not None:
            dict_fim = {}
            for i, bu in enumerate(m.para_set):
                for j, un in enumerate(m.para_set):
                    dict_fim[(bu, un)] = self.fim_initial[i][j]

            def initialize_fim(m, j, d):
                return dict_fim[(j,d)]
            m.FIM = Var(m.para_set, m.para_set, initialize=initialize_fim)
        else:
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
            # check if j is a measurement with extra index by checking if there is '_index_' in its name
            up_C_name, lo_C_name, legal_t_option = self.measure.SP_measure_name(j,t,scenario_all=scenario_all, mode='simultaneous_finite', p=p)
            if legal_t_option:
                up_C = eval(up_C_name)
                lo_C = eval(lo_C_name)
                if self.scale_nominal_param_value:
                    return m.jac[j, p, t] == (up_C - lo_C) / scenario_all['eps-abs'][p] * self.param_init[p] * self.scale_constant_value
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

        #if m.Obj.available():
        #m.Obj.deactivate()

            # Only giving the objective function when there's Degree of freedom. Make OBJ=0 when it's a square problem, which helps converge.
        if no_obj:
            m.Obj = Objective(expr=0)
        else:
            if self.optimize:
                # if cholesky, calculating L and evaluate the OBJ with Cholesky decomposition
                if self.Cholesky_option:
                    m.cholesky_cons = Constraint(m.para_set, m.para_set, rule=cholesky_imp)
                    m.Obj = Objective(expr=2*sum(log(m.L_ele[j,j]) for j in m.para_set), sense=maximize)
                # if not cholesky but determinant, calculating det and evaluate the OBJ with det
                elif (self.objective_option=='det'):
                    m.det_rule =  Constraint(rule=det_general)
                    m.Obj = Objective(expr=log(m.det), sense=maximize)
                # if not determinant or cholesky, calculating the OBJ with trace
                elif (self.objective_option=='trace'):
                    m.trace_rule = Constraint(rule=trace_calc)
                    m.Obj = Objective(expr=log(m.trace), sense=maximize)
                elif (self.objective_option=='zero'):
                    m.Obj = Objective(expr=0)
            #else:
            #    m.Obj = Objective(expr=0)

        return m

    def __add_objective(self, m, deactive_obj= True):

        if deactive_obj:
            m.Obj.deactivate()

        def cholesky_imp(m, c, d):
            '''
            Calculate Cholesky L matrix using algebraic constraints
            '''
        # If it is the left bottom half of L
            if (self.param_name.index(c) >= self.param_name.index(d)):
                return m.FIM[c, d] == sum(
                    m.L_ele[c, self.param_name[k]] * m.L_ele[d, self.param_name[k]] for k in range(self.param_name.index(d) + 1))
            else:
        # This is the empty half of L above the diagonal
                return Constraint.Skip


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
            det_perm = sum( self.__sgn(list_p[d])*sum(m.FIM[each, name_order[b]] for b, each in enumerate(m.para_set)) for d in range(len(list_p)))
            return m.det == det_perm

        if self.Cholesky_option:
            m.cholesky_cons = Constraint(m.para_set, m.para_set, rule=cholesky_imp)
            m.Obj = Objective(expr=2 * sum(log(m.L_ele[j, j]) for j in m.para_set), sense=maximize)
        # if not cholesky but determinant, calculating det and evaluate the OBJ with det
        elif (self.objective_option == 'det'):
            m.det_rule = Constraint(rule=det_general)
            m.Obj = Objective(expr=log(m.det), sense=maximize)
        # if not determinant or cholesky, calculating the OBJ with trace
        elif (self.objective_option == 'trace'):
            m.trace_rule = Constraint(rule=trace_calc)
            m.Obj = Objective(expr=log(m.trace), sense=maximize)
        elif (self.objective_option == 'zero'):
            m.Obj = Objective(expr=0)

        return m

    def __fix_design(self, m, design_val, fix_opt=True, optimize_option=None):
        ''' Fix design variable

        Parameters:
        -----------
        m: model
        design_val: design variable values dict
        fix_opt: if True, fix. Else, unfix
        optimize: a dictionary, keys are design variable name, values are True or False, deciding if this design variable is optimized as DOF this time

        Returns:
        --------
        m: model
        '''
        # loop over the design variables and time index and to fix values specified in design_val
        for d, dname in enumerate(self.design_name):
            # if design variables are indexed by time
            if self.design_time[d] is not None:
                for t, time in enumerate(self.design_time[d]):
                    newvar = eval('m.' + dname + '[' + str(time) + ']')
                    fix_v = design_val[dname][time]

                    if fix_opt:
                        newvar.fix(fix_v)
                        #print(newvar, 'is fixed at ', fix_v)
                    else:
                        if optimize_option is None:
                            newvar.unfix()
                        else:
                            if optimize_option[dname]:
                                newvar.unfix()
            else:
                newvar = eval('m.' + dname)
                fix_v = design_val[dname][0]

                if fix_opt:
                    newvar.fix(fix_v)
                    #print(newvar, 'is fixed at ', fix_v)
                else:
                    newvar.unfix()
        return m

    def __get_default_ipopt_solver(self):
        ''' Default solver
        '''
        solver = SolverFactory('ipopt')
        solver.options['linear_solver'] = 'ma57'
        solver.options['halt_on_ampl_error'] = 'yes'
        solver.options['max_iter'] = 3000
        return solver

    def __solve_doe(self, m, fix=False, opt_option=None):
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
        mod = self.__fix_design(m, self.design_values, fix_opt=fix, optimize_option=opt_option)

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

        Parameters
        -----------
        para_dict:
            a ``dict`` of parameter, keys are names of ''string'', values are their nominal value of ''float''. for e.g., {'A1': 84.79, 'A2': 371.72, 'E1': 7.78, 'E2': 15.05}
        formula:
            choose from 'central', 'forward', 'backward', None.
        step:
            Sensitivity perturbation step size, a fraction between [0,1]. default is 0.001
        store:
            if True, store results.
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
    def __init__(self, para_name, measure_object, jacobian_info=None, all_jacobian_info=None, prior_FIM=None, store_FIM=None, scale_constant_value=1, max_condition_number=1.0E12,
                 verbose=True):
        '''Analyze the FIM result for a single run

        Parameters
        -----------
        para_name:
            A ``list`` of parameter names
        measure_object:
            measurement information object
        jacobian_info:
            the jacobian for this measurement object
        all_jacobian_info:
            the overall jacobian
        prior_FIM:
            if there's prior FIM to be added
        store_FIM:
            if storing the FIM in a .csv or .txt, give the file name here as a string
        scale_constant_value:
            scale all elements in Jacobian matrix, default is 1.
        max_condition_number:
            max condition number
        verbose:
            if True, print statements are used
        '''
        self.para_name = para_name
        self.measure_object = measure_object
        self.measurement_variables = measure_object.measurement_name
        self.measurement_timeset = measure_object.flatten_measure_timeset
        self.flatten_all_measure = measure_object.flatten_measure_name

        if jacobian_info is None:
            self.jaco_information = all_jacobian_info
        else:
            self.jaco_information = jacobian_info
        self.all_jacobian_info = all_jacobian_info

        self.prior_FIM = prior_FIM
        self.store_FIM = store_FIM
        self.scale_constant_value = scale_constant_value
        self.fim_scale_constant_value = scale_constant_value ** 2
        self.max_condition_number = max_condition_number
        self.verbose = verbose

    def calculate_FIM(self, dv_values, result=None):
        '''Calculate FIM from Jacobian information. This is for grid search (combined models) results

        Parameters
        ----------
        dv_values:
            a ``dict`` where keys are design variable names, values are a dict whose keys are time point and values are the design variable value at that time point
        result:
            solver status returned by IPOPT


        Return
        ------
        fim_info: a FIM dictionary
        solver_info: a solver information dictionary
        '''
        self.result = result
        self.doe_result = None

        # get number of parameters
        no_param = len(self.para_name)

        # reform jacobian, split the overall Q into Q_r, each r is a flattened measurement name
        Q_response_list, variance_list = self.__jac_reform_3D(self.jaco_information, Q_response=True)

        fim = np.zeros((no_param, no_param))

        for i in range(len(Q_response_list)):
            #print(1/variance_list[i])
            #print(np.shape(Q_response_list[i]))
            #print(np.shape(Q_response_list[i]@Q_response_list[i].T))
            fim += ((1/variance_list[i])*(Q_response_list[i]@Q_response_list[i].T))
        #print('shape of fim:', np.shape(fim))

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

    def subset(self, measurement_subset):
        ''' Create new FIM_result object corresponding to provided measurement_subset.
    
        This requires that measurement_subset is a true subset of the original measurement object.
    
        Arguments:
            measurement_subset: Instance of Measurements class
    
        Returns:
            new_result: New instance of FIM_result
        '''

        # Check that measurement_subset is a valid subset of self.measurement
        self.measure_object.check_subset(measurement_subset)

        # Split Jacobian (should already be 3D)
        small_jac = self.__split_jacobian(measurement_subset)

        # create a new subject
        FIM_subclass = FIM_result(self.para_name, measurement_subset, jacobian_info=small_jac, prior_FIM=self.prior_FIM, store_FIM=self.store_FIM, scale_constant_value=self.scale_constant_value, max_condition_number=self.max_condition_number)

        return FIM_subclass

    def __split_jacobian(self, measurement_subset):
        '''
        Split jacobian
        Args:
            measure_subclass: the class of the measurement subsets

        Returns:
            jaco_info: splitted Jacobian
        '''
        # create a dict for FIM. It has the same keys as the Jacobian dict.
        jaco_info = {}

        # convert the form of jacobian for split
        jaco_3D = self.__jac_reform_3D(self.jacobian_info)

        involved_flatten_index = measurement_subset.flatten_measure_name
        if self.verbose:
            print('involved flatten name:', involved_flatten_index)

        # reorganize the jacobian subset with the same form of the jacobian
        # loop over parameters
        for p, par in enumerate(self.para_name):
            jaco_info[par] = []
            # loop over flatten measurements
            for n, nam in enumerate(involved_flatten_index):
                if nam in self.flatten_all_measure:
                    #print('get :', nam)
                    n_all_measure = self.flatten_all_measure.index(nam)
                    # loop over time
                    for d in range(len(jaco_3D[n_all_measure, p, :])):
                        jaco_info[par].append(jaco_3D[n_all_measure, p, d])
        return jaco_info

    def __jac_reform_3D(self, jac_original, Q_response=False):
        '''
        Reform the Jacobian returned by __finite_calculation() to be a 3D numpy array, [measurements, parameters, time]
        '''
        # 3-D array form of jacobian [measurements, parameters, time]
        self.measure_timeset = list(self.measurement_timeset.values())[0]
        no_time = len(self.measure_timeset)
        jac_3Darray = np.zeros((len(self.flatten_all_measure), len(self.para_name), no_time))
        # reorganize the matrix
        for m, mname in enumerate(self.flatten_all_measure):
            for p, para in enumerate(self.para_name):
                for t, tim in enumerate(self.measure_timeset):
                    jac_3Darray[m, p, t] = jac_original[para][m * no_time + t]
        if Q_response:
            Qr_list = []
            var_list = []
            for m, mname in enumerate(self.flatten_all_measure):
                Qr_list.append(jac_3Darray[m, :, :])
                var_list.append(self.measure_object.flatten_variance[mname])
            #print(type(Qr_list[0]))
            #print(Qr_list[0])
            #print(var_list)

            return Qr_list, var_list
        else:
            return jac_3Darray


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
            if dv_times[d] is not None:
                for t, time in enumerate(dv_times[d]):
                    newvar = eval('m.' + dname + '[' + str(time) + ']')
                    sol.append(value(newvar))
            else:
                newvar = eval('m.' + dname)
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
        '''This class deals with the FIM results from grid search, providing A, D, E, ME-criteria results for each design variable.
        Can choose to draw 1D sensitivity curves and 2D heatmaps.

        Parameters:
        -----------
        design_ranges:
            a ``dict`` whose keys are design variable names, values are a list of design variable values to go over
        design_dimension_names:
            a ``list`` of design variables names
        design_control_time:
            a ``list`` of design control timesets
        FIM_result_list:
            a ``dict`` containing FIM results, keys are a tuple of design variable values, values are FIM result objects
        store_optimality_name:
            a .csv file name containing all four optimalities value
        verbose:
            if True, print statements
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
        # create a dictionary for sensitivity dimensions
        sensitivity_dict = {}
        for i, nam in enumerate(self.design_names):
            if nam in self.sensitivity_dimension:
                sensitivity_dict[nam] = self.design_ranges[i]

        x_range = sensitivity_dict[self.sensitivity_dimension[0]]
        y_range = sensitivity_dict[self.sensitivity_dimension[1]]


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
    
    Args:
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

    
    
    
