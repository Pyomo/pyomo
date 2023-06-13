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


from pyomo.common.dependencies import numpy as np, pandas as pd, matplotlib as plt
from pyomo.core.expr.numvalue import value

from itertools import product
import logging
from pyomo.opt import SolverStatus, TerminationCondition


class FisherResults:
    def __init__(
        self,
        parameter_names,
        measurements,
        jacobian_info=None,
        all_jacobian_info=None,
        prior_FIM=None,
        store_FIM=None,
        scale_constant_value=1,
        max_condition_number=1.0e12,
    ):
        """Analyze the FIM result for a single run

        Parameters
        ----------
        parameter_names:
            A ``list`` of parameter names
        measurements:
            A ``MeasurementVariables`` which contains the Pyomo variable names and their corresponding indices and
            bounds for experimental measurements
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
        """
        self.parameter_names = parameter_names
        self.measurements = measurements
        self.measurement_variables = measurements.variable_names

        if jacobian_info is None:
            self.jaco_information = all_jacobian_info
        else:
            self.jaco_information = jacobian_info
        self.all_jacobian_info = all_jacobian_info

        self.prior_FIM = prior_FIM
        self.store_FIM = store_FIM
        self.scale_constant_value = scale_constant_value
        self.fim_scale_constant_value = scale_constant_value**2
        self.max_condition_number = max_condition_number
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(level=logging.WARN)

    def result_analysis(self, result=None):
        """Calculate FIM from Jacobian information. This is for grid search (combined models) results

        Parameters
        ----------
        result:
            solver status returned by IPOPT
        """
        self.result = result
        self.doe_result = None

        # get number of parameters
        no_param = len(self.parameter_names)

        fim = np.zeros((no_param, no_param))

        # convert dictionary to a numpy array
        Q_all = []
        for par in self.parameter_names:
            Q_all.append(self.jaco_information[par])
        n = len(self.parameter_names)

        Q_all = np.array(list(self.jaco_information[p] for p in self.parameter_names)).T
        # add the FIM for each measurement variables together
        for i, mea_name in enumerate(self.measurement_variables):
            fim += (
                1
                / self.measurements.variance[str(mea_name)]  # variance of measurement
                * (
                    Q_all[i, :].reshape(n, 1) @ Q_all[i, :].reshape(n, 1).T
                )  # Q.T @ Q for each measurement variable
            )

        # add prior information
        if self.prior_FIM is not None:
            try:
                fim = fim + self.prior_FIM
                self.logger.info('Existed information has been added.')
            except:
                raise ValueError('Check the shape of prior FIM.')

        if np.linalg.cond(fim) > self.max_condition_number:
            self.logger.info(
                "Warning: FIM is near singular. The condition number is: %s ;",
                np.linalg.cond(fim),
            )
            self.logger.info(
                'A condition number bigger than %s is considered near singular.',
                self.max_condition_number,
            )

        # call private methods
        self._print_FIM_info(fim)
        if self.result is not None:
            self._get_solver_info()

        # if given store file name, store the FIM
        if self.store_FIM is not None:
            self._store_FIM()

    def subset(self, measurement_subset):
        """Create new FisherResults object corresponding to provided measurement_subset.
        This requires that measurement_subset is a true subset of the original measurement object.

        Parameters
        ----------
        measurement_subset: Instance of Measurements class

        Returns
        -------
        new_result: New instance of FisherResults
        """

        # Check that measurement_subset is a valid subset of self.measurement
        self.measurements.check_subset(measurement_subset)

        # Split Jacobian (should already be 3D)
        small_jac = self._split_jacobian(measurement_subset)

        # create a new subject
        FIM_subset = FisherResults(
            self.parameter_names,
            measurement_subset,
            jacobian_info=small_jac,
            prior_FIM=self.prior_FIM,
            store_FIM=self.store_FIM,
            scale_constant_value=self.scale_constant_value,
            max_condition_number=self.max_condition_number,
        )

        return FIM_subset

    def _split_jacobian(self, measurement_subset):
        """
        Split jacobian

        Parameters
        ----------
        measurement_subset: the object of the measurement subsets

        Returns
        -------
        jaco_info: split Jacobian
        """
        # create a dict for FIM. It has the same keys as the Jacobian dict.
        jaco_info = {}

        # reorganize the jacobian subset with the same form of the jacobian
        # loop over parameters
        for par in self.parameter_names:
            jaco_info[par] = []
            # loop over measurements
            for name in measurement_subset.variable_names:
                try:
                    n_all_measure = self.measurement_variables.index(name)
                    jaco_info[par].append(self.all_jacobian_info[par][n_all_measure])
                except:
                    raise ValueError(
                        "Measurement ", name, " is not in original measurement set."
                    )

        return jaco_info

    def _print_FIM_info(self, FIM):
        """
        using a dictionary to store all FIM information

        Parameters
        ----------
        FIM: the Fisher Information Matrix, needs to be P.D. and symmetric

        Returns
        -------
        fim_info: a FIM dictionary containing the following key:value pairs
            ~['FIM']: a list of FIM itself
            ~[design variable name]: a list of design variable values at each time point
            ~['Trace']: a scalar number of Trace
            ~['Determinant']: a scalar number of determinant
            ~['Condition number:']: a scalar number of condition number
            ~['Minimal eigen value:']: a scalar number of minimal eigen value
            ~['Eigen values:']: a list of all eigen values
            ~['Eigen vectors:']: a list of all eigen vectors
        """
        eig = np.linalg.eigvals(FIM)
        self.FIM = FIM
        self.trace = np.trace(FIM)
        self.det = np.linalg.det(FIM)
        self.min_eig = min(eig)
        self.cond = max(eig) / min(eig)
        self.eig_vals = eig
        self.eig_vecs = np.linalg.eig(FIM)[1]

        self.logger.info(
            'FIM: %s; \n Trace: %s; \n Determinant: %s;', self.FIM, self.trace, self.det
        )
        self.logger.info(
            'Condition number: %s; \n Min eigenvalue: %s.', self.cond, self.min_eig
        )

    def _solution_info(self, m, dv_set):
        """
        Solution information. Only for optimization problem

        Parameters
        ----------
        m: model
        dv_set: design variable dictionary

        Returns
        -------
        model_info: model solutions dictionary containing the following key:value pairs
            -['obj']: a scalar number of objective function value
            -['det']: a scalar number of determinant calculated by the model (different from FIM_info['det'] which
            is calculated by numpy)
            -['trace']: a scalar number of trace calculated by the model
            -[design variable name]: a list of design variable solution
        """
        self.obj_value = value(m.obj)

        # When scaled with constant values, the effect of the scaling factors are removed here
        # For determinant, the scaling factor to determinant is scaling factor ** (Dim of FIM)
        # For trace, the scaling factor to trace is the scaling factor.
        if self.obj == 'det':
            self.obj_det = np.exp(value(m.obj)) / (self.fim_scale_constant_value) ** (
                len(self.parameter_names)
            )
        elif self.obj == 'trace':
            self.obj_trace = np.exp(value(m.obj)) / (self.fim_scale_constant_value)

        design_variable_names = list(dv_set.keys())
        dv_times = list(dv_set.values())

        solution = {}
        for d, dname in enumerate(design_variable_names):
            sol = []
            if dv_times[d] is not None:
                for t, time in enumerate(dv_times[d]):
                    newvar = getattr(m, dname)[time]
                    sol.append(value(newvar))
            else:
                newvar = getattr(m, dname)
                sol.append(value(newvar))

            solution[dname] = sol
        self.solution = solution

    def _store_FIM(self):
        # if given store file name, store the FIM
        store_dict = {}
        for i, name in enumerate(self.parameter_names):
            store_dict[name] = self.FIM[i]
        FIM_store = pd.DataFrame(store_dict)
        FIM_store.to_csv(self.store_FIM, index=False)

    def _get_solver_info(self):
        """
        Solver information dictionary

        Return:
        ------
        solver_status: a solver information dictionary containing the following key:value pairs
            -['square']: a string of square result solver status
            -['doe']: a string of doe result solver status
        """

        if (self.result.solver.status == SolverStatus.ok) and (
            self.result.solver.termination_condition == TerminationCondition.optimal
        ):
            self.status = 'converged'
        elif (
            self.result.solver.termination_condition == TerminationCondition.infeasible
        ):
            self.status = 'infeasible'
        else:
            self.status = self.result.solver.status


class GridSearchResult:
    def __init__(
        self,
        design_ranges,
        design_dimension_names,
        FIM_result_list,
        store_optimality_name=None,
    ):
        """
        This class deals with the FIM results from grid search, providing A, D, E, ME-criteria results for each design variable.
        Can choose to draw 1D sensitivity curves and 2D heatmaps.

        Parameters
        ----------
        design_ranges:
            a ``dict`` whose keys are design variable names, values are a list of design variable values to go over
        design_dimension_names:
            a ``list`` of design variables names
        FIM_result_list:
            a ``dict`` containing FIM results, keys are a tuple of design variable values, values are FIM result objects
        store_optimality_name:
            a .csv file name containing all four optimalities value
        """
        # design variables
        self.design_names = design_dimension_names
        self.design_ranges = design_ranges
        self.FIM_result_list = FIM_result_list

        self.store_optimality_name = store_optimality_name

    def extract_criteria(self):
        """
        Extract design criteria values for every 'grid' (design variable combination) searched.

        Returns
        -------
        self.store_all_results_dataframe: a pandas dataframe with columns as design variable names and A, D, E, ME-criteria names.
            Each row contains the design variable value for this 'grid', and the 4 design criteria value for this 'grid'.
        """

        # a list store all results
        store_all_results = []

        # generate combinations of design variable values to go over
        search_design_set = product(*self.design_ranges)

        # loop over deign value combinations
        for design_set_iter in search_design_set:
            # locate this grid in the dictionary of combined results
            result_object_asdict = {
                k: v for k, v in self.FIM_result_list.items() if k == design_set_iter
            }
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
        for i in self.design_names:
            # if design variables share the same value, use the first name as the column name
            if type(i) is list:
                column_names.append(i[0])
            else:
                column_names.append(i)

        # Each design criteria has a column to store values
        column_names.append('A')
        column_names.append('D')
        column_names.append('E')
        column_names.append('ME')
        # generate the dataframe
        store_all_results = np.asarray(store_all_results)
        self.store_all_results_dataframe = pd.DataFrame(
            store_all_results, columns=column_names
        )
        # if needs to store the values
        if self.store_optimality_name is not None:
            self.store_all_results_dataframe.to_csv(
                self.store_optimality_name, index=False
            )

    def figure_drawing(
        self,
        fixed_design_dimensions,
        sensitivity_dimension,
        title_text,
        xlabel_text,
        ylabel_text,
        font_axes=16,
        font_tick=14,
        log_scale=True,
    ):
        """
        Extract results needed for drawing figures from the overall result dataframe.
        Draw 1D sensitivity curve or 2D heatmap.
        It can be applied to results of any dimensions, but requires design variable values in other dimensions be fixed.

        Parameters
        ----------
        fixed_design_dimensions: a dictionary, keys are the design variable names to be fixed, values are the value of it to be fixed.
        sensitivity_dimension: a list of design variable names to draw figures.
            If only one name is given, a 1D sensitivity curve is drawn
            if two names are given, a 2D heatmap is drawn.
        title_text: name of the figure, a string
        xlabel_text: x label title, a string.
            In a 1D sensitivity curve, it is the design variable by which the curve is drawn.
            In a 2D heatmap, it should be the second design variable in the design_ranges
        ylabel_text: y label title, a string.
            A 1D sensitivity curve does not need it. In a 2D heatmap, it should be the first design variable in the dv_ranges
        font_axes: axes label font size
        font_tick: tick label font size
        log_scale: if True, the result matrix will be scaled by log10

        Returns
        --------
        None
        """
        self.fixed_design_names = list(fixed_design_dimensions.keys())
        self.fixed_design_values = list(fixed_design_dimensions.values())
        self.sensitivity_dimension = sensitivity_dimension

        if len(self.fixed_design_names) + len(self.sensitivity_dimension) != len(
            self.design_names
        ):
            raise ValueError(
                'Error: All dimensions except for those the figures are drawn by should be fixed.'
            )

        if len(self.sensitivity_dimension) not in [1, 2]:
            raise ValueError("Error: Either 1D or 2D figures can be drawn.")

        # generate a combination of logic sentences to filter the results of the DOF needed.
        # an example filter: (self.store_all_results_dataframe["CA0"]==5).
        if len(self.fixed_design_names) != 0:
            filter = ''
            for i in range(len(self.fixed_design_names)):
                filter += '(self.store_all_results_dataframe['
                filter += str(self.fixed_design_names[i])
                filter += ']=='
                filter += str(self.fixed_design_values[i])
                filter += ')'
                if i != (len(self.fixed_design_names) - 1):
                    filter += '&'
            # extract results with other dimensions fixed
            figure_result_data = self.store_all_results_dataframe.loc[eval(filter)]
        # if there is no other fixed dimensions
        else:
            figure_result_data = self.store_all_results_dataframe

        # add results for figures
        self.figure_result_data = figure_result_data

        # if one design variable name is given as DOF, draw 1D sensitivity curve
        if len(sensitivity_dimension) == 1:
            self._curve1D(
                title_text, xlabel_text, font_axes=16, font_tick=14, log_scale=True
            )
        # if two design variable names are given as DOF, draw 2D heatmaps
        elif len(sensitivity_dimension) == 2:
            self._heatmap(
                title_text,
                xlabel_text,
                ylabel_text,
                font_axes=16,
                font_tick=14,
                log_scale=True,
            )

    def _curve1D(
        self, title_text, xlabel_text, font_axes=16, font_tick=14, log_scale=True
    ):
        """
        Draw 1D sensitivity curves for all design criteria

        Parameters
        ----------
        title_text: name of the figure, a string
        xlabel_text: x label title, a string.
            In a 1D sensitivity curve, it is the design variable by which the curve is drawn.
        font_axes: axes label font size
        font_tick: tick label font size
        log_scale: if True, the result matrix will be scaled by log10

        Returns
        --------
        4 Figures of 1D sensitivity curves for each criteria
        """

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
        fig = plt.pyplot.figure()
        plt.pyplot.rc('axes', titlesize=font_axes)
        plt.pyplot.rc('axes', labelsize=font_axes)
        plt.pyplot.rc('xtick', labelsize=font_tick)
        plt.pyplot.rc('ytick', labelsize=font_tick)
        ax = fig.add_subplot(111)
        params = {'mathtext.default': 'regular'}
        # plt.rcParams.update(params)
        ax.plot(x_range, y_range_A)
        ax.scatter(x_range, y_range_A)
        ax.set_ylabel('$log_{10}$ Trace')
        ax.set_xlabel(xlabel_text)
        plt.pyplot.title(title_text + ' - A optimality')
        plt.pyplot.show()

        # Draw D-optimality
        fig = plt.pyplot.figure()
        plt.pyplot.rc('axes', titlesize=font_axes)
        plt.pyplot.rc('axes', labelsize=font_axes)
        plt.pyplot.rc('xtick', labelsize=font_tick)
        plt.pyplot.rc('ytick', labelsize=font_tick)
        ax = fig.add_subplot(111)
        params = {'mathtext.default': 'regular'}
        # plt.rcParams.update(params)
        ax.plot(x_range, y_range_D)
        ax.scatter(x_range, y_range_D)
        ax.set_ylabel('$log_{10}$ Determinant')
        ax.set_xlabel(xlabel_text)
        plt.pyplot.title(title_text + ' - D optimality')
        plt.pyplot.show()

        # Draw E-optimality
        fig = plt.pyplot.figure()
        plt.pyplot.rc('axes', titlesize=font_axes)
        plt.pyplot.rc('axes', labelsize=font_axes)
        plt.pyplot.rc('xtick', labelsize=font_tick)
        plt.pyplot.rc('ytick', labelsize=font_tick)
        ax = fig.add_subplot(111)
        params = {'mathtext.default': 'regular'}
        # plt.rcParams.update(params)
        ax.plot(x_range, y_range_E)
        ax.scatter(x_range, y_range_E)
        ax.set_ylabel('$log_{10}$ Minimal eigenvalue')
        ax.set_xlabel(xlabel_text)
        plt.pyplot.title(title_text + ' - E optimality')
        plt.pyplot.show()

        # Draw Modified E-optimality
        fig = plt.pyplot.figure()
        plt.pyplot.rc('axes', titlesize=font_axes)
        plt.pyplot.rc('axes', labelsize=font_axes)
        plt.pyplot.rc('xtick', labelsize=font_tick)
        plt.pyplot.rc('ytick', labelsize=font_tick)
        ax = fig.add_subplot(111)
        params = {'mathtext.default': 'regular'}
        # plt.rcParams.update(params)
        ax.plot(x_range, y_range_ME)
        ax.scatter(x_range, y_range_ME)
        ax.set_ylabel('$log_{10}$ Condition number')
        ax.set_xlabel(xlabel_text)
        plt.pyplot.title(title_text + ' - Modified E optimality')
        plt.pyplot.show()

    def _heatmap(
        self,
        title_text,
        xlabel_text,
        ylabel_text,
        font_axes=16,
        font_tick=14,
        log_scale=True,
    ):
        """
        Draw 2D heatmaps for all design criteria

        Parameters
        ----------
        title_text: name of the figure, a string
        xlabel_text: x label title, a string.
            In a 2D heatmap, it should be the second design variable in the design_ranges
        ylabel_text: y label title, a string.
            In a 2D heatmap, it should be the first design variable in the dv_ranges
        font_axes: axes label font size
        font_tick: tick label font size
        log_scale: if True, the result matrix will be scaled by log10

        Returns
        --------
        4 Figures of 2D heatmap for each criteria
        """

        # achieve the design variable ranges this figure needs
        # create a dictionary for sensitivity dimensions
        sensitivity_dict = {}
        for i, name in enumerate(self.design_names):
            if name in self.sensitivity_dimension:
                sensitivity_dict[name] = self.design_ranges[i]
            elif name[0] in self.sensitivity_dimension:
                sensitivity_dict[name[0]] = self.design_ranges[i]

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
        fig = plt.pyplot.figure()
        plt.pyplot.rc('axes', titlesize=font_axes)
        plt.pyplot.rc('axes', labelsize=font_axes)
        plt.pyplot.rc('xtick', labelsize=font_tick)
        plt.pyplot.rc('ytick', labelsize=font_tick)
        ax = fig.add_subplot(111)
        params = {'mathtext.default': 'regular'}
        plt.pyplot.rcParams.update(params)
        ax.set_yticks(range(len(yLabel)))
        ax.set_yticklabels(yLabel)
        ax.set_ylabel(ylabel_text)
        ax.set_xticks(range(len(xLabel)))
        ax.set_xticklabels(xLabel)
        ax.set_xlabel(xlabel_text)
        im = ax.imshow(hes_a.T, cmap=plt.pyplot.cm.hot_r)
        ba = plt.pyplot.colorbar(im)
        ba.set_label('log10(trace(FIM))')
        plt.pyplot.title(title_text + ' - A optimality')
        plt.pyplot.show()

        # D-optimality
        fig = plt.pyplot.figure()
        plt.pyplot.rc('axes', titlesize=font_axes)
        plt.pyplot.rc('axes', labelsize=font_axes)
        plt.pyplot.rc('xtick', labelsize=font_tick)
        plt.pyplot.rc('ytick', labelsize=font_tick)
        ax = fig.add_subplot(111)
        params = {'mathtext.default': 'regular'}
        plt.pyplot.rcParams.update(params)
        ax.set_yticks(range(len(yLabel)))
        ax.set_yticklabels(yLabel)
        ax.set_ylabel(ylabel_text)
        ax.set_xticks(range(len(xLabel)))
        ax.set_xticklabels(xLabel)
        ax.set_xlabel(xlabel_text)
        im = ax.imshow(hes_d.T, cmap=plt.pyplot.cm.hot_r)
        ba = plt.pyplot.colorbar(im)
        ba.set_label('log10(det(FIM))')
        plt.pyplot.title(title_text + ' - D optimality')
        plt.pyplot.show()

        # E-optimality
        fig = plt.pyplot.figure()
        plt.pyplot.rc('axes', titlesize=font_axes)
        plt.pyplot.rc('axes', labelsize=font_axes)
        plt.pyplot.rc('xtick', labelsize=font_tick)
        plt.pyplot.rc('ytick', labelsize=font_tick)
        ax = fig.add_subplot(111)
        params = {'mathtext.default': 'regular'}
        plt.pyplot.rcParams.update(params)
        ax.set_yticks(range(len(yLabel)))
        ax.set_yticklabels(yLabel)
        ax.set_ylabel(ylabel_text)
        ax.set_xticks(range(len(xLabel)))
        ax.set_xticklabels(xLabel)
        ax.set_xlabel(xlabel_text)
        im = ax.imshow(hes_e.T, cmap=plt.pyplot.cm.hot_r)
        ba = plt.pyplot.colorbar(im)
        ba.set_label('log10(minimal eig(FIM))')
        plt.pyplot.title(title_text + ' - E optimality')
        plt.pyplot.show()

        # modified E-optimality
        fig = plt.pyplot.figure()
        plt.pyplot.rc('axes', titlesize=font_axes)
        plt.pyplot.rc('axes', labelsize=font_axes)
        plt.pyplot.rc('xtick', labelsize=font_tick)
        plt.pyplot.rc('ytick', labelsize=font_tick)
        ax = fig.add_subplot(111)
        params = {'mathtext.default': 'regular'}
        plt.pyplot.rcParams.update(params)
        ax.set_yticks(range(len(yLabel)))
        ax.set_yticklabels(yLabel)
        ax.set_ylabel(ylabel_text)
        ax.set_xticks(range(len(xLabel)))
        ax.set_xticklabels(xLabel)
        ax.set_xlabel(xlabel_text)
        im = ax.imshow(hes_e2.T, cmap=plt.pyplot.cm.hot_r)
        ba = plt.pyplot.colorbar(im)
        ba.set_label('log10(cond(FIM))')
        plt.pyplot.title(title_text + ' - Modified E-optimality')
        plt.pyplot.show()
