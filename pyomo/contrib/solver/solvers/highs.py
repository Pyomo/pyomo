#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import logging
import io
from typing import List, Optional

from pyomo.common.collections import ComponentMap
from pyomo.common.dependencies import attempt_import
from pyomo.common.errors import ApplicationError
from pyomo.common.flags import NOTSET
from pyomo.common.tee import TeeStream, capture_output
from pyomo.core.kernel.objective import minimize, maximize
from pyomo.core.base.var import VarData
from pyomo.core.base.constraint import ConstraintData
from pyomo.core.base.sos import SOSConstraintData
from pyomo.core.base.param import ParamData
from pyomo.core.expr.numvalue import value, is_constant
from pyomo.repn import generate_standard_repn
from pyomo.core.expr.numeric_expr import NPV_MaxExpression, NPV_MinExpression
from pyomo.common.dependencies import numpy as np
from pyomo.core.staleflag import StaleFlagManager

from pyomo.contrib.solver.common.base import PersistentSolverBase, Availability
from pyomo.contrib.solver.common.results import (
    Results,
    TerminationCondition,
    SolutionStatus,
)
from pyomo.contrib.solver.common.config import PersistentBranchAndBoundConfig
from pyomo.contrib.solver.common.persistent import (
    PersistentSolverUtils,
    PersistentSolverMixin,
)
from pyomo.contrib.solver.common.solution_loader import PersistentSolutionLoader
from pyomo.contrib.solver.common.util import (
    NoFeasibleSolutionError,
    NoOptimalSolutionError,
    NoDualsError,
    NoReducedCostsError,
    NoSolutionError,
    IncompatibleModelError,
)

logger = logging.getLogger(__name__)

highspy, highspy_available = attempt_import('highspy')


class _MutableVarBounds:
    def __init__(self, lower_expr, upper_expr, pyomo_var_id, var_map, highs):
        self.pyomo_var_id = pyomo_var_id
        self.lower_expr = lower_expr
        self.upper_expr = upper_expr
        self.var_map = var_map
        self.highs = highs

    def update(self):
        col_ndx = self.var_map[self.pyomo_var_id]
        lb = value(self.lower_expr)
        ub = value(self.upper_expr)
        self.highs.changeColBounds(col_ndx, lb, ub)


class _MutableLinearCoefficient:
    def __init__(self, pyomo_con, pyomo_var_id, con_map, var_map, expr, highs):
        self.expr = expr
        self.highs = highs
        self.pyomo_var_id = pyomo_var_id
        self.pyomo_con = pyomo_con
        self.con_map = con_map
        self.var_map = var_map

    def update(self):
        row_ndx = self.con_map[self.pyomo_con]
        col_ndx = self.var_map[self.pyomo_var_id]
        self.highs.changeCoeff(row_ndx, col_ndx, value(self.expr))


class _MutableObjectiveCoefficient:
    def __init__(self, pyomo_var_id, var_map, expr, highs):
        self.expr = expr
        self.highs = highs
        self.pyomo_var_id = pyomo_var_id
        self.var_map = var_map

    def update(self):
        col_ndx = self.var_map[self.pyomo_var_id]
        self.highs.changeColCost(col_ndx, value(self.expr))


class _MutableQuadraticCoefficient:
    def __init__(self, expr, v1_id, v2_id):
        self.expr = expr
        self.v1_id = v1_id
        self.v2_id = v2_id


class _MutableObjective:
    def __init__(
        self,
        highs,
        constant,
        linear_coefs,
        quadratic_coefs,
        pyomo_var_to_solver_var_map,
    ):
        self.highs = highs
        self.constant = constant
        self.linear_coefs = linear_coefs
        self.quadratic_coefs = quadratic_coefs
        self._pyomo_var_to_solver_var_map = pyomo_var_to_solver_var_map
        # Store the quadratic coefficients in dictionary format
        self._initialize_quad_coef_dicts()
        # Flag to force first update of quadratic coefficients
        self._first_update = True

    def _initialize_quad_coef_dicts(self):
        self.quad_coef_dict = {}
        for coef in self.quadratic_coefs:
            self.quad_coef_dict[(coef.v1_id, coef.v2_id)] = value(coef.expr)
        self.previous_quad_coef_dict = self.quad_coef_dict.copy()

    def update(self):
        """
        Update the quadratic objective expression.
        """
        needs_quadratic_update = self._first_update

        self.constant.update()
        for coef in self.linear_coefs:
            coef.update()

        for coef in self.quadratic_coefs:
            current_val = value(coef.expr)
            previous_val = self.previous_quad_coef_dict.get((coef.v1_id, coef.v2_id))
            if previous_val is not None and current_val != previous_val:
                needs_quadratic_update = True
                self.quad_coef_dict[(coef.v1_id, coef.v2_id)] = current_val
                self.previous_quad_coef_dict[(coef.v1_id, coef.v2_id)] = current_val

        # If anything changed, rebuild and pass the Hessian
        if needs_quadratic_update:
            self._build_and_pass_hessian()
            self._first_update = False

    def _build_and_pass_hessian(self):
        """Build and pass the Hessian to HiGHS in CSC format"""
        if not self.quad_coef_dict:
            return

        dim = self.highs.getNumCol()

        # Build CSC format for the lower triangular part
        hessian_value = []
        hessian_index = []
        hessian_start = [0] * dim

        quad_coef_idx_dict = {}
        for (v1_id, v2_id), coef in self.quad_coef_dict.items():
            v1_ndx = self._pyomo_var_to_solver_var_map[v1_id]
            v2_ndx = self._pyomo_var_to_solver_var_map[v2_id]
            # Ensure we're storing the lower triangular part
            row = max(v1_ndx, v2_ndx)
            col = min(v1_ndx, v2_ndx)
            # Adjust the diagonal to match Highs' expected format
            if v1_ndx == v2_ndx:
                coef *= 2.0
            quad_coef_idx_dict[(row, col)] = coef

        sorted_entries = sorted(
            quad_coef_idx_dict.items(), key=lambda x: (x[0][1], x[0][0])
        )

        last_col = -1
        for (row, col), val in sorted_entries:
            while col > last_col:
                last_col += 1
                if last_col < dim:
                    hessian_start[last_col] = len(hessian_value)

            # Add the entry
            hessian_index.append(row)
            hessian_value.append(val)

        while last_col < dim - 1:
            last_col += 1
            hessian_start[last_col] = len(hessian_value)

        nnz = len(hessian_value)
        status = self.highs.passHessian(
            dim,
            nnz,
            highspy.HessianFormat.kTriangular,
            np.array(hessian_start, dtype=np.int32),
            np.array(hessian_index, dtype=np.int32),
            np.array(hessian_value, dtype=np.double),
        )

        if status != highspy.HighsStatus.kOk:
            logger.warning(
                f"HiGHS returned non-OK status when passing Hessian: {status}"
            )


class _MutableObjectiveOffset:
    def __init__(self, expr, highs):
        self.expr = expr
        self.highs = highs

    def update(self):
        self.highs.changeObjectiveOffset(value(self.expr))


class _MutableConstraintBounds:
    def __init__(self, lower_expr, upper_expr, pyomo_con, con_map, highs):
        self.lower_expr = lower_expr
        self.upper_expr = upper_expr
        self.con = pyomo_con
        self.con_map = con_map
        self.highs = highs

    def update(self):
        row_ndx = self.con_map[self.con]
        lb = value(self.lower_expr)
        ub = value(self.upper_expr)
        self.highs.changeRowBounds(row_ndx, lb, ub)


class Highs(PersistentSolverMixin, PersistentSolverUtils, PersistentSolverBase):
    """
    Interface to HiGHS
    """

    CONFIG = PersistentBranchAndBoundConfig()

    _available = None

    def __init__(self, **kwds):
        treat_fixed_vars_as_params = kwds.pop('treat_fixed_vars_as_params', True)
        PersistentSolverBase.__init__(self, **kwds)
        PersistentSolverUtils.__init__(
            self, treat_fixed_vars_as_params=treat_fixed_vars_as_params
        )
        self._solver_model = None
        self._pyomo_var_to_solver_var_map = {}
        self._pyomo_con_to_solver_con_map = {}
        self._solver_con_to_pyomo_con_map = {}
        self._mutable_helpers = {}
        self._mutable_bounds = {}
        self._last_results_object: Optional[Results] = None
        self._sol = None

    def available(self):
        if highspy_available:
            return Availability.FullLicense
        return Availability.NotFound

    def version(self):
        try:
            version = (
                highspy.HIGHS_VERSION_MAJOR,
                highspy.HIGHS_VERSION_MINOR,
                highspy.HIGHS_VERSION_PATCH,
            )
        except AttributeError:
            # Older versions of Highs do not have the above attributes
            # and the solver version can only be obtained by making
            # an instance of the solver class.
            tmp = highspy.Highs()
            version = (tmp.versionMajor(), tmp.versionMinor(), tmp.versionPatch())

        return version

    def _solve(self):
        config = self._active_config
        timer = config.timer
        options = config.solver_options
        ostreams = [io.StringIO()] + config.tee

        with capture_output(output=TeeStream(*ostreams), capture_fd=True):
            self._solver_model.setOptionValue('log_to_console', True)

            if config.threads is not None:
                self._solver_model.setOptionValue('threads', config.threads)
            if config.time_limit is not None:
                self._solver_model.setOptionValue('time_limit', config.time_limit)
            if config.rel_gap is not None:
                self._solver_model.setOptionValue('mip_rel_gap', config.rel_gap)
            if config.abs_gap is not None:
                self._solver_model.setOptionValue('mip_abs_gap', config.abs_gap)

            for key, option in options.items():
                self._solver_model.setOptionValue(key, option)
            timer.start('optimize')
            if self.version()[:2] >= (1, 8):
                self._solver_model.HandleKeyboardInterrupt = True
            self._solver_model.run()
            timer.stop('optimize')

        return self._postsolve()

    def _process_domain_and_bounds(self, var_id):
        _v, _lb, _ub, _fixed, _domain_interval, _value = self._vars[var_id]
        lb, ub, step = _domain_interval
        if lb is None:
            lb = -highspy.kHighsInf
        if ub is None:
            ub = highspy.kHighsInf
        if step == 0:
            vtype = highspy.HighsVarType.kContinuous
        elif step == 1:
            vtype = highspy.HighsVarType.kInteger
        else:
            raise ValueError(
                f'Unrecognized domain step: {step} (should be either 0 or 1)'
            )
        if _fixed:
            lb = _value
            ub = _value
        else:
            if _lb is not None or _ub is not None:
                if not is_constant(_lb) or not is_constant(_ub):
                    if _lb is None:
                        tmp_lb = -highspy.kHighsInf
                    else:
                        tmp_lb = _lb
                    if _ub is None:
                        tmp_ub = highspy.kHighsInf
                    else:
                        tmp_ub = _ub
                    mutable_bound = _MutableVarBounds(
                        lower_expr=NPV_MaxExpression((tmp_lb, lb)),
                        upper_expr=NPV_MinExpression((tmp_ub, ub)),
                        pyomo_var_id=var_id,
                        var_map=self._pyomo_var_to_solver_var_map,
                        highs=self._solver_model,
                    )
                    self._mutable_bounds[var_id] = (_v, mutable_bound)
            if _lb is not None:
                lb = max(value(_lb), lb)
            if _ub is not None:
                ub = min(value(_ub), ub)

        return lb, ub, vtype

    def _add_variables(self, variables: List[VarData]):
        self._sol = None
        if self._last_results_object is not None:
            self._last_results_object.solution_loader.invalidate()
        lbs = []
        ubs = []
        indices = []
        vtypes = []

        current_num_vars = len(self._pyomo_var_to_solver_var_map)
        for v in variables:
            v_id = id(v)
            lb, ub, vtype = self._process_domain_and_bounds(v_id)
            lbs.append(lb)
            ubs.append(ub)
            vtypes.append(vtype)
            indices.append(current_num_vars)
            self._pyomo_var_to_solver_var_map[v_id] = current_num_vars
            current_num_vars += 1

        self._solver_model.addVars(
            len(lbs), np.array(lbs, dtype=np.double), np.array(ubs, dtype=np.double)
        )
        self._solver_model.changeColsIntegrality(
            len(vtypes), np.array(indices), np.array(vtypes)
        )

    def _add_parameters(self, params: List[ParamData]):
        pass

    def _reinit(self):
        saved_config = self.config
        saved_active_config = self._active_config
        self.__init__(treat_fixed_vars_as_params=self._treat_fixed_vars_as_params)
        self.config = saved_config
        self._active_config = saved_active_config

    def set_instance(self, model):
        config = self._active_config
        ostreams = config.tee

        if self._last_results_object is not None:
            self._last_results_object.solution_loader.invalidate()
        if not self.available():
            c = self.__class__
            raise ApplicationError(
                f'Solver {c.__module__}.{c.__qualname__} is not available '
                f'({self.available()}).'
            )

        with capture_output(TeeStream(*ostreams), capture_fd=True):
            self._reinit()
            self._model = model

            self._solver_model = highspy.Highs()
            self.add_block(model)
            if self._objective is None:
                self.set_objective(None)

    def _add_constraints(self, cons: List[ConstraintData]):
        self._sol = None
        if self._last_results_object is not None:
            self._last_results_object.solution_loader.invalidate()
        current_num_cons = len(self._pyomo_con_to_solver_con_map)
        lbs = []
        ubs = []
        starts = []
        var_indices = []
        coef_values = []

        for con in cons:
            repn = generate_standard_repn(
                con.body, quadratic=False, compute_values=False
            )
            if repn.nonlinear_expr is not None:
                raise IncompatibleModelError(
                    f'Highs interface does not support expressions of degree {repn.polynomial_degree()}'
                )

            starts.append(len(coef_values))
            for ndx, coef in enumerate(repn.linear_coefs):
                v = repn.linear_vars[ndx]
                v_id = id(v)
                coef_val = value(coef)
                if not is_constant(coef):
                    mutable_linear_coefficient = _MutableLinearCoefficient(
                        pyomo_con=con,
                        pyomo_var_id=v_id,
                        con_map=self._pyomo_con_to_solver_con_map,
                        var_map=self._pyomo_var_to_solver_var_map,
                        expr=coef,
                        highs=self._solver_model,
                    )
                    if con not in self._mutable_helpers:
                        self._mutable_helpers[con] = []
                    self._mutable_helpers[con].append(mutable_linear_coefficient)
                    if coef_val == 0:
                        continue
                var_indices.append(self._pyomo_var_to_solver_var_map[v_id])
                coef_values.append(coef_val)

            if con.has_lb():
                lb = con.lower - repn.constant
            else:
                lb = -highspy.kHighsInf
            if con.has_ub():
                ub = con.upper - repn.constant
            else:
                ub = highspy.kHighsInf

            if not is_constant(lb) or not is_constant(ub):
                mutable_con_bounds = _MutableConstraintBounds(
                    lower_expr=lb,
                    upper_expr=ub,
                    pyomo_con=con,
                    con_map=self._pyomo_con_to_solver_con_map,
                    highs=self._solver_model,
                )
                if con not in self._mutable_helpers:
                    self._mutable_helpers[con] = [mutable_con_bounds]
                else:
                    self._mutable_helpers[con].append(mutable_con_bounds)

            lbs.append(value(lb))
            ubs.append(value(ub))
            self._pyomo_con_to_solver_con_map[con] = current_num_cons
            self._solver_con_to_pyomo_con_map[current_num_cons] = con
            current_num_cons += 1

        self._solver_model.addRows(
            len(lbs),
            np.array(lbs, dtype=np.double),
            np.array(ubs, dtype=np.double),
            len(coef_values),
            np.array(starts),
            np.array(var_indices),
            np.array(coef_values, dtype=np.double),
        )

    def _add_sos_constraints(self, cons: List[SOSConstraintData]):
        if cons:
            raise NotImplementedError(
                'Highs interface does not support SOS constraints'
            )

    def _remove_constraints(self, cons: List[ConstraintData]):
        self._sol = None
        if self._last_results_object is not None:
            self._last_results_object.solution_loader.invalidate()
        indices_to_remove = []
        for con in cons:
            con_ndx = self._pyomo_con_to_solver_con_map.pop(con)
            del self._solver_con_to_pyomo_con_map[con_ndx]
            indices_to_remove.append(con_ndx)
            self._mutable_helpers.pop(con, None)
        self._solver_model.deleteRows(
            len(indices_to_remove), np.array(list(sorted(indices_to_remove)))
        )
        con_ndx = 0
        new_con_map = {}
        for c in self._pyomo_con_to_solver_con_map:
            new_con_map[c] = con_ndx
            con_ndx += 1
        self._pyomo_con_to_solver_con_map.clear()
        self._pyomo_con_to_solver_con_map.update(new_con_map)
        self._solver_con_to_pyomo_con_map.clear()
        self._solver_con_to_pyomo_con_map.update(
            {v: k for k, v in self._pyomo_con_to_solver_con_map.items()}
        )

    def _remove_sos_constraints(self, cons: List[SOSConstraintData]):
        if cons:
            raise NotImplementedError(
                'Highs interface does not support SOS constraints'
            )

    def _remove_variables(self, variables: List[VarData]):
        self._sol = None
        if self._last_results_object is not None:
            self._last_results_object.solution_loader.invalidate()
        indices_to_remove = []
        for v in variables:
            v_id = id(v)
            v_ndx = self._pyomo_var_to_solver_var_map.pop(v_id)
            indices_to_remove.append(v_ndx)
            self._mutable_bounds.pop(v_id, None)
        indices_to_remove.sort()
        self._solver_model.deleteVars(
            len(indices_to_remove), np.array(list(sorted(indices_to_remove)))
        )
        v_ndx = 0
        new_var_map = {}
        for v_id in self._pyomo_var_to_solver_var_map:
            new_var_map[v_id] = v_ndx
            v_ndx += 1
        self._pyomo_var_to_solver_var_map.clear()
        self._pyomo_var_to_solver_var_map.update(new_var_map)

    def _remove_parameters(self, params: List[ParamData]):
        pass

    def _update_variables(self, variables: List[VarData]):
        self._sol = None
        if self._last_results_object is not None:
            self._last_results_object.solution_loader.invalidate()
        indices = []
        lbs = []
        ubs = []
        vtypes = []

        for v in variables:
            v_id = id(v)
            self._mutable_bounds.pop(v_id, None)
            v_ndx = self._pyomo_var_to_solver_var_map[v_id]
            lb, ub, vtype = self._process_domain_and_bounds(v_id)
            lbs.append(lb)
            ubs.append(ub)
            vtypes.append(vtype)
            indices.append(v_ndx)

        self._solver_model.changeColsBounds(
            len(indices),
            np.array(indices),
            np.array(lbs, dtype=np.double),
            np.array(ubs, dtype=np.double),
        )
        self._solver_model.changeColsIntegrality(
            len(indices), np.array(indices), np.array(vtypes)
        )

    def update_parameters(self):
        self._sol = None
        if self._last_results_object is not None:
            self._last_results_object.solution_loader.invalidate()

        for con, helpers in self._mutable_helpers.items():
            for helper in helpers:
                helper.update()
        for k, (v, helper) in self._mutable_bounds.items():
            helper.update()

        self._mutable_objective.update()

    def _set_objective(self, obj):
        self._sol = None
        if self._last_results_object is not None:
            self._last_results_object.solution_loader.invalidate()
        n = len(self._pyomo_var_to_solver_var_map)
        indices = np.arange(n)
        costs = np.zeros(n, dtype=np.double)

        # Initialize empty lists for all coefficient types
        mutable_linear_coefficients = []
        mutable_quadratic_coefficients = []

        if obj is None:
            sense = highspy.ObjSense.kMinimize
            mutable_constant = _MutableObjectiveOffset(expr=0, highs=self._solver_model)
        else:
            if obj.sense == minimize:
                sense = highspy.ObjSense.kMinimize
            elif obj.sense == maximize:
                sense = highspy.ObjSense.kMaximize
            else:
                raise ValueError(f'Objective sense is not recognized: {obj.sense}')

            repn = generate_standard_repn(
                obj.expr, quadratic=True, compute_values=False
            )
            if repn.nonlinear_expr is not None or repn.polynomial_degree() > 2:
                raise IncompatibleModelError(
                    f'Highs interface does not support expressions of degree {repn.polynomial_degree()}'
                )

            for coef, v in zip(repn.linear_coefs, repn.linear_vars):
                v_id = id(v)
                v_ndx = self._pyomo_var_to_solver_var_map[v_id]
                costs[v_ndx] = value(coef)
                if not is_constant(coef):
                    mutable_objective_coef = _MutableObjectiveCoefficient(
                        pyomo_var_id=v_id,
                        var_map=self._pyomo_var_to_solver_var_map,
                        expr=coef,
                        highs=self._solver_model,
                    )
                    mutable_linear_coefficients.append(mutable_objective_coef)

            mutable_constant = _MutableObjectiveOffset(
                expr=repn.constant, highs=self._solver_model
            )

            if repn.quadratic_vars and len(repn.quadratic_vars) > 0:
                for ndx, (v1, v2) in enumerate(repn.quadratic_vars):
                    coef = repn.quadratic_coefs[ndx]

                    mutable_quadratic_coefficients.append(
                        _MutableQuadraticCoefficient(
                            expr=coef, v1_id=id(v1), v2_id=id(v2)
                        )
                    )

        self._solver_model.changeObjectiveSense(sense)
        self._solver_model.changeColsCost(n, indices, costs)
        self._mutable_objective = _MutableObjective(
            self._solver_model,
            mutable_constant,
            mutable_linear_coefficients,
            mutable_quadratic_coefficients,
            self._pyomo_var_to_solver_var_map,
        )
        self._mutable_objective.update()

    def _postsolve(self):
        config = self._active_config
        timer = config.timer
        timer.start('load solution')

        highs = self._solver_model
        status = highs.getModelStatus()

        results = Results()
        results.solution_loader = PersistentSolutionLoader(self)
        results.timing_info.highs_time = highs.getRunTime()

        self._sol = highs.getSolution()
        has_feasible_solution = self._sol.value_valid
        if status == highspy.HighsModelStatus.kOptimal:
            results.solution_status = SolutionStatus.optimal
        elif has_feasible_solution:
            results.solution_status = SolutionStatus.feasible
        else:
            results.solution_status = SolutionStatus.noSolution

        if status == highspy.HighsModelStatus.kNotset:
            results.termination_condition = TerminationCondition.unknown
        elif status == highspy.HighsModelStatus.kLoadError:
            results.termination_condition = TerminationCondition.error
        elif status == highspy.HighsModelStatus.kModelError:
            results.termination_condition = TerminationCondition.error
        elif status == highspy.HighsModelStatus.kPresolveError:
            results.termination_condition = TerminationCondition.error
        elif status == highspy.HighsModelStatus.kSolveError:
            results.termination_condition = TerminationCondition.error
        elif status == highspy.HighsModelStatus.kPostsolveError:
            results.termination_condition = TerminationCondition.error
        elif status == highspy.HighsModelStatus.kModelEmpty:
            results.termination_condition = TerminationCondition.unknown
        elif status == highspy.HighsModelStatus.kOptimal:
            results.termination_condition = (
                TerminationCondition.convergenceCriteriaSatisfied
            )
        elif status == highspy.HighsModelStatus.kInfeasible:
            results.termination_condition = TerminationCondition.provenInfeasible
        elif status == highspy.HighsModelStatus.kUnboundedOrInfeasible:
            results.termination_condition = TerminationCondition.infeasibleOrUnbounded
        elif status == highspy.HighsModelStatus.kUnbounded:
            results.termination_condition = TerminationCondition.unbounded
        elif status == highspy.HighsModelStatus.kObjectiveBound:
            results.termination_condition = TerminationCondition.objectiveLimit
        elif status == highspy.HighsModelStatus.kObjectiveTarget:
            results.termination_condition = TerminationCondition.objectiveLimit
        elif status == highspy.HighsModelStatus.kTimeLimit:
            results.termination_condition = TerminationCondition.maxTimeLimit
        elif status == highspy.HighsModelStatus.kIterationLimit:
            results.termination_condition = TerminationCondition.iterationLimit
        elif status == getattr(highspy.HighsModelStatus, "kSolutionLimit", NOTSET):
            # kSolutionLimit was introduced in HiGHS v1.5.3 for MIP-related limits
            results.termination_condition = TerminationCondition.iterationLimit
        elif status == highspy.HighsModelStatus.kUnknown:
            results.termination_condition = TerminationCondition.unknown
        else:
            logger.warning(f'Received unhandled {status=} from solver HiGHS.')
            results.termination_condition = TerminationCondition.unknown

        if (
            results.termination_condition
            != TerminationCondition.convergenceCriteriaSatisfied
            and config.raise_exception_on_nonoptimal_result
        ):
            raise NoOptimalSolutionError()

        results.incumbent_objective = None
        results.objective_bound = None
        info = highs.getInfo()
        if self._objective is not None:
            if has_feasible_solution:
                results.incumbent_objective = info.objective_function_value
            if info.mip_node_count == -1:
                if has_feasible_solution:
                    results.objective_bound = info.objective_function_value
                else:
                    results.objective_bound = None
            else:
                results.objective_bound = info.mip_dual_bound

        if config.load_solutions:
            if has_feasible_solution:
                self._load_vars()
            else:
                raise NoFeasibleSolutionError()
        timer.stop('load solution')

        return results

    def _load_vars(self, vars_to_load=None):
        for v, val in self._get_primals(vars_to_load=vars_to_load).items():
            v.set_value(val, skip_validation=True)
        StaleFlagManager.mark_all_as_stale(delayed=True)

    def _get_primals(self, vars_to_load=None):
        if self._sol is None or not self._sol.value_valid:
            raise NoSolutionError()

        res = ComponentMap()
        if vars_to_load is None:
            var_ids_to_load = []
            for v, ref_info in self._referenced_variables.items():
                using_cons, using_sos, using_obj = ref_info
                if using_cons or using_sos or (using_obj is not None):
                    var_ids_to_load.append(v)
        else:
            var_ids_to_load = [id(v) for v in vars_to_load]

        var_vals = self._sol.col_value

        for v_id in var_ids_to_load:
            v = self._vars[v_id][0]
            v_ndx = self._pyomo_var_to_solver_var_map[v_id]
            res[v] = var_vals[v_ndx]

        return res

    def _get_reduced_costs(self, vars_to_load=None):
        if self._sol is None or not self._sol.dual_valid:
            raise NoReducedCostsError()
        res = ComponentMap()
        if vars_to_load is None:
            var_ids_to_load = list(self._vars.keys())
        else:
            var_ids_to_load = [id(v) for v in vars_to_load]

        var_vals = self._sol.col_dual

        for v_id in var_ids_to_load:
            v = self._vars[v_id][0]
            v_ndx = self._pyomo_var_to_solver_var_map[v_id]
            res[v] = var_vals[v_ndx]

        return res

    def _get_duals(self, cons_to_load=None):
        if self._sol is None or not self._sol.dual_valid:
            raise NoDualsError()

        res = {}
        if cons_to_load is None:
            cons_to_load = list(self._pyomo_con_to_solver_con_map.keys())

        duals = self._sol.row_dual

        for c in cons_to_load:
            c_ndx = self._pyomo_con_to_solver_con_map[c]
            res[c] = duals[c_ndx]

        return res
