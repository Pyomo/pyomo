#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import logging
import re
import sys
import pyomo.common
from pyutilib.misc import Bunch
from pyutilib.services import TempfileManager
from pyomo.core.expr.numvalue import is_fixed
from pyomo.core.expr.numvalue import value
from pyomo.repn import generate_standard_repn
from pyomo.solvers.plugins.solvers.direct_solver import \
    DirectSolver
from pyomo.solvers.plugins.solvers.direct_or_persistent_solver import \
    DirectOrPersistentSolver
from pyomo.core.kernel.conic import (_ConicBase,
                                     quadratic,
                                     rotated_quadratic,
                                     primal_exponential,
                                     primal_power,
                                     dual_exponential,
                                     dual_power)
from pyomo.core.kernel.objective import minimize, maximize
from pyomo.core.kernel.component_set import ComponentSet
from pyomo.core.kernel.component_map import ComponentMap
from pyomo.opt.results.results_ import SolverResults
from pyomo.opt.results.solution import Solution, SolutionStatus
from pyomo.opt.results.solver import TerminationCondition, SolverStatus
from pyomo.opt.base import SolverFactory
from pyomo.core.base.suffix import Suffix
import pyomo.core.base.var


logger = logging.getLogger('pyomo.solvers')


class DegreeError(ValueError):
    pass


def _is_numeric(x):
    try:
        float(x)
    except ValueError:
        return False
    return True


@SolverFactory.register('mosek', doc='Direct python interface to Mosek')
class MosekDirect(DirectSolver):

    def __init__(self, **kwds):
        kwds['type'] = 'mosek'
        DirectSolver.__init__(self, **kwds)
        self._pyomo_var_to_solver_var_map = ComponentMap()
        self._solver_var_to_pyomo_var_map = ComponentMap()
        self._pyomo_con_to_solver_con_map = dict()
        self._solver_con_to_pyomo_con_map = ComponentMap()
        self._pyomo_cone_to_solver_cone_map = dict()
        self._solver_cone_to_pyomo_cone_map = ComponentMap()
        self._init()

    def _init(self):
        self._name = None
        try:
            import mosek
            self._mosek = mosek
            self._mosek_env = self._mosek.Env()
            self._python_api_exists = True
            self._version = self._mosek_env.getversion()
            if self._version[0] > 8:
                self._name = "Mosek %s.%s.%s" % self._version
                while len(self._version) < 3:
                    self._version += (0,)
            else:
                self._name = "Mosek %s.%s.%s.%s" % self._version
                while len(self._version) < 4:
                    self._version += (0,)

            self._version_major = self._version[0]
        except ImportError:
            self._python_api_exists = False
        except Exception as e:
            print("Import of mosek failed - mosek message=" + str(e) + "\n")
            self._python_api_exists = False

        self._range_constraints = set()

        self._max_obj_degree = 2
        self._max_constraint_degree = 2
        self._termcode = None

        # Note: Undefined capabilites default to None
        self._capabilities.linear = True
        self._capabilities.quadratic_objective = True
        self._capabilities.quadratic_constraint = True
        self._capabilities.integer = True
        self._capabilities.sos1 = False
        self._capabilities.sos2 = False

    @staticmethod
    def license_is_valid():
        """
        Runs a check for a valid Mosek license. Returns False
        if Mosek fails to run on a trivial test case.
        """
        try:
            import mosek
        except ImportError:
            return False
        try:
            mosek.Env().Task(0,0).optimize()
        except mosek.Error:
            return False
        return True

    def _apply_solver(self):
        if not self._save_results:
            for block in self._pyomo_model.block_data_objects(descend_into=True,
                                                              active=True):
                for var in block.component_data_objects(ctype=pyomo.core.base.var.Var,
                                                        descend_into=False,
                                                        active=True,
                                                        sort=False):
                    var.stale = True
        if self._tee:
            def _process_stream(msg):
                sys.stdout.write(msg)
                sys.stdout.flush()
            self._solver_model.set_Stream(
                self._mosek.streamtype.log, _process_stream)

        if self._keepfiles:
            print("Solver log file: "+self._log_file)

        for key, option in self.options.items():

            param = self._mosek

            try:
                for sub_key in key.split('.'):
                    param = getattr(param, sub_key)
            except (TypeError, AttributeError):
                raise

            if 'sparam' in key.split('.'):
                self._solver_model.putstrparam(param, option)
            else:
                if 'iparam' in key.split('.'):
                    self._solver_model.putintparam(param, option)
                elif 'dparam' in key.split('.'):
                    self._solver_model.putdouparam(param, option)
                else:
                    raise AttributeError(
                        "Unknown parameter type. Type sparam, iparam or dparam expected.")

        self._termcode = self._solver_model.optimize()
        self._solver_model.solutionsummary(self._mosek.streamtype.msg)

        # FIXME: can we get a return code indicating if Mosek had a significant failure?
        return Bunch(rc=None, log=None)

    def _get_cone_data(self, con):
        # if the cone is not recognized, this function
        # will return None for cone_type and cone_members
        cone_type = None
        cone_param = 0
        cone_members = None
        if isinstance(con, quadratic):
            assert con.has_ub() and \
                (con.ub == 0) and (not con.has_lb())
            assert con.check_convexity_conditions(relax=True)
            cone_type = self._mosek.conetype.quad
            cone_members = [con.r] + list(con.x)
        elif isinstance(con, rotated_quadratic):
            assert con.has_ub() and \
                (con.ub == 0) and (not con.has_lb())
            assert con.check_convexity_conditions(relax=True)
            cone_type = self._mosek.conetype.rquad
            cone_members = [con.r1, con.r2] + list(con.x)
        elif self._version >= (9, 0, 0):
            if isinstance(con, primal_exponential):
                assert con.has_ub() and \
                    (con.ub == 0) and (not con.has_lb())
                assert con.check_convexity_conditions(
                    relax=False)
                cone_type = self._mosek.conetype.pexp
                cone_members = [con.r, con.x1, con.x2]
            elif isinstance(con, primal_power):
                assert con.has_ub() and \
                    (con.ub == 0) and (not con.has_lb())
                assert con.check_convexity_conditions(
                    relax=False)
                cone_type = self._mosek.conetype.ppow
                cone_param = value(con.alpha)
                cone_members = [con.r1, con.r2] + list(con.x)
            elif isinstance(con, dual_exponential):
                assert con.has_ub() and \
                    (con.ub == 0) and (not con.has_lb())
                assert con.check_convexity_conditions(
                    relax=False)
                cone_type = self._mosek.conetype.dexp
                cone_members = [con.r, con.x1, con.x2]
            elif isinstance(con, dual_power):
                assert con.has_ub() and \
                    (con.ub == 0) and (not con.has_lb())
                assert con.check_convexity_conditions(
                    relax=False)
                cone_type = self._mosek.conetype.dpow
                cone_param = value(con.alpha)
                cone_members = [con.r1, con.r2] + list(con.x)

        return cone_type, cone_param, cone_members

    def _get_expr_from_pyomo_repn(self, repn, max_degree=2):
        referenced_vars = ComponentSet()

        degree = repn.polynomial_degree()
        if (degree is None) or (degree > max_degree):
            raise DegreeError(
                'Mosek does not support expressions of degree {0}.'.format(degree))

        # if len(repn.linear_vars) > 0:
        referenced_vars.update(repn.linear_vars)

        indexes = []
        [indexes.append(self._pyomo_var_to_solver_var_map[i])
         for i in repn.linear_vars]

        new_expr = [list(repn.linear_coefs), indexes, repn.constant]

        qsubi = []
        qsubj = []
        qval = []
        for i, v in enumerate(repn.quadratic_vars):
            x, y = v
            qsubj.append(self._pyomo_var_to_solver_var_map[x])
            qsubi.append(self._pyomo_var_to_solver_var_map[y])
            qval.append(repn.quadratic_coefs[i]*((qsubi==qsubj)+1))
            referenced_vars.add(x)
            referenced_vars.add(y)
        new_expr.extend([qval, qsubi, qsubj])

        return new_expr, referenced_vars

    def _get_expr_from_pyomo_expr(self, expr, max_degree=2):
        if max_degree == 2:
            repn = generate_standard_repn(expr, quadratic=True)
        else:
            repn = generate_standard_repn(expr, quadratic=False)

        try:
            mosek_expr, referenced_vars = self._get_expr_from_pyomo_repn(
                repn, max_degree)
        except DegreeError as e:
            msg = e.args[0]
            msg += '\nexpr: {0}'.format(expr)
            raise DegreeError(msg)

        return mosek_expr, referenced_vars

    def _add_var(self, var):
        varname = self._symbol_map.getSymbol(var, self._labeler)
        vtype = self._mosek_vtype_from_var(var)
        if var.has_lb():
            lb = value(var.lb)
        else:
            lb = '0'
        if var.has_ub():
            ub = value(var.ub)
        else:
            ub = '0'

        bound_type = self.set_var_boundtype(var, ub, lb)
        self._solver_model.appendvars(1)
        index = self._solver_model.getnumvar()-1
        self._solver_model.putvarbound(index, bound_type, float(lb), float(ub))
        self._solver_model.putvartype(index, vtype)
        self._solver_model.putvarname(index, varname)

        self._pyomo_var_to_solver_var_map[var] = index
        self._solver_var_to_pyomo_var_map[index] = var
        self._referenced_variables[var] = 0

    def _set_instance(self, model, kwds={}):
        self._range_constraints = set()
        DirectOrPersistentSolver._set_instance(self, model, kwds)
        self._pyomo_con_to_solver_con_map = dict()
        self._solver_con_to_pyomo_con_map = ComponentMap()
        self._pyomo_cone_to_solver_cone_map = dict()
        self._solver_cone_to_pyomo_cone_map = ComponentMap()
        self._pyomo_var_to_solver_var_map = ComponentMap()
        self._solver_var_to_pyomo_var_map = ComponentMap()
        self._whichsol = getattr(
            self._mosek.soltype, kwds.pop('soltype', 'bas'))

        try:
            self._solver_model = self._mosek_env.Task(0, 0)
        except Exception:
            e = sys.exc_info()[1]
            msg = ("Unable to create Mosek Task. "
                   "Have you installed the Python "
                   "bindings for Mosek?\n\n\t" +
                   "Error message: {0}".format(e))
            raise Exception(msg)

        self._add_block(model)

    def _add_block(self, block):
        DirectOrPersistentSolver._add_block(self, block)

    def _add_constraint(self, con):
        if not con.active:
            return None

        if is_fixed(con.body):
            if self._skip_trivial_constraints:
                return None

        conname = self._symbol_map.getSymbol(con, self._labeler)

        mosek_expr = None
        referenced_vars = None
        cone_type = None
        cone_param = 0
        cone_members = None
        if con._linear_canonical_form:
            mosek_expr, referenced_vars = self._get_expr_from_pyomo_repn(
                con.canonical_form(),
                self._max_constraint_degree)
        elif isinstance(con, _ConicBase):
            cone_type, cone_param, cone_members = \
                self._get_cone_data(con)
            if cone_type is not None:
                assert cone_members is not None
                referenced_vars = ComponentSet(cone_members)
            else:
                logger.warning("Cone %s was not recognized by Mosek"
                               % (str(con)))
                # the cone was not recognized, treat
                # it like a standard constraint, which
                # will in all likelihood lead to Mosek
                # reporting a helpful error message
                assert mosek_expr is None
        if (mosek_expr is None) and (cone_type is None):
            mosek_expr, referenced_vars = \
                self._get_expr_from_pyomo_expr(
                    con.body,
                    self._max_constraint_degree)

        assert referenced_vars is not None
        if mosek_expr is not None:
            assert cone_type is None
            self._solver_model.appendcons(1)
            con_index = self._solver_model.getnumcon()-1
            con_type, ub, lb = self.set_con_bounds(con, mosek_expr[2])

            if con.has_lb():
                if not is_fixed(con.lower):
                    raise ValueError("Lower bound of constraint {0} "
                                     "is not constant.".format(con))
            if con.has_ub():
                if not is_fixed(con.upper):
                    raise ValueError("Upper bound of constraint {0} "
                                     "is not constant.".format(con))

            self._solver_model.putarow(con_index, mosek_expr[1], mosek_expr[0])
            self._solver_model.putqconk(
                con_index, mosek_expr[4], mosek_expr[5], mosek_expr[3])
            self._solver_model.putconbound(con_index, con_type, lb, ub)
            self._solver_model.putconname(con_index, conname)
            self._pyomo_con_to_solver_con_map[con] = con_index
            self._solver_con_to_pyomo_con_map[con_index] = con
        else:
            assert cone_type is not None
            members = [self._pyomo_var_to_solver_var_map[v_]
                       for v_ in cone_members]
            self._solver_model.appendcone(cone_type,
                                          cone_param,
                                          members)
            cone_index = self._solver_model.getnumcone()-1
            self._solver_model.putconename(cone_index, conname)
            self._pyomo_cone_to_solver_cone_map[con] = cone_index
            self._solver_cone_to_pyomo_cone_map[cone_index] = con

        for var in referenced_vars:
            self._referenced_variables[var] += 1
        self._vars_referenced_by_con[con] = referenced_vars

    def _mosek_vtype_from_var(self, var):
        """
        This function takes a pyomo variable and returns the appropriate mosek variable type
        :param var: pyomo.core.base.var.Var
        :return: mosek.variabletype.type_int or mosek.variabletype.type_cont
        """

        if var.is_integer() or var.is_binary():
            vtype = self._mosek.variabletype.type_int
        elif var.is_continuous():
            vtype = self._mosek.variabletype.type_cont
        else:
            raise ValueError(
                'Variable domain type is not recognized for {0}'.format(var.domain))
        return vtype

    def set_var_boundtype(self, var, ub, lb):

        if var.is_fixed():
            return self._mosek.boundkey.fx
        elif ub != '0' and lb != '0':
            return self._mosek.boundkey.ra
        elif ub == '0' and lb == '0':
            return self._mosek.boundkey.fr
        elif ub != '0' and lb == '0':
            return self._mosek.boundkey.up
        return self._mosek.boundkey.lo

    def set_con_bounds(self, con, constant):

        if con.equality:
            ub = value(con.upper) - constant
            lb = value(con.lower) - constant
            con_type = self._mosek.boundkey.fx
        elif con.has_lb() and con.has_ub():
            ub = value(con.upper) - constant
            lb = value(con.lower) - constant
            con_type = self._mosek.boundkey.ra
        elif con.has_lb():
            ub = 0
            lb = value(con.lower) - constant
            con_type = self._mosek.boundkey.lo
        elif con.has_ub():
            ub = value(con.upper) - constant
            lb = 0
            con_type = self._mosek.boundkey.up
        else:
            ub = 0
            lb = 0
            con_type = self._mosek.boundkey.fr
        return con_type, ub, lb

    def _set_objective(self, obj):

        if self._objective is not None:
            for var in self._vars_referenced_by_obj:
                self._referenced_variables[var] -= 1
            self._vars_referenced_by_obj = ComponentSet()
            self._objective = None

        if obj.active is False:
            raise ValueError('Cannot add inactive objective to solver.')

        if obj.sense == minimize:
            self._solver_model.putobjsense(self._mosek.objsense.minimize)
        elif obj.sense == maximize:
            self._solver_model.putobjsense(self._mosek.objsense.maximize)
        else:
            raise ValueError(
                'Objective sense is not recognized: {0}'.format(obj.sense))

        mosek_expr, referenced_vars = self._get_expr_from_pyomo_expr(
            obj.expr, self._max_obj_degree)

        for var in referenced_vars:
            self._referenced_variables[var] += 1

        for i, j in enumerate(mosek_expr[1]):
            self._solver_model.putcj(j, mosek_expr[0][i])

        self._solver_model.putqobj(mosek_expr[4], mosek_expr[5], mosek_expr[3])
        self._solver_model.putcfix(mosek_expr[2])
        self._objective = obj
        self._vars_referenced_by_obj = referenced_vars

    def _postsolve(self):

        extract_duals = False
        extract_slacks = False
        extract_reduced_costs = False
        for suffix in self._suffixes:
            flag = False
            if re.match(suffix, "dual"):
                extract_duals = True
                flag = True
            if re.match(suffix, "slack"):
                extract_slacks = True
                flag = True
            if re.match(suffix, "rc"):
                extract_reduced_costs = True
                flag = True
            if not flag:
                raise RuntimeError(
                    "***The mosek solver plugin cannot extract solution suffix="+suffix)

        msk_task = self._solver_model
        msk = self._mosek

        itr_soltypes = [msk.problemtype.qo,
                        msk.problemtype.qcqo,
                        msk.problemtype.conic]

        if (msk_task.getnumintvar() >= 1):
            self._whichsol = msk.soltype.itg
            if extract_reduced_costs:
                logger.warning("Cannot get reduced costs for MIP.")
            if extract_duals:
                logger.warning("Cannot get duals for MIP.")
            extract_reduced_costs = False
            extract_duals = False
        elif (msk_task.getprobtype() in itr_soltypes):
            self._whichsol = msk.soltype.itr

        whichsol = self._whichsol
        sol_status = msk_task.getsolsta(whichsol)
        pro_status = msk_task.getprosta(whichsol)

        self.results = SolverResults()
        soln = Solution()

        self.results.solver.name = self._name
        self.results.solver.wallclock_time = msk_task.getdouinf(
            msk.dinfitem.optimizer_time)

        SOLSTA_MAP = {
            msk.solsta.unknown: 'unknown',
            msk.solsta.optimal: 'optimal',
            msk.solsta.prim_and_dual_feas: 'pd_feas',
            msk.solsta.prim_feas: 'p_feas',
            msk.solsta.dual_feas: 'd_feas',
            msk.solsta.prim_infeas_cer: 'p_infeas',
            msk.solsta.dual_infeas_cer: 'd_infeas',
            msk.solsta.prim_illposed_cer: 'p_illposed',
            msk.solsta.dual_illposed_cer: 'd_illposed',
            msk.solsta.integer_optimal: 'optimal'
        }
        PROSTA_MAP = {
            msk.prosta.unknown: 'unknown',
            msk.prosta.prim_and_dual_feas: 'pd_feas',
            msk.prosta.prim_feas: 'p_feas',
            msk.prosta.dual_feas: 'd_feas',
            msk.prosta.prim_infeas: 'p_infeas',
            msk.prosta.dual_infeas: 'd_infeas',
            msk.prosta.prim_and_dual_infeas: 'pd_infeas',
            msk.prosta.ill_posed: 'illposed',
            msk.prosta.prim_infeas_or_unbounded: 'p_inf_unb'
        }

        if self._version_major < 9:
            SOLSTA_OLD = {
                msk.solsta.near_optimal: 'optimal',
                msk.solsta.near_integer_optimal: 'optimal',
                msk.solsta.near_prim_feas: 'p_feas',
                msk.solsta.near_dual_feas: 'd_feas',
                msk.solsta.near_prim_and_dual_feas: 'pd_feas',
                msk.solsta.near_prim_infeas_cer: 'p_infeas',
                msk.solsta.near_dual_infeas_cer: 'd_infeas'
            }
            PROSTA_OLD = {
                msk.prosta.near_prim_and_dual_feas: 'pd_feas',
                msk.prosta.near_prim_feas: 'p_feas',
                msk.prosta.near_dual_feas: 'd_feas'
            }
            SOLSTA_MAP.update(SOLSTA_OLD)
            PROSTA_MAP.update(PROSTA_OLD)

        if self._termcode == msk.rescode.ok:
            self.results.solver.status = SolverStatus.ok
            self.results.solver.termination_message = ""

        elif self._termcode == msk.rescode.trm_max_iterations:
            self.results.solver.status = SolverStatus.ok
            self.results.solver.termination_message = "Optimization terminated because the total number " \
                "iterations performed exceeded the value specified in the " \
                "IterationLimit parameter."
            self.results.solver.termination_condition = TerminationCondition.maxIterations
            soln.status = SolutionStatus.stoppedByLimit

        elif self._termcode == msk.rescode.trm_max_time:
            self.results.solver.status = SolverStatus.ok
            self.results.solver.termination_message = "Optimization terminated because the time expended exceeded " \
                "the value specified in the TimeLimit parameter."
            self.results.solver.termination_condition = TerminationCondition.maxTimeLimit
            soln.status = SolutionStatus.stoppedByLimit

        elif self._termcode == msk.rescode.trm_user_callback:
            self.results.solver.status = SolverStatus.Aborted
            self.results.solver.termination_message = "Optimization terminated because of the user callback "
            self.results.solver.termination_condition = TerminationCondition.userInterrupt
            soln.status = SolutionStatus.unknown

        elif self._termcode in [msk.rescode.trm_mio_num_relaxs,
                                msk.rescode.trm_mio_num_branches,
                                msk.rescode.trm_num_max_num_int_solutions]:
            self.results.solver.status = SolverStatus.ok
            self.results.solver.termination_message = "Optimization terminated because maximum number of relaxations" \
                " / branches / integer solutions exceeded " \
                "the value specified in the TimeLimit parameter."
            self.results.solver.termination_condition = TerminationCondition.maxEvaluations
            soln.status = SolutionStatus.stoppedByLimit

        else:
            self.results.solver.termination_message = " Optimization terminated %s response code." \
                "Check Mosek response code documentation for further explanation." % self._termcode
            self.results.solver.termination_condition = TerminationCondition.unknown

        if SOLSTA_MAP[sol_status] == 'unknown':
            self.results.solver.status = SolverStatus.warning
            self.results.solver.termination_message += " Unknown solution status."
            self.results.solver.Message = self.results.solver.termination_message
            self.results.solver.termination_condition = TerminationCondition.unknown
            soln.status = SolutionStatus.unknown

        if PROSTA_MAP[pro_status] == 'd_infeas':
            self.results.solver.status = SolverStatus.warning
            self.results.solver.termination_message += " Problem proven to be dual infeasible"
            self.results.solver.Message = self.results.solver.termination_message
            self.results.solver.termination_condition = TerminationCondition.unbounded
            soln.status = SolutionStatus.unbounded

        elif PROSTA_MAP[pro_status] == 'p_infeas':
            self.results.solver.status = SolverStatus.warning
            self.results.solver.termination_message += " Problem proven to be primal infeasible."
            self.results.solver.Message = self.results.solver.termination_message
            self.results.solver.termination_condition = TerminationCondition.infeasible
            soln.status = SolutionStatus.infeasible

        elif PROSTA_MAP[pro_status] == 'pd_infeas':
            self.results.solver.status = SolverStatus.warning
            self.results.solver.termination_message += " Problem proven to be primal and dual infeasible."
            self.results.solver.Message = self.results.solver.termination_message
            self.results.solver.termination_condition = TerminationCondition.infeasible
            soln.status = SolutionStatus.infeasible

        elif PROSTA_MAP[pro_status] == 'p_inf_unb':
            self.results.solver.status = SolverStatus.warning
            self.results.solver.termination_message += " Problem proven to be infeasible or unbounded."
            self.results.solver.Message = self.results.solver.termination_message
            self.results.solver.termination_condition = TerminationCondition.infeasibleOrUnbounded
            soln.status = SolutionStatus.unsure

        if SOLSTA_MAP[sol_status] == 'optimal':
            self.results.solver.status = SolverStatus.ok
            self.results.solver.termination_message += " Model was solved to optimality, " \
                "and an optimal solution is available."
            self.results.solver.termination_condition = TerminationCondition.optimal
            soln.status = SolutionStatus.optimal

        elif SOLSTA_MAP[sol_status] == 'pd_feas':
            self.results.solver.status = SolverStatus.ok
            self.results.solver.termination_message += " The solution is both primal and dual feasible"
            self.results.solver.termination_condition = TerminationCondition.feasible
            soln.status = SolutionStatus.feasible

        elif SOLSTA_MAP[sol_status] == 'p_feas':
            self.results.solver.status = SolverStatus.ok
            self.results.solver.termination_message += " Primal feasible solution is available."
            self.results.solver.termination_condition = TerminationCondition.feasible
            soln.status = SolutionStatus.feasible

        elif SOLSTA_MAP[sol_status] == 'd_feas':
            self.results.solver.status = SolverStatus.ok
            self.results.solver.termination_message += " Dual feasible solution is available."
            self.results.solver.termination_condition = TerminationCondition.feasible
            soln.status = SolutionStatus.feasible

        elif SOLSTA_MAP[sol_status] == 'd_infeas':
            self.results.solver.status = SolverStatus.warning
            self.results.solver.termination_message += " The solution is dual infeasible."
            self.results.solver.Message = self.results.solver.termination_message
            self.results.solver.termination_condition = TerminationCondition.unbounded
            soln.status = SolutionStatus.infeasible

        elif SOLSTA_MAP[sol_status] == 'p_infeas':
            self.results.solver.status = SolverStatus.warning
            self.results.solver.termination_message += " The solution is primal infeasible."
            self.results.solver.Message = self.results.solver.termination_message
            self.results.solver.termination_condition = TerminationCondition.infeasible
            soln.status = SolutionStatus.infeasible

        self.results.problem.name = msk_task.gettaskname()

        if msk_task.getobjsense() == msk.objsense.minimize:
            self.results.problem.sense = minimize
        elif msk_task.getobjsense() == msk.objsense.maximize:
            self.results.problem.sense = maximize
        else:
            raise RuntimeError(
                'Unrecognized Mosek objective sense: {0}'.format(msk_task.getobjname()))

        self.results.problem.upper_bound = None
        self.results.problem.lower_bound = None

        if msk_task.getnumintvar() == 0:
            try:
                if msk_task.getobjsense() == msk.objsense.minimize:
                    self.results.problem.upper_bound = msk_task.getprimalobj(
                        whichsol)
                    self.results.problem.lower_bound = msk_task.getdualobj(
                        whichsol)
                elif msk_task.getobjsense() == msk.objsense.maximize:
                    self.results.problem.upper_bound = msk_task.getprimalobj(
                        whichsol)
                    self.results.problem.lower_bound = msk_task.getdualobj(
                        whichsol)

            except (msk.MosekException, AttributeError):
                pass
        elif msk_task.getobjsense() == msk.objsense.minimize:  # minimizing
            try:
                self.results.problem.upper_bound = msk_task.getprimalobj(
                    whichsol)
            except (msk.MosekException, AttributeError):
                pass
            try:
                self.results.problem.lower_bound = msk_task.getdouinf(
                    msk.dinfitem.mio_obj_bound)
            except (msk.MosekException, AttributeError):
                pass
        elif msk_task.getobjsense() == msk.objsense.maximize:  # maximizing
            try:
                self.results.problem.upper_bound = msk_task.getdouinf(
                    msk.dinfitem.mio_obj_bound)
            except (msk.MosekException, AttributeError):
                pass
            try:
                self.results.problem.lower_bound = msk_task.getprimalobj(
                    whichsol)
            except (msk.MosekException, AttributeError):
                pass
        else:
            raise RuntimeError(
                'Unrecognized Mosek objective sense: {0}'.format(msk_task.getobjsense()))

        try:
            soln.gap = self.results.problem.upper_bound - self.results.problem.lower_bound
        except TypeError:
            soln.gap = None

        self.results.problem.number_of_constraints = msk_task.getnumcon()
        self.results.problem.number_of_nonzeros = msk_task.getnumanz()
        self.results.problem.number_of_variables = msk_task.getnumvar()
        self.results.problem.number_of_integer_variables = msk_task.getnumintvar()
        self.results.problem.number_of_continuous_variables = msk_task.getnumvar() - \
            msk_task.getnumintvar()
        self.results.problem.number_of_objectives = 1
        self.results.problem.number_of_solutions = 1

        # if a solve was stopped by a limit, we still need to check to
        # see if there is a solution available - this may not always
        # be the case, both in LP and MIP contexts.
        if self._save_results:
            """
            This code in this if statement is only needed for backwards compatability. It is more efficient to set
            _save_results to False and use load_vars, load_duals, etc.
            """
            if self.results.problem.number_of_solutions > 0:
                soln_variables = soln.variable
                soln_constraints = soln.constraint

                mosek_vars = list(range(msk_task.getnumvar()))
                mosek_vars = list(set(mosek_vars).intersection(
                    set(self._pyomo_var_to_solver_var_map.values())))
                var_vals = [0.0] * len(mosek_vars)
                self._solver_model.getxx(whichsol, var_vals)
                names = []
                for i in mosek_vars:
                    names.append(msk_task.getvarname(i))

                for mosek_var, val, name in zip(mosek_vars, var_vals, names):
                    pyomo_var = self._solver_var_to_pyomo_var_map[mosek_var]
                    if self._referenced_variables[pyomo_var] > 0:
                        pyomo_var.stale = False
                        soln_variables[name] = {"Value": val}

                if extract_reduced_costs:
                    vals = [0.0]*len(mosek_vars)
                    msk_task.getreducedcosts(
                        whichsol, 0, len(mosek_vars), vals)
                    for mosek_var, val, name in zip(mosek_vars, vals, names):
                        pyomo_var = self._solver_var_to_pyomo_var_map[mosek_var]
                        if self._referenced_variables[pyomo_var] > 0:
                            soln_variables[name]["Rc"] = val

                if extract_duals or extract_slacks:
                    mosek_cons = list(range(msk_task.getnumcon()))
                    con_names = []
                    for con in mosek_cons:
                        con_names.append(msk_task.getconname(con))
                    for name in con_names:
                        soln_constraints[name] = {}
                    """TODO wrong length, needs to be getnumvars()
                    mosek_cones = list(range(msk_task.getnumcone()))
                    cone_names = []
                    for cone in mosek_cones:
                        cone_names.append(msk_task.getconename(cone))
                    for name in cone_names:
                        soln_constraints[name] = {}
                    """

                if extract_duals:
                    ncon = msk_task.getnumcon()
                    if ncon > 0:
                        vals = [0.0]*ncon
                        msk_task.gety(whichsol, vals)
                        for val, name in zip(vals, con_names):
                            soln_constraints[name]["Dual"] = val
                    """TODO: wrong length, needs to be getnumvars()
                    ncone = msk_task.getnumcone()
                    if ncone > 0:
                        vals = [0.0]*ncone
                        msk_task.getsnx(whichsol, vals)
                        for val, name in zip(vals, cone_names):
                            soln_constraints[name]["Dual"] = val
                    """

                if extract_slacks:
                    Ax = [0]*len(mosek_cons)
                    msk_task.getxc(self._whichsol, Ax)
                    for con, name in zip(mosek_cons, con_names):
                        Us = Ls = 0

                        bk, lb, ub = msk_task.getconbound(con)

                        if bk in [msk.boundkey.fx, msk.boundkey.ra, msk.boundkey.up]:
                            Us = ub - Ax[con]
                        if bk in [msk.boundkey.fx, msk.boundkey.ra, msk.boundkey.lo]:
                            Ls = Ax[con] - lb

                        if Us > Ls:
                            soln_constraints[name]["Slack"] = Us
                        else:
                            soln_constraints[name]["Slack"] = -Ls

        elif self._load_solutions:
            if self.results.problem.number_of_solutions > 0:

                self._load_vars()

                if extract_reduced_costs:
                    self._load_rc()

                if extract_duals:
                    self._load_duals()

                if extract_slacks:
                    self._load_slacks()

        self.results.solution.insert(soln)

        # finally, clean any temporary files registered with the temp file
        # manager, created populated *directly* by this plugin.
        TempfileManager.pop(remove=not self._keepfiles)

        return DirectOrPersistentSolver._postsolve(self)

    def warm_start_capable(self):
        return True

    def _warm_start(self):
        for pyomo_var, mosek_var in self._pyomo_var_to_solver_var_map.items():
            if pyomo_var.value is not None:
                for solType in self._mosek.soltype._values:
                    self._solver_model.putxxslice(
                        solType, mosek_var, mosek_var+1, [(pyomo_var.value)])

    def _load_vars(self, vars_to_load=None):
        var_map = self._pyomo_var_to_solver_var_map
        ref_vars = self._referenced_variables
        if vars_to_load is None:
            vars_to_load = var_map.keys()

        mosek_vars_to_load = [var_map[pyomo_var] for pyomo_var in vars_to_load]
        var_vals = [0.0] * len(mosek_vars_to_load)
        self._solver_model.getxx(self._whichsol, var_vals)

        for var, val in zip(vars_to_load, var_vals):
            if ref_vars[var] > 0:
                var.stale = False
                var.value = val

    def _load_rc(self, vars_to_load=None):
        if not hasattr(self._pyomo_model, 'rc'):
            self._pyomo_model.rc = Suffix(direction=Suffix.IMPORT)
        var_map = self._pyomo_var_to_solver_var_map
        ref_vars = self._referenced_variables
        rc = self._pyomo_model.rc
        if vars_to_load is None:
            vars_to_load = var_map.keys()

        mosek_vars_to_load = [var_map[pyomo_var] for pyomo_var in vars_to_load]
        vals = [0.0]*len(mosek_vars_to_load)
        self._solver_model.getreducedcosts(
            self._whichsol, 0, len(mosek_vars_to_load), vals)

        for var, val in zip(vars_to_load, vals):
            if ref_vars[var] > 0:
                rc[var] = val

    def _load_duals(self, objs_to_load=None):
        if not hasattr(self._pyomo_model, 'dual'):
            self._pyomo_model.dual = Suffix(direction=Suffix.IMPORT)
        con_map = self._pyomo_con_to_solver_con_map
        reverse_con_map = self._solver_con_to_pyomo_con_map
        cone_map = self._pyomo_cone_to_solver_cone_map
        reverse_cone_map = self._solver_cone_to_pyomo_cone_map
        dual = self._pyomo_model.dual

        if objs_to_load is None:
            # constraints
            mosek_cons_to_load = range(self._solver_model.getnumcon())
            vals = [0.0]*len(mosek_cons_to_load)
            self._solver_model.gety(self._whichsol, vals)
            for mosek_con, val in zip(mosek_cons_to_load, vals):
                pyomo_con = reverse_con_map[mosek_con]
                dual[pyomo_con] = val
            """TODO wrong length, needs to be getnumvars()
            # cones
            mosek_cones_to_load = range(self._solver_model.getnumcone())
            vals = [0.0]*len(mosek_cones_to_load)
            self._solver_model.getsnx(self._whichsol, vals)
            for mosek_cone, val in zip(mosek_cones_to_load, vals):
                pyomo_cone = reverse_cone_map[mosek_cone]
                dual[pyomo_cone] = val
            """
        else:
            mosek_cons_to_load = []
            mosek_cones_to_load = []
            for obj in objs_to_load:
                if obj in con_map:
                    mosek_cons_to_load.append(con_map[obj])
                else:
                    # assume it is a cone
                    mosek_cones_to_load.append(cone_map[obj])
            # constraints
            mosek_cons_first = min(mosek_cons_to_load)
            mosek_cons_last = max(mosek_cons_to_load)
            vals = [0.0]*(mosek_cons_last - mosek_cons_first + 1)
            self._solver_model.getyslice(self._whichsol,
                                         mosek_cons_first,
                                         mosek_cons_last,
                                         vals)
            for mosek_con in mosek_cons_to_load:
                slice_index = mosek_con - mosek_cons_first
                val = vals[slice_index]
                pyomo_con = reverse_con_map[mosek_con]
                dual[pyomo_con] = val
            """TODO wrong length, needs to be getnumvars()
            # cones
            mosek_cones_first = min(mosek_cones_to_load)
            mosek_cones_last = max(mosek_cones_to_load)
            vals = [0.0]*(mosek_cones_last - mosek_cones_first + 1)
            self._solver_model.getsnxslice(self._whichsol,
                                           mosek_cones_first,
                                           mosek_cones_last,
                                           vals)
            for mosek_cone in mosek_cones_to_load:
                slice_index = mosek_cone - mosek_cones_first
                val = vals[slice_index]
                pyomo_cone = reverse_cone_map[mosek_cone]
                dual[pyomo_cone] = val
            """

    def _load_slacks(self, cons_to_load=None):
        if not hasattr(self._pyomo_model, 'slack'):
            self._pyomo_model.slack = Suffix(direction=Suffix.IMPORT)
        con_map = self._pyomo_con_to_solver_con_map
        reverse_con_map = self._solver_con_to_pyomo_con_map
        slack = self._pyomo_model.slack
        msk = self._mosek

        if cons_to_load is None:
            mosek_cons_to_load = range(self._solver_model.getnumcon())
        else:
            mosek_cons_to_load = set([con_map[pyomo_con]
                                      for pyomo_con in cons_to_load])

        Ax = [0]*len(mosek_cons_to_load)
        self._solver_model.getxc(self._whichsol, Ax)
        for con in mosek_cons_to_load:
            pyomo_con = reverse_con_map[con]
            Us = Ls = 0

            bk, lb, ub = self._solver_model.getconbound(con)

            if bk in [msk.boundkey.fx, msk.boundkey.ra, msk.boundkey.up]:
                Us = ub - Ax[con]
            if bk in [msk.boundkey.fx, msk.boundkey.ra, msk.boundkey.lo]:
                Ls = Ax[con] - lb

            if Us > Ls:
                slack[pyomo_con] = Us
            else:
                slack[pyomo_con] = -Ls

    def load_duals(self, cons_to_load=None):
        """
        Load the duals into the 'dual' suffix. The 'dual' suffix must live on the parent model.

        Parameters
        ----------
        cons_to_load: list of Constraint
        """
        self._load_duals(cons_to_load)

    def load_rc(self, vars_to_load):
        """
        Load the reduced costs into the 'rc' suffix. The 'rc' suffix must live on the parent model.

        Parameters
        ----------
        vars_to_load: list of Var
        """
        self._load_rc(vars_to_load)

    def load_slacks(self, cons_to_load=None):
        """
        Load the values of the slack variables into the 'slack' suffix. The 'slack' suffix must live on the parent
        model.

        Parameters
        ----------
        cons_to_load: list of Constraint
        """
        self._load_slacks(cons_to_load)
