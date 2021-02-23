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
import six
import sys
import itertools
import operator
import pyomo.core.base.var
import pyomo.core.base.constraint
from pyomo.common.tempfiles import TempfileManager
from pyomo.core import is_fixed, value, minimize, maximize
from pyomo.repn import generate_standard_repn
from pyomo.core.base.suffix import Suffix
from pyomo.opt.base.solvers import OptSolver
from pyomo.solvers.plugins.solvers.direct_solver import DirectSolver
from pyomo.solvers.plugins.solvers.direct_or_persistent_solver import \
    DirectOrPersistentSolver
from pyomo.common.collections import ComponentMap, ComponentSet, Bunch
from pyomo.opt import SolverFactory
from pyomo.core.kernel.conic import (_ConicBase, quadratic, rotated_quadratic,
                                     primal_exponential, primal_power,
                                     dual_exponential, dual_power)
from pyomo.opt.results.results_ import SolverResults
from pyomo.opt.results.solution import Solution, SolutionStatus
from pyomo.opt.results.solver import TerminationCondition, SolverStatus
logger = logging.getLogger('pyomo.solvers')
inf = float('inf')

if six.PY2:
    def accumulate(it):
        total = 0
        for x in it:
            total += x
            yield total
else:
    from itertools import accumulate

class DegreeError(ValueError):
    pass


def _is_numeric(x):
    try:
        float(x)
    except ValueError:
        return False
    return True


@SolverFactory.register('mosek', doc='The MOSEK LP/QP/SOCP/MIP solver')
class MOSEK(OptSolver):
    """
    The MOSEK LP/QP/SOCP/MIP solver
    """

    def __new__(cls, *args, **kwds):
        mode = kwds.pop('solver_io', 'direct')
        if mode in {'python', 'direct'}:
            opt = SolverFactory('mosek_direct', **kwds)
            if opt is None:
                logger.error(
                    'MOSEK\'s Optimizer API for python is not installed.')
            return opt
        if mode == 'persistent':
            opt = SolverFactory('mosek_persistent', **kwds)
            if opt is None:
                logger.error(
                    'MOSEK\'s Optimizer API for python is not installed.')
            return opt
        else:
            logger.error(
                'Unknown solver interface: \"{}\"'.format(mode))
            return None


@SolverFactory.register('mosek_direct', doc='Direct python interface to MOSEK')
class MOSEKDirect(DirectSolver):
    """
    A class to provide a direct interface between pyomo and MOSEK's Optimizer API.
    Due to direct python bindings interacting with each other, there is no need for
    file IO.
    """

    def __init__(self, **kwds):
        kwds.setdefault('type', 'mosek_direct')
        DirectSolver.__init__(self, **kwds)
        self._pyomo_cone_to_solver_cone_map = dict()
        self._solver_cone_to_pyomo_cone_map = ComponentMap()
        self._name = None
        try:
            import mosek
            self._mosek = mosek
            self._mosek_env = self._mosek.Env()
            self._python_api_exists = True
            self._version = self._mosek_env.getversion()
            self._name = "MOSEK " + ".".join(str(i) for i in self._version)
        except ImportError:
            self._python_api_exists = False
        except Exception as e:
            print("Import of MOSEK failed - MOSEK message = " + str(e) + "\n")
            self._python_api_exists = False

        self._range_constraint = set()
        self._max_obj_degree = 2
        self._max_constraint_degree = 2
        self._termcode = None

        # Undefined capabilities default to None.
        self._capabilities.linear = True
        self._capabilities.quadratic_objective = True
        self._capabilities.quadratic_constraint = True
        self._capabilities.integer = True
        self._capabilities.conic_constraints = True
        self._capabilities.sos1 = False
        self._capabilities.sos2 = False

    def license_is_valid(self):
        """
        Runs a check for a valid MOSEK license. Returns False if MOSEK fails
        to run on a trivial test case.
        """
        try:
            import mosek
        except ImportError:
            return False
        try:
            mosek.Env().checkoutlicense(mosek.feature.pton)
            mosek.Env().checkinlicense(mosek.feature.pton)
        except mosek.Error:
            return False
        return True

    def _apply_solver(self):
        if not self._save_results:
            for block in self._pyomo_model.block_data_objects(
                    descend_into=True, active=True):
                for var in block.component_data_objects(
                        ctype=pyomo.core.base.var.Var, descend_into=False,
                        active=True, sort=False):
                    var.stale = True

        if self._tee:
            def _process_stream(msg):
                sys.stdout.write(msg)
                sys.stdout.flush()
            self._solver_model.set_Stream(
                self._mosek.streamtype.log, _process_stream)

        if self._keepfiles:
            logger.info("Solver log file: {}".format(self._log_file))

        for key, option in self.options.items():
            try:
                param = key.split('.')
                if param[0] == 'mosek':
                    param.pop(0)
                param = getattr(self._mosek, param[0])(param[1])
                if 'sparam' in key.split('.'):
                    self._solver_model.putstrparam(param, option)
                elif 'dparam' in key.split('.'):
                    self._solver_model.putdouparam(param, option)
                elif 'iparam' in key.split('.'):
                    if isinstance(option, str):
                        option = option.split('.')
                        if option[0] == 'mosek':
                            option.pop('mosek')
                        option = getattr(self._mosek, option[0])(option[1])
                    else:
                        self._solver_model.putintparam(param, option)
            except (TypeError, AttributeError):
                raise
        try:
            self._termcode = self._solver_model.optimize()
            self._solver_model.solutionsummary(self._mosek.streamtype.msg)
        except self._mosek.Error as e:
            logger.error(e)
            raise
        return Bunch(rc=None, log=None)

    def _set_instance(self, model, kwds={}):
        self._range_constraints = set()
        super(MOSEKDirect, self)._set_instance(model, kwds)
        self._pyomo_cone_to_solver_cone_map = dict()
        self._solver_cone_to_pyomo_cone_map = ComponentMap()
        self._whichsol = getattr(self._mosek.soltype, kwds.pop(
            'soltype', 'bas'))
        try:
            self._solver_model = self._mosek.Env().Task()
        except:
            err_msg = sys.exc_info()[1]
            logger.error("MOSEK task creation failed. "
                         + "Reason: {}".format(err_msg))
            raise
        self._add_block(model)

    def _get_cone_data(self, con):
        cone_type, cone_param, cone_members = None, 0, None
        if isinstance(con, quadratic):
            cone_type = self._mosek.conetype.quad
            cone_members = [con.r] + list(con.x)
        elif isinstance(con, rotated_quadratic):
            cone_type = self._mosek.conetype.rquad
            cone_members = [con.r1, con.r2] + list(con.x)
        elif self._version[0] >= 9:
            if isinstance(con, primal_exponential):
                cone_type = self._mosek.conetype.pexp
                cone_members = [con.r, con.x1, con.x2]
            elif isinstance(con, primal_power):
                cone_type = self._mosek.conetype.ppow
                cone_param = value(con.alpha)
                cone_members = [con.r1, con.r2] + list(con.x)
            elif isinstance(con, dual_exponential):
                cone_type = self._mosek.conetype.dexp
                cone_members = [con.r, con.x1, con.x2]
            elif isinstance(con, dual_power):
                cone_type = self._mosek.conetype.dpow
                cone_param = value(con.alpha)
                cone_members = [con.r1, con.r2] + list(con.x)
        return(cone_type, cone_param, ComponentSet(cone_members))

    def _get_expr_from_pyomo_repn(self, repn, max_degree=2):
        degree = repn.polynomial_degree()
        if (degree is None) or degree > max_degree:
            raise DegreeError(
                'MOSEK does not support expressions of degree {}.'.format(degree))

        referenced_vars = ComponentSet(repn.linear_vars)
        indices = tuple(self._pyomo_var_to_solver_var_map[i]
                        for i in repn.linear_vars)
        mosek_arow = (indices, tuple(repn.linear_coefs), repn.constant)

        if len(repn.quadratic_vars) == 0:
            mosek_qexp = ((), (), ())
            return mosek_arow, mosek_qexp, referenced_vars
        else:
            q_vars = itertools.chain.from_iterable(repn.quadratic_vars)
            referenced_vars.update(q_vars)
            qsubi = tuple(
                self._pyomo_var_to_solver_var_map[i] for i, j in repn.quadratic_vars)
            qsubj = tuple(
                self._pyomo_var_to_solver_var_map[j] for i, j in repn.quadratic_vars)
            qvals = tuple(v * 2 if qsubi[i] is qsubj[i] else v
                          for i, v in enumerate(repn.quadratic_coefs))
            mosek_qexp = (qsubi, qsubj, qvals)
        return mosek_arow, mosek_qexp, referenced_vars

    def _get_expr_from_pyomo_expr(self, expr, max_degree=2):
        repn = generate_standard_repn(expr, quadratic=(max_degree == 2))
        try:
            mosek_arow, mosek_qexp, referenced_vars = self._get_expr_from_pyomo_repn(
                repn, max_degree)
        except DegreeError as e:
            msg = e.args[0]
            msg += '\nexpr: {}'.format(expr)
            logger.error(DegreeError(msg))
            raise e
        return mosek_arow, mosek_qexp, referenced_vars

    def _mosek_vartype_from_var(self, var):
        if var.is_integer():
            return self._mosek.variabletype.type_int
        return self._mosek.variabletype.type_cont

    def _mosek_bounds(self, lb, ub, fixed_bool):
        if fixed_bool:
            return self._mosek.boundkey.fx
        if lb == -inf:
            if ub == inf:
                return self._mosek.boundkey.fr
            else:
                return self._mosek.boundkey.up
        elif ub == inf:
            return self._mosek.boundkey.lo
        return self._mosek.boundkey.ra

    def _add_var(self, var):
        self._add_vars((var,))

    def _add_vars(self, var_seq):
        if not var_seq:
            return
        var_num = self._solver_model.getnumvar()
        vnames = tuple(self._symbol_map.getSymbol(
            v, self._labeler) for v in var_seq)
        vtypes = tuple(map(self._mosek_vartype_from_var, var_seq))
        lbs = tuple(-inf if value(v.lb) is None else value(v.lb)
                    for v in var_seq)
        ubs = tuple(inf if value(v.ub) is None else value(v.ub)
                    for v in var_seq)
        fxs = tuple(v.is_fixed() for v in var_seq)
        bound_types = tuple(map(self._mosek_bounds, lbs, ubs, fxs))
        self._solver_model.appendvars(len(var_seq))
        var_ids = range(var_num,
                        var_num + len(var_seq))
        _vnames = tuple(map(self._solver_model.putvarname, var_ids, vnames))
        self._solver_model.putvartypelist(var_ids, vtypes)
        self._solver_model.putvarboundlist(var_ids, bound_types, lbs, ubs)
        self._pyomo_var_to_solver_var_map.update(zip(var_seq, var_ids))
        self._solver_var_to_pyomo_var_map.update(zip(var_ids, var_seq))
        self._referenced_variables.update(zip(var_seq, [0]*len(var_seq)))

    def _add_constraint(self, con):
        self._add_constraints((con,))

    def _add_constraints(self, con_seq):
        if not con_seq:
            return
        active_seq = tuple(filter(operator.attrgetter('active'), con_seq))
        if len(active_seq) != len(con_seq):
            logger.warning("Inactive constraints will be skipped.")
        con_seq = active_seq
        if self._skip_trivial_constraints:
            con_seq = tuple(filter(is_fixed(
                operator.attrgetter('body')), con_seq))

        lq = tuple(filter(operator.attrgetter("_linear_canonical_form"),
                          con_seq))
        conic = tuple(filter(lambda x: isinstance(x, _ConicBase), con_seq))
        lq_ex = tuple(six.moves.filterfalse(lambda x: isinstance(
            x, _ConicBase) or (x._linear_canonical_form), con_seq))
        lq_all = lq + lq_ex
        num_lq = len(lq) + len(lq_ex)
        num_cones = len(conic)
        if num_lq > 0:
            con_num = self._solver_model.getnumcon()
            lq_data = [self._get_expr_from_pyomo_repn(c.canonical_form())
                       for c in lq]
            lq_data.extend(
                self._get_expr_from_pyomo_expr(c.body) for c in lq_ex)
            arow, qexp, referenced_vars = zip(*lq_data)
            q_is, q_js, q_vals = zip(*qexp)
            l_ids, l_coefs, constants = zip(*arow)
            lbs = tuple(-inf if value(lq_all[i].lower) is None else value(
                lq_all[i].lower) - constants[i] for i in range(num_lq))
            ubs = tuple(inf if value(lq_all[i].upper) is None else value(
                lq_all[i].upper) - constants[i] for i in range(num_lq))
            fxs = tuple(c.equality for c in lq_all)
            bound_types = tuple(map(self._mosek_bounds, lbs, ubs, fxs))
            sub = range(con_num, con_num + num_lq)
            sub_names = tuple(self._symbol_map.getSymbol(c, self._labeler)
                              for c in lq_all)
            ptre = tuple(accumulate(list(map(len, l_ids))))
            ptrb = (0,) + ptre[:-1]
            asubs = tuple(itertools.chain.from_iterable(l_ids))
            avals = tuple(itertools.chain.from_iterable(l_coefs))
            qcsubi = tuple(itertools.chain.from_iterable(q_is))
            qcsubj = tuple(itertools.chain.from_iterable(q_js))
            qcval = tuple(itertools.chain.from_iterable(q_vals))
            qcsubk = tuple(i*len(q_is[i - con_num]) for i in sub)
            self._solver_model.appendcons(num_lq)
            self._solver_model.putarowlist(sub, ptrb, ptre, asubs, avals)
            self._solver_model.putqcon(qcsubk, qcsubi, qcsubj, qcval)
            self._solver_model.putconboundlist(sub, bound_types, lbs, ubs)
            for i, s_n in enumerate(sub_names):
                self._solver_model.putconname(sub[i], s_n)
            self._pyomo_con_to_solver_con_map.update(zip(lq_all, sub))
            self._solver_con_to_pyomo_con_map.update(zip(sub, lq_all))

            for i, c in enumerate(lq_all):
                self._vars_referenced_by_con[c] = referenced_vars[i]
                for v in referenced_vars[i]:
                    self._referenced_variables[v] += 1

        if num_cones > 0:
            cone_num = self._solver_model.getnumcone()
            cone_indices = range(cone_num,
                                 cone_num + num_cones)
            cone_names = tuple(self._symbol_map.getSymbol(
                c, self._labeler) for c in conic)
            cone_type, cone_param, cone_members = zip(*map(
                self._get_cone_data, conic))
            for i in range(num_cones):
                members = tuple(self._pyomo_var_to_solver_var_map[c_m]
                                for c_m in cone_members[i])
                self._solver_model.appendcone(
                    cone_type[i], cone_param[i], members)
                self._solver_model.putconename(
                    cone_indices[i], cone_names[i])
            self._pyomo_cone_to_solver_cone_map.update(
                zip(conic, cone_indices))
            self._solver_cone_to_pyomo_cone_map.update(
                zip(cone_indices, conic))

            for i, c in enumerate(conic):
                self._vars_referenced_by_con[c] = cone_members[i]
                for v in cone_members[i]:
                    self._referenced_variables[v] += 1

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
            raise ValueError("Objective sense not recognized.")

        mosek_arow, mosek_qexp, referenced_vars = self._get_expr_from_pyomo_expr(
            obj.expr, self._max_obj_degree)

        for var in referenced_vars:
            self._referenced_variables[var] += 1

        self._solver_model.putclist(mosek_arow[0], mosek_arow[1])
        self._solver_model.putqobj(mosek_qexp[0], mosek_qexp[1], mosek_qexp[2])
        self._solver_model.putcfix(mosek_arow[2])

        self._objective = obj
        self._vars_referenced_by_obj = referenced_vars

    def _add_block(self, block):
        """
        Overrides the _add_block method to utilize _add_vars/_add_constraints.

        This will keep any existing model components intact.

        Use this method when adding conic domains. The add_constraint method
        is compatible with conic-constraints, not conic-domains.

        Parameters
        ----------
        block: Block (scalar Block or single _BlockData)
        """
        var_seq = tuple(block.component_data_objects(
            ctype=pyomo.core.base.var.Var,
            descend_into=True, active=True,
            sort=True))
        self._add_vars(var_seq)
        for sub_block in block.block_data_objects(descend_into=True,
                                                  active=True):
            con_list = []
            for con in sub_block.component_data_objects(
                    ctype=pyomo.core.base.constraint.Constraint,
                    descend_into=False,
                    active=True,
                    sort=True):
                if (not con.has_lb()) and \
                   (not con.has_ub()):
                    assert not con.equality
                    continue  # non-binding, so skip
                con_list.append(con)
            self._add_constraints(con_list)

            for con in sub_block.component_data_objects(
                    ctype=pyomo.core.base.sos.SOSConstraint,
                    descend_into=False,
                    active=True,
                    sort=True):
                self._add_sos_constraint(con)

            obj_counter = 0
            for obj in sub_block.component_data_objects(
                    ctype=pyomo.core.base.objective.Objective,
                    descend_into=False,
                    active=True):
                obj_counter += 1
                if obj_counter > 1:
                    raise ValueError("Solver interface does not "
                                     "support multiple objectives.")
                self._set_objective(obj)

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
                    "***MOSEK solver plugin cannot extract solution suffix = "
                    + suffix)

        msk_task = self._solver_model
        msk = self._mosek

        itr_soltypes = [msk.problemtype.qo, msk.problemtype.qcqo,
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

        if self._version[0] < 9:
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
            self.results.solver.termination_message = "Optimizer terminated at the maximum number of iterations."
            self.results.solver.termination_condition = TerminationCondition.maxIterations
            soln.status = SolutionStatus.stoppedByLimit

        elif self._termcode == msk.rescode.trm_max_time:
            self.results.solver.status = SolverStatus.ok
            self.results.solver.termination_message = "Optimizer terminated at the maximum amount of time."
            self.results.solver.termination_condition = TerminationCondition.maxTimeLimit
            soln.status = SolutionStatus.stoppedByLimit

        elif self._termcode == msk.rescode.trm_user_callback:
            self.results.solver.status = SolverStatus.aborted
            self.results.solver.termination_message = "Optimizer terminated due to the return of the "\
                "user-defined callback function."
            self.results.solver.termination_condition = TerminationCondition.userInterrupt
            soln.status = SolutionStatus.unknown

        elif self._termcode in [msk.rescode.trm_mio_num_relaxs,
                                msk.rescode.trm_mio_num_branches,
                                msk.rescode.trm_num_max_num_int_solutions]:
            self.results.solver.status = SolverStatus.ok
            self.results.solver.termination_message = "The mixed-integer optimizer terminated as the maximum number "\
                "of relaxations/branches/feasible solutions was reached."
            self.results.solver.termination_condition = TerminationCondition.maxEvaluations
            soln.status = SolutionStatus.stoppedByLimit
        else:
            self.results.solver.termination_message = " Optimization terminated with {} response code." \
                "Check MOSEK response code documentation for more information.".format(
                    self._termcode)
            self.results.solver.termination_condition = TerminationCondition.unknown

        if SOLSTA_MAP[sol_status] == 'unknown':
            self.results.solver.status = SolverStatus.warning
            self.results.solver.termination_message += " The solution status is unknown."
            self.results.solver.Message = self.results.solver.termination_message
            self.results.solver.termination_condition = TerminationCondition.unknown
            soln.status = SolutionStatus.unknown

        if PROSTA_MAP[pro_status] == 'd_infeas':
            self.results.solver.status = SolverStatus.warning
            self.results.solver.termination_message += " Problem is dual infeasible"
            self.results.solver.Message = self.results.solver.termination_message
            self.results.solver.termination_condition = TerminationCondition.unbounded
            soln.status = SolutionStatus.unbounded

        elif PROSTA_MAP[pro_status] == 'p_infeas':
            self.results.solver.status = SolverStatus.warning
            self.results.solver.termination_message += " Problem is primal infeasible."
            self.results.solver.Message = self.results.solver.termination_message
            self.results.solver.termination_condition = TerminationCondition.infeasible
            soln.status = SolutionStatus.infeasible

        elif PROSTA_MAP[pro_status] == 'pd_infeas':
            self.results.solver.status = SolverStatus.warning
            self.results.solver.termination_message += " Problem is primal and dual infeasible."
            self.results.solver.Message = self.results.solver.termination_message
            self.results.solver.termination_condition = TerminationCondition.infeasible
            soln.status = SolutionStatus.infeasible

        elif PROSTA_MAP[pro_status] == 'p_inf_unb':
            self.results.solver.status = SolverStatus.warning
            self.results.solver.termination_message += " Problem is either primal infeasible or unbounded."\
                " This may happen for MIPs."
            self.results.solver.Message = self.results.solver.termination_message
            self.results.solver.termination_condition = TerminationCondition.infeasibleOrUnbounded
            soln.status = SolutionStatus.unsure

        if SOLSTA_MAP[sol_status] == 'optimal':
            self.results.solver.status = SolverStatus.ok
            self.results.solver.termination_message += " Model was solved to optimality and an optimal solution is available."
            self.results.solver.termination_condition = TerminationCondition.optimal
            soln.status = SolutionStatus.optimal

        elif SOLSTA_MAP[sol_status] == 'pd_feas':
            self.results.solver.status = SolverStatus.ok
            self.results.solver.termination_message += " The solution is both primal and dual feasible."
            self.results.solver.termination_condition = TerminationCondition.feasible
            soln.status = SolutionStatus.feasible

        elif SOLSTA_MAP[sol_status] == 'p_feas':
            self.results.solver.status = SolverStatus.ok
            self.results.solver.termination_message += " The solution is primal feasible."
            self.results.solver.termination_condition = TerminationCondition.feasible
            soln.status = SolutionStatus.feasible

        elif SOLSTA_MAP[sol_status] == 'd_feas':
            self.results.solver.status = SolverStatus.ok
            self.results.solver.termination_message += " The solution is dual feasible."
            self.results.solver.termination_condition = TerminationCondition.feasible
            soln.status = SolutionStatus.feasible

        elif SOLSTA_MAP[sol_status] == 'd_infeas':
            self.results.solver.status = SolverStatus.warning
            self.results.solver.termination_message += " The solution is a certificate of dual infeasibility."
            self.results.solver.Message = self.results.solver.termination_message
            self.results.solver.termination_condition = TerminationCondition.unbounded
            soln.status = SolutionStatus.infeasible

        elif SOLSTA_MAP[sol_status] == 'p_infeas':
            self.results.solver.status = SolverStatus.warning
            self.results.solver.termination_message += " The solution is a certificate of primal infeasibility."
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
                names = list(map(msk_task.getvarname, mosek_vars))

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
                    con_names = list(map(msk_task.getconname, mosek_cons))
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
                for solType in self._mosek.soltype.values:
                    self._solver_model.putxxslice(
                        solType, mosek_var, mosek_var + 1, [(pyomo_var.value)])

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
        #reverse_cone_map = self._solver_cone_to_pyomo_cone_map
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
