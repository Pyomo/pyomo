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
import pyomo.core.base.var
from pyutilib.misc import Bunch
from pyomo.core.expr.numvalue import (is_fixed,value)
from pyomo.core.kernel.objective import minimize, maximize
from pyomo.repn import generate_standard_repn
from pyomo.core.base.suffix import Suffix
from pyomo.solvers.plugins.solvers.direct_solver import DirectSolver
from pyomo.solvers.plugins.solvers.direct_or_persistent_solver import \
        DirectOrPersistentSolver
from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.opt import SolverFactory
from pyomo.core.kernel.conic import (_ConicBase,quadratic,rotated_quadratic,
                                     primal_exponential,primal_power,
                                     dual_exponential,dual_power)
from pyomo.opt.results.results_ import SolverResults
from pyomo.opt.results.solution import Solution, SolutionStatus
from pyomo.opt.results.solver import TerminationCondition, SolverStatus
logger = logging.getLogger('pyomo.solvers')

class DegreeError(ValueError):
    pass

def _is_numeric(x):
    try:
        float(x)
    except ValueError:
        return False
    return True

@SolverFactory.register('mosek_direct',doc='Direct python interface to MOSEK')
class MosekDirect(DirectSolver):
    
    def __init__(self,**kwds):
        kwds['type'] = 'mosek_direct'
        DirectSolver.__init__(self,**kwds)
        self._pyomo_cone_to_solver_cone_map = dict()
        self._solver_cone_to_pyomo_cone_map = ComponentMap()
        self._name None
        try:
            import mosek
            self._mosek = mosek
            self._mosek_env = self._mosek.Env()
            self._python_api_exists = True
            self._version = self._mosek_env.getversion()
            self._version_major = self._version[0]
            self._name = "MOSEK " + ".".join([str(i) for i in self._version])
        except ImportError:
            self._python_api_exists = False
        except Exception as e:
            print("Import of MOSEK failed - MOSEK message = "+ str(e) + "\n")
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

    def _apply_solver(self):
        try:
            if not self._save_results:
                for block in self._pyomo_model.block_data_objects(descend_into=True,active=True):
                    for var in block.component_data_objects(ctype=pyomo.core.base.var.Var,
                                                            descend_into=False,
                                                            active=True,sort=False):
                        var.stale = True
            if self.tee:
                def _process_stream(msg):
                    sys.stdout.write(msg)
                    sys.stdout.flush()
                self._solver_model.set_Stream(self._mosek.streamtype.log,_process_stream)

            if self._keepfiles:
                print("Solver log file : " + self._log_file)

            for key, option in self.options.items():
                try:
                    param = ".".join(("mosek",key))
                    if 'sparam' in key.split('.'):
                        self._solver_model.putstrparam(eval(param),option)
                    elif 'dparam' in key.split('.'):
                        self._solver_model.putdouparam(eval(param),option)
                    elif 'iparam' in key.split('.'):
                        self._solver_model.putintparam(eval(param),option)
                except (TypeError,AttributeError):
                    raise
            self._termcode = self._solver_model.optimize()
            self._solver_model.solutionsummary(self._mosek.streamtype.msg)
        except self._mosek.Error as e:
            # Catching any expressions thrown by MOSEK:
            print("ERROR : "+str(e.errno))
            if e.msg is not None:
                print("\t"+e.msg)
                sys.exit(1)
        return Bunch(rc=None,log=None)

    def _get_cone_data(self, con):
        assert isinstance(con, _ConicBase)
        assert con.check_convexity_conditions(relax = True)
        cone_type, cone_param, cone_members = None, None, None
        if isinstance(con, quadratic):
            cone_type = self._mosek.conetype.quad
            cone_members = [con.r] + list(con.x)
        elif isinstance(con, rotated_quadratic):
            cone_type = self._mosek.conetype.rquad
            cone_members = [con.r1, con.r2] + list(con.x)
        elif isinstance(con, primal_exponential) and 
                        (self._version_major>=9):
            cone_type = self._mosek.conetype.pexp
            cone_members = [con.r, con.x1, con.x2]
        elif isinstance(con, primal_power) and (self._version_major>=9):
            cone_type = self._mosek.conetype.ppow
            cone_param = con.alpha
            cone_members = [con.r1, con.r2] + list(con.x)
        elif isinstance(con, dual_exponential) and (self._version_major>=9):
            cone_type = self._mosek.conetype.dexp
            cone_members = [con.r, con.x1, con.x2]
        elif isinstance(con, dual_power) and (self._version_major>=9):
            cone_type = self._mosek.conetype.dpow
            cone_param = con.alpha
            cone_members = [con.r1, con.r2] + list(con.x)
        return(cone_type, cone_param, cone_members)

    def _get_expr_from_pyomo_repn(self,repn,max_degree=2):
        # TODO: time it vs the old implementation of this method
        degree = repn.polynomial_degree()
        if (degree is None) or degree>max_degree:
            raise DegreeError('MOSEK does not support expressions of degree {}.'.format(degree))
        
        referenced_vars = ComponentSet(repn.linear_vars)
        referenced_vars.update(repn.quadratic_vars)

        indices = [self._pyomo_var_to_solver_var_map[i] for i in repn.linear_vars]
        mosek_arow = [indices,list(repn.linear_coefs),repn.constant]
        qsubi,qsubj = zip(*[[self._pyomo_var_to_solver_var_map[xi],self._pyomo_var_to_solver_var_map[xj]] 
                            for xi,xj in repn.quadratic_vars])
        qval = list(map(lambda v,i,j: v*((i==j)+1),repn.quadratic_coefs,qsubi,qsubj))
        mosek_qexp = [qsubi,qsubj,qval]
        return mosek_arow, mosek_qexp, referenced_vars
    
    def _get_expr_from_pyomo_expr(self,expr,max_degree=2):
        repn = generate_standard_repn(expr,quadratic=(max_degree==2))
        try:
            mosek_arow, mosek_qexp, referenced_vars = self._get_expr_from_pyomo_repn(repn,max_degree)
        except DegreeError as e:
            msg = e.args[0]
            msg += '\nexpr: {}'.format(expr)
            raise DegreeError(msg)
        return mosek_arow, mosek_qexp, referenced_vars
    
    def _add_var(self,var):
        vname = self._symbol_map.getSymbol(var,self._labeler)
        vtype = self._mosek.variabletype.type_cont
        if var.is_integer() or var.is_binary():
            vtype = self._mosek.variabletype.type_int
        elif not var.is_continuous():
            raise ValueError(
                'Variable domain type not recognized for {0}'.format(var.domain))

        boundtype = self._mosek.boundkey.fr
        lb,ub = -float('inf'),float('inf')
        if var.is_fixed():
            boundtype = self._mosek.boundkey.fx
            lb = value(var.lb)
            ub = value(var.ub)
        elif var.has_lb() and var.has_ub():
            boundtype = self._mosek.boundkey.ra
            lb,ub = value(var.lb),value(var.ub)
        elif var.has_lb() and not var.has_ub():
            boundtype = self._mosek.boundkey.lo
            lb = value(var.lb)
        elif not var.has_lb() and var.has_ub():
            boundtype = self._mosek.boundkey.up
            ub = value(var.ub())
        
        self._solver_model.appendvars(1)
        index = self._solver_model.getnumvar() - 1
        self._solver_model.putvarname(index,vname)
        self._solver_model.putvartype(index,vtype)
        self._solver_model.putvarbound(index,boundtype,lb,ub)

        self._pyomo_var_to_solver_var_map[var] = index
        self._solver_var_to_pyomo_var_map[index] = var
        self._referenced_variables[var] = 0 
    
    def _set_instance(self,model,kwds={}):
        self._range_constraints = set()
        DirectOrPersistentSolver._set_instance(self,model,kwds)
        self._whichsol = getattr(self._mosek.soltype,kwds.pop('soltype','bas'))
        try:
            self._solver_model = self._mosek.Env().Task()
        except Exception:
            e = sys.exc_info()[1]
            msg = ("Unable to create MOSEK task. Make sure that MOSEK's "
                   "Optimizer API for python is installed correctly.\n\n\t"+
                   "Error message: {}".format(e))
            raise Exception(msg)
        self._add_block(model)

    def _con_bounds(self,con,constant):
        if con.has_lb() and not is_fixed(con.lower):
            raise ValueError("Lower bound for constraint {} "
                             "is not constant.".format(con))
        if con.has_ub() and not is_fixed(con.upper):
            raise ValueError("Upper bound for constraint {} "
                             "is not constant.".format(con))
        lb, ub = -float("inf"),float("inf")
        con_bound_type = self._mosek.boundkey.fr
        if con.equality:
            ub = value(con.upper) - constant
            lb = value(con.lower) - constant
            con_bound_type = self._mosek.boundkey.fx
        elif con.has_lb() and con.has_ub():
            ub = value(con.upper) - constant
            lb = value(con.lower) - constant
            con_bound_type = self._mosek.boundkey.ra
        elif con.has_lb():
            lb = value(con.lower) - constant
            con_bound_type = self._mosek.boundkey.lo
        elif con.has_ub():
            ub = value(con.upper) - constant
            con_bound_type = self._mosek.boundkey.up
        return(con_bound_type,lb,ub)

    def _add_constraint(self,con):
        if not con.active or (is_fixed(con.body) and
                              self._skip_trivial_constraints):
            return None
            
        con_name = self._symbol_map.getSymbol(con,self._labeler)
        mosek_arow, mosek_qexp = None, None
        referenced_vars = None
        cone_type, cone_param, cone_members = None, None, None
        if isinstance(con,_ConicBase):
            cone_type, cone_param, cone_members = self._get_cone_data(con)
            assert cone_members is not None
            referenced_vars = ComponentSet(cone_members)
            if cone_type is None:
                logger.warning("Cone {0} is not supported by MOSEK v{1}.\n".format(
                               str(con),self._version_major) + "Exponential/"+
                               "Power cones are supported ONLY for MOSEK v>9.")
        elif con._linear_canonical_form:
            mosek_arow, mosek_qexp, referenced_vars = self._get_expr_from_pyomo_repn(
                con.canonical_form(),self._max_constraint_degree)
        else:
            mosek_arow, mosek_qexp, referenced_vars = self._get_expr_from_pyomo_expr(
                con.body,self._max_constraint_degree)
        
        assert referenced_vars is not None
        if mosek_arow is not None and cone_type is None:
            self._solver_model.appendcons(1)
            con_index = self._solver_model.getnumcon() - 1
            con_bound_type, lb, ub = self._con_bounds(con,mosek_arow[2])
            self._solver_model.putarow(con_index,mosek_arow[0],mosek_arow[1])
            self._solver_model.putqconk(con_index,mosek_qexp[0],mosek_qexp[1],
                                        mosek_qexp[2])
            self._solver_model.putconbound(con_index,con_bound_type,lb,ub)
            self._solver_model.putconname(con_index,con_name)
            self._pyomo_con_to_solver_con_map[con] = con_index
            self._solver_con_to_pyomo_con_map[con_index] = con
        elif cone_type is not None:
            members = [self._pyomo_var_to_solver_var_map[v_]
                        for v_ in cone_members]
            self._solver_model.appendcone(cone_type,cone_param,
                                          cone_members)
            cone_index = self._solver_model.getnumcone() - 1
            self._solver_model.putconename(cone_index,con_name)
            self._pyomo_cone_to_solver_cone_map[con] = cone_index
            self._solver_cone_to_pyomo_cone_map[cone_index] = con
        
        for var in referenced_vars:
            self._referenced_variables[var] += 1
        self._vars_referenced_by_con[con] = referenced_vars

    def _set_objective(self,obj):
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
        
        for i,j in enumerate(mosek_arow[0]):
            self._solver_model.putcj(j, mosek_arow[1][i])

        self._solver_model.putqobj(mosek_qexp[0],mosek_qexp[1],mosek_qexp[2])
        self._solver_model.putcfix(mosek_arow[2])
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
            if re.match(suffix, "slacks"):
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
        mks = self._mosek

        itr_soltypes = [msk.problemtype.qo,msk.problemtype.qcqo,
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
            self.results.solver.termination_message = "The optimizer terminated at the maximum number of iterations."
            self.results.solver.termination_condition = TerminationCondition.maxIterations
            soln.status = SolutionStatus.stoppedByLimit

        elif self._termcode == msk.rescode.trm_max_time:
            self.results.solver.status = SolverStatus.ok
            self.results.solver.termination_message = "The optimizer terminated at the maximum amount of time."
            self.results.solver.termination_condition = TerminationCondition.maxTimeLimit
            soln.status = SolutionStatus.stoppedByLimit

        elif self._termcode == msk.rescode.trm_user_callback:
            self.results.solver.status = SolverStatus.aborted
            self.results.solver.termination_message = "The optimizer terminated due to the return of the "\
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
            self.results.solver.termination_message = " Optimization terminated {} response code." \
                "Check MOSEK response code documentation for more information.".format(self._termcode)
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
            self.results.solver.termination_message += " The solution is optimal."
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

        