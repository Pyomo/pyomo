#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import operator
import itertools
import logging
logger = logging.getLogger('pyomo.solvers')
from pyomo.core.expr.numvalue import (is_fixed,value)
from pyomo.core.base.PyomoModel import ConcreteModel
from pyomo.core.base.constraint import Constraint
from pyomo.core.base.var import Var
from pyomo.solvers.plugins.solvers.mosek_direct import MOSEKDirect
from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver
from pyomo.opt.base import SolverFactory
from pyomo.core.kernel.conic import _ConicBase

@SolverFactory.register('mosek_persistent', doc = 'Persistent python interface to MOSEK.')
class MOSEKPersistent(PersistentSolver, MOSEKDirect):
    
    def __init__(self, **kwds):
        kwds['type'] = 'mosek_persistent'
        MOSEKDirect.__init__(self, **kwds)

        self._pyomo_model = kwds.pop('model', None)
        if self._pyomo_model is not None:
            self.set_instance(self._pyomo_model, **kwds)

    def _warm_start(self):
        MOSEKDirect._warm_start(self)
    
    def _remove_vars(self, *solver_vars):
        try:
            self._solver_model.removevars(
                [self._pyomo_var_to_solver_var_map[v] for v in solver_vars])
            [self._pyomo_var_to_solver_var_map.pop(v, None) for v in solver_vars]
            var_num = self._solver_model.getnumvar()
            self._solver_var_to_pyomo_var_map = dict(zip((
                range(var_num)),self._pyomo_var_to_solver_var_map.keys()))
            self._vars_append_offset = var_num
        except self._mosek.Error as e:
            raise e 
    
    def _remove_constraints(self, *solver_cons):
        """
        Method to remove a constraint/cone from the solver model.
        User can pass several constraints as arguments, each of them 
        separated by commas (for instance, unpack a list of constraints).
        It is important to remember that removing constraints is an 
        expensive operation, that may not necessarily provide a huge 
        advantage over re-doing the entire model from scratch, especially
        if the Interior-point optimizer is being used.
        """
        try:
            lq = list(itertools.filterfalse(
                lambda x: isinstance(x,_ConicBase), solver_cons))
            lq = [self._pyomo_con_to_solver_con_map[c] for c in lq]
            cones = list(filter(lambda x: isinstance(x,_ConicBase), solver_cons))
            cones = [self._pyomo_cone_to_solver_cone_map[c] for c in cones]
            self._solver_model.removecons(lq)
            self._solver_model.removecones(cones)
            lq_num = self._solver_model.getnumcon()
            cone_num = self._solver_model.getnumcone()
            [self._pyomo_con_to_solver_con_map.pop(c, None) for c in lq]
            [self._pyomo_cone_to_solver_cone_map.pop(c, None) for c in cones]
            self._solver_con_to_pyomo_con_map = dict(zip(
                range(lq_num),self._pyomo_con_to_solver_con_map.keys()))
            self._solver_cone_to_pyomo_cone_map = dict(zip(
                range(cone_num),self._pyomo_cone_to_solver_cone_map.keys()))
            self._cons_append_offset = lq_num
            self._cones_append_offset = cone_num
            
            for c in solver_cons:
                for v in self._vars_referenced_by_con[c]:
                    self._referenced_variables[v] -= 1
        except self._mosek.Error as e:
            raise e
    
    def update_vars(self, *solver_vars):
        """
        Update variable(s) in the solver's model.
        This method allows fixing/unfixing variables, changing the 
        variable bounds and updating the variable types.
        Passing a single scalar var, much like other interfaces
        is perfectly valid, but a user can also pass several variables
        as arguments to change several variables at a time.
        Ideal use case would be by unpacking a list, for instance.

        Parameters:
        *solver_vars: scalar Vars or single _VarData, separated by commas
        """
        for var in solver_vars:
            if var not in self._pyomo_var_to_solver_var_map.keys():
                raise ValueError('Variable {} not found. It needs to be added '
                                    'before it can be modified.'.format(var))
        var_ids = [self._pyomo_var_to_solver_var_map[v] for v in solver_vars]
        vtypes = list(map(self._mosek_vartype_from_var, solver_vars))
        lbs, ubs, bound_types = zip(*[self._mosek_bounds(
            *p.bounds) for p in solver_vars])
        self._solver_model.putvartypelist(var_ids, vtypes)
        self._solver_model.putvarboundlist(var_ids, bound_types, lbs, ubs)
        
    def update_constraints(self, *solver_cons):
        """
        Update constraint(s) in the solver's model by changing constraint
        bounds and/or changing the coefficients.

        Note: this method does not accept conic constraints, or conic domains.
        Only linear/quadratic constraints are allowed.
        """
        con_list = list(filter(operator.attrgetter('active'),solver_cons))
        if len(con_list)!=len(solver_cons):
            logger.warning('Inactive constraints cannot be updated and will be skipped.')
        if self._skip_trivial_constraints:
            con_list = list(filter(is_fixed(
                operator.attrgetter('body')),con_list))

        lq = list(filter(operator.attrgetter("_linear_canonical_form"),
                  con_list))
        lq_ex = list(itertools.filterfalse(lambda x: isinstance(
            x,_ConicBase) or (x._linear_canonical_form),con_list))
        lq_all = lq + lq_ex
        num_lq = len(lq) + len(lq_ex)
        if len(lq_all)!= len(con_list):
            logger.warning("Any conic constraints/domains passed to this"//
                " method will be ignored. Only valid inputs are linear"//
                " or quadratic constraints.")
        try:
            if num_lq>0:
                for c in num_lq:
                    v_c = self._vars_referenced_by_con[c]
                    for v in v_c:
                        self._referenced_variables -= 1
                lq_canon = list(map(operator.attrgetter('_linear_canonical_form'),lq))
                lq_ex_body = list(map(operator.attrgetter('body'),lq_ex))
                lq_data = list(map(self._get_expr_from_pyomo_repn,lq_canon))
                lq_data.extend(list(map(self._get_expr_from_pyomo_expr,lq_ex_body)))
                arow, qexp, referenced_vars = zip(*lq_data)
                qcsubi, qcsubj, qcval = zip(*qexp)
                subi, vali, constants = zip(*arow)
                lbs, ubs, bound_types = zip(*[self._mosek_bounds(
                    lq_all[i].lower, lq_all[i].upper, constants[i]) 
                    for i in range(num_lq)])
                con_ids = [self._pyomo_con_to_solver_con_map[c] for c in lq_all]
                map(self._solver_model.putarow, con_ids, subi, vali)
                map(self._solver_model.putqconk,con_ids, qcsubi, qcsubj, qcval)
                self._solver_model.putconboundlist(subi, bound_types, lbs, ubs)
        
            for i,c in enumerate(con_list):
                self._vars_referenced_by_con[c] = referenced_vars[i]
                for v in referenced_vars[i]:
                    self._referenced_variables[v] += 1

        except self._mosek.Error as e:
            raise e