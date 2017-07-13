#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.solvers.plugins.solvers.gurobi_direct import GurobiDirect
from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver
from pyomo.util.plugin import alias
from pyomo.core.base.constraint import Constraint
from pyomo.core.base.var import Var
from pyomo.core.base.sos import SOSConstraint


class GurobiPersistent(PersistentSolver, GurobiDirect):
    alias('gurobi_persistent', doc='Persistent python interface to Gurobi')

    def __init__(self, **kwds):
        kwds['type'] = 'gurobi_persistent'
        PersistentSolver.__init__(self, **kwds)
        GurobiDirect._init(self)

        self._pyomo_model = kwds.pop('model', None)
        if self._pyomo_model is not None:
            self.compile_instance(self._pyomo_model, **kwds)

    def _presolve(self, *args, **kwds):
        PersistentSolver._presolve(self, *args, **kwds)

    def _apply_solver(self):
        return GurobiDirect._apply_solver(self)

    def _postsolve(self):
        results = GurobiDirect._postsolve(self)
        return results

    def _compile_instance(self, model, **kwds):
        GurobiDirect._compile_instance(self, model, **kwds)

    def _add_block(self, block):
        GurobiDirect._add_block(self, block)

    def _compile_objective(self):
        GurobiDirect._compile_objective(self)

    def _add_constraint(self, con):
        GurobiDirect._add_constraint(self, con)

    def _add_var(self, var):
        GurobiDirect._add_var(self, var)

    def _add_sos_constraint(self, con):
        GurobiDirect._add_sos_constraint(self, con)

    def remove_block(self, block):
        for sub_block in block.block_data_objects(descend_into=True, active=True):
            for con in sub_block.component_data_objects(ctype=Constraint, descend_into=False, active=True):
                self.remove_constraint(con)

            for con in sub_block.component_data_objects(ctype=SOSConstraint, descend_into=False, active=True):
                self.remove_sos_constraint(con)

        for var in block.component_data_objects(ctype=Var, descend_into=True, active=True):
            self.remove_var(var)

    def remove_constraint(self, con):
        gurobipy_con = self._pyomo_con_to_solver_con_map[con]
        self._solver_model.remove(gurobipy_con)
        self._symbol_map.removeSymbol(con)
        for var in self._vars_referenced_by_con[con]:
            self._referenced_variables[var] -= 1
        del self._vars_referenced_by_con[con]
        del self._pyomo_con_to_solver_con_map[con]

    def remove_var(self, var):
        if self._referenced_variables[var] != 0:
            raise ValueError('Cannot remove Var {0} because it is still referenced by the '
                             'objective or one or more constraints')
        gurobipy_var = self._pyomo_var_to_solver_var_map[var]
        self._solver_model.remove(gurobipy_var)
        self._symbol_map.removeSymbol(var)
        del self._referenced_variables[var]
        del self._pyomo_var_to_solver_var_map[var]

    def remove_sos_constraint(self, con):
        gurobipy_con = self._pyomo_con_to_solver_con_map[con]
        self._solver_model.remove(gurobipy_con)
        self._symbol_map.removeSymbol(con)
        for var in self._vars_referenced_by_con[con]:
            self._referenced_variables[var] -= 1
        del self._vars_referenced_by_con[con]
        del self._pyomo_con_to_solver_con_map[con]

    def _get_expr_from_pyomo_repn(self, repn, max_degree=None):
        return GurobiDirect._get_expr_from_pyomo_repn(self, repn, max_degree)

    def _get_expr_from_pyomo_expr(self, expr, max_degree=None):
        return GurobiDirect._get_expr_from_pyomo_expr(self, expr, max_degree)

    def _load_vars(self, vars_to_load=None):
        GurobiDirect._load_vars(self, vars_to_load)

    def warm_start_capable(self):
        return GurobiDirect.warm_start_capable(self)

    def _warm_start(self):
        GurobiDirect._warm_start(self)

    def compile_instance(self, model, **kwds):
        return self._compile_instance(model, **kwds)

    def add_block(self, block):
        return self._add_block(block)

    def compile_objective(self):
        return self._compile_objective()

    def add_constraint(self, con):
        return self._add_constraint(con)

    def add_var(self, var):
        return self._add_var(var)

    def add_sos_constraint(self, con):
        return self._add_sos_constraint(con)

    def _gurobi_vtype_from_var(self, var):
        return GurobiDirect._gurobi_vtype_from_var(self, var)