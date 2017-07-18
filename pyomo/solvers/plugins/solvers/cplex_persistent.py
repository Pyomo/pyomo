#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.solvers.plugins.solvers.cplex_direct import CPLEXDirect
from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver
from pyomo.util.plugin import alias
from pyomo.core.base.constraint import Constraint
from pyomo.core.base.var import Var
from pyomo.core.base.sos import SOSConstraint
from pyomo.core.kernel.numvalue import value


class CPLEXPersistent(PersistentSolver, CPLEXDirect):
    alias('cplex_persistent', doc='Persistent python interface to CPLEX')

    def __init__(self, **kwds):
        kwds['type'] = 'cplex_persistent'
        PersistentSolver.__init__(self, **kwds)
        CPLEXDirect._init(self)

        self._pyomo_model = kwds.pop('model', None)
        if self._pyomo_model is not None:
            self.compile_instance(self._pyomo_model, **kwds)

    def _remove_constraint(self, solver_con):
        try:
            self._solver_model.linear_constraints.delete(solver_con)
        except self._cplex.exceptions.CplexError:
            try:
                self._solver_model.quadratic_constraints.delete(solver_con)
            except self._cplex.exceptions.CplexError:
                raise ValueError('Failed to find the cplex constraint {0}'.format(solver_con))

    def _remove_sos_constraint(self, solver_sos_con):
        self._solver_model.SOS.delete(solver_sos_con)

    def _remove_var(self, solver_var):
        self._solver_model.variables.delete(solver_var)

    def _warm_start(self):
        GurobiDirect._warm_start(self)

    def compile_var(self, var):
        if var.is_indexed():
            for child_var in var.values():
                self.compile_var(child_var)
            return
        if var not in self._pyomo_var_to_solver_var_map:
            raise ValueError('The Var provided to compile_var needs to be added first: {0}'.format(var))
        cplex_var = self._pyomo_var_to_solver_var_map[var]
        vtype = self._cplex_vtype_from_var(var)
        if var.is_fixed():
            lb = var.value
            ub = var.value
        else:
            lb = value(var.lb)
            ub = value(var.ub)
        if lb is None:
            lb = -self._cplex.infinity
        if ub is None:
            ub = self._cplex.infinity

        self._solver_model.variables.set_lower_bounds(cplex_var, lb)
        self._solver_model.variables.set_upper_bounds(cplex_var, ub)
        self._solver_model.variables.set_types(cplex_var, vtype)

    def write(self, filename, filetype=''):
        self._solver_model.write(filename, filetype=filetype)
