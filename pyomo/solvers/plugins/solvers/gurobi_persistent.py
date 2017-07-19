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
from pyomo.core.kernel.numvalue import value


class GurobiPersistent(PersistentSolver, GurobiDirect):
    alias('gurobi_persistent', doc='Persistent python interface to Gurobi')

    def __init__(self, **kwds):
        kwds['type'] = 'gurobi_persistent'
        PersistentSolver.__init__(self, **kwds)
        GurobiDirect._init(self)

        self._pyomo_model = kwds.pop('model', None)
        if self._pyomo_model is not None:
            self.compile_instance(self._pyomo_model, **kwds)

    def _remove_constraint(self, solver_con):
        self._solver_model.remove(solver_con)

    def _remove_sos_constraint(self, solver_sos_con):
        self._solver_model.remove(solver_sos_con)

    def _remove_var(self, solver_var):
        self._solver_model.remove(solver_var)

    def add_var(self, var):
        if var.is_indexed():
            for child_var in var.values():
                self.add_var(child_var)
            return
        self._add_var(var)
        self._solver_model.update()

    def _warm_start(self):
        GurobiDirect._warm_start(self)

    def compile_var(self, var):
        if var.is_indexed():
            for child_var in var.values():
                self.compile_var(child_var)
            return
        if var not in self._pyomo_var_to_solver_var_map:
            raise ValueError('The Var provided to compile_var needs to be added first: {0}'.format(var))
        gurobipy_var = self._pyomo_var_to_solver_var_map[var]
        vtype = self._gurobi_vtype_from_var(var)
        if var.is_fixed():
            lb = var.value
            ub = var.value
        else:
            lb = value(var.lb)
            ub = value(var.ub)
        if lb is None:
            lb = -self._gurobipy.GRB.INFINITY
        if ub is None:
            ub = self._gurobipy.GRB.INFINITY

        gurobipy_var.setAttr('lb', lb)
        gurobipy_var.setAttr('ub', ub)
        gurobipy_var.setAttr('vtype', vtype)

    def write(self, filename):
        self._solver_model.write(filename)
