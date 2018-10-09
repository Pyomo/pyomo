#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.core.expr.numvalue import value
from pyomo.core.base.PyomoModel import ConcreteModel
from pyomo.core.base.constraint import Constraint
from pyomo.core.base.var import Var
from pyomo.core.base.sos import SOSConstraint
from pyomo.solvers.plugins.solvers.cplex_direct import CPLEXDirect
from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver
from pyomo.opt.base import SolverFactory


@SolverFactory.register('cplex_persistent', doc='Persistent python interface to CPLEX')
class CPLEXPersistent(PersistentSolver, CPLEXDirect):
    """
    A class that provides a persistent interface to Cplex. Direct solver interfaces do not use any file io.
    Rather, they interface directly with the python bindings for the specific solver. Persistent solver interfaces
    are similar except that they "remember" their model. Thus, persistent solver interfaces allow incremental changes
    to the solver model (e.g., the gurobi python model or the cplex python model). Note that users are responsible
    for notifying the persistent solver interfaces when changes are made to the corresponding pyomo model.

    Keyword Arguments
    -----------------
    model: ConcreteModel
        Passing a model to the constructor is equivalent to calling the set_instance mehtod.
    type: str
        String indicating the class type of the solver instance.
    name: str
        String representing either the class type of the solver instance or an assigned name.
    doc: str
        Documentation for the solver
    options: dict
        Dictionary of solver options
    """

    def __init__(self, **kwds):
        kwds['type'] = 'cplex_persistent'
        PersistentSolver.__init__(self, **kwds)
        CPLEXDirect._init(self)

        self._pyomo_model = kwds.pop('model', None)
        if self._pyomo_model is not None:
            self.set_instance(self._pyomo_model, **kwds)

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
        pyomo_var = self._solver_var_to_pyomo_var_map[solver_var]
        ndx = self._pyomo_var_to_ndx_map[pyomo_var]
        for tmp_var, tmp_ndx in self._pyomo_var_to_ndx_map.items():
            if tmp_ndx > ndx:
                self._pyomo_var_to_ndx_map[tmp_var] -= 1
        self._ndx_count -= 1
        del self._pyomo_var_to_ndx_map[pyomo_var]
        self._solver_model.variables.delete(solver_var)

    def _warm_start(self):
        CPLEXDirect._warm_start(self)

    def update_var(self, var):
        """Update a single variable in the solver's model.

        This will update bounds, fix/unfix the variable as needed, and
        update the variable type.

        Parameters
        ----------
        var: Var (scalar Var or single _VarData)

        """
        # see PR #366 for discussion about handling indexed
        # objects and keeping compatibility with the
        # pyomo.kernel objects
        #if var.is_indexed():
        #    for child_var in var.values():
        #        self.compile_var(child_var)
        #    return
        if var not in self._pyomo_var_to_solver_var_map:
            raise ValueError('The Var provided to compile_var needs to be added first: {0}'.format(var))
        cplex_var = self._pyomo_var_to_solver_var_map[var]
        vtype = self._cplex_vtype_from_var(var)
        if var.is_fixed():
            lb = var.value
            ub = var.value
        else:
            lb = -self._cplex.infinity
            ub = self._cplex.infinity
            if var.has_lb():
                lb = value(var.lb)
            if var.has_ub():
                ub = value(var.ub)
        self._solver_model.variables.set_lower_bounds(cplex_var, lb)
        self._solver_model.variables.set_upper_bounds(cplex_var, ub)
        self._solver_model.variables.set_types(cplex_var, vtype)

    def write(self, filename, filetype=''):
        """
        Write the model to a file (e.g., and lp file).

        Parameters
        ----------
        filename: str
            Name of the file to which the model should be written.
        filetype: str
            The file type (e.g., lp).
        """
        self._solver_model.write(filename, filetype=filetype)
