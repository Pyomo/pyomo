#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.core.base.PyomoModel import ConcreteModel
from pyomo.solvers.plugins.solvers.gurobi_direct import GurobiDirect
from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver
from pyomo.core.expr.numvalue import value
from pyomo.opt.base import SolverFactory


@SolverFactory.register('gurobi_persistent', doc='Persistent python interface to Gurobi')
class GurobiPersistent(PersistentSolver, GurobiDirect):
    """
    A class that provides a persistent interface to Gurobi. Direct solver interfaces do not use any file io.
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
        kwds['type'] = 'gurobi_persistent'
        PersistentSolver.__init__(self, **kwds)
        GurobiDirect._init(self)

        self._pyomo_model = kwds.pop('model', None)
        if self._pyomo_model is not None:
            self.set_instance(self._pyomo_model, **kwds)

    def _remove_constraint(self, solver_con):
        self._solver_model.remove(solver_con)

    def _remove_sos_constraint(self, solver_sos_con):
        self._solver_model.remove(solver_sos_con)

    def _remove_var(self, solver_var):
        self._solver_model.remove(solver_var)

    def add_var(self, var):
        """
        Add a variable to the solver's model. This will keep any existing model components intact.

        Parameters
        ----------
        var: Var
            The variable to add to the solver's model.
        """
        PersistentSolver.add_var(self, var)
        self._solver_model.update()

    def add_constraint(self, con):
        """
        Add a constraint to the solver's model. This will keep any existing model components intact.

        Parameters
        ----------
        con: Constraint
        """
        PersistentSolver.add_constraint(self, con)
        self._solver_model.update()

    def add_sos_constraint(self, con):
        """
        Add an SOS constraint to the solver's model (if supported). This will keep any existing model components intact.

        Parameters
        ----------
        con: SOSConstraint
        """
        PersistentSolver.add_sos_constraint(self, con)
        self._solver_model.update()

    def _warm_start(self):
        GurobiDirect._warm_start(self)

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
        #        self.update_var(child_var)
        #    return
        if var not in self._pyomo_var_to_solver_var_map:
            raise ValueError('The Var provided to update_var needs to be added first: {0}'.format(var))
        gurobipy_var = self._pyomo_var_to_solver_var_map[var]
        vtype = self._gurobi_vtype_from_var(var)
        if var.is_fixed():
            lb = var.value
            ub = var.value
        else:
            lb = -self._gurobipy.GRB.INFINITY
            ub = self._gurobipy.GRB.INFINITY
            if var.has_lb():
                lb = value(var.lb)
            if var.has_ub():
                ub = value(var.ub)
        gurobipy_var.setAttr('lb', lb)
        gurobipy_var.setAttr('ub', ub)
        gurobipy_var.setAttr('vtype', vtype)

    def write(self, filename):
        """
        Write the model to a file (e.g., and lp file).

        Parameters
        ----------
        filename: str
            Name of the file to which the model should be written.
        """
        self._solver_model.write(filename)
