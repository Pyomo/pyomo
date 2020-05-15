#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.core.expr.numvalue import value, native_numeric_types
from pyomo.core.base.PyomoModel import ConcreteModel
from pyomo.core.base.constraint import Constraint
from pyomo.core.base.var import Var
from pyomo.core.base.sos import SOSConstraint
from pyomo.solvers.plugins.solvers.cplex_direct import CPLEXDirect
from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver
from pyomo.pysp.phutils import find_active_objective
from pyomo.solvers.plugins.solvers.xpress_persistent import _convert_to_const
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

    def add_column(self, var, obj_term, constraints, coefficients):
        """Add a column to the solver's and Pyomo model

        This will add the Pyomo variable var to the solver's
        model, and put the coefficients on the associated 
        constraints in the solver model. If the obj_term is
        not zero, it will add obj_term*var to the objective 
        of both the Pyomo and solver's model.

        Parameters
        ----------
        var: Var (scalar Var or single _VarData)
        obj_term: float, pyo.Param

        constraints: list of scalar Constraints of single _ConstraintDatas  
        coefficients: the coefficient to put on var in the associated constraint
        """
        
        ## process the objective
        obj_term_const = False
        if obj_term.__class__ in native_numeric_types and obj_term == 0.:
            pass ## nothing to do
        else:
            obj = find_active_objective(self._pyomo_model, True)
            obj.expr += obj_term*var

        obj_coef = _convert_to_const(obj_term)

        ## add the constraints, collect the
        ## column information
        coeff_list = list()
        constr_list = list()
        for val,c in zip(coefficients,constraints):
            c._body += val*var
            self._vars_referenced_by_con[c].add(var)

            cval = _convert_to_const(val)
            coeff_list.append(cval)
            constr_list.append(self._pyomo_con_to_solver_con_map[c])

        ## set-up add var
        varname = self._symbol_map.getSymbol(var, self._labeler)
        vtype = self._cplex_vtype_from_var(var)
        if var.has_lb():
            lb = value(var.lb)
        else:
            lb = -self._cplex.infinity
        if var.has_ub():
            ub = value(var.ub)
        else:
            ub = self._cplex.infinity

        ## do column addition
        self._solver_model.variables.add(obj=[obj_coef], lb=[lb], ub=[ub], types=[vtype], names=[varname],
                            columns=[self._cplex.SparsePair(ind=constr_list, val=coeff_list)])

        self._pyomo_var_to_solver_var_map[var] = varname
        self._solver_var_to_pyomo_var_map[varname] = var
        self._pyomo_var_to_ndx_map[var] = self._ndx_count
        self._ndx_count += 1
        self._referenced_variables[var] = len(coeff_list)
