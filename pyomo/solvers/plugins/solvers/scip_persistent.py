#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
from pyomo.solvers.plugins.solvers.scip_direct import SCIPDirect
from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver
from pyomo.opt.base import SolverFactory


@SolverFactory.register("scip_persistent", doc="Persistent python interface to SCIP")
class SCIPPersistent(PersistentSolver, SCIPDirect):
    """
    A class that provides a persistent interface to SCIP. Direct solver interfaces do not use any file io.
    Rather, they interface directly with the python bindings for the specific solver. Persistent solver interfaces
    are similar except that they "remember" their model. Thus, persistent solver interfaces allow incremental changes
    to the solver model (e.g., the gurobi python model or the cplex python model). Note that users are responsible
    for notifying the persistent solver interfaces when changes are made to the corresponding pyomo model.

    Keyword Arguments
    -----------------
    model: ConcreteModel
        Passing a model to the constructor is equivalent to calling the set_instance method.
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
        kwds["type"] = "scip_persistent"
        PersistentSolver.__init__(self, **kwds)
        SCIPDirect._init(self)

        self._pyomo_model = kwds.pop("model", None)
        if self._pyomo_model is not None:
            self.set_instance(self._pyomo_model, **kwds)

    def _remove_constraint(self, solver_conname):
        con = self._solver_con_to_pyomo_con_map[solver_conname]
        scip_con = self._pyomo_con_to_solver_con_expr_map[con]
        self._solver_model.delCons(scip_con)
        del self._pyomo_con_to_solver_con_expr_map[con]

    def _remove_sos_constraint(self, solver_sos_conname):
        con = self._solver_con_to_pyomo_con_map[solver_sos_conname]
        scip_con = self._pyomo_con_to_solver_con_expr_map[con]
        self._solver_model.delCons(scip_con)
        del self._pyomo_con_to_solver_con_expr_map[con]

    def _remove_var(self, solver_varname):
        var = self._solver_var_to_pyomo_var_map[solver_varname]
        scip_var = self._pyomo_var_to_solver_var_expr_map[var]
        self._solver_model.delVar(scip_var)
        del self._pyomo_var_to_solver_var_expr_map[var]

    def _warm_start(self):
        SCIPDirect._warm_start(self)

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
        # if var.is_indexed():
        #    for child_var in var.values():
        #        self.compile_var(child_var)
        #    return
        if var not in self._pyomo_var_to_solver_var_map:
            raise ValueError(
                f"The Var provided to compile_var needs to be added first: {var}"
            )
        scip_var = self._pyomo_var_to_solver_var_map[var]
        vtype = self._scip_vtype_from_var(var)
        lb, ub = self._scip_lb_ub_from_var(var)

        self._solver_model.chgVarLb(scip_var, lb)
        self._solver_model.chgVarUb(scip_var, ub)
        self._solver_model.chgVarType(scip_var, vtype)

    def write(self, filename, filetype=""):
        """
        Write the model to a file (e.g., an lp file).

        Parameters
        ----------
        filename: str
            Name of the file to which the model should be written.
        filetype: str
            The file type (e.g., lp).
        """
        self._solver_model.writeProblem(filename + filetype)

    def set_scip_param(self, param, val):
        """
        Set a SCIP parameter.

        Parameters
        ----------
        param: str
            The SCIP parameter to set. Options include any SCIP parameter.
            Please see the SCIP documentation for options.
            Link at: https://www.scipopt.org/doc/html/PARAMETERS.php
        val: any
            The value to set the parameter to. See SCIP documentation for possible values.
        """
        self._solver_model.setParam(param, val)

    def get_scip_param(self, param):
        """
        Get the value of the SCIP parameter.

        Parameters
        ----------
        param: str or int or float
            The SCIP parameter to get the value of. See SCIP documentation for possible options.
            Link at: https://www.scipopt.org/doc/html/PARAMETERS.php
        """
        return self._solver_model.getParam(param)

    def _add_column(self, var, obj_coef, constraints, coefficients):
        """Add a column to the solver's model

        This will add the Pyomo variable var to the solver's
        model, and put the coefficients on the associated
        constraints in the solver model. If the obj_coef is
        not zero, it will add obj_coef*var to the objective
        of the solver's model.

        Parameters
        ----------
        var: Var (scalar Var or single _VarData)
        obj_coef: float
        constraints: list of solver constraints
        coefficients: list of coefficients to put on var in the associated constraint
        """

        # Set-up add var
        varname = self._symbol_map.getSymbol(var, self._labeler)
        vtype = self._scip_vtype_from_var(var)
        lb, ub = self._scip_lb_ub_from_var(var)

        # Add the variable to the model and then to all the constraints
        scip_var = self._solver_model.addVar(lb=lb, ub=ub, vtype=vtype, name=varname)
        self._pyomo_var_to_solver_var_expr_map[var] = scip_var
        self._solver_var_to_pyomo_var_map[varname] = var
        self._referenced_variables[var] = len(coefficients)

        # Get the SCIP cons by passing through two dictionaries
        pyomo_cons = [self._solver_con_to_pyomo_con_map[con] for con in constraints]
        scip_cons = [
            self._pyomo_con_to_solver_con_expr_map[pyomo_con]
            for pyomo_con in pyomo_cons
        ]

        for i, scip_con in enumerate(scip_cons):
            if not scip_con.isLinear():
                raise ValueError(
                    "_add_column functionality not supported for non-linear constraints"
                )
            self._solver_model.addConsCoeff(scip_con, scip_var, coefficients[i])
            con = self._solver_con_to_pyomo_con_map[scip_con.name]
            self._vars_referenced_by_con[con].add(var)

        sense = self._solver_model.getObjectiveSense()
        self._solver_model.setObjective(obj_coef * scip_var, sense=sense, clear=False)

    def reset(self):
        """This function is necessary to call before making any changes to the
        SCIP model after optimizing. It frees solution run specific information
        that is not automatically done when changes to an already solved model
        are made. Making changes to an already optimized model, e.g. adding additional
        constraints will raise an error unless this function is called."""
        self._solver_model.freeTransform()
