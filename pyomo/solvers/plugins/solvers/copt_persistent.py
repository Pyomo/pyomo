#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.solvers.plugins.solvers.copt_direct import CoptDirect, coptpy_available
from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver
from pyomo.opt.base import SolverFactory

if coptpy_available:
    import coptpy

@SolverFactory.register('copt_persistent', doc='Persistent python interface to COPT')
class CoptPersistent(PersistentSolver, CoptDirect):
    """
    A class that provides a persistent interface to COPT. Persistent interface is similar to direct
    interface excepts that it 'remember' their model, thus it allows incremental changes to the model.
    Note that users are responsible for notifying the persistent solver interface when changes were
    made to the corresponding pyomo model.

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
        kwds['type'] = 'copt_persistent'
        CoptDirect.__init__(self, **kwds)

        self._pyomo_model = kwds.pop('model', None)
        if self._pyomo_model is not None:
            self.set_instance(self._pyomo_model, **kwds)

    def _remove_constraint(self, solver_con):
        self._solver_model.remove(solver_con)

    def _remove_sos_constraint(self, solver_sos_con):
        self._solver_model.remove(solver_sos_con)

    def _remove_var(self, solver_var):
        self._solver_model.remove(solver_var)

    def _warm_start(self):
        CoptDirect._warm_start(self)

    def update_var(self, var):
        """
        Update a single variable in the solver's model.
        This will update bounds, fix/unfix the variable as needed, and update the variable type.

        Parameters
        ----------
        var: Var (scalar Var or single _VarData)

        """
        if var not in self._pyomo_var_to_solver_var_map:
            raise ValueError(
                'The Var provided to update_var needs to be added first: {0}'.format(
                    var
                )
            )

        coptpy_var = self._pyomo_var_to_solver_var_map[var]
        vtype = self._copt_vtype_from_var(var)
        lb, ub = self._copt_lb_ub_from_var(var)

        coptpy_var.lb = lb
        coptpy_var.ub = ub
        coptpy_var.vtype = vtype

    def write(self, filename):
        """
        Write the model to a file.

        Parameters
        ----------
        filename: str
            Name of the file to which the model should be written.
        """
        self._solver_model.write(filename)

    def set_linear_constraint_attr(self, con, attr, val):
        """
        Set the value of information on a COPT linear constraint.

        Parameters
        ----------
        con: pyomo.core.base.constraint._GeneralConstraintData
            The pyomo constraint for which the corresponding COPT constraint information should be modified.
        attr: str
            The information to be modified. See COPT documentation for details.
        val: any
            See COPT documentation for acceptable values.
        """
        setattr(self._pyomo_con_to_solver_con_map[con], attr, val)

    def set_var_attr(self, var, attr, val):
        """
        Set the value of information on a COPT variable.

        Parameters
        ----------
        con: pyomo.core.base.var._GeneralVarData
            The pyomo var for which the corresponding COPT variable information should be modified.
        attr: str
            The information to be modified. See COPT documentation for details.
        val: any
            See COPT documentation for acceptable values.
        """
        setattr(self._pyomo_var_to_solver_var_map[var], attr, val)

    def get_model_attr(self, attr):
        """
        Get the value of an attribute on the COPT model.

        Parameters
        ----------
        attr: str
            The attribute to get. See COPT documentation for descriptions of the attributes.
        """
        return getattr(self._solver_model, attr)

    def get_var_attr(self, var, attr):
        """
        Get the value of information on a COPT variable.

        Parameters
        ----------
        var: pyomo.core.base.var._GeneralVarData
            The pyomo var for which the corresponding COPT variable attribute should be retrieved.
        attr: str
            The information to get. See COPT documentation for details.
        """
        return getattr(self._pyomo_var_to_solver_var_map[var], attr)

    def get_linear_constraint_attr(self, con, attr):
        """
        Get the value of information on a COPT linear constraint.

        Parameters
        ----------
        con: pyomo.core.base.constraint._GeneralConstraintData
            The pyomo constraint for which the corresponding COPT constraint information should be retrieved.
        attr: str
            The attribute to get. See COPT documentation for details.
        """
        return getattr(self._pyomo_con_to_solver_con_map[con], attr)

    def get_sos_attr(self, con, attr):
        """
        Get the value of information on a COPT sos constraint.

        Parameters
        ----------
        con: pyomo.core.base.sos._SOSConstraintData
            The pyomo SOS constraint for which the corresponding COPT SOS constraint information
            should be retrieved.
        attr: str
            The information to get. See COPT documentation for details.
        """
        return getattr(self._pyomo_con_to_solver_con_map[con], attr)

    def get_quadratic_constraint_attr(self, con, attr):
        """
        Get the value of information on a COPT quadratic constraint.

        Parameters
        ----------
        con: pyomo.core.base.constraint._GeneralConstraintData
            The pyomo constraint for which the corresponding COPT quadratic constraint information
            should be retrieved.
        attr: str
            The information to get. See COPT documentation for details.
        """
        return getattr(self._pyomo_con_to_solver_con_map[con], attr)

    def set_copt_param(self, param, val):
        """
        Set a COPT parameter.

        Parameters
        ----------
        param: str
            The COPT parameter to set. Options include any COPT parameter.
            Please see the COPT documentation for options.
        val: any
            The value to set the parameter to. See COPT documentation for possible values.
        """
        self._solver_model.setParam(param, val)

    def get_copt_param_info(self, param):
        """
        Get information about a COPT parameter.

        Parameters
        ----------
        param: str
            The COPT parameter to get information for. See COPT documentation for possible options.

        Returns
        -------
        A 5-tuple containing the parameter name, current value, default value, minimum value and maximum value.
        """
        return self._solver_model.getParamInfo(param)

    def _add_column(self, var, obj_coef, constraints, coefficients):
        """
        Add a column to the solver's model

        This will add the Pyomo variable var to the solver's model, and put the coefficients on the
        associated constraints in the solver model. If the obj_coef is not zero, it will add
        obj_coef*var to the objective of the solver's model.

        Parameters
        ----------
        var: Var (scalar Var or single _VarData)
        obj_coef: float
        constraints: list of solver constraints
        coefficients: list of coefficients to put on var in the associated constraint
        """
        varname = self._symbol_map.getSymbol(var, self._labeler)
        vtype = self._copt_vtype_from_var(var)
        lb, ub = self._copt_lb_ub_from_var(var)

        coptpy_var = self._solver_model.addVar(
            obj=obj_coef,
            lb=lb,
            ub=ub,
            vtype=vtype,
            name=varname,
            column=coptpy.Column(constrs=constraints, coeffs=coefficients),
        )

        self._pyomo_var_to_solver_var_map[var] = coptpy_var
        self._solver_var_to_pyomo_var_map[coptpy_var] = var
        self._referenced_variables[var] = len(coefficients)

    def reset(self):
        self._solver_model.reset()
