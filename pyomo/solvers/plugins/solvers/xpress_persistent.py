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
from pyomo.solvers.plugins.solvers.xpress_direct import XpressDirect
from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver
from pyomo.core.expr.numvalue import value, is_fixed
from pyomo.core.expr import current as EXPR
from pyomo.opt.base import SolverFactory
import collections


@SolverFactory.register('xpress_persistent', doc='Persistent python interface to Xpress')
class XpressPersistent(PersistentSolver, XpressDirect):
    """
    A class that provides a persistent interface to Xpress. Direct solver interfaces do not use any file io.
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
        kwds['type'] = 'xpress_persistent'
        XpressDirect.__init__(self, **kwds)

        self._pyomo_model = kwds.pop('model', None)
        if self._pyomo_model is not None:
            self.set_instance(self._pyomo_model, **kwds)

    def _remove_constraint(self, solver_con):
        self._solver_model.delConstraint(solver_con)

    def _remove_sos_constraint(self, solver_sos_con):
        self._solver_model.delSOS(solver_sos_con)

    def _remove_var(self, solver_var):
        self._solver_model.delVariable(solver_var)

    def _warm_start(self):
        XpressDirect._warm_start(self)

    def _xpress_chgcoltype_from_var(self, var):
        """
        This function takes a pyomo variable and returns the appropriate xpress variable type
        for use in xpress.problem.chgcoltype
        :param var: pyomo.core.base.var.Var
        :return: xpress.continuous or xpress.binary or xpress.integer
        """
        if var.is_binary():
            vartype = 'B'
        elif var.is_integer():
            vartype = 'I'
        elif var.is_continuous():
            vartype = 'C'
        else:
            raise ValueError('Variable domain type is not recognized for {0}'.format(var.domain))
        return vartype

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
        xpress_var = self._pyomo_var_to_solver_var_map[var]
        qctype = self._xpress_chgcoltype_from_var(var)
        lb, ub = self._xpress_lb_ub_from_var(var)

        self._solver_model.chgbounds([xpress_var, xpress_var], ['L', 'U'], [lb, ub])
        self._solver_model.chgcoltype([xpress_var], [qctype])

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

        ## set-up add var
        varname = self._symbol_map.getSymbol(var, self._labeler)
        vartype = self._xpress_chgcoltype_from_var(var)
        lb, ub = self._xpress_lb_ub_from_var(var)

        self._solver_model.addcols(objx=[obj_coef], mstart=[0,len(coefficients)],
                                    mrwind=constraints, dmatval=coefficients, 
                                    bdl=[lb], bdu=[ub], names=[varname], 
                                    types=[vartype])

        xpress_var = self._solver_model.getVariable(
                        index=self._solver_model.getIndexFromName(type=2, name=varname))

        self._pyomo_var_to_solver_var_map[var] = xpress_var
        self._solver_var_to_pyomo_var_map[xpress_var] = var
        self._referenced_variables[var] = len(coefficients)

    def get_xpress_attribute(self, *args):
        """
        Get xpress atrributes.

        Parameters
        ----------
        control(s): str, strs, list, None
            The xpress attribute to get. Options include any xpress attribute.
            Can also be list of xpress controls or None for every atrribute
            Please see the Xpress documentation for options.

        See the Xpress documentation for xpress.problem.getAttrib for other
        uses of this function

        Returns
        -------
        control value or dictionary of control values
        """
        return self._solver_model.getAttrib(*args)

    def set_xpress_control(self, *args):
        """
        Set xpress controls.

        Parameters
        ----------
        control: str
            The xpress control to set. Options include any xpree control.
            Please see the Xpress documentation for options.
        val: any
            The value to set the control to. See Xpress documentation for possible values.

        If one argument, it must be a dictionary with control keys and control values
        """
        self._solver_model.setControl(*args)

    def get_xpress_control(self, *args):
        """
        Get xpress controls.

        Parameters
        ----------
        control(s): str, strs, list, None
            The xpress control to get. Options include any xpress control.
            Can also be list of xpress controls or None for every contorl
            Please see the Xpress documentation for options.

        See the Xpress documentation for xpress.problem.getControl for other
        uses of this function

        Returns
        -------
        control value or dictionary of control values
        """
        return self._solver_model.getControl(*args)

    def write(self, filename, flags=''):
        """
        Write the model to a file (e.g., a lp file).

        Parameters
        ----------
        filename: str
            Name of the file to which the model should be written.
        flags: str
            Flags for xpress.problem.write
        """
        self._solver_model.write(filename, flags)
