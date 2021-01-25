#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.solvers.plugins.solvers.direct_or_persistent_solver import DirectOrPersistentSolver
from pyomo.core.base.block import _BlockData
from pyomo.core.kernel.block import IBlock
from pyomo.core.base.suffix import active_import_suffix_generator
from pyomo.core.kernel.suffix import import_suffix_generator
from pyomo.core.expr.numvalue import native_numeric_types, value
from pyomo.core.expr.visitor import evaluate_expression
from pyomo.core.base.constraint import Constraint
from pyomo.core.base.var import Var
from pyomo.core.base.sos import SOSConstraint

from pyomo.common.errors import ApplicationError
from pyomo.common.collections import Options

import time
import logging

logger = logging.getLogger('pyomo.solvers')

def _convert_to_const(val):
    if val.__class__ in native_numeric_types:
        return val
    elif val.is_expression_type():
        return evaluate_expression(val)
    else:
        return value(val)

class PersistentSolver(DirectOrPersistentSolver):
    """
    A base class for persistent solvers. Direct solver interfaces do not use any file io.
    Rather, they interface directly with the python bindings for the specific solver. Persistent solver interfaces
    are similar except that they "remember" their model. Thus, persistent solver interfaces allow incremental changes
    to the solver model (e.g., the gurobi python model or the cplex python model). Note that users are responsible
    for notifying the persistent solver interfaces when changes are made to the corresponding pyomo model.

    Keyword Arguments
    -----------------
    type: str
        String indicating the class type of the solver instance.
    name: str
        String representing either the class type of the solver instance or an assigned name.
    doc: str
        Documentation for the solver
    options: dict
        Dictionary of solver options
    """

    def _presolve(self, **kwds):
        DirectOrPersistentSolver._presolve(self, **kwds)

    def set_instance(self, model, **kwds):
        """
        This method is used to translate the Pyomo model provided to an instance of the solver's Python model. This
        discards any existing model and starts from scratch.

        Parameters
        ----------
        model: ConcreteModel
            The pyomo model to be used with the solver.

        Keyword Arguments
        -----------------
        symbolic_solver_labels: bool
            If True, the solver's components (e.g., variables, constraints) will be given names that correspond to
            the Pyomo component names.
        skip_trivial_constraints: bool
            If True, then any constraints with a constant body will not be added to the solver model.
            Be careful with this. If a trivial constraint is skipped then that constraint cannot be removed from
            a persistent solver (an error will be raised if a user tries to remove a non-existent constraint).
        output_fixed_variable_bounds: bool
            If False then an error will be raised if a fixed variable is used in one of the solver constraints.
            This is useful for catching bugs. Ordinarily a fixed variable should appear as a constant value in the
            solver constraints. If True, then the error will not be raised.
        """
        return self._set_instance(model, kwds)

    def add_block(self, block):
        """Add a single Pyomo Block to the solver's model.

        This will keep any existing model components intact.

        Parameters
        ----------
        block: Block (scalar Block or single _BlockData)

        """
        if self._pyomo_model is None:
            raise RuntimeError('You must call set_instance before calling add_block.')
        # see PR #366 for discussion about handling indexed
        # objects and keeping compatibility with the
        # pyomo.kernel objects
        #if block.is_indexed():
        #    for sub_block in block.values():
        #        self._add_block(block)
        #    return
        self._add_block(block)

    def set_objective(self, obj):
        """
        Set the solver's objective. Note that, at least for now, any existing objective will be discarded. Other than
        that, any existing model components will remain intact.

        Parameters
        ----------
        obj: Objective
        """
        if self._pyomo_model is None:
            raise RuntimeError('You must call set_instance before calling set_objective.')
        return self._set_objective(obj)

    def add_constraint(self, con):
        """Add a single constraint to the solver's model.

        This will keep any existing model components intact.

        Parameters
        ----------
        con: Constraint (scalar Constraint or single _ConstraintData)

        """
        if self._pyomo_model is None:
            raise RuntimeError('You must call set_instance before calling add_constraint.')
        # see PR #366 for discussion about handling indexed
        # objects and keeping compatibility with the
        # pyomo.kernel objects
        #if con.is_indexed():
        #    for child_con in con.values():
        #        self._add_constraint(child_con)
        #else:
        self._add_constraint(con)

    def add_var(self, var):
        """Add a single variable to the solver's model.

        This will keep any existing model components intact.

        Parameters
        ----------
        var: Var

        """
        if self._pyomo_model is None:
            raise RuntimeError('You must call set_instance before calling add_var.')
        if id(self._pyomo_model) != id(var.model()):
            raise RuntimeError('The pyomo var must be attached to the solver model')
        # see PR #366 for discussion about handling indexed
        # objects and keeping compatibility with the
        # pyomo.kernel objects
        #if var.is_indexed():
        #    for child_var in var.values():
        #        self._add_var(child_var)
        #else:
        self._add_var(var)

    def add_sos_constraint(self, con):
        """Add a single SOS constraint to the solver's model (if supported).

        This will keep any existing model components intact.

        Parameters
        ----------
        con: SOSConstraint

        """
        if self._pyomo_model is None:
            raise RuntimeError('You must call set_instance before calling add_sos_constraint.')
        # see PR #366 for discussion about handling indexed
        # objects and keeping compatibility with the
        # pyomo.kernel objects
        #if con.is_indexed():
        #    for child_con in con.values():
        #        self._add_sos_constraint(child_con)
        #else:
        self._add_sos_constraint(con)

    def add_column(self, model, var, obj_coef, constraints, coefficients):
        """Add a column to the solver's and Pyomo model

        This will add the Pyomo variable var to the solver's
        model, and put the coefficients on the associated 
        constraints in the solver model. If the obj_coef is
        not zero, it will add obj_coef*var to the objective 
        of both the Pyomo and solver's model.

        Parameters
        ----------
        model: pyomo ConcreteModel to which the column will be added
        var: Var (scalar Var or single _VarData)
        obj_coef: float, pyo.Param
        constraints: list of scalar Constraints of single _ConstraintDatas  
        coefficients: list of the coefficient to put on var in the associated constraint

        """
        if self._pyomo_model is None:
            raise RuntimeError('You must call set_instance before calling add_column.')
        if id(self._pyomo_model) != id(model):
            raise RuntimeError('The pyomo model which the column is being added to '
                                'must be the same as the pyomo model attached to this '
                                'PersistentSolver instance; i.e., the same pyomo model '
                                'used in set_instance.')
        if id(self._pyomo_model) != id(var.model()):
            raise RuntimeError('The pyomo var must be attached to the solver model')
        if var in self._pyomo_var_to_solver_var_map:
            raise RuntimeError('The pyomo var must not have been already added to '
                                'the solver model')
        if len(constraints) != len(coefficients):
            raise RuntimeError('The list of constraints and the list of coefficents '
                               'be of equal length')
        obj_coef, constraints, coefficients = self._add_and_collect_column_data(
                var, obj_coef, constraints, coefficients)
        self._add_column(var, obj_coef, constraints, coefficients)

    """ This method should be implemented by subclasses."""
    def _add_column(self, var, obj_coef, constraints, coefficients):
        raise NotImplementedError('This method should be implemented by subclasses.')

    def _add_and_collect_column_data(self, var, obj_coef, constraints, coefficients):
        """
        Update the objective Pyomo objective function and constraints, and update
        the _vars_referenced_by Maps

        Returns the column and objective coefficient data to pass to the solver
        """
        ## process the objective
        if obj_coef.__class__ in native_numeric_types and obj_coef == 0.:
            pass ## nothing to do
        else:
            self._objective.expr += obj_coef*var
            self._vars_referenced_by_obj.add(var)
            obj_coef = _convert_to_const(obj_coef)

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

        return obj_coef, constr_list, coeff_list

    """ This method should be implemented by subclasses."""
    def _remove_constraint(self, solver_con):
        raise NotImplementedError('This method should be implemented by subclasses.')

    """ This method should be implemented by subclasses."""
    def _remove_sos_constraint(self, solver_sos_con):
        raise NotImplementedError('This method should be implemented by subclasses.')

    """ This method should be implemented by subclasses."""
    def _remove_var(self, solver_var):
        raise NotImplementedError('This method should be implemented by subclasses.')

    def remove_block(self, block):
        """Remove a single block from the solver's model.

        This will keep any other model components intact.

        WARNING: Users must call remove_block BEFORE modifying the block.

        Parameters
        ----------
        block: Block (scalar Block or a single _BlockData)

        """
        # see PR #366 for discussion about handling indexed
        # objects and keeping compatibility with the
        # pyomo.kernel objects
        #if block.is_indexed():
        #    for sub_block in block.values():
        #        self.remove_block(sub_block)
        #    return
        for sub_block in block.block_data_objects(descend_into=True, active=True):
            for con in sub_block.component_data_objects(ctype=Constraint, descend_into=False, active=True):
                self.remove_constraint(con)

            for con in sub_block.component_data_objects(ctype=SOSConstraint, descend_into=False, active=True):
                self.remove_sos_constraint(con)

        for var in block.component_data_objects(ctype=Var, descend_into=True, active=True):
            self.remove_var(var)

    def remove_constraint(self, con):
        """Remove a single constraint from the solver's model.

        This will keep any other model components intact.

        Parameters
        ----------
        con: Constraint (scalar Constraint or single _ConstraintData)

        """
        # see PR #366 for discussion about handling indexed
        # objects and keeping compatibility with the
        # pyomo.kernel objects
        #if con.is_indexed():
        #    for child_con in con.values():
        #        self.remove_constraint(child_con)
        #    return
        solver_con = self._pyomo_con_to_solver_con_map[con]
        self._remove_constraint(solver_con)
        self._symbol_map.removeSymbol(con)
        self._labeler.remove_obj(con)
        for var in self._vars_referenced_by_con[con]:
            self._referenced_variables[var] -= 1
        del self._vars_referenced_by_con[con]
        del self._pyomo_con_to_solver_con_map[con]
        del self._solver_con_to_pyomo_con_map[solver_con]

    def remove_sos_constraint(self, con):
        """Remove a single SOS constraint from the solver's model.

        This will keep any other model components intact.

        Parameters
        ----------
        con: SOSConstraint

        """
        # see PR #366 for discussion about handling indexed
        # objects and keeping compatibility with the
        # pyomo.kernel objects
        #if con.is_indexed():
        #    for child_con in con.values():
        #        self.remove_sos_constraint(child_con)
        #    return
        solver_con = self._pyomo_con_to_solver_con_map[con]
        self._remove_sos_constraint(solver_con)
        self._symbol_map.removeSymbol(con)
        self._labeler.remove_obj(con)
        for var in self._vars_referenced_by_con[con]:
            self._referenced_variables[var] -= 1
        del self._vars_referenced_by_con[con]
        del self._pyomo_con_to_solver_con_map[con]
        del self._solver_con_to_pyomo_con_map[solver_con]

    def remove_var(self, var):
        """Remove a single variable from the solver's model.

        This will keep any other model components intact.

        Parameters
        ----------
        var: Var (scalar Var or single _VarData)

        """
        # see PR #366 for discussion about handling indexed
        # objects and keeping compatibility with the
        # pyomo.kernel objects
        #if var.is_indexed():
        #    for child_var in var.values():
        #        self.remove_var(child_var)
        #    return
        if self._referenced_variables[var] != 0:
            raise ValueError('Cannot remove Var {0} because it is still referenced by the '.format(var) +
                             'objective or one or more constraints')
        solver_var = self._pyomo_var_to_solver_var_map[var]
        self._remove_var(solver_var)
        self._symbol_map.removeSymbol(var)
        self._labeler.remove_obj(var)
        del self._referenced_variables[var]
        del self._pyomo_var_to_solver_var_map[var]
        del self._solver_var_to_pyomo_var_map[solver_var]

    """ This method should be implemented by subclasses."""
    def update_var(self, var):
        """
        Update a variable in the solver's model. This will update bounds, fix/unfix the variable as needed, and update
        the variable type.

        Parameters
        ----------
        var: Var
        """
        raise NotImplementedError('This method should be implemented by subclasses.')

    def solve(self, *args, **kwds):
        """
        Solve the model.

        Keyword Arguments
        -----------------
        suffixes: list of str
            The strings should represnt suffixes support by the solver. Examples include 'dual', 'slack', and 'rc'.
        options: dict
            Dictionary of solver options. See the solver documentation for possible solver options.
        warmstart: bool
            If True, the solver will be warmstarted.
        keepfiles: bool
            If True, the solver log file will be saved.
        logfile: str
            Name to use for the solver log file.
        load_solutions: bool
            If True and a solution exists, the solution will be loaded into the Pyomo model.
        report_timing: bool
            If True, then timing information will be printed.
        tee: bool
            If True, then the solver log will be printed.
        """
        if self._pyomo_model is None:
            msg = 'Please use set_instance to set the instance before calling solve with the persistent'
            msg += ' solver interface.'
            raise RuntimeError(msg)
        if len(args) != 0:
            if self._pyomo_model is not args[0]:
                msg = 'The problem instance provided to the solve method is not the same as the instance provided'
                msg += ' to the set_instance method in the persistent solver interface. '
                raise ValueError(msg)

        self.available(exception_flag=True)

        # Collect suffix names to try and import from solution.
        if isinstance(self._pyomo_model, _BlockData):
            model_suffixes = list(name for (name, comp) in active_import_suffix_generator(self._pyomo_model))

        else:
            assert isinstance(self._pyomo_model, IBlock)
            model_suffixes = list(comp.storage_key for comp in
                                  import_suffix_generator(self._pyomo_model,
                                                          active=True,
                                                          descend_into=False))

        if len(model_suffixes) > 0:
            kwds_suffixes = kwds.setdefault('suffixes', [])
            for name in model_suffixes:
                if name not in kwds_suffixes:
                    kwds_suffixes.append(name)

        #
        # Handle ephemeral solvers options here. These
        # will override whatever is currently in the options
        # dictionary, but we will reset these options to
        # their original value at the end of this method.
        #

        orig_options = self.options

        self.options = Options()
        self.options.update(orig_options)
        self.options.update(kwds.pop('options', {}))
        self.options.update(self._options_string_to_dict(kwds.pop('options_string', '')))
        try:

            # we're good to go.
            initial_time = time.time()

            self._presolve(**kwds)

            presolve_completion_time = time.time()
            if self._report_timing:
                print("      %6.2f seconds required for presolve" % (presolve_completion_time - initial_time))

            if self._pyomo_model is not None:
                self._initialize_callbacks(self._pyomo_model)

            _status = self._apply_solver()
            if hasattr(self, '_transformation_data'):
                del self._transformation_data
            if not hasattr(_status, 'rc'):
                logger.warning(
                    "Solver (%s) did not return a solver status code.\n"
                    "This is indicative of an internal solver plugin error.\n"
                    "Please report this to the Pyomo developers.")
            elif _status.rc:
                logger.error(
                    "Solver (%s) returned non-zero return code (%s)"
                    % (self.name, _status.rc,))
                if self._tee:
                    logger.error(
                        "See the solver log above for diagnostic information.")
                elif hasattr(_status, 'log') and _status.log:
                    logger.error("Solver log:\n" + str(_status.log))
                raise ApplicationError(
                    "Solver (%s) did not exit normally" % self.name)
            solve_completion_time = time.time()
            if self._report_timing:
                print("      %6.2f seconds required for solver" % (solve_completion_time - presolve_completion_time))

            result = self._postsolve()
            # ***********************************************************
            # The following code is only needed for backwards compatability of load_solutions=False.
            # If we ever only want to support the load_vars, load_duals, etc. methods, then this can be deleted.
            if self._save_results:
                result._smap_id = self._smap_id
                result._smap = None
                _model = self._pyomo_model
                if _model:
                    if isinstance(_model, IBlock):
                        if len(result.solution) == 1:
                            result.solution(0).symbol_map = \
                                getattr(_model, "._symbol_maps")[result._smap_id]
                            result.solution(0).default_variable_value = \
                                self._default_variable_value
                            if self._load_solutions:
                                _model.load_solution(result.solution(0))
                        else:
                            assert len(result.solution) == 0
                        # see the hack in the write method
                        # we don't want this to stick around on the model
                        # after the solve
                        assert len(getattr(_model, "._symbol_maps")) == 1
                        delattr(_model, "._symbol_maps")
                        del result._smap_id
                        if self._load_solutions and \
                           (len(result.solution) == 0):
                            logger.error("No solution is available")
                    else:
                        if self._load_solutions:
                            _model.solutions.load_from(
                                result,
                                select=self._select_index,
                                default_variable_value=self._default_variable_value)
                            result._smap_id = None
                            result.solution.clear()
                        else:
                            result._smap = _model.solutions.symbol_map[self._smap_id]
                            _model.solutions.delete_symbol_map(self._smap_id)
            # ********************************************************
            postsolve_completion_time = time.time()

            if self._report_timing:
                print("      %6.2f seconds required for postsolve" % (postsolve_completion_time -
                                                                      solve_completion_time))

        finally:
            #
            # Reset the options dict
            #
            self.options = orig_options

        return result

    def has_instance(self):
        """
        True if set_instance has been called and this solver interface has a pyomo model and a solver model.

        Returns
        -------
        tmp: bool
        """
        return self._pyomo_model is not None
