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
from pyomo.core.kernel.component_block import IBlockStorage
from pyomo.core.base.suffix import active_import_suffix_generator
from pyomo.core.kernel.component_suffix import import_suffix_generator
import pyutilib.misc
import pyutilib.common
import time
import logging
from pyomo.core.base.constraint import Constraint
from pyomo.core.base.var import Var
from pyomo.core.base.sos import SOSConstraint


logger = logging.getLogger('pyomo.solvers')


class PersistentSolver(DirectOrPersistentSolver):

    def __init__(self, **kwds):
        DirectOrPersistentSolver.__init__(self, **kwds)

        # Ensure any subclasses inherit from PersistentSolver before any direct solver
        assert type(self).__bases__[0] is PersistentSolver

    def _presolve(self, *args, **kwds):
        if len(args) != 0:
            msg = 'The persistent solver interface does not accept a problem instance in the solve method.'
            msg += ' The problem instance should be compiled before the solve using the compile_instance method.'
            raise ValueError(msg)

        DirectOrPersistentSolver._presolve(self, *args, **kwds)

    def compile_instance(self, model, **kwds):
        return self._compile_instance(model, kwds)

    def add_block(self, block):
        if block.is_indexed():
            for sub_block in block.values():
                self._add_block(block)
            return
        self._add_block(block)

    def add_objective(self, obj):
        return self._add_objective(obj)

    def add_constraint(self, con):
        if con.is_indexed():
            for child_con in con.values():
                self._add_constraint(child_con)
        else:
            self._add_constraint(con)

    def add_var(self, var):
        if var.is_indexed():
            for child_var in var.values():
                self._add_var(child_var)
        else:
            self._add_var(var)

    def add_sos_constraint(self, con):
        if con.is_indexed():
            for child_con in con.values():
                self._add_sos_constraint(child_con)
        else:
            self._add_sos_constraint(con)

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
        if block.is_indexed():
            for sub_block in block.values():
                self.remove_block(sub_block)
            return
        for sub_block in block.block_data_objects(descend_into=True, active=True):
            for con in sub_block.component_data_objects(ctype=Constraint, descend_into=False, active=True):
                self.remove_constraint(con)

            for con in sub_block.component_data_objects(ctype=SOSConstraint, descend_into=False, active=True):
                self.remove_sos_constraint(con)

        for var in block.component_data_objects(ctype=Var, descend_into=True, active=True):
            self.remove_var(var)

    def remove_constraint(self, con):
        if con.is_indexed():
            for child_con in con.values():
                self.remove_constraint(child_con)
            return
        solver_con = self._pyomo_con_to_solver_con_map[con]
        self._remove_constraint(solver_con)
        self._symbol_map.removeSymbol(con)
        self._labeler.remove_obj(con)
        for var in self._vars_referenced_by_con[con]:
            self._referenced_variables[var] -= 1
        del self._vars_referenced_by_con[con]
        del self._pyomo_con_to_solver_con_map[con]

    def remove_sos_constraint(self, con):
        if con.is_indexed():
            for child_con in con.values():
                self.remove_sos_constraint(child_con)
            return
        solver_con = self._pyomo_con_to_solver_con_map[con]
        self._remove_sos_constraint(solver_con)
        self._symbol_map.removeSymbol(con)
        self._labeler.remove_obj(con)
        for var in self._vars_referenced_by_con[con]:
            self._referenced_variables[var] -= 1
        del self._vars_referenced_by_con[con]
        del self._pyomo_con_to_solver_con_map[con]

    def remove_var(self, var):
        if var.is_indexed():
            for child_var in var.values():
                self.remove_var(child_var)
            return
        if self._referenced_variables[var] != 0:
            raise ValueError('Cannot remove Var {0} because it is still referenced by the '.format(var) +
                             'objective or one or more constraints')
        solver_var = self._pyomo_var_to_solver_var_map[var]
        self._remove_var(solver_var)
        self._symbol_map.removeSymbol(var)
        self._labeler.remove_obj(var)
        del self._referenced_variables[var]
        del self._pyomo_var_to_solver_var_map[var]

    """ This method should be implemented by subclasses."""
    def compile_var(self, var):
        raise NotImplementedError('This method should be implemented by subclasses.')

    def load_vars(self, vars_to_load):
        self._load_vars(vars_to_load)

    def solve(self, *args, **kwds):
        if len(args) != 0:
            msg = 'The persistent solver interface does not accept a problem instance in the solve method.'
            msg += ' The problem instance should be compiled before the solve using the compile_instance method.'
            raise ValueError(msg)

        self.available(exception_flag=True)

        # Collect suffix names to try and import from solution.
        if isinstance(self._pyomo_model, _BlockData):
            model_suffixes = list(name for (name, comp) in active_import_suffix_generator(self._pyomo_model))

        else:
            assert isinstance(self._pyomo_model, IBlockStorage)
            model_suffixes = list(name for (name, comp) in
                                  import_suffix_generator(self._pyomo_model, active=True,
                                                          descend_into=False, return_key=True))

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

        self.options = pyutilib.misc.Options()
        self.options.update(orig_options)
        self.options.update(kwds.pop('options', {}))
        self.options.update(self._options_string_to_dict(kwds.pop('options_string', '')))
        try:

            # we're good to go.
            initial_time = time.time()

            self._presolve(*args, **kwds)

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
                raise pyutilib.common.ApplicationError(
                    "Solver (%s) did not exit normally" % self.name)
            solve_completion_time = time.time()
            if self._report_timing:
                print("      %6.2f seconds required for solver" % (solve_completion_time - presolve_completion_time))

            result = self._postsolve()
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
