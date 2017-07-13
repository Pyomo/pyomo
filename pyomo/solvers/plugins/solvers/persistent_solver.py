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

    def _apply_solver(self):
        raise NotImplementedError('The subclass should implement this method.')

    def _postsolve(self):
        return DirectOrPersistentSolver._postsolve(self)

    def _compile_instance(self, model, kwds={}):
        DirectOrPersistentSolver._compile_instance(self, model, kwds)

    def _add_block(self, block):
        raise NotImplementedError('The subclass should implement this method.')

    def _compile_objective(self):
        raise NotImplementedError('The subclass should implement this method.')

    def _add_constraint(self, con):
        raise NotImplementedError('The subclass should implement this method.')

    def _add_var(self, var):
        raise NotImplementedError('The subclass should implement this method.')

    def _add_sos_constraint(self, con):
        raise NotImplementedError('The subclass should implement this method.')

    def compile_instance(self, model, **kwds):
        raise NotImplementedError('The subclass should implement this method.')

    def add_block(self, block):
        raise NotImplementedError('The subclass should implement this method.')

    def compile_objective(self):
        raise NotImplementedError('The subclass should implement this method.')

    def add_constraint(self, con):
        raise NotImplementedError('The subclass should implement this method.')

    def add_var(self, var):
        raise NotImplementedError('The subclass should implement this method.')

    def add_sos_constraint(self, con):
        raise NotImplementedError('The subclass should implement this method.')

    def remove_block(self, block):
        raise NotImplementedError('The subclass should implement this method.')

    def remove_constraint(self, con):
        raise NotImplementedError('The subclass should implement this method.')

    def remove_sos_constraint(self, con):
        raise NotImplementedError('The subclass should implement this method.')

    def remove_var(self, var):
        raise NotImplementedError('The subclass should implement this method.')

    def _get_expr_from_pyomo_repn(self, repn, max_degree=None):
        raise NotImplementedError('The subclass should implement this method.')

    def _get_expr_from_pyomo_expr(self, expr, max_degree=None):
        raise NotImplementedError('The subclass should implement this method.')

    def _load_vars(self, vars_to_load):
        raise NotImplementedError('The subclass should implement this method.')

    def warm_start_capable(self):
        raise NotImplementedError('The subclass should implement this method.')

    def _warm_start(self):
        raise NotImplementedError('If a subclass can warmstart, then it should implement this method.')

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
            result._smap_id = self._smap_id
            result._smap = None
            if self._pyomo_model:
                if isinstance(self._pyomo_model, IBlockStorage):
                    if len(result.solution) == 1:
                        result.solution(0).symbol_map = \
                            getattr(self._pyomo_model, "._symbol_maps")[result._smap_id]
                        result.solution(0).default_variable_value = \
                            self._default_variable_value
                        if self._load_solutions:
                            self._pyomo_model.load_solution(result.solution(0))
                            result.solution.clear()
                    else:
                        assert len(result.solution) == 0
                    # see the hack in the write method
                    # we don't want this to stick around on the model
                    # after the solve
                    assert len(getattr(self._pyomo_model, "._symbol_maps")) == 1
                    delattr(self._pyomo_model, "._symbol_maps")
                    del result._smap_id
                else:
                    if self._load_solutions:
                        self._pyomo_model.solutions.load_from(
                            result,
                            select=self._select_index,
                            default_variable_value=self._default_variable_value)
                        result._smap_id = None
                        result.solution.clear()
                    else:
                        result._smap = self._pyomo_model.solutions.symbol_map[self._smap_id]
                        self._pyomo_model.solutions.delete_symbol_map(self._smap_id)
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
