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


class DirectSolver(DirectOrPersistentSolver):
    """
    Subclasses need to:
    1.) Initialize self._solver_model during _presolve before calling DirectSolver._presolve
    """
    def __init__(self, **kwds):
        DirectOrPersistentSolver.__init__(self, **kwds)

    def _presolve(self, *args, **kwds):
        """
        kwds not consumed here or at the beginning of OptSolver._presolve will raise an error in
        OptSolver._presolve.

        args
        ----
        pyomo Model or IBlockStorage

        kwds
        ----
        warmstart: bool
            can only be True if the subclass is warmstart capable; if not, an error will be raised
        symbolic_solver_labels: bool
            if True, the model will be translated using the names from the pyomo model; otherwise, the variables and
            constraints will be numbered with a generic xi
        skip_trivial_constraints: bool
            if True, any trivial constraints (e.g., 1 == 1) will be skipped (i.e., not passed to the solver).
        output_fixed_variable_bounds: bool
            if False, an error will be raised if a fixed variable is used in any expression rather than the value of the
            fixed variable.
        keepfiles: bool
            if True, the solver log file will be saved and the name of the file will be printed.

        kwds accepted by OptSolver._presolve
        """
        model = args[0]
        if len(args) != 1:
            msg = ("The {0} plugin method '_presolve' must be supplied a single problem instance - {1} were " +
                   "supplied.").format(type(self), len(args))
            raise ValueError(msg)

        self._set_instance(model, kwds)

        DirectOrPersistentSolver._presolve(self, *args, **kwds)

    def solve(self, *args, **kwds):
        """ Solve the problem """

        self.available(exception_flag=True)
        #
        # If the inputs are models, then validate that they have been
        # constructed! Collect suffix names to try and import from solution.
        #
        _model = None
        for arg in args:
            if isinstance(arg, (_BlockData, IBlockStorage)):
                if isinstance(arg, _BlockData):
                    if not arg.is_constructed():
                        raise RuntimeError(
                            "Attempting to solve model=%s with unconstructed "
                            "component(s)" % (arg.name,) )

                _model = arg
                # import suffixes must be on the top-level model
                if isinstance(arg, _BlockData):
                    model_suffixes = list(name for (name,comp) in active_import_suffix_generator(arg))
                else:
                    assert isinstance(arg, IBlockStorage)
                    model_suffixes = list(name for (name,comp) in import_suffix_generator(arg, active=True,
                                                                                          descend_into=False,
                                                                                          return_key=True))

                if len(model_suffixes) > 0:
                    kwds_suffixes = kwds.setdefault('suffixes',[])
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
        self.options.update(
            self._options_string_to_dict(kwds.pop('options_string', '')))
        try:

            # we're good to go.
            initial_time = time.time()

            self._presolve(*args, **kwds)

            presolve_completion_time = time.time()
            if self._report_timing:
                print("      %6.2f seconds required for presolve" % (presolve_completion_time - initial_time))

            if not _model is None:
                self._initialize_callbacks(_model)

            _status = self._apply_solver()
            if hasattr(self, '_transformation_data'):
                del self._transformation_data
            if not hasattr(_status, 'rc'):
                logger.warning(
                    "Solver (%s) did not return a solver status code.\n"
                    "This is indicative of an internal solver plugin error.\n"
                    "Please report this to the Pyomo developers." )
            elif _status.rc:
                logger.error(
                    "Solver (%s) returned non-zero return code (%s)"
                    % (self.name, _status.rc,))
                if self._tee:
                    logger.error(
                        "See the solver log above for diagnostic information." )
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
                print("      %6.2f seconds required for postsolve" % (postsolve_completion_time - solve_completion_time))

        finally:
            #
            # Reset the options dict
            #
            self.options = orig_options

        return result


