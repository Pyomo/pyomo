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

        self._compile_instance(model, kwds)

        DirectOrPersistentSolver._presolve(self, *args, **kwds)
