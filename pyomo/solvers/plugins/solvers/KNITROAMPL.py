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
import logging

from pyomo.common import Executable
from pyomo.common.collections import Bunch
from pyomo.common.dependencies import pathlib
from pyomo.opt.base.formats import ProblemFormat, ResultsFormat
from pyomo.opt.base.solvers import SolverFactory, OptSolver
from pyomo.solvers.plugins.solvers.ASL import ASL

logger = logging.getLogger('pyomo.solvers')


@SolverFactory.register(
    'knitroampl', doc='The Knitro solver for NLP/MINLP and their subclasses'
)
class KNITROAMPL(ASL):
    """An interface to the Knitro optimizer that uses the AMPL Solver Library."""

    def __init__(self, **kwds):
        """Constructor"""
        executable = kwds.pop('executable', None)
        validate = kwds.pop('validate', True)
        kwds["type"] = "knitroampl"
        kwds["options"] = {'solver': "knitroampl"}
        OptSolver.__init__(self, **kwds)
        self._keepfiles = False
        self._results_file = None
        self._timer = ''
        self._user_executable = None
        # broadly useful for reporting, and in cases where
        # a solver plugin may not report execution time.
        self._last_solve_time = None
        self._define_signal_handlers = None
        self._version_timeout = 2

        if executable == 'knitroampl':
            self.set_executable(name=None, validate=validate)
        elif executable is not None:
            self.set_executable(name=executable, validate=validate)
        #
        # Setup valid problem formats, and valid results for each problem format.
        # Also set the default problem and results formats.
        #
        self._valid_problem_formats = [ProblemFormat.nl]
        self._valid_result_formats = {ProblemFormat.nl: [ResultsFormat.sol]}
        self.set_problem_format(ProblemFormat.nl)
        #
        # Note: Undefined capabilities default to 'None'
        #
        self._capabilities = Bunch()
        self._capabilities.linear = True
        self._capabilities.integer = True
        self._capabilities.quadratic_objective = True
        self._capabilities.quadratic_constraint = True
        self._capabilities.sos1 = False
        self._capabilities.sos2 = False

    def _default_executable(self):
        try:
            # If knitro Python package is available, use the executable it contains
            import knitro

            package_knitroampl_path = (
                pathlib.Path(knitro.__file__).resolve().parent
                / 'knitroampl'
                / 'knitroampl'
            )
            executable = Executable(str(package_knitroampl_path))
        except ModuleNotFoundError:
            # Otherwise, search usual path list
            executable = Executable('knitroampl')
        if not executable:
            logger.warning(
                "Could not locate the 'knitroampl' executable, "
                "which is required for solver %s" % self.name
            )
            self.enable = False
            return None
        return executable.path()
