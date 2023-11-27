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
"""
The cyipopt_solver module includes two solvers that call CyIpopt. One,
CyIpoptSolver, is a solver that operates on a CyIpoptProblemInterface
(such as CyIpoptNLP). The other, PyomoCyIpoptSolver, operates directly on a
Pyomo model.

"""
import io
import sys
import logging
import os
import abc

from pyomo.common.deprecation import relocated_module_attribute
from pyomo.common.dependencies import attempt_import, numpy as np, numpy_available
from pyomo.common.tee import redirect_fd, TeeStream

# Because pynumero.interfaces requires numpy, we will leverage deferred
# imports here so that the solver can be registered even when numpy is
# not available.
pyomo_nlp = attempt_import("pyomo.contrib.pynumero.interfaces.pyomo_nlp")[0]
pyomo_grey_box = attempt_import("pyomo.contrib.pynumero.interfaces.pyomo_grey_box_nlp")[
    0
]
egb = attempt_import("pyomo.contrib.pynumero.interfaces.external_grey_box")[0]

# Defer this import so that importing this module (PyomoCyIpoptSolver in
# particular) does not rely on an attempted cyipopt import.
cyipopt_interface, _ = attempt_import(
    "pyomo.contrib.pynumero.interfaces.cyipopt_interface"
)

# These attributes should no longer be imported from this module. These
# deprecation paths provide a deferred import to these attributes so (a) they
# can still be used until these paths are removed, and (b) the imports are not
# triggered when this module is imported.
relocated_module_attribute(
    "cyipopt_available",
    "pyomo.contrib.pynumero.interfaces.cyipopt_interface.cyipopt_available",
    "6.6.0",
)
relocated_module_attribute(
    "CyIpoptProblemInterface",
    "pyomo.contrib.pynumero.interfaces.cyipopt_interface.CyIpoptProblemInterface",
    "6.6.0",
)
relocated_module_attribute(
    "CyIpoptNLP",
    "pyomo.contrib.pynumero.interfaces.cyipopt_interface.CyIpoptNLP",
    "6.6.0",
)

from pyomo.common.config import ConfigBlock, ConfigValue
from pyomo.common.timing import TicTocTimer
from pyomo.core.base import Block, Objective, minimize
from pyomo.opt import SolverStatus, SolverResults, TerminationCondition, ProblemSense
from pyomo.opt.results.solution import Solution

logger = logging.getLogger(__name__)

# This maps the cyipopt STATUS_MESSAGES back to string representations
# of the Ipopt ApplicationReturnStatus enum
_cyipopt_status_enum = [
    "Solve_Succeeded",
    (
        b"Algorithm terminated successfully at a locally "
        b"optimal point, satisfying the convergence tolerances "
        b"(can be specified by options)."
    ),
    "Solved_To_Acceptable_Level",
    (
        b"Algorithm stopped at a point that was "
        b'converged, not to "desired" tolerances, '
        b'but to "acceptable" tolerances (see the '
        b"acceptable-... options)."
    ),
    "Infeasible_Problem_Detected",
    (
        b"Algorithm converged to a point of local "
        b"infeasibility. Problem may be "
        b"infeasible."
    ),
    "Search_Direction_Becomes_Too_Small",
    (b"Algorithm proceeds with very little progress."),
    "Diverging_Iterates",
    b"It seems that the iterates diverge.",
    "User_Requested_Stop",
    (
        b"The user call-back function intermediate_callback "
        b"(see Section 3.3.4 in the documentation) returned "
        b"false, i.e., the user code requested a premature "
        b"termination of the optimization."
    ),
    "Feasible_Point_Found",
    b"Feasible point for square problem found.",
    "Maximum_Iterations_Exceeded",
    (b"Maximum number of iterations exceeded (can be specified by an option)."),
    "Restoration_Failed",
    (b"Restoration phase failed, algorithm doesn't know how to proceed."),
    "Error_In_Step_Computation",
    (
        b"An unrecoverable error occurred while Ipopt "
        b"tried to compute the search direction."
    ),
    "Maximum_CpuTime_Exceeded",
    b"Maximum CPU time exceeded.",
    "Not_Enough_Degrees_Of_Freedom",
    b"Problem has too few degrees of freedom.",
    "Invalid_Problem_Definition",
    b"Invalid problem definition.",
    "Invalid_Option",
    b"Invalid option encountered.",
    "Invalid_Number_Detected",
    (
        b"Algorithm received an invalid number (such as "
        b"NaN or Inf) from the NLP; see also option "
        b"check_derivatives_for_naninf."
    ),
    # Note that the concluding "." was missing before cyipopt 1.0.3
    "Invalid_Number_Detected",
    (
        b"Algorithm received an invalid number (such as "
        b"NaN or Inf) from the NLP; see also option "
        b"check_derivatives_for_naninf"
    ),
    "Unrecoverable_Exception",
    b"Some uncaught Ipopt exception encountered.",
    "NonIpopt_Exception_Thrown",
    b"Unknown Exception caught in Ipopt.",
    # Note that the concluding "." was missing before cyipopt 1.0.3
    "NonIpopt_Exception_Thrown",
    b"Unknown Exception caught in Ipopt",
    "Insufficient_Memory",
    b"Not enough memory.",
    "Internal_Error",
    (
        b"An unknown internal error occurred. Please contact "
        b"the Ipopt authors through the mailing list."
    ),
]
_cyipopt_status_enum = {
    _cyipopt_status_enum[i + 1]: _cyipopt_status_enum[i]
    for i in range(0, len(_cyipopt_status_enum), 2)
}

# This maps Ipopt ApplicationReturnStatus enum strings to an appropriate
# Pyomo TerminationCondition
_ipopt_term_cond = {
    "Solve_Succeeded": TerminationCondition.optimal,
    "Solved_To_Acceptable_Level": TerminationCondition.feasible,
    "Infeasible_Problem_Detected": TerminationCondition.infeasible,
    "Search_Direction_Becomes_Too_Small": TerminationCondition.minStepLength,
    "Diverging_Iterates": TerminationCondition.unbounded,
    "User_Requested_Stop": TerminationCondition.userInterrupt,
    "Feasible_Point_Found": TerminationCondition.feasible,
    "Maximum_Iterations_Exceeded": TerminationCondition.maxIterations,
    "Restoration_Failed": TerminationCondition.noSolution,
    "Error_In_Step_Computation": TerminationCondition.solverFailure,
    "Maximum_CpuTime_Exceeded": TerminationCondition.maxTimeLimit,
    "Not_Enough_Degrees_Of_Freedom": TerminationCondition.invalidProblem,
    "Invalid_Problem_Definition": TerminationCondition.invalidProblem,
    "Invalid_Option": TerminationCondition.error,
    "Invalid_Number_Detected": TerminationCondition.internalSolverError,
    "Unrecoverable_Exception": TerminationCondition.internalSolverError,
    "NonIpopt_Exception_Thrown": TerminationCondition.error,
    "Insufficient_Memory": TerminationCondition.resourceInterrupt,
    "Internal_Error": TerminationCondition.internalSolverError,
}


class CyIpoptSolver(object):
    def __init__(self, problem_interface, options=None):
        """Create an instance of the CyIpoptSolver. You must
        provide a problem_interface that corresponds to
        the abstract class CyIpoptProblemInterface
        options can be provided as a dictionary of key value
        pairs
        """
        self._problem = problem_interface

        self._options = options
        if options is not None:
            assert isinstance(options, dict)
        else:
            self._options = dict()

    def solve(self, x0=None, tee=False):
        if x0 is None:
            x0 = self._problem.x_init()
        xstart = x0
        cyipopt_solver = self._problem

        # check if we need scaling
        obj_scaling, x_scaling, g_scaling = self._problem.scaling_factors()
        if any(_ is not None for _ in (obj_scaling, x_scaling, g_scaling)):
            # need to set scaling factors
            if obj_scaling is None:
                obj_scaling = 1.0
            if x_scaling is None:
                x_scaling = np.ones(nx)
            if g_scaling is None:
                g_scaling = np.ones(ng)
            try:
                set_scaling = cyipopt_solver.set_problem_scaling
            except AttributeError:
                # Fall back to pre-1.0.0 API
                set_scaling = cyipopt_solver.setProblemScaling
            set_scaling(obj_scaling, x_scaling, g_scaling)

        # add options
        try:
            add_option = cyipopt_solver.add_option
        except AttributeError:
            # Fall back to pre-1.0.0 API
            add_option = cyipopt_solver.addOption
        for k, v in self._options.items():
            add_option(k, v)

        # We preemptively set up the TeeStream, even if we aren't
        # going to use it: the implementation is such that the
        # context manager does nothing (i.e., doesn't start up any
        # processing threads) until after a client accesses
        # STDOUT/STDERR
        with TeeStream(sys.stdout) as _teeStream:
            if tee:
                try:
                    fd = sys.stdout.fileno()
                except (io.UnsupportedOperation, AttributeError):
                    # If sys,stdout doesn't have a valid fileno,
                    # then create one using the TeeStream
                    fd = _teeStream.STDOUT.fileno()
            else:
                fd = None
            with redirect_fd(fd=1, output=fd, synchronize=False):
                x, info = cyipopt_solver.solve(xstart)

        return x, info


def _numpy_vector(val):
    ans = np.array(val, np.float64)
    if len(ans.shape) != 1:
        raise ValueError(
            "expected a vector, but received a matrix with shape %s" % (ans.shape,)
        )
    return ans


class PyomoCyIpoptSolver(object):
    CONFIG = ConfigBlock("cyipopt")
    CONFIG.declare(
        "tee",
        ConfigValue(
            default=False, domain=bool, description="Stream solver output to console"
        ),
    )
    CONFIG.declare(
        "load_solutions",
        ConfigValue(
            default=True,
            domain=bool,
            description="Store the final solution into the original Pyomo model",
        ),
    )
    CONFIG.declare(
        "return_nlp",
        ConfigValue(
            default=False,
            domain=bool,
            description="Return the results object and the underlying nlp"
            " NLP object from the solve call.",
        ),
    )
    CONFIG.declare("options", ConfigBlock(implicit=True))
    CONFIG.declare(
        "intermediate_callback",
        ConfigValue(
            default=None,
            description="Set the function that will be called each iteration.",
        ),
    )
    CONFIG.declare(
        "halt_on_evaluation_error",
        ConfigValue(
            default=None,
            description="Whether to halt if a function or derivative evaluation fails",
        ),
    )

    def __init__(self, **kwds):
        """Create an instance of the CyIpoptSolver. You must
        provide a problem_interface that corresponds to
        the abstract class CyIpoptProblemInterface

        options can be provided as a dictionary of key value
        pairs
        """
        self.config = self.CONFIG(kwds)

    def _set_model(self, model):
        self._model = model

    def available(self, exception_flag=False):
        return bool(numpy_available and cyipopt_interface.cyipopt_available)

    def license_is_valid(self):
        return True

    def version(self):
        return tuple(int(_) for _ in cyipopt.__version__.split("."))

    def solve(self, model, **kwds):
        config = self.config(kwds, preserve_implicit=True)

        if not isinstance(model, Block):
            raise ValueError(
                "PyomoCyIpoptSolver.solve(model): model must be a Pyomo Block"
            )

        # If this is a Pyomo model / block, then we need to create
        # the appropriate PyomoNLP, then wrap it in a CyIpoptNLP
        grey_box_blocks = list(
            model.component_data_objects(egb.ExternalGreyBoxBlock, active=True)
        )
        if grey_box_blocks:
            # nlp = pyomo_nlp.PyomoGreyBoxNLP(model)
            nlp = pyomo_grey_box.PyomoNLPWithGreyBoxBlocks(model)
        else:
            nlp = pyomo_nlp.PyomoNLP(model)

        problem = cyipopt_interface.CyIpoptNLP(
            nlp,
            intermediate_callback=config.intermediate_callback,
            halt_on_evaluation_error=config.halt_on_evaluation_error,
        )
        ng = len(problem.g_lb())
        nx = len(problem.x_lb())
        cyipopt_solver = problem

        # check if we need scaling
        obj_scaling, x_scaling, g_scaling = problem.scaling_factors()
        if any(_ is not None for _ in (obj_scaling, x_scaling, g_scaling)):
            # need to set scaling factors
            if obj_scaling is None:
                obj_scaling = 1.0
            if x_scaling is None:
                x_scaling = np.ones(nx)
            if g_scaling is None:
                g_scaling = np.ones(ng)
            try:
                set_scaling = cyipopt_solver.set_problem_scaling
            except AttributeError:
                # Fall back to pre-1.0.0 API
                set_scaling = cyipopt_solver.setProblemScaling
            set_scaling(obj_scaling, x_scaling, g_scaling)

        # add options
        try:
            add_option = cyipopt_solver.add_option
        except AttributeError:
            # Fall back to pre-1.0.0 API
            add_option = cyipopt_solver.addOption
        for k, v in config.options.items():
            add_option(k, v)

        timer = TicTocTimer()
        try:
            # We preemptively set up the TeeStream, even if we aren't
            # going to use it: the implementation is such that the
            # context manager does nothing (i.e., doesn't start up any
            # processing threads) until after a client accesses
            # STDOUT/STDERR
            with TeeStream(sys.stdout) as _teeStream:
                if config.tee:
                    try:
                        fd = sys.stdout.fileno()
                    except (io.UnsupportedOperation, AttributeError):
                        # If sys,stdout doesn't have a valid fileno,
                        # then create one using the TeeStream
                        fd = _teeStream.STDOUT.fileno()
                else:
                    fd = None
                with redirect_fd(fd=1, output=fd, synchronize=False):
                    x, info = cyipopt_solver.solve(problem.x_init())
            solverStatus = SolverStatus.ok
        except:
            msg = "Exception encountered during cyipopt solve:"
            logger.error(msg, exc_info=sys.exc_info())
            solverStatus = SolverStatus.unknown
            raise

        wall_time = timer.toc(None)

        results = SolverResults()

        if config.load_solutions:
            nlp.set_primals(x)
            nlp.set_duals(info["mult_g"])
            nlp.load_state_into_pyomo(
                bound_multipliers=(info["mult_x_L"], info["mult_x_U"])
            )
        else:
            soln = Solution()
            sm = nlp.symbol_map
            soln.variable.update(
                (sm.getSymbol(i), {'Value': j, 'ipopt_zL_out': zl, 'ipopt_zU_out': zu})
                for i, j, zl, zu in zip(
                    nlp.get_pyomo_variables(), x, info['mult_x_L'], info['mult_x_U']
                )
            )
            soln.constraint.update(
                (sm.getSymbol(i), {'Dual': j})
                for i, j in zip(nlp.get_pyomo_constraints(), info['mult_g'])
            )
            model.solutions.add_symbol_map(sm)
            results._smap_id = id(sm)
            results.solution.insert(soln)

        results.problem.name = model.name
        obj = next(model.component_data_objects(Objective, active=True))
        if obj.sense == minimize:
            results.problem.sense = ProblemSense.minimize
            results.problem.upper_bound = info["obj_val"]
        else:
            results.problem.sense = ProblemSense.maximize
            results.problem.lower_bound = info["obj_val"]
        results.problem.number_of_objectives = 1
        results.problem.number_of_constraints = ng
        results.problem.number_of_variables = nx
        results.problem.number_of_binary_variables = 0
        results.problem.number_of_integer_variables = 0
        results.problem.number_of_continuous_variables = nx
        # TODO: results.problem.number_of_nonzeros

        results.solver.name = "cyipopt"
        results.solver.return_code = info["status"]
        results.solver.message = info["status_msg"]
        results.solver.wallclock_time = wall_time
        status_enum = _cyipopt_status_enum[info["status_msg"]]
        results.solver.termination_condition = _ipopt_term_cond[status_enum]
        results.solver.status = TerminationCondition.to_solver_status(
            results.solver.termination_condition
        )

        problem.close()

        if config.return_nlp:
            return results, nlp

        return results

    #
    # Support "with" statements.
    #
    def __enter__(self):
        return self

    def __exit__(self, t, v, traceback):
        pass
