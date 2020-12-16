#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

# TODO: have smps convert create symbols files for second stage
# TODO: parse second-stage solution and load into scenario tree workers
# TODO: objective, cost, stage_costs
# TODO: make output_scenario_tree_solution work
# TODO: results object statuses

# TODO Fix the seed defaults. The I think the default
#      behavior should be to provide no needs and enable
#      AUTO_SEED. There are so many seeds to provide that
#      I'm not sure where to start on this. Perhaps, the
#      SD developers can updates these defaults. It would be
#      nice if a single seed could be provided and the rest
#      generated from this by SD, when deterministic behavior
#      is required.

import os
import sys
import time
import traceback
import logging

import pyutilib.subprocess

from pyomo.core import maximize
from pyomo.opt import (TerminationCondition,
                       SolverStatus,
                       SolutionStatus)
from pyomo.pysp.util.configured_object import PySPConfiguredObject
from pyomo.pysp.util.config import (PySPConfigValue,
                                    PySPConfigBlock,
                                    safe_register_common_option,
                                    safe_declare_common_option,
                                    safe_declare_unique_option,
                                    _domain_positive,
                                    _domain_positive_integer,
                                    _domain_nonnegative_integer)
from pyomo.pysp.util.misc import (parse_command_line,
                                  launch_command)
from pyomo.pysp.scenariotree.manager import \
    ScenarioTreeManagerFactory
import pyomo.pysp.convert.smps
from pyomo.pysp.embeddedsp import EmbeddedSP
from pyomo.pysp.solvers.spsolver import (SPSolverResults,
                                         SPSolverFactory)
from pyomo.pysp.solvers.spsolvershellcommand import \
    SPSolverShellCommand

logger = logging.getLogger('pyomo.pysp')

_sd_group_label = "SD Options"

# maps the sd status message to tuples of the form
# (SolutionStatus, SolverStatus, TerminationCondition)
# feel free to modify if you have opinions one these
_sd_status_map = {}
_sd_status_map["COMPROMISE SOLUTION"] = (SolutionStatus.optimal,
                                         SolverStatus.ok,
                                         TerminationCondition.optimal)
_sd_status_map["MEAN SOLUTION"] = (SolutionStatus.optimal,
                                   SolverStatus.ok,
                                   TerminationCondition.optimal)
_sd_status_map["REPLICATION SOLUTION"] = (SolutionStatus.optimal,
                                          SolverStatus.ok,
                                          TerminationCondition.optimal)
_sd_status_map["NO  SOLUTION"] = (SolutionStatus.unknown,
                                  SolverStatus.warning,
                                  TerminationCondition.noSolution)
_sd_status_map["???"] = (SolutionStatus.unknown,
                         SolverStatus.error,
                         TerminationCondition.unknown)

sd_advanced_config_section = \
"""
// ---------------Changes made below here are not recommended-----------------


// 0 for LP master, 1 for QP master
MASTER_TYPE 1

// 0 for zero lower bound, 1 for mean value lower bound
LB_TYPE 1

// Amount of improvement which must be observed in order to update incumbent X.
R 0.2

// For updating the scaling factor. zl
R2 0.95

// For updating the scaling factor. zl
R3 2.0

// The Minimum value of the cell->quad_scalar. zl
MIN_QUAD_SCALAR 0.001

// The Maximum value of the cell->quad_scalar. zl
MAX_QUAD_SCALAR 10000.0

// Number of iterations after which incumbent cut is reevaluated.
TAU 2

// Ratio used in pre_test for optimality: (Sm - Lm) / Fk  < PRE_EPSILON.
PRE_EPSILON 0.01

// MIN_ITER will be max(ITER_FACT*xdim, MIN_ITER)   JH 3/13/98
ITER_FACT 0

// Percent of resampled solutions which must meet test for optimality.
PERCENT_PASS 0.95

// Number of resampling to take when checking optimality.
M 50

// Multiplier for the number of cuts; with QP, SD sets this to 1
CUT_MULT  5

// Level of confidence when choosing the set of cuts, T, for optimality test.
CONFID_HI 1.0

// Level of confidence when conducting the optimality pre-test.
CONFID_LO 1.45

// Percent of the number by which two "equal" numbers may differ.
TOLERANCE 0.001

// number by which two "equal" numbers may differ (used in judging feasibility).
FEA_TOLER 0.05

// Like tolerance, but used when trying to thin omegas.
THIN_TOLER 0.001

// Number of iterations before SD tries to thin the data structures.
START_THIN 9001

// Number of iterations between successive thinning of structures.
THIN_CYCLE 200


// Number of consecutive iterations a cut must be slack before its dropped.
DROP_TIME 16300

// 0 for pre-tests only, 1 for full_test
TEST_TYPE 1

// Number of iterations before checking new pi's impact
PI_EVAL_START 1

// Print out detailed solutions, set this to 1
DETAILED_SOLN 1

// Number of iterations between evaluation of the impact of new dual vertex
PI_CYCLE 1

// The flag for subproblem lower bound checker.
// 0 -- no subproblem LB checking.
// 1 -- check subproblem LB.
SUB_LB_CHECK 0

// The flag for seed generation method
// 0 -- user provide seeds manually
// 1 -- SD generates seeds automatically
AUTO_SEED  0

// 16 digits are recommended for the seed
// Random number seed for generating observations of omega.
// RUN_SEED1     9495518635394380
RUN_SEED1     9495518635394380
RUN_SEED2     4650175399072632
RUN_SEED3     6070772756632709
RUN_SEED4     5451675876709589
RUN_SEED5     5285327724846206
RUN_SEED6     5588857889468088
RUN_SEED7     1098833779416153
RUN_SEED8     6192593982049265
RUN_SEED9     4756774140130874
RUN_SEED10    6784592265109609
RUN_SEED11    9728429908537680
RUN_SEED12    1163479388309571
RUN_SEED13    3279282318700126
RUN_SEED14    8773753208032360
RUN_SEED15    9337302665697748
RUN_SEED16    4415169667296773
RUN_SEED17    4220432037464045
RUN_SEED18    3554548844580680
RUN_SEED19    1814300451929103
RUN_SEED20    5339672949292608
RUN_SEED21    5638710736762732
RUN_SEED22    3154245808720589
RUN_SEED23    2414929536171258
RUN_SEED24    7998609999427572
RUN_SEED25    7080145164625719
RUN_SEED26    3648862740490586
RUN_SEED27    7772725003305823
RUN_SEED28    5982768791029230
RUN_SEED29    1395182510837913
RUN_SEED30    3735836402047426
"""

def _domain_sd_tolerance(val):
    val = str(val)
    if val not in _domain_sd_tolerance._values:
        raise ValueError(
            "Value %s is not one of %s."
            % (val, str(_domain_sd_tolerance._values)))
    return val
_domain_sd_tolerance._values = ('loose','nominal','tight')
_domain_sd_tolerance.doc = \
    "<domain: %s>" % (str(_domain_sd_tolerance._values))

class SDSolver(SPSolverShellCommand, PySPConfiguredObject):

    @classmethod
    def _declare_options(cls, options=None):
        if options is None:
            options = PySPConfigBlock()
        safe_declare_unique_option(
            options,
            "stopping_rule_tolerance",
            PySPConfigValue(
                "nominal",
                domain=_domain_sd_tolerance,
                description=(
                    "Stopping rule tolerance used by the SD solver. "
                    "Must be one of: %s. Default is 'nominal'."
                    % (str(_domain_sd_tolerance._values))
                ),
                doc=None,
                visibility=0),
            ap_group=_sd_group_label)
        safe_declare_unique_option(
            options,
            "single_replication",
            PySPConfigValue(
                False,
                domain=bool,
                description=(
                    "Disables multiple replication procedure in "
                    "SD and uses a single replication."
                ),
                doc=None,
                visibility=0),
            ap_group=_sd_group_label)
        safe_declare_unique_option(
            options,
            "print_cycle",
            PySPConfigValue(
                100,
                domain=_domain_positive_integer,
                description=(
                    "Number of iterations between output of "
                    "solution data to screen and file."
                ),
                doc=None,
                visibility=0),
            ap_group=_sd_group_label)
        safe_declare_unique_option(
            options,
            "eval_run_flag",
            PySPConfigValue(
                False,
                domain=bool,
                description=(
                    "Set to evaluate on the run. This should be "
                    "only used for instances with relatively complete "
                    "recourse. This flag is not recommended because "
                    "accurate function evaluations are unnecessarily "
                    "time consuming. It is best to use a large print "
                    "cycle when this option is activated."
                ),
                doc=None,
                visibility=0),
            ap_group=_sd_group_label)
        safe_declare_unique_option(
            options,
            "eval_flag",
            PySPConfigValue(
                False,
                domain=bool,
                description=(
                    "Set to get an estimated objective function value "
                    "for the final incumbent of each replication."
                ),
                doc=None,
                visibility=0),
            ap_group=_sd_group_label)
        safe_declare_unique_option(
            options,
            "eval_seed1",
            PySPConfigValue(
                2668655841019641,
                domain=int,
                description=(
                    "Random number seed for re-sampling omegas during "
                    "optimality test. Default is None, meaning no "
                    "seed will be provided."
                ),
                doc=None,
                visibility=0),
            ap_group=_sd_group_label)
        safe_declare_unique_option(
            options,
            "eval_error",
            PySPConfigValue(
                0.01,
                domain=_domain_positive,
                description=(
                    "Objective evaluation is accurate to within "
                    "this much, with 95%% confidence. Default is 0.01."
                ),
                doc=None,
                visibility=0),
            ap_group=_sd_group_label)
        safe_declare_unique_option(
            options,
            "mean_dev",
            PySPConfigValue(
                0.05,
                domain=_domain_positive,
                description=(
                    "Solution tolerance for deciding the usage of "
                    "mean solution. Default is 0.05."
                ),
                doc=None,
                visibility=0),
            ap_group=_sd_group_label)
        safe_declare_unique_option(
            options,
            "min_iterations",
            PySPConfigValue(
                None,
                domain=_domain_nonnegative_integer,
                description=(
                    "Number of iterations which must pass before "
                    "optimality is checked. Default is None, meaning "
                    "no minimum is given."
                ),
                doc=None,
                visibility=0),
            ap_group=_sd_group_label)
        safe_declare_unique_option(
            options,
            "max_iterations",
            PySPConfigValue(
                5000,
                domain=_domain_positive_integer,
                description=(
                    "Maximum number of iterations for any given "
                    "problem. Default is 5000."
                ),
                doc=None,
                visibility=0),
            ap_group=_sd_group_label)
        safe_declare_common_option(options,
                                   "verbose",
                                   ap_group=_sd_group_label)

        return options

    def __init__(self):
        super(SDSolver, self).__init__(self.register_options())
        self.set_options_to_default()
        self._executable = "sd"

    def set_options_to_default(self):
        self._options = self.register_options()

    @property
    def options(self):
        return self._options

    @property
    def name(self):
        return "sd"

    def _solve_impl(self,
                    sp,
                    output_solver_log=False,
                    **kwds):
        """
        Solve a stochastic program with the SD solver.

        See the 'solve' method on the base class for
        additional keyword documentation.

        Args:
            sp: The stochastic program to solve.
            output_solver_log (bool): Stream the solver
                output during the solve.
            **kwds: Passed to the SMPS file writer as I/O
              options (e.g., symbolic_solver_labels=True).

        Returns: A results object with information about the solution.
        """

        if sp.objective_sense == maximize:
            raise ValueError("SD solver does not yet handle "
                             "maximization problems")

        #
        # Setup the SD working directory
        #

        working_directory = self._create_tempdir("workdir")
        config_filename = self._files["config"] = \
                          os.path.join(working_directory,
                                       "config.sd")
        self._add_tempfile("config", config_filename)
        sdinput_directory = os.path.join(working_directory,
                                         "sdinput",
                                         "pysp_model")
        self._add_tempfile("stdinput", sdinput_directory)
        sdoutput_directory = os.path.join(working_directory,
                                          "sdoutput",
                                          "pysp_model")
        self._add_tempfile("sdoutput", sdoutput_directory)
        logfile = os.path.join(working_directory, "sd.log")
        self._add_tempfile("log", logfile)

        if self.get_option("verbose"):
            print("Writing solver files in directory: %s"
                  % (working_directory))

        os.makedirs(sdinput_directory)
        assert os.path.exists(sdinput_directory)
        assert not os.path.exists(sdoutput_directory)
        self._write_config(config_filename)

        if self.get_option('single_replication'):
            solution_filename = os.path.join(
                sdoutput_directory,
                "pysp_model.detailed_rep_soln.out")
        else:
            solution_filename = os.path.join(
                sdoutput_directory,
                "pysp_model.detailed_soln.out")
        self._add_tempfile("solution", solution_filename)

        #
        # Create the SD input files
        #

        if isinstance(sp, EmbeddedSP):
            input_files, _ = pyomo.pysp.convert.smps.\
                convert_embedded(
                    sdinput_directory,
                    "pysp_model",
                    sp,
                    core_format='mps',
                    io_options=kwds)
        else:
            input_files = pyomo.pysp.convert.smps.\
                convert_external(
                    sdinput_directory,
                    "pysp_model",
                    sp,
                    core_format='mps',
                    io_options=kwds)
        for key in input_files:
            self._add_tempfile(key, input_files[key])

        #
        # Launch SD
        #

        _cmd_string = self.executable+" < pysp_model"
        if self.get_option("verbose"):
            print("Launching SD solver with command: %s"
                  % (_cmd_string))

        start = time.time()
        rc, log = pyutilib.subprocess.run(
            self.executable,
            cwd=working_directory,
            stdin="pysp_model",
            outfile=logfile,
            tee=output_solver_log)
        stop = time.time()
        assert os.path.exists(sdoutput_directory)

        #
        # Parse the SD solution
        #

        if self.get_option("verbose"):
            print("Reading SD solution from file: %s"
                  % (solution_filename))

        xhat, results = self._read_solution(input_files["symbols"],
                                            solution_filename)
        results.solver.time = stop - start

        results.xhat = None
        if xhat is not None:
            if isinstance(sp, EmbeddedSP):
                results.xhat = {sp.time_stages[0]: xhat}
            else:
                results.xhat = {sp.scenario_tree.findRootNode().name: xhat}

        return results

    def _write_config(self, filename):
        """ Writes an SD config file """
        with open(filename, "w") as f:
            if self.get_option("stopping_rule_tolerance") == "loose":
                f.write("// nominal tolerance\n")
                f.write("EPSILON 0.01\n")
                f.write("SCAN_LEN 64\n")
            elif self.get_option("stopping_rule_tolerance") == "nominal":
                f.write("// nominal tolerance\n")
                f.write("EPSILON 0.001\n")
                f.write("SCAN_LEN 256\n")
            elif self.get_option("stopping_rule_tolerance") == "tight":
                f.write("// nominal tolerance\n")
                f.write("EPSILON 0.00001\n")
                f.write("SCAN_LEN 512\n")
            else:
                assert False
            f.write("MULTIPLE_REP %d\n"
                    % (0 if self.get_option("single_replication") else 1))
            f.write("PRINT_CYCLE %d\n" % (self.get_option("print_cycle")))
            f.write("EVAL_RUN_FLAG %d\n"
                    % (1 if self.get_option("eval_run_flag") else 0))
            f.write("EVAL_FLAG %d\n"
                    % (1 if self.get_option("eval_flag") else 0))
            f.write("EVAL_SEED1 %d\n" % (self.get_option("eval_seed1")))
            f.write("EVAL_ERROR %r\n" % (self.get_option("eval_error")))
            f.write("MEAN_DEV %r\n" % (self.get_option("mean_dev")))
            if self.get_option("min_iterations") is not None:
                f.write("MIN_ITER %d\n" % (self.get_option("min_iterations")))
            f.write("MAX_ITER %d\n" % (self.get_option("max_iterations")))
            f.write(sd_advanced_config_section)

    def _read_solution(self,
                       symbols_filename,
                       solution_filename):
        """ Parses an SD solution file """

        # parse the symbol map
        symbol_map = {}
        with open(symbols_filename) as f:
            for line in f:
                lp_symbol, scenario_tree_id = line.strip().split()
                symbol_map[lp_symbol] = scenario_tree_id

        results = SPSolverResults()
        results.status = None
        results.solver.status = None
        results.solver.termination_condition = None
        results.solver.message = None
        xhat = {}
        try:
            with open(solution_filename, 'r') as f:
                line = f.readline()
                assert line.startswith("Problem:")
                assert line.split()[1].strip() == "pysp_model"
                line = f.readline()
                assert line.startswith("First Stage Rows:")
                line = f.readline()
                assert line.startswith("First Stage Columns:")
                line = f.readline()
                assert line.startswith("First Stage Non-zeros:")
                line = f.readline()
                assert line.startswith("Replication No.") or \
                    line.startswith("Number of replications:")
                line = f.readline()
                assert line.startswith("Status:")
                results.solver.message = line.split(":")[1].strip()
                (results.status,
                 results.solver.status,
                 results.solver.termination_condition) = \
                    _sd_status_map.get(results.solver.message,
                                       (SolutionStatus.unknown,
                                        SolverStatus.unknown,
                                        TerminationCondition.unknown))

                #
                # Objective and Bound
                #

                line = f.readline()
                assert line.startswith("Total Objective Function Upper Bound:")
                line = line.split(':')
                if line[1].strip() == '':
                    pass
                else:
                    assert len(line) == 4
                    line = line[1]
                    if "half-width" in line:
                        # we are given confidence intervals on the objective
                        line = line.split(',')
                        assert len(line) == 4
                        results.objective = float(line[0])
                        assert line[1].startswith('[')
                        assert line[2].endswith(']')
                        results.objective_interval = (float(line[1][1:]),
                                              float(line[2][:-1]))
                    else:
                        results.objective = float(line[1])
                line = f.readline()
                assert line.startswith("Total Objective Function Lower Bound:")
                line = line.split(':')
                if line[1].strip() == '':
                    pass
                else:
                    if "half-width" in line[1]:
                        # we are given confidence intervals on the bound
                        line = line[1].split(',')
                        assert len(line) == 4
                        results.bound = float(line[0])
                        assert line[1].startswith('[')
                        assert line[2].endswith(']')
                        results.bound_interval = (float(line[1][1:]),
                                          float(line[2][:-1]))
                    else:
                        results.bound = float(line[1].strip())

                #
                # Xhat
                #

                line = f.readline()
                assert line.strip() == ''
                line = f.readline()
                assert line.startswith('First Stage Solutions:')
                line = f.readline()
                assert line.startswith('   No.   Row name   Activity      Lower bound   Upper bound   Dual          Dual STDEV')
                line = f.readline()
                assert line.startswith('------ ------------ ------------- ------------- ------------- ------------- -------------')

                xhat_start_line = '   No. Column name  Activity      Lower bound   Upper bound   Reduced Cost  RC STDEV'
                line = f.readline()
                while not line.startswith(xhat_start_line):
                    line = f.readline()
                line = f.readline()
                assert line.startswith('------ ------------ ------------- ------------- ------------- ------------- -------------')
                line = f.readline().strip().split()
                while line:
                    varlabel, varvalue = line[1:3]
                    varlabel = varlabel.strip()
                    varvalue = float(varvalue)
                    xhat[symbol_map[varlabel]] = varvalue
                    line = f.readline().strip().split()

        except (IOError, OSError):
            logger.warn(
                "Exception encountered while parsing sd "
                "solution file '%s':\n%s'"
                % (solution_filename, traceback.format_exc()))
            xhat = None

        return xhat, results

def runsd_register_options(options=None):
    if options is None:
        options = PySPConfigBlock()
    SDSolver.register_options(options)
    ScenarioTreeManagerFactory.register_options(options)
    safe_register_common_option(options,
                               "verbose")
    safe_register_common_option(options,
                               "disable_gc")
    safe_register_common_option(options,
                               "profile")
    safe_register_common_option(options,
                               "traceback")
    safe_register_common_option(options,
                                "output_scenario_tree_solution")
    safe_register_common_option(options,
                                "keep_solver_files")
    safe_register_common_option(options,
                                "output_solver_log")
    safe_register_common_option(options,
                                "symbolic_solver_labels")
    return options

def runsd(options):
    """
    Construct a senario tree manager and solve it
    with the SD solver.
    """
    start_time = time.time()
    with ScenarioTreeManagerFactory(options) as sp:
        sp.initialize()
        print("")
        print("Running SD solver for stochastic "
              "programming problems")
        sd = SDSolver()
        sd_options = sd.extract_user_options_to_dict(options,
                                                     sparse=True)
        results = sd.solve(
            sp,
            options=sd_options,
            output_solver_log=options.output_solver_log,
            keep_solver_files=options.keep_solver_files,
            symbolic_solver_labels=options.symbolic_solver_labels)
        xhat = results.xhat
        del results.xhat
        print("")
        print(results)

        if options.output_scenario_tree_solution:
            print("")
            sp.scenario_tree.snapshotSolutionFromScenarios()
            sp.scenario_tree.pprintSolution()
            sp.scenario_tree.pprintCosts()

    print("")
    print("Total execution time=%.2f seconds"
          % (time.time() - start_time))
    return 0

#
# the main driver routine
#

def main(args=None):
    #
    # Top-level command that executes everything
    #

    #
    # Import plugins
    #
    import pyomo.environ

    #
    # Parse command-line options.
    #
    try:
        options = parse_command_line(
            args,
            runsd_register_options,
            prog='runsd',
            description=(
"""Optimize a stochastic program using the SD solver."""
            ))

    except SystemExit as _exc:
        # the parser throws a system exit if "-h" is specified
        # - catch it to exit gracefully.
        return _exc.code

    return launch_command(runsd,
                          options,
                          error_label="runsd: ",
                          disable_gc=options.disable_gc,
                          profile_count=options.profile,
                          traceback=options.traceback)

SPSolverFactory.register_solver("sd", SDSolver)

if __name__ == "__main__":
    sys.exit(main())
