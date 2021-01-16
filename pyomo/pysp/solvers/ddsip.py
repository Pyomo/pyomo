#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

# TODO: have ddsip convert create symbols files for second stage
# TODO: parse second-stage solution and load into scenario tree workers
# TODO: objective, cost, stage_costs
# TODO: make output_scenario_tree_solution work

import io
import os
import sys
import time
import logging
import traceback

import pyutilib.subprocess

from pyomo.opt import (TerminationCondition,
                       SolverStatus,
                       SolutionStatus)
from pyomo.pysp.util.configured_object import PySPConfiguredObject
from pyomo.pysp.util.config import (PySPConfigValue,
                                    PySPConfigBlock,
                                    safe_register_common_option,
                                    safe_register_unique_option,
                                    safe_declare_common_option,
                                    safe_declare_unique_option,
                                    _domain_tuple_of_str_or_dict,
                                    _domain_must_be_str)
from pyomo.pysp.util.misc import (parse_command_line,
                                  launch_command)
from pyomo.pysp.scenariotree.manager import ScenarioTreeManagerFactory
import pyomo.pysp.convert.ddsip
from pyomo.pysp.solvers.spsolver import (SPSolverResults,
                                         SPSolverFactory)
from pyomo.pysp.solvers.spsolvershellcommand import \
    SPSolverShellCommand

logger = logging.getLogger('pyomo.pysp')

thisfile = os.path.abspath(__file__)

_ddsip_group_label = "DDSIP Options"
_firststage_var_suffix = "__DDSIP_FIRSTSTAGE"

# maps the ddsip status code to tuples of the form
# (SolutionStatus, SolverStatus, TerminationCondition, message)
# feel free to modify if you have opinions one these
_ddsip_status_map = {}
# the process was terminated by the user
_ddsip_status_map[-1] = (SolutionStatus.unknown,
                         SolverStatus.aborted,
                         TerminationCondition.userInterrupt,
                         "Termination signal received.")
# the node limit has been reached
_ddsip_status_map[1] = (SolutionStatus.stoppedByLimit,
                        SolverStatus.ok,
                        TerminationCondition.maxEvaluations,
                        "Node limit reached (total number of nodes).")
# the gap (absolute or relative) has been reached
_ddsip_status_map[2] = (SolutionStatus.optimal,
                        SolverStatus.ok,
                        TerminationCondition.optimal,
                        "Gap reached.")
# the time limit has been reached
_ddsip_status_map[3] = (SolutionStatus.stoppedByLimit,
                        SolverStatus.ok,
                        TerminationCondition.maxTimeLimit,
                        "Time limit reached.")
# the maximal dispersion, i.e. the maximal difference of the
# first-stage components within all remaining front nodes,
# was less then the parameter NULLDISP (null dispersion)
_ddsip_status_map[4] = (SolutionStatus.optimal,
                        SolverStatus.ok,
                        TerminationCondition.minStepLength,
                        "Maximal dispersion equals zero.")
# the whole branching tree was backtracked.
_ddsip_status_map[5] = (SolutionStatus.optimal,
                        SolverStatus.ok,
                        TerminationCondition.minStepLength,
                        ("The whole branching tree was "
                         "backtracked. Probably due to MIP "
                         "gaps (see below) the specified "
                         "gap tolerance could not be reached."))
# no valid lower bound
_ddsip_status_map[6] = (SolutionStatus.unknown,
                        SolverStatus.warning,
                        TerminationCondition.invalidProblem,
                        "No valid lower bound.")
# problem infeasible
_ddsip_status_map[7] = (SolutionStatus.infeasible,
                        SolverStatus.warning,
                        TerminationCondition.infeasible,
                        "Problem infeasible.")
# problem unbounded
_ddsip_status_map[8] = (SolutionStatus.unbounded,
                        SolverStatus.warning,
                        TerminationCondition.unbounded,
                        "Problem unbounded.")

def _load_solution(manager,
                   scenario,
                   solution_filename,
                   info_filename,
                   firststage_symbols_filename,
                   secondstage_symbols_filename,
                   scenario_id):

    # parse the symbol map
    firststage_symbol_map = {}
    with open(firststage_symbols_filename) as f:
        for line in f:
            lp_symbol, scenario_tree_id = line.strip().split()
            firststage_symbol_map[lp_symbol] = scenario_tree_id
    secondstage_symbol_map = {}
    with open(secondstage_symbols_filename) as f:
        for line in f:
            lp_symbol, scenario_tree_id = line.strip().split()
            secondstage_symbol_map[lp_symbol] = scenario_tree_id

    x_firststage = scenario._x[scenario.node_list[0].name]
    assert scenario.node_list[0] is manager.scenario_tree.findRootNode()
    x_secondstage = scenario._x[scenario.node_list[1].name]
    with open(solution_filename, 'r') as f:
        line = f.readline()
        while line.strip() != "1. Best Solution":
            line = f.readline()
        line = f.readline()
        assert line.startswith("Variable name                Value")
        line = f.readline()
        assert line.startswith("-----------------------------------")
        line = f.readline().strip()
        while line != "":
            varlabel, varsol = line.split()
            x_firststage[firststage_symbol_map[varlabel]] = \
                float(varsol)
            line = f.readline().strip()
        while line != "4. Second-stage solutions":
            line = f.readline().strip()
        scenario_label = ("Scenario %d:" % scenario_id)
        line = f.readline().strip()
        while line != scenario_label:
            line = f.readline().strip()
        line = f.readline().strip()
        while (line != "") and (not line.startswith("Scenario ")):
            varlabel, varsol = line.split()
            if varlabel != "ONE_VAR_CONSTANT":
                x_secondstage[secondstage_symbol_map[varlabel]] = \
                    float(varsol)
            line = f.readline().strip()
    return x

class DDSIPSolver(SPSolverShellCommand, PySPConfiguredObject):

    @classmethod
    def _declare_options(cls, options=None):
        if options is None:
            options = PySPConfigBlock()
        safe_declare_unique_option(
            options,
            "firststage_suffix",
            PySPConfigValue(
                "__DDSIP_FIRSTSTAGE",
                domain=_domain_must_be_str,
                description=(
                    "The suffix used to identity first-stage "
                    "variables to DDSIP. Default is "
                    "'__DDSIP_FIRSTSTAGE'"
                ),
                doc=None,
                visibility=0),
            ap_group=_ddsip_group_label)
        safe_declare_unique_option(
            options,
            "config_file",
            PySPConfigValue(
                None,
                domain=_domain_must_be_str,
                description=(
                    "The name of a partial DDSIP configuration file "
                    "that contains option specifications unrelated to "
                    "the problem structure. If specified, the contents "
                    "of this file will be appended to the "
                    "configuration created by this solver interface. "
                    "Default is None."
                ),
                doc=None,
                visibility=0),
            ap_group=_ddsip_group_label)
        safe_declare_common_option(options,
                                   "verbose",
                                   ap_group=_ddsip_group_label)

        return options

    def __init__(self):
        super(DDSIPSolver, self).__init__(self.register_options())
        self.set_options_to_default()
        self._executable = "ddsip"

    def set_options_to_default(self):
        self._options = self.register_options()
        self._options._implicit_declaration = True

    @property
    def options(self):
        return self._options

    @property
    def name(self):
        return "ddsip"

    def _solve_impl(self,
                    sp,
                    output_solver_log=False,
                    **kwds):
        """
        Solve a stochastic program with the DDSIP solver.

        See the 'solve' method on the base class for
        additional keyword documentation.

        Args:
            sp: The stochastic program to solve.
            output_solver_log (bool): Stream the solver
                output during the solve.
            **kwds: Passed to the DDSIP file writer as I/O
              options (e.g., symbolic_solver_labels=True).

        Returns: A results object with information about the solution.
        """

        if len(sp.scenario_tree.stages) > 2:
            raise ValueError("DDSIP solver does not handle more "
                             "than 2 time-stages")

        #
        # Setup the DDSIP working directory
        #

        working_directory = self._create_tempdir("workdir")
        input_directory = os.path.join(working_directory,
                                       "ddsip_files")
        output_directory = os.path.join(input_directory,
                                        "sipout")

        logfile = self._files['logfile'] = \
                  os.path.join(working_directory,
                               "ddsip.log")

        os.makedirs(input_directory)
        assert os.path.exists(input_directory)
        assert not os.path.exists(output_directory)
        info_filename = os.path.join(output_directory,
                                     "sip.out")
        solution_filename = os.path.join(output_directory,
                                         "solution.out")

        #
        # Create the DDSIP input files
        #

        if self.get_option("verbose"):
            print("Writing solver files in directory: %s"
                  % (working_directory))

        input_files = pyomo.pysp.convert.ddsip.\
            convert_external(
                input_directory,
                self.options.firststage_suffix,
                sp,
                io_options=kwds)
        for key in input_files:
            self._add_tempfile(key, input_files[key])

        self._update_config(input_files["config"])

        #
        # Launch DDSIP
        #

        _cmd_string = self.executable+" < "+input_files["script"]
        if self.get_option("verbose"):
            print("Launching DDSIP solver with command: %s"
                  % (_cmd_string))
        ddsipstdin = None
        with open(input_files["script"]) as f:
            ddsipstdin = f.read()
        assert ddsipstdin is not None

        start = time.time()
        rc, log = pyutilib.subprocess.run(
            self.executable,
            cwd=input_directory,
            stdin=ddsipstdin,
            outfile=logfile,
            tee=output_solver_log)
        stop = time.time()

        if rc:
            if not self.available():
                raise ValueError(
                    "Solver executable does not exist: '%s'. "
                    "(note that the default executable generated "
                    "by the DDSIP build system does not have this name)"
                    % (self.executable))
            else:
                raise RuntimeError(
                    "A nonzero return code (%s) was encountered after "
                    "launching the command: %s. Check the solver log file "
                    "for more information: %s"
                    % (rc, _cmd_string, logfile))

        #
        # Parse the DDSIP solution
        #

        if self.get_option("verbose"):
            print("Reading DDSIP solution from file: %s"
                  % (solution_filename))
        assert os.path.exists(output_directory)

        async_xhat = None
        async_responses = []

        # TODO: The solution symbols are not being translated
        #       back to scenario tree ids. Fix this and also have
        #       the first and second stage solutions be loaded into
        #       the Pyomo models
        #for scenario_id, scenario in enumerate(sp.scenario_tree.scenarios, 1):
        #    async_responses.append(sp.invoke_function(
        #        "_load_solution",
        #        thisfile,
        #        invocation_type=InvocationType.OnScenario(scenario.name),
        #        function_args=(solution_filename, scenario_id),
        #        async_call=True))

        xhat, results = self._read_solution(sp,
                                            input_files["symbols"],
                                            info_filename,
                                            solution_filename)

        results.xhat = None
        if xhat is not None:
            results.xhat = {sp.scenario_tree.findRootNode().name: xhat}

        for res in async_responses:
            res.complete()

        return results

    def _update_config(self, config_filename):
        """ Writes a DDSIP config file """

        # find the byte position where
        # we start appending to the config file
        # (just before END)
        append_pos = 0
        with io.open(config_filename,
                     mode='rb',
                     buffering=0) as f:
            f.seek(0)
            append_pos = f.tell()
            for line in f:
                if line.strip().decode() == "END":
                    break
                append_pos = f.tell()
        assert append_pos > 0

        config_lines = {}
        config_lines[None] = []
        config_lines['CPLEX'] = []
        config_lines['CPLEXEEV'] = []
        config_lines['CPLEXLB'] = []
        config_lines['CPLEX2LB'] = []
        config_lines['CPLEXUB'] = []
        config_lines['CPLEX2UB'] = []
        config_lines['CPLEXDUAL'] = []
        config_lines['CPLEX2DUAL'] = []
        has_cplex_opts = False

        # parse the user specified config file
        if self.options.config_file is not None:
            with open(self.options.config_file) as fconfig:
                section = config_lines[None]
                for line in fconfig:
                    stripped = line.strip()
                    if (stripped == "BEGIN") and \
                       (stripped == "END"):
                        continue
                    if "CPLEXBEGIN" in stripped:
                        section = config_lines['CPLEX']
                        has_cplex_opts = True
                        continue
                    elif "CPLEXEEV" in stripped:
                        section = config_lines['CPLEXEEV']
                        has_cplex_opts = True
                        continue
                    elif "CPLEXLB" in stripped:
                        section = config_lines['CPLEXLB']
                        has_cplex_opts = True
                        continue
                    elif "CPLEX2LB" in stripped:
                        section = config_lines['CPLEX2LB']
                        has_cplex_opts = True
                        continue
                    elif "CPLEXUB" in stripped:
                        section = config_lines['CPLEXUB']
                        has_cplex_opts = True
                        continue
                    elif "CPLEX2UB" in stripped:
                        section = config_lines['CPLEX2UB']
                        has_cplex_opts = True
                        continue
                    elif "CPLEXDUAL" in stripped:
                        section = config_lines['CPLEXDUAL']
                        has_cplex_opts = True
                        continue
                    elif "CPLEX2DUAL" in stripped:
                        section = config_lines['CPLEX2DUAL']
                        has_cplex_opts = True
                        continue
                    elif "CPLEXEND" in stripped:
                        section = config_lines[None]
                        continue
                    section.append(line)

        for key in self.options:
            if (key != "firststage_suffix") and \
               (key != "config_file") and \
               (key != "verbose"):
                if key.startswith("CPLEX_"):
                    section = config_lines["CPLEX"]
                    prefix = "CPLEX_"
                    has_cplex_opts = True
                elif key.startswith("CPLEXEEV_"):
                    section = config_lines["CPLEXEEV"]
                    prefix = "CPLEXEEV_"
                    has_cplex_opts = True
                elif key.startswith("CPLEXLB_"):
                    section = config_lines["CPLEXLB"]
                    prefix = "CPLEXLB_"
                    has_cplex_opts = True
                elif key.startswith("CPLEX2LB_"):
                    section = config_lines["CPLEX2LB"]
                    prefix = "CPLEX2LB_"
                    has_cplex_opts = True
                elif key.startswith("CPLEXUB_"):
                    section = config_lines["CPLEXUB"]
                    prefix = "CPLEXUB_"
                    has_cplex_opts = True
                elif key.startswith("CPLEX2UB_"):
                    section = config_lines["CPLEX2UB"]
                    prefix = "CPLEX2UB_"
                    has_cplex_opts = True
                elif key.startswith("CPLEXDUAL_"):
                    section = config_lines["CPLEXDUAL"]
                    prefix = "CPLEXDUAL_"
                    has_cplex_opts = True
                elif key.startswith("CPLEX2DUAL_"):
                    section = config_lines["CPLEX2DUAL"]
                    prefix = "CPLEX2DUAL_"
                    has_cplex_opts = True
                else:
                    section = config_lines[None]
                    prefix = ""
                val = self.options[key]
                line = "%s %s\n" % (key.replace(prefix,'',1), val)
                section.append(line)

        with open(config_filename, "r+") as f:
            f.seek(append_pos)
            for line in config_lines[None]:
                f.write(line)
            if has_cplex_opts:
                f.write("\nCPLEXBEGIN\n")
                for line in config_lines["CPLEX"]:
                    f.write(line)
                for section_key in ('CPLEXEEV',
                                    'CPLEXLB',
                                    'CPLEX2LB',
                                    'CPLEXUB',
                                    'CPLEX2UB',
                                    'CPLEXDUAL',
                                    'CPLEX2DUAL'):
                    section = config_lines[section_key]
                    if len(section) > 0:
                        f.write(section_key+"\n")
                        for line in section:
                            f.write(line)
                f.write("CPLEXEND\n")

            f.write("\n\nEND\n")

    def _read_solution(self,
                       sp,
                       symbols_filename,
                       info_filename,
                       solution_filename):
        """Parses a DDSIP solution file."""

        # parse the symbol map
        symbol_map = {}
        with open(symbols_filename) as f:
            for line in f:
                lp_symbol, scenario_tree_id = line.strip().split()
                symbol_map[lp_symbol] = scenario_tree_id

        #
        # Xhat
        #
        try:
            xhat = {}
            with open(solution_filename, 'r') as f:
                line = f.readline()
                while line.strip() != "1. Best Solution":
                    line = f.readline()
                line = f.readline()
                assert line.startswith("Variable name                Value")
                line = f.readline()
                assert line.startswith("-----------------------------------")
                line = f.readline().strip()
                while line != "":
                    line = line.split()
                    varlabel, varsol = line
                    xhat[symbol_map[varlabel]] = float(varsol)
                    line = f.readline().strip()
        except (IOError, OSError):
            logger.warn(
                "Exception encountered while parsing ddsip "
                "solution file '%s':\n%s'"
                % (solution_filename, traceback.format_exc()))
            xhat = None

        #
        # Objective, bound, status, etc.
        #
        results = SPSolverResults()
        results.solver.status_code = None
        results.status = None
        results.solver.status = None
        results.solver.termination_condition = None
        results.solver.message = None
        results.solver.time = None
        try:
            with open(info_filename, 'r') as f:
                line = f.readline()
                while True:
                    if line.startswith("Total CPU time:"):
                        break
                    line = f.readline()
                    if line == '':
                        # Unexpected file format or the solve failed
                        logger.warn(
                            "Unexpected ddsip info file format. No "
                            "status information will be returned")
                        return xhat, results
                line = f.readline().strip()
                while (line == "") or \
                      (line == "----------------------------------------------------------------------------------------"):
                    line = f.readline().strip()
                if line.startswith("EEV"):
                    results.eev = float(line.split()[1])
                    line = f.readline().strip()
                if line.startswith("VSS"):
                    results.vss = float(line.split()[1])
                    line = f.readline().strip()
                assert line.startswith("EVPI")
                line = line.split()
                results.evpi = float(line[1])
                line = f.readline().strip()
                assert line == ""
                line = f.readline().strip()
                assert line == "----------------------------------------------------------------------------------------"
                line = f.readline().strip()
                line = line.split()
                assert len(line) == 4
                assert line[0] == "Status"
                results.solver.status_code = int(line[1])
                assert line[2] == "Time"
                results.solver.time = float(line[3])
                (results.status,
                 results.solver.status,
                 results.solver.termination_condition,
                 results.solver.message) = \
                    _ddsip_status_map.get(results.solver.status_code,
                                          (SolutionStatus.unknown,
                                           SolverStatus.unknown,
                                           TerminationCondition.unknown,
                                           None))

                line = f.readline().strip()
                line = line.split()
                assert len(line) == 6
                assert line[0] == "Upper"
                assert line[3] == "Tree"
                results.tree_depth = int(line[5])
                line = f.readline().strip()
                line = line.split()
                assert len(line) == 2
                assert line[0] == "Nodes"
                results.nodes = int(line[1])
                line = f.readline().strip()
                while line != "----------------------------------------------------------------------------------------":
                    line = f.readline().strip()
                line = f.readline().strip()
                assert line.startswith("Best Value")
                results.objective = float(line.split()[2])
                # NOTE: I think DDSIP refers to the "bound" as
                #       "Lower Bound", even when the objective
                #       is being maximized.
                line = f.readline().strip()
                assert line.startswith("Lower Bound")
                results.bound = float(line.split()[2])
        except (IOError, OSError):
            logger.warn(
                "Exception encountered while parsing ddsip "
                "info file '%s':\n%s'"
                % (info_filename, traceback.format_exc()))

        return xhat, results

def runddsip_register_options(options=None):
    if options is None:
        options = PySPConfigBlock()
    DDSIPSolver.register_options(options)
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
    # used to populate the implicit DDSIP options
    safe_register_unique_option(
        options,
        "solver_options",
        PySPConfigValue(
            (),
            domain=_domain_tuple_of_str_or_dict,
            description=(
                "Unregistered solver options that will be passed "
                "to DDSIP via the config file (e.g., NODELIM=4, "
                "CPLEX_1067=1). This option can be used multiple "
                "times from the command line to specify more "
                "than one DDSIP option."
            ),
            doc=None,
            visibility=0),
        ap_kwds={'action': 'append'},
        ap_group=_ddsip_group_label)

    return options

def runddsip(options):
    """
    Construct a senario tree manager and solve it
    with the DDSIP solver.
    """
    start_time = time.time()
    with ScenarioTreeManagerFactory(options) as sp:
        sp.initialize()
        print("")
        print("Running DDSIP solver for stochastic "
              "programming problems")
        ddsip = DDSIPSolver()
        # add the implicit ddsip options
        solver_options = options.solver_options
        if len(solver_options) > 0:
            if type(solver_options) is tuple:
                for name_val in solver_options:
                    assert "=" in name_val
                    name, val = name_val.split("=")
                    ddsip.options[name.strip()] = val.strip()
            else:
                for key, val in solver_options.items():
                    ddsip.options[key] = val
        ddsip_options = ddsip.extract_user_options_to_dict(options,
                                                           sparse=True)
        results = ddsip.solve(
            sp,
            options=ddsip_options,
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
            runddsip_register_options,
            prog='runddsip',
            description=(
"""Optimize a stochastic program using the DDSIP solver."""
            ))

    except SystemExit as _exc:
        # the parser throws a system exit if "-h" is specified
        # - catch it to exit gracefully.
        return _exc.code

    return launch_command(runddsip,
                          options,
                          error_label="runddsip: ",
                          disable_gc=options.disable_gc,
                          profile_count=options.profile,
                          traceback=options.traceback)

SPSolverFactory.register_solver("ddsip", DDSIPSolver)

if __name__ == "__main__":
    sys.exit(main())
