#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

# TODO: finishing exposing DDSIP options by declaring them
#       on the solver config object
# TODO: parse second-stage solution and load into scenario tree workers

import os
import sys
import time
import shutil

import pyutilib.subprocess
import pyutilib.services

from pyomo.core import maximize
from pyomo.pysp.util.configured_object import PySPConfiguredObject
from pyomo.pysp.util.config import (PySPConfigValue,
                                    PySPConfigBlock,
                                    safe_register_common_option,
                                    safe_register_unique_option,
                                    safe_declare_common_option,
                                    safe_declare_unique_option,
                                    _domain_positive,
                                    _domain_nonnegative,
                                    _domain_positive_integer,
                                    _domain_nonnegative_integer,
                                    _domain_must_be_str,
                                    _domain_integer_interval)
from pyomo.pysp.util.misc import (parse_command_line,
                                  launch_command)
from pyomo.pysp.scenariotree.manager import InvocationType
from pyomo.pysp.scenariotree.manager_solver import \
    (ScenarioTreeManagerSolver,
     ScenarioTreeManagerFactory)
import pyomo.pysp.convert.ddsip
from pyomo.pysp.embeddedsp import EmbeddedSP
from pyomo.pysp.solvers.spsolver import (SPSolverResults,
                                         SPSolverFactory)
from pyomo.pysp.solvers.spsolvershellcommand import \
    SPSolverShellCommand

thisfile = os.path.abspath(__file__)

_ddsip_group_label = "DDSIP Options"
_firststage_var_suffix = "__DDSIP_FIRSTSTAGE"

def _load_solution(manager,
                   scenario,
                   solution_filename,
                   scenario_id):
    x_firststage = x[scenario.node_list[0].name] = {}
    assert scenario.node_list[0] is manager.scenario_tree.findRootNode()
    x_secondstage = x[scenario.node_list[1].name] = {}
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
            varname, varsol = line
            assert varname.endswith(_firststage_var_suffix)
            x_firststage[varname[:-len(_firststage_var_suffix)]] = float(varsol)
            line = f.readline().strip()
        while line != "4. Second-stage solutions":
            line = f.readline().strip()
        scenario_label = ("Scenario %d:" % scenario_id)
        line = f.readline().strip()
        while line != scenario_label:
            line = f.readline().strip()
        line = f.readline().strip()
        while (line != "") and (not line.startswith("Scenario ")):
            name, val = line.split()
            x_secondstage[name] = float(val)
            line = f.readline().strip()
    return x

class DDSIPSolver(SPSolverShellCommand, PySPConfiguredObject):

    @classmethod
    def _declare_options(cls, options=None):
        if options is None:
            options = PySPConfigBlock()
        safe_register_unique_option(
            options,
            "output_level",
            PySPConfigValue(
                5,
                domain=_domain_integer_interval(0,100),
                description=(
                    "Amount of output to more.out. Caution: "
                    "The file more.out may become large for "
                    "high values of outlevel. Default is 5."
                ),
                doc=None,
                visibility=0),
            ap_group=_ddsip_group_label)
        safe_register_unique_option(
            options,
            "output_files",
            PySPConfigValue(
                1,
                domain=_domain_integer_interval(0,6),
                description=(
                    "Amount of output files. See DDSIP "
                    "manual for more information. Caution: "
                    "If outfiles is greater than 3, lp- or "
                    "sav- files are written at each node and "
                    "for each scenario! Default is 1."
                ),
                doc=None,
                visibility=0),
            ap_group=_ddsip_group_label)
        safe_register_unique_option(
            options,
            "log_frequency",
            PySPConfigValue(
                1,
                domain=_domain_positive_integer,
                description=(
                    "A line of output is printed every i-th "
                    "iteration. Default is 1."
                ),
                doc=None,
                visibility=0),
            ap_group=_ddsip_group_label)
        safe_register_unique_option(
            options,
            "node_limit",
            PySPConfigValue(
                10000,
                domain=_domain_nonnegative_integer,
                description=(
                    "The node limit for the branch-and-bound "
                    "procedure. Default is 10000."
                ),
                doc=None,
                visibility=0),
            ap_group=_ddsip_group_label)
        safe_register_unique_option(
            options,
            "time_limit",
            PySPConfigValue(
                86400,
                domain=_domain_nonnegative_integer,
                description=(
                    "The total time limit in seconds (CPU-time) "
                    "including the time needed to solve the "
                    "EEV problem. Default is 86400 (1 day)."
                ),
                doc=None,
                visibility=0),
            ap_group=_ddsip_group_label)
        safe_register_unique_option(
            options,
            "absolute_gap",
            PySPConfigValue(
                0,
                domain=_domain_nonnegative,
                description=(
                    "The absolute duality gap. Default is 0."
                ),
                doc=None,
                visibility=0),
            ap_group=_ddsip_group_label)
        safe_register_unique_option(
            options,
            "relative_gap",
            PySPConfigValue(
                1e-4,
                domain=_domain_nonnegative,
                description=(
                    "The relative duality gap. Default is 1e-4."
                ),
                doc=None,
                visibility=0),
            ap_group=_ddsip_group_label)
        safe_register_unique_option(
            options,
            "eev_problem",
            PySPConfigValue(
                False,
                domain=bool,
                description=(
                    "Solve the EEV-problem and report the "
                    "VSS, cf. Birge and Louveaux (1997). "
                    "Default is False."
                ),
                doc=None,
                visibility=0),
            ap_group=_ddsip_group_label)
        safe_register_unique_option(
            options,
            "deterministic_equivalent",
            PySPConfigValue(
                False,
                domain=bool,
                description=(
                    "Produce a deterministic equivalent as "
                    "detequ.lp.gz. Only for expectation based "
                    "problems (and it takes quite a while). "
                    "Default is False."
                ),
                doc=None,
                visibility=0),
            ap_group=_ddsip_group_label)
        safe_register_unique_option(
            options,
            "max_inherit",
            PySPConfigValue(
                5,
                domain=_domain_nonnegative_integer,
                description=(
                    "Maximal level of inheritance of "
                    "scenario solutions in the "
                    "branch-and-bound tree. Default is 5."
                ),
                doc=None,
                visibility=0),
            ap_group=_ddsip_group_label)
        safe_register_unique_option(
            options,
            "hotstart_strategy",
            PySPConfigValue(
                1,
                domain=_domain_integer_interval(0,6),
                description=(
                    "Hotstart strategy used for solutions in "
                    "branch-and-bound tree. See DDSIP manual "
                    "for more information. Valid values fall in "
                    "the integer range [0,6]. Default is 1."
                ),
                doc=None,
                visibility=0),
            ap_group=_ddsip_group_label)

        return options

    def __init__(self, *args, **kwds):
        super(DDSIPSolver, self).__init__(*args, **kwds)
        self._name = "ddsip"
        self._executable = "ddsip"
        self._firststage_var_suffix = '__DDSIP_FIRSTSTAGE'

    def _solve_impl(self,
                    sp,
                    output_solver_log=False,
                    verbose=False,
                    **kwds):

        if len(sp.scenario_tree.stages) > 2:
            raise ValueError("DDSIP solver does not handle more "
                             "than 2 time-stages")

        #
        # Setup the DDSIP working directory
        #

        working_directory = self._create_tempdir("workdir")
        input_directory = os.path.join(working_directory,
                                       "sipin")
        output_directory = os.path.join(working_directory,
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

        if verbose:
            print("Writing solver files in directory: %s"
                  % (working_directory))

        input_files = pyomo.pysp.convert.ddsip.\
            convert_external(
                input_directory,
                _firststage_var_suffix,
                sp,
                io_options=kwds)
        for key in input_files:
            self._add_tempfile(key, input_files[key])

        self._update_config(input_files["config"])

        #
        # Launch DDSIP
        #

        _cmd_string = self.executable+" < "+input_files["script"]
        if verbose:
            print("Launching DDSIP solver with command: %s"
                  % (_cmd_string))
        ddsipstdin = None
        with open(input_files["script"]) as f:
            ddsipstdin = f.read()
        assert ddsipstdin is not None

        start = time.time()
        rc, log = pyutilib.subprocess.run(
            self.executable,
            cwd=working_directory,
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

        if verbose:
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
        #        async=True))

        xhat, results = self._read_solution(sp,
                                            input_files["symbols"],
                                            info_filename,
                                            solution_filename)

        results.xhat = {sp.scenario_tree.findRootNode().name: xhat}

        results.solver_time = stop - start

        for res in async_responses:
            res.complete()

        return results

    def _update_config(self, config_filename):
        """ Writes an DDSIP config file """

        # remove "END" from the config file
        new_config_filename = \
            os.path.join(os.path.dirname(config_filename),
                         os.path.basename(config_filename)+".tmp")
        with open(new_config_filename, "w") as f:
            with open(config_filename, "r") as forig:
                for line in forig:
                    if line.strip() != "END":
                        f.write(line)

            f.write("\n\nCPLEXBEGIN\n")
            f.write("1035 0 * Output on screen indicator\n")
            f.write("2008 0.001 * Absolute Gap\n")
            f.write("2009 0.001 * Relative Gap\n")
            f.write("1039 1200 * Time limit\n")
            f.write("1016 1e-9 * simplex feasibility tolerance\n")
            f.write("1014 1e-9 * simplex optimality tolerance\n")
            f.write("1065 40000 * Memory available for working storage\n")
            f.write("2010 1e-20 * integrality tolerance\n")
            f.write("2008 0 * Absolute gap\n")
            f.write("2020 0 * Priority order\n")
            f.write("2012 4 * MIP display level\n")
            f.write("2053 2 * disjunctive cuts\n")
            f.write("2040 2 * flow cover cuts\n")
            f.write("2060 3 *DiveType mip strategy dive (probe=3)\n")
            f.write("CPLEXEND\n\n")

            f.write("OUTLEVEL %s * Print info to more.out\n"
                    % (self.get_option("output_level")))
            f.write("OUTFILES %s * Print files (models and output)\n"
                    % (self.get_option("output_files")))
            f.write("LOGFREQ %s * Output log frequency\n"
                    % (self.get_option("log_frequency")))
            f.write("NODELIM %s * Node limit\n"
                    % (self.get_option("node_limit")))
            f.write("TIMELIMIT %s * Time limit\n"
                    % (self.get_option("time_limit")))
            f.write("ABSOLUTEGAP %r * Absolute duality gap\n"
                    % (self.get_option("absolute_gap")))
            f.write("RELATIVEGAP %r * Relative duality gap\n"
                    % (self.get_option("relative_gap")))
            f.write("EEVPROB %d * Solve EEV, cf. Birge and Louveaux (1997)\n"
                    % (self.get_option("eev_problem")))
            f.write("DETEQU %d * Produce deterministic equivalent\n"
                    % (self.get_option("deterministic_equivalent")))
            # TODO
            f.write("PORDER 0 * Use branching priority order\n")
            # TODO
            f.write("STARTI 0 * Use start solution/bound\n")
            f.write("MAXINHERIT %s * Levels to inherit in B&B tree\n"
                    % (self.get_option("max_inherit")))
            f.write("HOTSTART %d * Hotstart strategy used in B&B tree\n"
                    % (self.get_option("hotstart_strategy")))

            # TODO: finish declaring these remaining options
            f.write("HEURISTIC 99 3 7 * Heuristics: Down, Up, Near, Common, Byaverage ...(12)\n")
            f.write("BRADIRECTION -1 * Branching direction in DD\n")
            f.write("BRASTRATEGY 1 * Branching strategy in DD (1 = unsolved nodes first, 0 = best bound)\n")
            f.write("EPSILON 1e-10 * Branch epsilon for cont. var.\n")
            f.write("ACCURACY 1e-13 * Accuracy\n")
            f.write("BOUSTRATEGY 1 * Bounding strategy in DD\n")
            f.write("NULLDISP 5e-10\n")
            f.write("RELAXF 0\n")
            f.write("INTFIRST 1 * Branch first on integer\n")

            f.write("\n\nRISKMO 0 * Risk Model\n")
            f.write("RISKALG 1\n")
            f.write("WEIGHT 1\n")
            f.write("TARGET 54 * target if needed\n")
            f.write("PROBLEV .8 * probability level\n")
            f.write("RISKBM 11000000 * big M in \n")

            f.write("\n\nCBFREQ 0 * (50) Conic Bundle in every ith node\n")
            f.write("CBITLIM 20 * (10) Descent iteration limit for conic bundle method\n")
            f.write("CBTOTITLIM 50 * (1000) Total iteration limit for conic bundle method\n")
            f.write("NONANT 1 * Non-anticipativity representation\n")

            f.write("\n\nEND\n")

        os.remove(config_filename)
        shutil.copyfile(new_config_filename, config_filename)
        os.remove(new_config_filename)

    def _read_solution(self,
                       sp,
                       symbols_filename,
                       info_filename,
                       solution_filename):
        """Parses a DDSIP solution file."""

        results = SPSolverResults()

        # parse the symbol map
        symbol_map = {}
        with open(symbols_filename) as f:
            for line in f:
                lp_symbol, scenario_tree_id = line.strip().split()
                symbol_map[lp_symbol] = scenario_tree_id

        #
        # Xhat
        #
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

        #
        # Objective, bound, status, etc.
        #
        with open(info_filename, 'r') as f:

            line = f.readline()
            while not line.startswith("Total CPU time:"):
                line = f.readline()
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
            assert line.startswith("Status")
            results.status = int(line.split()[1])
            line = f.readline().strip()
            assert line.startswith("Upper")
            line = f.readline().strip()
            assert line.startswith("Nodes")
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

        return xhat, results

def runddsip_register_options(options=None):
    if options is None:
        options = PySPConfigBlock()
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
                                "symbolic_solver_labels")
    ScenarioTreeManagerFactory.register_options(options)
    DDSIPSolver.register_options(options)

    return options

def runddsip(options):
    """
    Construct a senario tree manager and solve it
    with the DDSIP solver.
    """
    start_time = time.time()
    with ScenarioTreeManagerFactory(options) as manager:
        manager.initialize()
        print("")
        print("Running DDSIP solver for stochastic "
              "programming problems")
        ddsip = DDSIPSolver(options)
        results = ddsip.solve(manager,
                              output_solver_log=True,
                              keep_solver_files=options.keep_solver_files,
                              symbolic_solver_labels=options.symbolic_solver_labels,
                              verbose=options.verbose)
        print(results)

        if options.output_scenario_tree_solution:
            print("Final solution (scenario tree format):")
            manager.scenario_tree.snapshotSolutionFromScenarios()
            manager.scenario_tree.pprintSolution()

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
