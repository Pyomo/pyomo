#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

# TODO: Workaround when mpi4py is not available or COMM_WORLD is
# TODO: Add option to launch without calling MPI_Comm_spawn
# TODO: Figure out what to do with working_directory, logfile, and output_solver_log
#       when MPI_Comm_spawn is called.

import io
import os
import sys
import time
import array

import pyutilib.subprocess

from pyomo.core import ComponentUID
from pyomo.opt import (ReaderFactory,
                       ResultsFormat,
                       UndefinedData)
from pyomo.pysp.util.configured_object import PySPConfiguredObject
from pyomo.pysp.util.config import (PySPConfigValue,
                                    PySPConfigBlock,
                                    safe_register_common_option,
                                    safe_register_unique_option,
                                    _domain_tuple_of_str_or_dict)
from pyomo.pysp.util.misc import (parse_command_line,
                                  launch_command)
from pyomo.pysp.scenariotree.manager import \
    (InvocationType,
     ScenarioTreeManagerFactory,
     ScenarioTreeManagerClientPyro,
     ScenarioTreeSolveResults)
from pyomo.pysp.scenariotree.manager_solver import \
    ScenarioTreeManagerSolverClientPyro
from pyomo.pysp.solvers.spsolver import (SPSolverResults,
                                         SPSolverFactory)
from pyomo.pysp.solvers.spsolvershellcommand import \
    SPSolverShellCommand
from pyomo.pysp.convert.schuripopt import \
    (_write_bundle_nl,
     _write_scenario_nl,
     _write_problem_list_file)


# generate an absolute path to this file
thisfile = os.path.abspath(__file__)

_schuripopt_group_label = "SchurIpoptSolver Options"

def EXTERNAL_invoke_solve(worker,
                          working_directory,
                          subproblem_type,
                          logfile,
                          problem_list_filename,
                          executable,
                          output_solver_log,
                          io_options,
                          command_line_options,
                          options_filename,
                          suffixes=None):
    assert os.path.exists(working_directory)
    if suffixes is None:
        suffixes = [".*"]

    #
    # Write the NL files for the subproblems local to
    # this worker
    #

    filedata = {}
    write_time = {}
    load_function = None
    if subproblem_type == 'bundles':
        assert worker.scenario_tree.contains_bundles()
        load_function = worker._process_bundle_solve_result
        for bundle in worker.scenario_tree.bundles:
            start = time.time()
            filedata[bundle.name] = _write_bundle_nl(
                worker,
                bundle,
                working_directory,
                io_options)
            stop = time.time()
            write_time[bundle.name] = stop - start
    else:
        assert subproblem_type == 'scenarios'
        load_function = worker._process_scenario_solve_result
        for scenario in worker.scenario_tree.scenarios:
            start = time.time()
            filedata[scenario.name] = _write_scenario_nl(
                worker,
                scenario,
                working_directory,
                io_options)
            stop = time.time()
            write_time[scenario.name] = stop - start
    assert load_function is not None
    assert len(filedata) > 0

    args = []
    args.append(problem_list_filename)
    args.append("-AMPL")
    args.append("use_problem_file=yes")
    args.append("option_file_name="+options_filename)
    root_node = worker.scenario_tree.findRootNode()
    if hasattr(worker, "MPI"):
        args.append("mpi_spawn_mode=yes")
        if worker.mpi_comm_tree[root_node.name].rank == 0:
            args.append("output_file="+str(logfile))

    for key, val in command_line_options:
        key = key.strip()
        if key == "use_problem_file":
            raise ValueError(
                "Use of the 'use_problem_file' command-line "
                "option is disallowed.")
        elif key == "mpi_spawn_mode":
            raise ValueError(
                "Use of the 'mpi_spawn_mode' command-line "
                "option is disallowed.")
        elif key == "option_file_name":
            raise ValueError(
                "Use of the 'option_file_name' command-line "
                "option is disallowed.")
        elif key == "output_file":
            raise ValueError(
                "Use of the 'output_file' command-line "
                "option is disallowed.")
        elif key == '-AMPL':
            raise ValueError(
                "Use of the '-AMPL' command-line "
                "option is disallowed.")
        else:
            args.append(key+"="+str(val))

    start = time.time()
    if hasattr(worker, "MPI"):
        currdir = os.getcwd()
        try:
            os.chdir(working_directory)
            spawn = worker.mpi_comm_tree[root_node.name].Spawn(
                executable,
                args=args,
                maxprocs=worker.mpi_comm_tree[root_node.name].size)
            rc = None
            if worker.mpi_comm_tree[root_node.name].rank == 0:
                rc = array.array("i", [0])
                spawn.Reduce(sendbuf=None,
                             recvbuf=[rc, worker.MPI.INT],
                             op=worker.MPI.SUM,
                             root=worker.MPI.ROOT)
            rc = worker.mpi_comm_tree[root_node.name].bcast(rc, root=0)
            spawn.Disconnect()
            assert len(rc) == 1
            rc = rc[0]
        finally:
            os.chdir(currdir)
    else:
        rc, msg = pyutilib.subprocess.run(
            [executable]+args,
            cwd=working_directory,
            outfile=logfile,
            tee=output_solver_log)
    assert rc == 0, str(msg)

    stop = time.time()
    solve_time = stop - start

    #
    # Parse the SOL files for the subproblems local to
    # this worker and load the results
    #
    worker_results = {}
    with ReaderFactory(ResultsFormat.sol) as reader:
        for object_name in filedata:
            start = time.time()
            nl_filename, symbol_map = filedata[object_name]
            assert nl_filename.endswith(".nl")
            sol_filename = nl_filename[:-2]+"sol"
            results = reader(sol_filename, suffixes=suffixes)
            stop = time.time()
            # tag the results object with the symbol_map
            results._smap = symbol_map
            results.solver.time = solve_time
            results.pyomo_solve_time = (stop - start) + \
                                       solve_time + \
                                       write_time[object_name]
            if str(results.solver.termination_condition) == "infeasible":
                if len(results.solution) > 0:
                    results.solution.clear()
            # TODO: Re-architect ScenarioTreeManagerSolver
            #       to better support this
            worker_results[object_name] = \
                load_function(object_name, results)

    return worker_results

def EXTERNAL_collect_solution(worker, node_name):
    solution = {}
    tmp = {}
    node = worker.scenario_tree.get_node(node_name)
    scenario = node.scenarios[0]
    stage = node.stage
    instance = scenario.instance
    assert instance is not None
    bySymbol = instance._ScenarioTreeSymbolMap.bySymbol
    for id_ in node._variable_ids:
        # TODO
        #cost_variable_name, cost_variable_index = \
        #    stage._cost_variable
        #stage_cost_obj = instance.find_component(cost_variable_name)\
        #                 [cost_variable_index]
        #if not stage_cost_obj.is_expression_type():
        #    solution[ComponentUID(stage_cost_obj,cuid_buffer=tmp)] = \
        #        (stage_cost_obj.value, stage_cost_obj.stale)
        for variable_id in node._variable_ids:
            var = bySymbol[variable_id]
            if var.is_expression_type():
                continue
            solution[ComponentUID(var, cuid_buffer=tmp)] = \
                (var.value, var.stale)

    return solution

class SchurIpoptSolver(SPSolverShellCommand, PySPConfiguredObject):

    def __init__(self):
        super(SchurIpoptSolver, self).__init__(
            self.register_options())
        self.set_options_to_default()
        self._executable = "schuripopt"

    def set_options_to_default(self):
        self._options = self.register_options()
        self._options._implicit_declaration = True

    @property
    def options(self):
        return self._options

    @property
    def name(self):
        return "schuripopt"

    def _launch_solver(self,
                       manager,
                       output_directory,
                       logfile,
                       ignore_bundles=False,
                       output_solver_log=False,
                       verbose=False,
                       io_options=None):

        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        problem_list_filename = os.path.join(output_directory,
                                             "PySP_Subproblems.txt")

        scenario_tree = manager.scenario_tree

        subproblem_type = _write_problem_list_file(
            scenario_tree,
            problem_list_filename,
            ignore_bundles=ignore_bundles)

        assert subproblem_type in ('scenarios','bundles')

        options_filename = os.path.join(output_directory,
                                        "schuripopt.opt")
        # just in case output_directory is not a tmpdir, make sure
        # we don't silently overwrite someone's options file
        assert not os.path.exists(options_filename)
        command_line_options = []
        with open(options_filename, "w") as f:
            for key, val in self.options.items():
                key = key.strip()
                if key.startswith("OF_"):
                    if key == "OF_output_file":
                        raise ValueError(
                            "Use of the 'output_file' option "
                            "is disallowed. Use the logfile "
                            "keyword instead.")
                    f.write(key[3:]+" "+str(val)+"\n")
                else:
                    command_line_options.append((key,val))

        if verbose:
            print("Schuripopt solver problem list file: %s"
                  % (problem_list_filename))
            print("Schuripopt solver options file: %s"
                  % (options_filename))
            print("Schuripopt solver problem type: %s"
                  % (subproblem_type))
            print("Sending solver invocation request to "
                  "workers")

        try:
            worker_results = manager.invoke_function(
                "EXTERNAL_invoke_solve",
                thisfile,
                invocation_type=InvocationType.Single,
                function_args=(output_directory,
                               subproblem_type,
                               logfile,
                               problem_list_filename,
                               self.executable,
                               output_solver_log,
                               io_options,
                               command_line_options,
                               options_filename))
        finally:
            if isinstance(manager,
                          (ScenarioTreeManagerClientPyro,
                           ScenarioTreeManagerSolverClientPyro)) and \
                output_solver_log:
                # If this is Pyro the best we can do is
                # dump the log file to the screen after the
                # solve completes because subprocesses are
                # spawned on the Pyro workers.
                if os.path.exists(logfile):
                    with io.open(logfile) as f:
                        print(f.read())

        results = ScenarioTreeSolveResults(subproblem_type)
        for worker_name in worker_results:
            worker_result = worker_results[worker_name]
            for object_name in worker_result:
                results.update(worker_result[object_name])

        return results

    def _solve_impl(self,
                    sp,
                    output_solver_log=False,
                    verbose=False,
                    logfile=None,
                    **kwds):
        """
        Solve a stochastic program with the SchurIpopt solver.

        See the 'solve' method on the base class for
        additional keyword documentation.

        Args:
            sp: The stochastic program to solve.
            output_solver_log (bool): Stream the solver
                output during the solve.
            logfile: The name of the logfile to save the
                solver output into.
            verbose: Report verbose status information to
                aid debugging.
            **kwds: Passed to the DDSIP file writer as I/O
              options (e.g., symbolic_solver_labels=True).

        Returns: A results object with information about the solution.
        """

        #
        # Setup the SchurIpopt working directory
        #
        problem_list_filename = "PySP_Subproblems.txt"
        working_directory = self._create_tempdir("workdir",
                                                 dir=os.getcwd())

        if logfile is None:
            logfile = os.path.join(working_directory,
                                   "schuripopt.log")
            self._add_tempfile("logfile", logfile)
        else:
            self._files["logfile"] = logfile

        if verbose:
            print("Schuripopt solver working directory: %s"
                  % (working_directory))
            print("Schuripopt solver logfile: %s"
                  % (logfile))

        #
        # Launch SchurIpopt from the worker processes
        # (assumed to be launched together using mpirun)
        #
        status = self._launch_solver(
            sp,
            working_directory,
            logfile=logfile,
            output_solver_log=output_solver_log,
            verbose=verbose,
            io_options=kwds)

        objective = 0.0
        solver_status = set()
        solver_message = set()
        termination_condition = set()
        solution_status = set()
        if status.solve_type == "bundles":
            assert sp.scenario_tree.contains_bundles()
            assert len(status.objective) == \
                len(sp.scenario_tree.bundles)
            for bundle in sp.scenario_tree.bundles:
                if objective is not None:
                    if isinstance(status.objective[bundle.name], UndefinedData):
                        objective = None
                    else:
                        objective += bundle.probability * \
                                     status.objective[bundle.name]
                solver_status.add(status.solver_status[bundle.name])
                solver_message.add(status.solver_message[bundle.name])
                termination_condition.add(status.termination_condition[bundle.name])
                if isinstance(status.solution_status[bundle.name], UndefinedData):
                    solution_status.add(None)
                else:
                    solution_status.add(status.solution_status[bundle.name])
        else:
            assert status.solve_type == "scenarios"
            assert len(status.objective) == \
                len(sp.scenario_tree.scenarios)
            for scenario in sp.scenario_tree.scenarios:
                if objective is not None:
                    if isinstance(status.objective[scenario.name], UndefinedData):
                        objective = None
                    else:
                        objective += scenario.probability * \
                                     status.objective[scenario.name]
                solver_status.add(status.solver_status[scenario.name])
                solver_message.add(status.solver_message[scenario.name])
                termination_condition.add(status.termination_condition[scenario.name])
                if isinstance(status.solution_status[scenario.name], UndefinedData):
                    solution_status.add(None)
                else:
                    solution_status.add(status.solution_status[scenario.name])

        assert len(solver_status) == 1
        assert len(solver_message) == 1
        assert len(termination_condition) == 1
        assert len(solution_status) == 1

        results = SPSolverResults()
        results.objective = None
        results.bound = None
        results.status = solution_status.pop()
        results.solver.status = solver_status.pop()
        results.solver.termination_condition = termination_condition.pop()
        results.solver.message = solver_message.pop()
        results.solver.time = max(status.solve_time.values())
        results.solver.pyomo_time = \
            max(status.pyomo_solve_time.values())

        results.xhat = None
        if str(results.solver.status) == "ok" and \
           str(results.solver.termination_condition) == "optimal":
            results.objective = objective
            xhat = results.xhat = {}
            for stage in sp.scenario_tree.stages[:-1]:
                for node in stage.nodes:
                    worker_name = sp.get_worker_for_scenario(
                        node.scenarios[0].name)
                    node_solution = sp.invoke_function_on_worker(
                        worker_name,
                        "EXTERNAL_collect_solution",
                        thisfile,
                        invocation_type=InvocationType.Single,
                        function_args=(node.name,))
                    node_xhat = xhat[node.name] = {}
                    for id_ in node_solution:
                        node_xhat[repr(id_)] = node_solution[id_][0]

        return results

def runschuripopt_register_options(options=None):
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
                                "output_solver_log")
    safe_register_common_option(options,
                                "keep_solver_files")
    safe_register_common_option(options,
                                "symbolic_solver_labels")
    ScenarioTreeManagerFactory.register_options(options)
    SchurIpoptSolver.register_options(options)
    # used to populate the implicit SchurIpopt options
    safe_register_unique_option(
        options,
        "solver_options",
        PySPConfigValue(
            (),
            domain=_domain_tuple_of_str_or_dict,
            description=(
                "Solver options to pass to SchurIpopt "
                "(e.g., relax_integrality=yes). This "
                "option can be used multiple times "
                "from the command line to specify more "
                "than one SchurIpopt option."
            ),
            doc=None,
            visibility=0),
        ap_kwds={'action': 'append'},
        ap_group=_schuripopt_group_label)

    return options

def runschuripopt(options):
    """
    Construct a senario tree manager and solve it
    with the SD solver.
    """
    start_time = time.time()
    with ScenarioTreeManagerFactory(options) as manager:
        manager.initialize()
        print("")
        print("Running SchurIpopt solver for stochastic "
              "programming problems")
        schuripopt = SchurIpoptSolver()
        # add the implicit schuripopt options
        solver_options = options.solver_options
        if len(solver_options) > 0:
            if type(solver_options) is tuple:
                for name_val in solver_options:
                    assert "=" in name_val
                    name, val = name_val.split("=")
                    schuripopt.options[name.strip()] = val.strip()
            else:
                for key, val in solver_options.items():
                    schuripopt.options[key] = val
        results = schuripopt.solve(
            manager,
            output_solver_log=options.output_solver_log,
            keep_solver_files=options.keep_solver_files,
            symbolic_solver_labels=options.symbolic_solver_labels)
        xhat = results.xhat
        del results.xhat
        print("")
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
            runschuripopt_register_options,
            prog='runschuripopt',
            description=(
"""Optimize a stochastic program using the SchurIpopt solver."""
            ))

    except SystemExit as _exc:
        # the parser throws a system exit if "-h" is specified
        # - catch it to exit gracefully.
        return _exc.code

    return launch_command(runschuripopt,
                          options,
                          error_label="runschuripopt: ",
                          disable_gc=options.disable_gc,
                          profile_count=options.profile,
                          traceback=options.traceback)

SPSolverFactory.register_solver("schuripopt", SchurIpoptSolver)

if __name__ == "__main__":
    sys.exit(main())
