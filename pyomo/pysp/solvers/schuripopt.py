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

import os
import sys
import time
import array

import pyutilib.subprocess
import pyutilib.services

from pyomo.core import (SymbolMap,
                        Block,
                        Suffix,
                        ComponentMap,
                        ComponentUID)
from pyomo.opt import (ReaderFactory,
                       ResultsFormat,
                       ProblemFormat)
from pyomo.pysp.util.configured_object import PySPConfiguredObject
from pyomo.pysp.util.config import (PySPConfigValue,
                                    PySPConfigBlock,
                                    safe_register_common_option,
                                    safe_register_unique_option,
                                    safe_declare_common_option,
                                    safe_declare_unique_option,
                                    _domain_must_be_str,
                                    _domain_tuple_of_str_or_dict)
from pyomo.pysp.util.misc import (parse_command_line,
                                  launch_command)
from pyomo.pysp.scenariotree.manager import InvocationType
from pyomo.pysp.scenariotree.manager_solver import \
    (ScenarioTreeManagerSolver,
     ScenarioTreeManagerSolverResults,
     ScenarioTreeManagerFactory)
from pyomo.pysp.phutils import indexToString
from pyomo.pysp.solvers.spsolver import (SPSolverResults,
                                         SPSolverFactory)
from pyomo.pysp.solvers.spsolvershellcommand import \
    SPSolverShellCommand

from six.moves import xrange
# use fast version of pickle (python 2 or 3)
from six.moves import cPickle as pickle

_mpi4py_available = False
try:
    import mpi4py
    _mpi4py_available = True
except:
    _mpi4py_available = False

# generate an absolute path to this file
thisfile = os.path.abspath(__file__)
_objective_weight_suffix_name = "schurip_objective_weight"
_variable_id_suffix_name = "schurip_variable_id"
_schuripopt_group_label = "SchurIpoptSolver Options"

def _write_bundle_nl(worker,
                     bundle,
                     output_directory,
                     io_options):

    assert os.path.exists(output_directory)

    bundle_instance = worker._bundle_binding_instance_map[bundle.name]
    assert not hasattr(bundle_instance, ".schuripopt")
    tmpblock = Block(concrete=True)
    bundle_instance.add_component(".schuripopt", tmpblock)

    #
    # linking variable suffix
    #
    tmpblock.add_component(_variable_id_suffix_name,
                           Suffix(direction=Suffix.EXPORT,
                                  datatype=Suffix.INT))
    linking_suffix = getattr(tmpblock, _variable_id_suffix_name)

    # Loop over all nodes for the bundle except the leaf nodes,
    # which have no blended variables
    scenario_tree = worker.scenario_tree
    for stage in bundle.scenario_tree.stages[:-1]:
        for _node in stage.nodes:
            # get the node of off the real scenario tree
            # as this has the linked variable information
            node = scenario_tree.get_node(_node.name)
            master_variable = bundle_instance.find_component(
                "MASTER_BLEND_VAR_"+str(node.name))
            for variable_id in node._standard_variable_ids:
                linking_suffix[master_variable[variable_id]] = variable_id

    #
    # objective weight suffix
    #
    tmpblock.add_component(_objective_weight_suffix_name,
                           Suffix(direction=Suffix.EXPORT))
    getattr(tmpblock, _objective_weight_suffix_name)[bundle_instance] = \
        bundle.probability

    # take care to disable any advanced preprocessing flags since we
    # are not going through the scenario tree manager solver interface
    # TODO: resolve this preprocessing mess
    block_attrs = []
    for block in bundle_instance.block_data_objects(active=True):
        attrs = []
        for attr_name in ("_gen_obj_ampl_repn",
                          "_gen_con_ampl_repn"):
            if hasattr(block, attr_name):
                attrs.append((attr_name, getattr(block, attr_name)))
                setattr(block, attr_name, True)
        if len(attrs):
            block_attrs.append((block, attrs))

    output_filename = os.path.join(output_directory,
                                   str(bundle.name)+".nl")
    # write the model and obtain the symbol_map
    _, smap_id = bundle_instance.write(
        output_filename,
        format=ProblemFormat.nl,
        io_options=io_options)
    symbol_map = bundle_instance.solutions.symbol_map[smap_id]

    # reset preprocessing flags
    # TODO: resolve this preprocessing mess
    for block, attrs in block_attrs:
        for attr_name, attr_val in attrs:
            setattr(block, attr_name, attr_val)

    bundle_instance.del_component(tmpblock)

    return output_filename, symbol_map

def _write_scenario_nl(worker,
                       scenario,
                       output_directory,
                       io_options):

    assert os.path.exists(output_directory)
    instance = scenario._instance
    assert not hasattr(instance, ".schuripopt")
    tmpblock = Block(concrete=True)
    instance.add_component(".schuripopt", tmpblock)

    #
    # linking variable suffix
    #
    bySymbol = instance._ScenarioTreeSymbolMap.bySymbol
    tmpblock.add_component(_variable_id_suffix_name,
                           Suffix(direction=Suffix.EXPORT,
                                  datatype=Suffix.INT))
    linking_suffix = getattr(tmpblock, _variable_id_suffix_name)

    # Loop over all nodes for the scenario except the leaf node,
    # which has no blended variables
    for node in scenario._node_list[:-1]:
        for variable_id in node._standard_variable_ids:
            linking_suffix[bySymbol[variable_id]] = variable_id

    #
    # objective weight suffix
    #
    tmpblock.add_component(_objective_weight_suffix_name,
                           Suffix(direction=Suffix.EXPORT))
    getattr(tmpblock, _objective_weight_suffix_name)[instance] = \
        scenario.probability

    # take care to disable any advanced preprocessing flags since we
    # are not going through the scenario tree manager solver interface
    # TODO: resolve this preprocessing mess
    block_attrs = []
    for block in instance.block_data_objects(active=True):
        attrs = []
        for attr_name in ("_gen_obj_ampl_repn",
                          "_gen_con_ampl_repn"):
            if hasattr(block, attr_name):
                attrs.append((attr_name, getattr(block, attr_name)))
                setattr(block, attr_name, True)
        if len(attrs):
            block_attrs.append((block, attrs))

    output_filename = os.path.join(output_directory,
                                   str(scenario.name)+".nl")

    # write the model and obtain the symbol_map
    _, smap_id = instance.write(
        output_filename,
        format=ProblemFormat.nl,
        io_options=io_options)
    symbol_map = instance.solutions.symbol_map[smap_id]

    # reset preprocessing flags
    # TODO: resolve this preprocessing mess
    for block, attrs in block_attrs:
        for attr_name, attr_val in attrs:
            setattr(block, attr_name, attr_val)

    instance.del_component(tmpblock)

    return output_filename, symbol_map

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
    import mpi4py.MPI
    assert _mpi4py_available
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
    args.append("use_problem_file=yes")
    args.append("mpi_spawn_mode=yes")
    args.append("option_file_name="+options_filename)
    if mpi4py.MPI.COMM_WORLD.rank == 0:
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
    args.append("-AMPL")

    # TODO
    print("Command: %s" % (' '.join([executable]+args)))
    start = time.time()
    spawn = mpi4py.MPI.COMM_WORLD.Spawn(
        executable,
        args=args,
        maxprocs=mpi4py.MPI.COMM_WORLD.size)
    rc = None
    if mpi4py.MPI.COMM_WORLD.rank == 0:
        rc = array.array("i", [0])
        spawn.Reduce(sendbuf=None,
                     recvbuf=[rc, mpi4py.MPI.INT],
                     op=mpi4py.MPI.SUM,
                     root=mpi4py.MPI.ROOT)
    rc = mpi4py.MPI.COMM_WORLD.bcast(rc, root=0)
    spawn.Disconnect()
    #rc, log = pyutilib.subprocess.run(
    #    [executable]+args,
    #    cwd=working_directory,
    #    outfile=logfile,
    #    tee=output_solver_log)
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
            # TODO: Re-architect ScenarioTreeManagerSolver
            #       to better support this
            worker_results[object_name] = \
                load_function(object_name, None, results)

    return worker_results

class SchurIpoptSolver(SPSolverShellCommand, PySPConfiguredObject):

    @classmethod
    def _declare_options(cls, options=None):
        if options is None:
            options = PySPConfigBlock()
        safe_declare_common_option(options,
                                   "verbose",
                                   ap_group=_schuripopt_group_label)
        safe_declare_unique_option(
            options,
            "executable",
            PySPConfigValue(
                "schuripopt",
                domain=_domain_must_be_str,
                description=(
                    "Name of the executable used when launching the "
                    "SchurIpopt solver. The default is 'schuripopt'. "
                    "This option can be set to an absolute or relative path."
                    " Otherwise, it is assumed that the named executable "
                    "will be found in the shell's search path."
                ),
                doc=None,
                visibility=0),
            ap_group=_schuripopt_group_label)

        return options

    def __init__(self, *args, **kwds):
        super(SchurIpoptSolver, self).__init__(*args, **kwds)
        self._name = "schuripopt"
        self._executable = "schuripopt"
        if not _mpi4py_available:
            raise RuntimeError(
                "The 'mpi4py' module is not available, but it "
                "is required by the %s solver" % (self.name))

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

        #
        # Write list of subproblems to file
        #
        subproblem_type = None
        with open(problem_list_filename, 'w') as f:
            if (not ignore_bundles) and scenario_tree.contains_bundles():
                subproblem_type = "bundles"
                for bundle in scenario_tree.bundles:
                    f.write(os.path.join(output_directory,
                                         str(bundle.name)+".nl"))
                    f.write("\n")
            else:
                subproblem_type = "scenarios"
                for scenario in scenario_tree.scenarios:
                    f.write(os.path.join(output_directory,
                                         str(scenario.name)+".nl"))
                    f.write("\n")

        assert subproblem_type is not None

        options_filename = os.path.join(output_directory, "schuripopt.opt")
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

        results = ScenarioTreeManagerSolverResults(subproblem_type)
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

        #
        # Setup the SchurIpopt working directory
        #

        problem_list_filename = "PySP_Subproblems.txt"
        working_directory = self._create_tempdir("workdir")
        logfile = self._files["logfile"] = \
            logfile if (logfile is not None) else \
            os.path.join(working_directory,
                         "schuripopt.log")
        if verbose:
            print("Schuripopt solver working directory: %s"
                  % (working_directory))
            print("Schuripopt solver logfile: %s"
                  % (logfile))

        #
        # Launch SchurIpopt from the worker processes
        # (assumed to be launched together using mpirun)
        #
        solve_results = self._launch_solver(
            sp,
            working_directory,
            logfile=logfile,
            output_solver_log=output_solver_log,
            verbose=verbose,
            io_options=kwds)

        objective = 0.0
        if solve_results.solve_type == "bundles":
            assert sp.scenario_tree.contains_bundles()
            assert len(solve_results.objective) == \
                len(sp.scenario_tree.bundles)
            for bundle in sp.scenario_tree.bundles:
                objective += bundle.probability * \
                             solve_results.objective[bundle.name]
        else:
            assert solve_results.solve_type == "scenarios"
            assert len(solve_results.objective) == \
                len(sp.scenario_tree.scenarios)
            for scenario in sp.scenario_tree.scenarios:
                objective += scenario.probability * \
                             solve_results.objective[scenario.name]

        results = SPSolverResults()
        results.objective = objective
        results.solver_time = max(solve_results.solve_time.values())
        results.pyomo_solve_time = \
            max(solve_results.pyomo_solve_time.values())
        # TODO
        results.xhat = dict(sp.scenario_tree.scenarios[0]._x)

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
                                "keep_solver_files")
    ScenarioTreeManagerFactory.register_options(options)
    SchurIpoptSolver.register_options(options)

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
        schuripopt = SchurIpoptSolver(options)
        results = schuripopt.solve(
            manager,
            keep_solver_files=options.keep_solver_files,
            output_solver_log=True)
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
