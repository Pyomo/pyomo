#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import time
import copy

from pyomo.common import pyomo_command
from pyomo.common.dependencies import yaml
from pyomo.pysp.util.config import (PySPConfigValue,
                                    PySPConfigBlock,
                                    safe_register_common_option,
                                    safe_register_unique_option,
                                    _extension_options_group_title,
                                    _domain_must_be_str)
from pyomo.pysp.util.misc import (parse_command_line,
                                  launch_command,
                                  sort_extensions_by_precedence)
from pyomo.pysp.scenariotree.manager import \
    ScenarioTreeManagerFactory
from pyomo.pysp.scenariotree.manager_solver import \
    ScenarioTreeManagerSolverFactory
from pyomo.pysp.solutionioextensions import \
    (IPySPSolutionSaverExtension,
     IPySPSolutionLoaderExtension)

#
# Fix all non-anticiptative variables to their current solution,
# solve, free all variables that weren't already fixed, and
# return the extensive form objective value
#
def evaluate_current_node_solution(sp, sp_solver, **solve_kwds):

    scenario_tree = sp.scenario_tree

    # Save the current fixed state and fix queue, then clear the fix queue
    fixed = {}
    fix_queue = {}
    for tree_node in scenario_tree.nodes:
        fixed[tree_node.name] = copy.deepcopy(tree_node._fixed)
        fix_queue[tree_node.name] = copy.deepcopy(tree_node._fix_queue)
        tree_node.clear_fix_queue()

    # Fix all non-anticipative variables to their
    # current value in the node solution
    for stage in scenario_tree.stages[:-1]:
        for tree_node in stage.nodes:
            for variable_id in tree_node._standard_variable_ids:
                if variable_id in tree_node._solution:
                    tree_node.fix_variable(variable_id,
                                           tree_node._solution[variable_id])
                else:
                    from pyomo.pysp.phutils import indexToString
                    name, index = tree_node._variable_ids[variable_id]
                    raise ValueError(
                        "Scenario tree variable with name %s (scenario_tree_id=%s) "
                        "does not have a solution stored on scenario tree node %s. "
                        "Unable to evaluate solution." % (name+indexToString(index),
                                                          variable_id,
                                                          tree_node.name))

    # Push fixed variable statuses on instances (or
    # transmit to the phsolverservers)
    sp.push_fix_queue_to_instances()

    failures = sp_solver.solve_subproblems(**solve_kwds)

    # Free all non-anticipative variables
    for stage in scenario_tree._stages[:-1]:
        for tree_node in stage.nodes:
            for variable_id in tree_node._standard_variable_ids:
                tree_node.free_variable(variable_id)

    # Refix all previously fixed variables
    for tree_node in scenario_tree.nodes:
        node_fixed = fixed[tree_node.name]
        for variable_id in node_fixed:
            tree_node.fix_variable(variable_id, node_fixed[variable_id])

    sp.push_fix_queue_to_instances()

    # Restore the fix_queue
    for tree_node in scenario_tree.nodes:
        tree_node._fix_queue.update(fix_queue[tree_node.name])

    return failures

def run_evaluate_xhat_register_options(options=None):
    if options is None:
        options = PySPConfigBlock()
    safe_register_common_option(options,
                               "disable_gc")
    safe_register_common_option(options,
                               "profile")
    safe_register_common_option(options,
                               "traceback")
    safe_register_common_option(options,
                               "scenario_tree_manager")
    safe_register_common_option(options,
                               "output_scenario_tree_solution")
    safe_register_common_option(options,
                               "solution_saver_extension")
    safe_register_common_option(options,
                               "solution_loader_extension")
    safe_register_unique_option(
        options,
        "disable_solution_loader_check",
        PySPConfigValue(
            False,
            domain=bool,
            description=(
                "Indicates that no solution loader extension is required to "
                "run this script, e.g., because the scenario tree manager "
                "is somehow pre-populated with a solution."
            ),
            doc=None,
            visibility=0),
        ap_group=_extension_options_group_title)
    safe_register_unique_option(
        options,
        "output_scenario_costs",
        PySPConfigValue(
            None,
            domain=_domain_must_be_str,
            description=(
                "A file name where individual scenario costs from the solution "
                "will be stored. The format is determined from the extension used "
                "in the filename. Recognized extensions: [.csv, .json, .yaml]"
            ),
            doc=None,
            visibility=0))
    ScenarioTreeManagerFactory.register_options(options)
    ScenarioTreeManagerSolverFactory.register_options(options,
                                                      options_prefix="subproblem_")

    return options

#
# Convert a PySP scenario tree formulation to SMPS input files
#

def run_evaluate_xhat(options,
                      solution_loaders=(),
                      solution_savers=()):

    start_time = time.time()
    import pyomo.environ
    solution_loaders = sort_extensions_by_precedence(solution_loaders)
    solution_savers = sort_extensions_by_precedence(solution_savers)

    with ScenarioTreeManagerFactory(options) as sp:
        sp.initialize()

        loaded = False
        for plugin in solution_loaders:
            ret = plugin.load(sp)
            if not ret:
                print("WARNING: Loader extension %s call did not return True. "
                      "This might indicate failure to load data." % (plugin))
            else:
                loaded = True

        if (not loaded) and (not options.disable_solution_loader_check):
            raise RuntimeError(
                "Either no solution loader extensions were provided or "
                "all solution loader extensions reported a bad return value. "
                "To disable this check use the disable_solution_loader_check "
                "option flag.")

        with ScenarioTreeManagerSolverFactory(sp, options, options_prefix="subproblem_") as sp_solver:
            evaluate_current_node_solution(sp, sp_solver)

        objective = sum(scenario.probability * \
                        scenario.get_current_objective()
                        for scenario in sp.scenario_tree.scenarios)
        sp.scenario_tree.snapshotSolutionFromScenarios()

        print("")
        print("***********************************************"
              "************************************************")
        print(">>>THE EXPECTED SUM OF THE STAGE COST VARIABLES="
              +str(sp.scenario_tree.findRootNode().\
                   computeExpectedNodeCost())+"<<<")
        print("***********************************************"
              "************************************************")

        # handle output of solution from the scenario tree.
        print("")
        print("Extensive form solution:")
        sp.scenario_tree.pprintSolution()
        print("")
        print("Extensive form costs:")
        sp.scenario_tree.pprintCosts()

        if options.output_scenario_tree_solution:
            print("Final solution (scenario tree format):")
            sp.scenario_tree.pprintSolution()

        if options.output_scenario_costs is not None:
            if options.output_scenario_costs.endswith('.json'):
                import json
                result = {}
                for scenario in sp.scenario_tree.scenarios:
                    result[str(scenario.name)] = scenario._cost
                with open(options.output_scenario_costs, 'w') as f:
                    json.dump(result, f, indent=2, sort_keys=True)
            elif options.output_scenario_costs.endswith('.yaml'):
                result = {}
                for scenario in sp.scenario_tree.scenarios:
                    result[str(scenario.name)] = scenario._cost
                with open(options.output_scenario_costs, 'w') as f:
                    yaml.dump(result, f)
            else:
                if not options.output_scenario_costs.endswith('.csv'):
                    print("Unrecognized file extension. Using CSV format "
                          "to store scenario costs")
                with open(options.output_scenario_costs, 'w') as f:
                    for scenario in sp.scenario_tree.scenarios:
                        f.write("%s,%r\n" % (scenario.name, scenario._cost))

        for plugin in solution_savers:
            if not plugin.save(sp):
                print("WARNING: Saver extension %s call did not return True. "
                      "This might indicate failure to save data." % (plugin))

    print("")
    print("Total execution time=%.2f seconds"
          % (time.time() - start_time))

    return 0

#
# the main driver routine for the evaluate_xhat script.
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
        options, extensions = parse_command_line(
            args,
            run_evaluate_xhat_register_options,
            with_extensions={'solution_loader_extension':
                             IPySPSolutionLoaderExtension,
                             'solution_saver_extension':
                             IPySPSolutionSaverExtension},
            prog='evaluate_xhat',
            description=(
"""Evaluate a non-anticipative solution over the given
scenario tree.  A solution is provided by specifying one or
more plugins implementing the IPySPSolutionLoaderExtension. E.g.,

evaluate_xhat -m ReferenceModel.py -s ScenarioStructure.dat \\
              --solution-loader-extension=pyomo.pysp.plugins.jsonio \\
              --jsonloader-input-name xhat.json

To include plugin specific options in the list of options
output after this message, declare them on the command-line
before the --help flag. E.g.,

evaluate_xhat --solution-loader-extension=pyomo.pysp.plugins.jsonio \\
              --help

This script will fix all non-derived, non-leaf stage
variables to their values specified in the loaded
solution. All other values are from the solution are
ignored."""
            ))

    except SystemExit as _exc:
        # the parser throws a system exit if "-h" is specified
        # - catch it to exit gracefully.
        return _exc.code

    return launch_command(run_evaluate_xhat,
                          options,
                          cmd_kwds={'solution_loaders':
                                    extensions['solution_loader_extension'],
                                    'solution_savers':
                                    extensions['solution_saver_extension']},
                          error_label="evaluate_xhat: ",
                          disable_gc=options.disable_gc,
                          profile_count=options.profile,
                          traceback=options.traceback)

@pyomo_command('evaluate_xhat', 'Evaluate a non-anticipative solution on a scenario tree.')
def EvaluateXhat_main(args=None):
    return main(args=args)
