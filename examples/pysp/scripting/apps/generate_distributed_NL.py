#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import os
import sys
import time
import argparse

from pyomo.common import pyomo_command
from pyomo.pysp.util.config import (PySPConfigValue,
                                    PySPConfigBlock,
                                    safe_register_common_option,
                                    safe_register_unique_option,
                                    _domain_must_be_str)
from pyomo.pysp.util.misc import (parse_command_line,
                                  launch_command)
from pyomo.pysp.scenariotree.manager import (ScenarioTreeManagerClientSerial,
                                             ScenarioTreeManagerClientPyro,
                                             InvocationType)
from pyomo.core import Suffix, Block

# generate an absolute path to this file
thisfile = os.path.abspath(__file__)

def _write_bundle_NL(worker,
                     bundle,
                     output_directory,
                     linking_suffix_name,
                     objective_suffix_name,
                     symbolic_solver_labels):

    assert os.path.exists(output_directory)

    bundle_instance = worker._bundle_binding_instance_map[bundle.name]
    assert not hasattr(bundle_instance, ".tmpblock")
    tmpblock = Block(concrete=True)
    bundle_instance.add_component(".tmpblock", tmpblock)

    #
    # linking variable suffix
    #
    tmpblock.add_component(linking_suffix_name,
                           Suffix(direction=Suffix.EXPORT))
    linking_suffix = getattr(tmpblock, linking_suffix_name)

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
    tmpblock.add_component(objective_suffix_name,
                           Suffix(direction=Suffix.EXPORT))

    getattr(tmpblock, objective_suffix_name)[bundle_instance] = \
        bundle._probability

    output_filename = os.path.join(output_directory, str(bundle.name)+".nl")
    bundle_instance.write(
        output_filename,
        io_options={'symbolic_solver_labels': symbolic_solver_labels})

    bundle_instance.del_component(tmpblock)

def _write_scenario_NL(worker,
                       scenario,
                       output_directory,
                       linking_suffix_name,
                       objective_suffix_name,
                       symbolic_solver_labels):

    assert os.path.exists(output_directory)
    instance = scenario._instance
    assert not hasattr(instance, ".tmpblock")
    tmpblock = Block(concrete=True)
    instance.add_component(".tmpblock", tmpblock)

    #
    # linking variable suffix
    #
    bySymbol = instance._ScenarioTreeSymbolMap.bySymbol
    tmpblock.add_component(linking_suffix_name,
                           Suffix(direction=Suffix.EXPORT))
    linking_suffix = getattr(tmpblock, linking_suffix_name)

    # Loop over all nodes for the scenario except the leaf node,
    # which has no blended variables
    for node in scenario._node_list[:-1]:
        for variable_id in node._standard_variable_ids:
            linking_suffix[bySymbol[variable_id]] = variable_id

    #
    # objective weight suffix
    #
    tmpblock.add_component(objective_suffix_name,
                           Suffix(direction=Suffix.EXPORT))
    getattr(tmpblock, objective_suffix_name)[instance] = \
        scenario._probability

    output_filename = os.path.join(output_directory,
                                   str(scenario.name)+".nl")
    instance.write(
        output_filename,
        io_options={'symbolic_solver_labels': symbolic_solver_labels})

    instance.del_component(tmpblock)

def write_distributed_NL_files(manager,
                               output_directory,
                               linking_suffix_name,
                               objective_suffix_name,
                               symbolic_solver_labels=False):

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    scenario_tree = manager.scenario_tree
    #
    # Write list of subproblems to file
    #
    with open(os.path.join(output_directory,
                           "PySP_Subproblems.txt"),'w') as f:

        if scenario_tree.contains_bundles():
            for bundle in scenario_tree.bundles:
                f.write(str(bundle.name)+".nl\n")
        else:
            for scenario in scenario_tree.scenarios:
                f.write(str(scenario.name)+".nl\n")

    if scenario_tree.contains_bundles():
        print("Executing bundle NL-file conversions")
        manager.invoke_function(
            "_write_bundle_NL",
            thisfile,
            invocation_type=InvocationType.PerBundle,
            function_args=(output_directory,
                           linking_suffix_name,
                           objective_suffix_name,
                           symbolic_solver_labels))

    else:
        print("Executing scenario NL-file conversions")
        manager.invoke_function(
            "_write_scenario_NL",
            thisfile,
            invocation_type=InvocationType.PerScenario,
            function_args=(output_directory,
                           linking_suffix_name,
                           objective_suffix_name,
                           symbolic_solver_labels))

def run_generate_distributed_NL_register_options(options=None):
    if options is None:
        options = PySPConfigBlock()
    safe_register_common_option(options, "disable_gc")
    safe_register_common_option(options, "profile")
    safe_register_common_option(options, "traceback")
    safe_register_common_option(options, "scenario_tree_manager")
    safe_register_common_option(options, "symbolic_solver_labels")
    safe_register_unique_option(
        options,
        "output_directory",
        PySPConfigValue(
            ".",
            domain=_domain_must_be_str,
            description=(
                "The directory in which to store all output files. "
                "Default is '.'."
            ),
            doc=None,
            visibility=0))
    safe_register_unique_option(
        options,
        "linking_suffix_name",
        PySPConfigValue(
            "variable_id",
            domain=_domain_must_be_str,
            description=(
                "The suffix name used to identify common variables "
                "across NL files. Default is 'ipopt_blend_id'."
            ),
            doc=None,
            visibility=0))
    safe_register_unique_option(
        options,
        "objective_suffix_name",
        PySPConfigValue(
            "objective_weight",
            domain=_domain_must_be_str,
            description=(
                "The suffix name used to identify the relative "
                "objective weight for each NL-file subproblem."
                "Default is 'ipopt_blend_weight'."
            ),
            doc=None,
            visibility=0))
    ScenarioTreeManagerClientSerial.register_options(options)
    ScenarioTreeManagerClientPyro.register_options(options)

    return options
#
# Convert a PySP scenario tree formulation to SMPS input files
#

def run_generate_distributed_NL(options):
    import pyomo.environ

    start_time = time.time()

    manager_class = None
    if options.scenario_tree_manager == 'serial':
        manager_class = ScenarioTreeManagerClientSerial
    elif options.scenario_tree_manager == 'pyro':
        manager_class = ScenarioTreeManagerClientPyro

    with manager_class(options) \
         as manager:
        manager.initialize()
        write_distributed_NL_files(
            manager,
            options.output_directory,
            options.linking_suffix_name,
            options.objective_suffix_name,
            symbolic_solver_labels=options.symbolic_solver_labels)

    print("")
    print("Total execution time=%.2f seconds"
          % (time.time() - start_time))

    return 0

#
# the main driver routine for the generate_distributed_NL script.
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
            run_generate_distributed_NL_register_options,
            prog='generate_distributed_NL',
            description=(
"""Converts a scenario tree into multiple NL files with linking
information specified with suffixes."""
            ))

    except SystemExit as _exc:
        # the parser throws a system exit if "-h" is specified
        # - catch it to exit gracefully.
        return _exc.code

    return launch_command(run_generate_distributed_NL,
                          options,
                          error_label="generate_distributed_NL: ",
                          disable_gc=options.disable_gc,
                          profile_count=options.profile,
                          traceback=options.traceback)

def generate_distributed_NL_main(args=None):
    return main(args=args)

if __name__ == "__main__":
    sys.exit(main())
