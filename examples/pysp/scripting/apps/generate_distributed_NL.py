#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

import os
import sys
import time
import argparse

from pyutilib.pyro import shutdown_pyro_components

from pyomo.util import pyomo_command
from pyomo.pysp.util.config import (PySPConfigValue,
                                    PySPConfigBlock,
                                    safe_declare_common_option,
                                    safe_declare_unique_option,
                                    _domain_must_be_str)
from pyomo.pysp.util.misc import (parse_command_line,
                                  launch_command)
from pyomo.pysp.scenariotree.scenariotreemanager import (ScenarioTreeManagerSerial,
                                                         ScenarioTreeManagerSPPyro)
from pyomo.pysp.scenariotree.scenariotreeserverutils import InvocationType
from pyomo.core import Suffix

# generate an absolute path to this file
thisfile = os.path.abspath(__file__)

def EXTERNAL_write_bundle_NL(manager,
                             scenario_tree,
                             bundle,
                             output_directory,
                             linking_suffix_name,
                             objective_suffix_name):

    from pyomo.repn.plugins.ampl import ProblemWriter_nl
    assert os.path.exists(output_directory)

    bundle_instance = manager._bundle_binding_instance_map[bundle.name]

    #
    # linking variable suffix
    #
    bundle_instance.del_component(linking_suffix_name)
    bundle_instance.add_component(linking_suffix_name,
                                  Suffix(direction=Suffix.EXPORT))
    linking_suffix = getattr(bundle_instance, linking_suffix_name)

    # Loop over all nodes for the bundle except the leaf nodes,
    # which have no blended variables
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
    bundle_instance.del_component(objective_suffix_name)
    bundle_instance.add_component(objective_suffix_name,
                                  Suffix(direction=Suffix.EXPORT))

    getattr(bundle_instance, objective_suffix_name)[bundle_instance] = \
        bundle._probability

    output_filename = os.path.join(output_directory, str(bundle.name)+".nl")
    with ProblemWriter_nl() as nl_writer:
        nl_writer(bundle_instance,
                  output_filename,
                  lambda x: True,
                  {})

    bundle_instance.del_component(linking_suffix_name)
    bundle_instance.del_component(objective_suffix_name)

def EXTERNAL_write_scenario_NL(manager,
                               scenario_tree,
                               scenario,
                               output_directory,
                               linking_suffix_name,
                               objective_suffix_name):

    from pyomo.repn.plugins.ampl import ProblemWriter_nl
    assert os.path.exists(output_directory)
    instance = scenario._instance

    #
    # linking variable suffix
    #
    bySymbol = instance._ScenarioTreeSymbolMap.bySymbol
    instance.del_component(linking_suffix_name)
    instance.add_component(linking_suffix_name,
                           Suffix(direction=Suffix.EXPORT))
    linking_suffix = getattr(instance, linking_suffix_name)

    # Loop over all nodes for the scenario except the leaf node,
    # which has no blended variables
    for node in scenario._node_list[:-1]:
        for variable_id in node._standard_variable_ids:
            linking_suffix[bySymbol[variable_id]] = variable_id

    #
    # objective weight suffix
    #
    instance.del_component(objective_suffix_name)
    instance.add_component(objective_suffix_name,
                           Suffix(direction=Suffix.EXPORT))
    getattr(instance, objective_suffix_name)[instance] = \
        scenario._probability

    output_filename = os.path.join(output_directory,
                                   str(scenario.name)+".nl")
    with ProblemWriter_nl() as nl_writer:
        nl_writer(instance,
                  output_filename,
                  lambda x: True,
                  {})

    instance.del_component(linking_suffix_name)
    instance.del_component(objective_suffix_name)

def write_distributed_NL_files(manager,
                               output_directory,
                               linking_suffix_name,
                               objective_suffix_name):

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
        manager.invoke_external_function(
            thisfile,
            "EXTERNAL_write_bundle_NL",
            invocation_type=InvocationType.PerBundle,
            function_args=(output_directory,
                           linking_suffix_name,
                           objective_suffix_name))

    else:
        print("Executing scenario NL-file conversions")
        manager.invoke_external_function(
            thisfile,
            "EXTERNAL_write_scenario_NL",
            invocation_type=InvocationType.PerScenario,
            function_args=(output_directory,
                           linking_suffix_name,
                           objective_suffix_name))

def run_generate_distributed_NL_register_options(options=None):
    if options is None:
        options = PySPConfigBlock()
    safe_declare_common_option(options, "disable_gc")
    safe_declare_common_option(options, "profile")
    safe_declare_common_option(options, "traceback")
    safe_declare_common_option(options, "scenario_tree_manager")
    safe_declare_unique_option(
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
    safe_declare_unique_option(
        options,
        "linking_suffix_name",
        PySPConfigValue(
            "ipopt_blend_id",
            domain=_domain_must_be_str,
            description=(
                "The suffix name used to identify common variables "
                "across NL files. Default is 'ipopt_blend_id'."
            ),
            doc=None,
            visibility=0))
    safe_declare_unique_option(
        options,
        "objective_suffix_name",
        PySPConfigValue(
            "ipopt_blend_weight",
            domain=_domain_must_be_str,
            description=(
                "The suffix name used to identify the relative "
                "objective weight for each NL-file subproblem."
                "Default is 'ipopt_blend_weight'."
            ),
            doc=None,
            visibility=0))
    ScenarioTreeManagerSerial.register_options(options)
    ScenarioTreeManagerSPPyro.register_options(options)

    return options
#
# Convert a PySP scenario tree formulation to SMPS input files
#

def run_generate_distributed_NL(options):
    import pyomo.environ

    start_time = time.time()

    try:

        ScenarioTreeManager_class = None
        if options.scenario_tree_manager == 'serial':
            ScenarioTreeManager_class = ScenarioTreeManagerSerial
        elif options.scenario_tree_manager == 'sppyro':
            ScenarioTreeManager_class = ScenarioTreeManagerSPPyro

        with ScenarioTreeManager_class(options) \
             as manager:
            manager.initialize()
            write_distributed_NL_files(manager,
                                       options.output_directory,
                                       options.linking_suffix_name,
                                       options.objective_suffix_name)

    finally:
        # shutdown-pyro components if requested
        if options.scenario_tree_manager == "sppyro":
            if options.shutdown_pyro:
                print("\n")
                print("Shutting down Pyro solver components.")
                shutdown_pyro_components(num_retries=0)

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
    main(args=sys.argv[1:])
