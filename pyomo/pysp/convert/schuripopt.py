#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

import os
import time
import sys
import logging

from pyomo.opt import ProblemFormat
from pyomo.core import (Block,
                        Suffix)
from pyomo.pysp.scenariotree.manager import InvocationType
from pyomo.pysp.util.config import (PySPConfigValue,
                                    PySPConfigBlock,
                                    safe_register_common_option,
                                    safe_register_unique_option,
                                    _domain_must_be_str)
from pyomo.pysp.scenariotree.manager import \
    (ScenarioTreeManagerClientSerial,
     ScenarioTreeManagerClientPyro)
from pyomo.pysp.scenariotree.util import \
    scenario_tree_id_to_pint32
from pyomo.pysp.util.misc import (parse_command_line,
                                  launch_command)

thisfile = os.path.abspath(__file__)

logger = logging.getLogger('pyomo.pysp')

_objective_weight_suffix_name = "schurip_objective_weight"
_variable_id_suffix_name = "schurip_variable_id"

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
            node_name = node.name
            master_variable = bundle_instance.find_component(
                "MASTER_BLEND_VAR_"+str(node.name))
            for variable_id in node._standard_variable_ids:
                # Assumes ASL uses 4-byte, signed integers to store suffixes,
                # and we need positive suffix values
                linking_suffix[master_variable[variable_id]] = \
                    scenario_tree_id_to_pint32(node_name, variable_id)
    # make sure the conversion from scenario tree id to int
    # did not have any collisions
    _ids = list(linking_suffix.values())
    assert len(_ids) == len(set(_ids))

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
        for attr_name in ("_gen_obj_repn",
                          "_gen_con_repn"):
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
        node_name = node.name
        for variable_id in node._standard_variable_ids:
            # Assumes ASL uses 4-byte, signed integers to store suffixes,
            # and we need positive suffix values
            linking_suffix[bySymbol[variable_id]] = \
                scenario_tree_id_to_pint32(node_name, variable_id)
    # make sure the conversion from scenario tree id to int
    # did not have any collisions
    _ids = list(linking_suffix.values())
    assert len(_ids) == len(set(_ids))

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
        for attr_name in ("_gen_obj_repn",
                          "_gen_con_repn"):
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

def _write_problem_list_file(scenario_tree,
                             problem_list_filename,
                             ignore_bundles=False):

    #
    # Write list of subproblems to file
    #
    subproblem_type = None
    with open(problem_list_filename, 'w') as f:
        if (not ignore_bundles) and scenario_tree.contains_bundles():
            subproblem_type = "bundles"
            for bundle in scenario_tree.bundles:
                f.write(str(bundle.name)+".nl")
                f.write("\n")
        else:
            subproblem_type = "scenarios"
            for scenario in scenario_tree.scenarios:
                f.write(str(scenario.name)+".nl")
                f.write("\n")
    assert subproblem_type is not None
    return subproblem_type

def EXTERNAL_write_nl(worker,
                      working_directory,
                      subproblem_type,
                      io_options):

    if subproblem_type == 'bundles':
        assert worker.scenario_tree.contains_bundles()
        for bundle in worker.scenario_tree.bundles:
            fname, symbol_map = _write_bundle_nl(
                worker,
                bundle,
                working_directory,
                io_options)
    else:
        assert subproblem_type == 'scenarios'
        orig_parents = {}
        if worker.scenario_tree.contains_bundles():
            for scenario in worker.scenario_tree.scenarios:
                if scenario._instance._parent is not None:
                    orig_parents[scenario] = scenario._instance._parent
                    scenario._instance._parent = None
                    assert not scenario._instance_objective.active
                    scenario._instance_objective.activate()
        try:
            for scenario in worker.scenario_tree.scenarios:
                fname, symbol_map = _write_scenario_nl(
                    worker,
                    scenario,
                    working_directory,
                    io_options)
        finally:
            for scenario, parent in orig_parents.items():
                scenario._instance._parent = parent
                assert scenario._instance_objective.active
                scenario._instance_objective.deactivate()

def write_schuripopt_files(manager,
                           output_directory,
                           ignore_bundles=False,
                           io_options=None,
                           verbose=False):

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

    manager.invoke_function(
        "EXTERNAL_write_nl",
        thisfile,
        invocation_type=InvocationType.Single,
        function_args=(output_directory,
                       subproblem_type,
                       io_options))

def convertschuripopt_register_options(options=None):
    if options is None:
        options = PySPConfigBlock()
    safe_register_common_option(options, "disable_gc")
    safe_register_common_option(options, "profile")
    safe_register_common_option(options, "traceback")
    safe_register_common_option(options, "verbose")
    safe_register_common_option(options, "symbolic_solver_labels")
    safe_register_unique_option(
        options,
        "output_directory",
        PySPConfigValue(
            ".",
            domain=_domain_must_be_str,
            description=(
                "The directory in which all SchurIpopt files "
                "will be stored. Default is '.'."
            ),
            doc=None,
            visibility=0))
    safe_register_unique_option(
        options,
        "ignore_bundles",
        PySPConfigValue(
            False,
            domain=bool,
            description=(
                "Ignore bundles when converting the SP to "
                "SchurIpopt input files."
            ),
            doc=None,
            visibility=0))
    safe_register_common_option(options, "scenario_tree_manager")
    ScenarioTreeManagerClientSerial.register_options(options)
    ScenarioTreeManagerClientPyro.register_options(options)

    return options

#
# Convert a PySP scenario tree formulation to DDSIP input files
#

def convertschuripopt(options):
    """
    Construct a senario tree manager and write the
    schuripopt input files.
    """

    start_time = time.time()

    io_options = {'symbolic_solver_labels':
                  options.symbolic_solver_labels}

    assert not options.compile_scenario_instances

    manager_class = None
    if options.scenario_tree_manager == 'serial':
        manager_class = ScenarioTreeManagerClientSerial
    elif options.scenario_tree_manager == 'pyro':
        manager_class = ScenarioTreeManagerClientPyro

    with manager_class(options) as scenario_tree_manager:
        scenario_tree_manager.initialize()
        files = write_schuripopt_files(
            scenario_tree_manager,
            options.output_directory,
            ignore_bundles=options.ignore_bundles,
            io_options=io_options,
            verbose=options.verbose)

    end_time = time.time()

    print("SchurIpopt files written to directory: %s"
          % (options.output_directory))
    print("")
    print("Total execution time=%.2f seconds"
          % (end_time - start_time))

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
            convertschuripopt_register_options,
            prog='convertschuripopt',
            description=(
"""Optimize a stochastic program using the SchurIpopt solver."""
            ))

    except SystemExit as _exc:
        # the parser throws a system exit if "-h" is specified
        # - catch it to exit gracefully.
        return _exc.code

    return launch_command(convertschuripopt,
                          options,
                          error_label="convertschuripopt: ",
                          disable_gc=options.disable_gc,
                          profile_count=options.profile,
                          traceback=options.traceback)

if __name__ == "__main__":
    sys.exit(main())
