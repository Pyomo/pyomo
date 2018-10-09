#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

__all__ = ('compile_scenario_tree',)

import os
import sys
import time
import argparse

from pyomo.common import pyomo_command
from pyomo.repn.beta.matrix import compile_block_linear_constraints
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

from six.moves import cPickle

# generate an absolute path to this file
thisfile = os.path.abspath(__file__)

#
# This function can be transmitted to scenario tree servers.
# NOTE: It is assumed the models are "clean", with the exception of
#       whatever objects are added by the ScenarioTree linking
#       processes. That is, these instances have not yet been handed
#       over to something like PH (e.g., annotated with PH specific
#       parameters, variables, and other objects).
#
def _pickle_compiled_scenario(worker,
                              scenario,
                              output_directory):
    from pyomo.core.base import (Var,
                                 Constraint,
                                 Objective,
                                 Block,
                                 Param,
                                 Set,
                                 BuildAction)

    assert output_directory is not None
    assert scenario._instance is not None
    scenario_instance = scenario._instance

    output_filename = os.path.join(output_directory,
                                   scenario.name+".compiled.pickle")

    #
    # Temporarily remove PySP objects added by the scenario tree
    # linking process, so we can pickle the original model.
    #
    scenario_instance_cost_expression = \
        scenario._instance_cost_expression
    assert scenario_instance_cost_expression.local_name in \
        ("_PySP_UserCostExpression", "_PySP_CostExpression")
    scenario_instance.del_component(scenario_instance_cost_expression)

    scenario_instance_objective = \
        scenario._instance_objective
    if scenario_instance_objective.local_name == "_PySP_CostObjective":
        scenario_instance.del_component(scenario_instance_objective)
    else:
        scenario_instance_objective.expr = scenario_instance_cost_expression.expr

    assert hasattr(scenario_instance, "_ScenarioTreeSymbolMap")
    scenario_tree_symbol_map = scenario_instance._ScenarioTreeSymbolMap
    del scenario_instance._ScenarioTreeSymbolMap

    if scenario._instance_original_objective_object is not None:
        assert scenario._instance_original_objective_object is not \
            scenario_instance_objective
        scenario._instance_original_objective_object.activate()

    # Delete all possible references to "rules"
    # declared in the original model file so we
    # don't have to reference it when we unpickle
    for block in scenario_instance.block_data_objects():
        if isinstance(block, Block):
            block._rule = None
        for var in block.component_objects(Var):
            var._domain_init_rule = None
        for con in block.component_objects(Constraint):
            con.rule = None
        for obj in block.component_objects(Objective):
            obj.rule = None
        for param in block.component_objects(Param):
            param._rule = None
            param._validate = None
        for set_ in block.component_objects(Set):
            set_.initialize = None
            set_.filter = None
        for ba in block.component_objects(BuildAction):
            ba._rule = None

    #
    # Pickle the scenario_instance
    #
    with open(output_filename, 'wb') as f:
        # in case it is in a bundle
        owning_block = scenario_instance._parent
        scenario_instance._parent = None
        cPickle.dump(scenario_instance,
                     f,
                     protocol=cPickle.HIGHEST_PROTOCOL)
        scenario_instance._parent = owning_block

    #
    # Re-add PySP generated model components
    #
    scenario_instance.add_component(scenario_instance_cost_expression.local_name,
                                    scenario_instance_cost_expression)
    if scenario_instance_objective.local_name == "_PySP_CostObjective":
        scenario_instance.add_component(scenario_instance_objective.local_name,
                                        scenario_instance_objective)
    scenario_instance_objective.expr = scenario_instance_cost_expression
    scenario_instance._ScenarioTreeSymbolMap = scenario_tree_symbol_map
    if scenario._instance_original_objective_object is not None:
        scenario._instance_original_objective_object.deactivate()

def pickle_compiled_scenario_tree(manager,
                                  output_directory,
                                  compiled_reference_model_filename):

    assert output_directory is not None
    output_directory = os.path.abspath(output_directory)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    scenario_tree = manager.scenario_tree
    if scenario_tree.contains_bundles():
        print("WARNING: This application ignores scenario bundles.")

    async_action = \
        manager.invoke_function(
            "_pickle_compiled_scenario",
            thisfile,
            function_args=(os.path.join(output_directory),),
            invocation_type=InvocationType.PerScenario,
            async_call=True)

    filename = os.path.join(output_directory,
                            compiled_reference_model_filename)
    print("Saving reference model for compiled scenario tree "
          "to file: "+str(filename))
    with open(filename, 'w') as f:
        f.write("import os\n")
        f.write("from six.moves import cPickle\n")
        f.write("thisdir = os.path.dirname(os.path.abspath(__file__))\n")
        f.write("def pysp_instance_creation_callback(scenario_name, node_names):\n")
        f.write("    scenario_filename = os.path.join(thisdir, scenario_name+'.compiled.pickle')\n")
        f.write("    with open(scenario_filename, 'rb') as f:\n")
        f.write("        return cPickle.load(f)\n")

    async_action.complete()
    for scenario in scenario_tree.scenarios:
        assert os.path.exists(os.path.join(output_directory,
                                           scenario.name+".compiled.pickle"))

def compile_scenario_tree_register_options(options=None):
    if options is None:
        options = PySPConfigBlock()
    safe_register_common_option(options, "disable_gc")
    safe_register_common_option(options, "profile")
    safe_register_common_option(options, "traceback")
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
        "compiled_reference_model_filename",
        PySPConfigValue(
            "PySP_CompiledReferenceModel.py",
            domain=_domain_must_be_str,
            description=(
                "The filename to use for the new reference model that uses "
                "the compiled scenarios. This will be prefixed by the "
                "output directory name where compiled scenarios are stored. "
                "Default is 'PySP_CompiledReferenceModel.py'."
            ),
            doc=None,
            visibility=0))
    safe_register_common_option(options, "scenario_tree_manager")
    ScenarioTreeManagerClientSerial.register_options(options)
    ScenarioTreeManagerClientPyro.register_options(options)

    return options

#
# Convert a PySP scenario tree formulation to SMPS input files
#

def run_compile_scenario_tree(options):

    import pyomo.environ

    start_time = time.time()

    manager_class = None
    if options.scenario_tree_manager == 'serial':
        manager_class = ScenarioTreeManagerClientSerial
    elif options.scenario_tree_manager == 'pyro':
        manager_class = ScenarioTreeManagerClientPyro

    options.compile_scenario_instances = True
    with manager_class(options) \
         as manager:
        manager.initialize()
        pickle_compiled_scenario_tree(
            manager,
            options.output_directory,
            options.compiled_reference_model_filename)

    print("")
    print("Total execution time=%.2f seconds"
          % (time.time() - start_time))

    return 0

#
# the main driver routine for the pysp_compile_scenario_tree script.
#

def main(args=None):
    #
    # Top-level command that executes the extensive form writer.
    # This is segregated from run_ef_writer to enable profiling.
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
            compile_scenario_tree_register_options,
            prog='compile_scenario_tree',
            description=(
"""Compresses linear constraints on all scenarios into
sparse matrix form and then pickles the resulting scenario
models. This should enable faster startup time with reduced
memory usage. This script will automatically activate the
'compile_scenario_instances' scenario tree manager flag."""
            ))

    except SystemExit as _exc:
        # the parser throws a system exit if "-h" is specified
        # - catch it to exit gracefully.
        return _exc.code

    return launch_command(run_compile_scenario_tree,
                          options,
                          error_label="compile_scenario_tree: ",
                          disable_gc=options.disable_gc,
                          profile_count=options.profile,
                          traceback=options.traceback)

def compile_scenario_tree_main(args=None):
    return main(args=args)

if __name__ == "__main__":
    sys.exit(main())
