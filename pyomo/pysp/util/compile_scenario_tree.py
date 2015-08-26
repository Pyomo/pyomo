#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

__all__ = ('compile_scenario_tree',)

import os
import sys
import time
import argparse

from pyutilib.misc.config import (ConfigValue,
                                  ConfigBlock)
from pyutilib.pyro import shutdown_pyro_components

from pyomo.util import pyomo_command
from pyomo.repn.beta.matrix import compile_block_linear_constraints
from pyomo.pysp.util.config import (safe_register_common_option,
                                    safe_register_unique_option,
                                    _domain_must_be_str)
from pyomo.pysp.util.misc import launch_command
from pyomo.pysp.scenariotree.scenariotreemanager import (ScenarioTreeManagerSerial,
                                                         ScenarioTreeManagerSPPyro)
from pyomo.pysp.scenariotree.scenariotreeserver import SPPyroScenarioTreeServer
from pyomo.pysp.scenariotree.scenariotreeworkerbasic import ScenarioTreeWorkerBasic
from pyomo.pysp.scenariotree.scenariotreeserverutils import InvocationType
#
# Compile all scenario models on this scenario tree manager. If output_directory
# is not none, the compiled scenario models will be pickled into individual
# files in that directory. If compiled_reference_model_filename is not none,
# a new reference model will be created that references the pickled scenario
# models.
#
# NOTE: It is assumed the models are "clean", with the exception of
#       whatever objects are added by the ScenarioTree linking
#       processes. That is, these instances have not yet been handed
#       over to something like PH (e.g., annotated with PH specific
#       parameters, variables, and other objects).
#
def compile_scenario_tree(scenario_tree_manager,
                          output_directory=None,
                          compiled_reference_model_filename=None,
                          verbose=False,
                          **kwds):

    import pyomo.environ
    import pyomo.solvers.plugins.smanager.phpyro
    import pyomo.pysp.scenariotree.scenariotreeserverutils as \
        scenariotreeserverutils
    from pyomo.pysp.scenariotree.scenariotreemanager import \
        (ScenarioTreeManagerSerial,
         ScenarioTreeManagerSPPyro)

    if output_directory is not None:
        output_directory = os.path.abspath(output_directory)
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

    scenario_tree = scenario_tree_manager._scenario_tree

    if isinstance(scenario_tree_manager,
                  ScenarioTreeManagerSPPyro):

        ahs = []
        for scenario in scenario_tree._scenarios:

            output_filename = None
            if output_directory is not None:
                output_filename = os.path.join(output_directory,
                                               scenario._name+".compiled.pickle")
            function_kwds = {'output_filename': output_filename,
                             'verbose': verbose}
            function_kwds.update(kwds)
            scenario_tree_manager.complete_actions(
                scenario_tree_manager.\
                transmit_external_function_invocation_to_worker(
                    scenario_tree_manager.get_worker_for_scenario(scenario._name),
                    "pyomo.pysp.util.compile_scenario_tree",
                    "EXTERNAL_compile_scenario",
                    invocation_type=InvocationType.PerScenarioInvocation,
                    function_kwds=function_kwds,
                    return_action_handle=True))

    else:

        for scenario in scenario_tree._scenarios:
            output_filename = None
            if output_directory is not None:
                output_filename = os.path.join(output_directory,
                                               scenario._name+".compiled.pickle")
            EXTERNAL_compile_scenario(scenario_tree_manager,
                                      scenario_tree,
                                      scenario,
                                      output_filename=output_filename,
                                      verbose=verbose,
                                      **kwds)

    if compiled_reference_model_filename is not None:
        assert output_directory is not None
        filename = os.path.join(output_directory,
                                compiled_reference_model_filename)
        if verbose:
            print("Saving reference model for compiled scenario tree "
                  "to file: "+str(filename))
        with open(filename, 'wb') as f:
            f.write("from six.moves import cPickle\n")
            f.write("def pysp_instance_creation_callback(scenario_name, node_names):\n")
            f.write("    with open(scenario_name+'.compiled.pickle', 'rb') as f:\n")
            f.write("        return cPickle.load(f)\n")

#
# This function can be transmitted to scenario tree servers.
# NOTE: It is assumed the models are "clean", with the exception of
#       whatever objects are added by the ScenarioTree linking
#       processes. That is, these instances have not yet been handed
#       over to something like PH (e.g., annotated with PH specific
#       parameters, variables, and other objects).
#
def EXTERNAL_compile_scenario(scenario_tree_manager,
                              scenario_tree,
                              scenario,
                              output_times=False,
                              **kwds):
    start_time = time.time()
    assert scenario._instance is not None
    scenario_instance = scenario._instance

    #
    # Temporarily PySP objects added by the scenario tree
    # linking process, just in case this model is pickled.
    # is pickled.
    #
    scenario_instance_cost_expression = \
        scenario._instance_cost_expression
    assert scenario_instance_cost_expression.name in \
        ("_PySP_UserCostExpression", "_PySP_CostExpression")
    scenario_instance.del_component(scenario_instance_cost_expression)

    scenario_instance_objective = \
        scenario._instance_objective
    assert scenario_instance_objective.name in \
        ("_PySP_UserCostObjective", "_PySP_CostObjective")
    scenario_instance.del_component(scenario_instance_objective)

    assert hasattr(scenario_instance, "_ScenarioTreeSymbolMap")
    scenario_tree_symbol_map = scenario_instance._ScenarioTreeSymbolMap
    del scenario_instance._ScenarioTreeSymbolMap

    if scenario._instance_original_objective_object is not None:
        assert scenario._instance_original_objective_object is not \
            scenario_instance_objective
        scenario._instance_original_objective_object.activate()

    assert not hasattr(scenario_instance,
                       "_PySP_compiled_linear_constraints")
    compile_block_linear_constraints(scenario_instance,
                                     "_PySP_compiled_linear_constraints",
                                     **kwds)

    #
    # Re-add PySP generated model components
    #
    scenario_instance.add_component(scenario_instance_cost_expression.name,
                                    scenario_instance_cost_expression)
    scenario_instance.add_component(scenario_instance_objective.name,
                                    scenario_instance_objective)
    scenario_instance._ScenarioTreeSymbolMap = scenario_tree_symbol_map
    if scenario._instance_original_objective_object is not None:
        scenario._instance_original_objective_object.deactivate()

    stop_time = time.time()
    if output_times:
        print("Total time to compile instance for scenario %s: %.2f seconds"
              % (scenario._name, stop_time-start_time))

def pysp_compile_scenario_tree_register_options(options):
    from pyomo.pysp.scenariotree.scenariotreemanager import \
        (ScenarioTreeManagerSerial,
         ScenarioTreeManagerSPPyro)

    safe_register_common_option(options, "disable_gc")
    safe_register_common_option(options, "profile")
    safe_register_common_option(options, "traceback")
    safe_register_common_option(options, "verbose")
    safe_register_common_option(options, "output_times")
    safe_register_unique_option(
        options,
        "output_directory",
        ConfigValue(
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
        ConfigValue(
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
    safe_register_unique_option(
        options,
        "single_precision_storage",
        ConfigValue(
            False,
            domain=bool,
            description=(
                "Use single precision memory for storing the compiled constraint "
                "coefficients. By default double precision floating point storage "
                "will be used. For more information, refer to the 'f' and 'd' type "
                "codes in the Python built-in 'array' module."
            ),
            doc=None,
            visibility=0))
    safe_register_common_option(options, "scenario_tree_manager")
    ScenarioTreeManagerSerial.register_options(options)
    ScenarioTreeManagerSPPyro.register_options(options)

#
# Convert a PySP scenario tree formulation to SMPS input files
#

def run_pysp_compile_scenario_tree(options):
    from pyomo.pysp.scenariotree.scenariotreemanager import \
        (ScenarioTreeManagerSerial,
         ScenarioTreeManagerSPPyro)

    ScenarioTreeManager_class = None
    if options.scenario_tree_manager == 'serial':
        ScenarioTreeManager_class = ScenarioTreeManagerSerial
    elif options.scenario_tree_manager == 'sppyro':
        ScenarioTreeManager_class = ScenarioTreeManagerSPPyro

    with ScenarioTreeManager_class(options) \
         as scenario_tree_manager:
        scenario_tree_manager.initialize()
        compile_scenario_tree(
            scenario_tree_manager,
            output_directory=options.output_directory,
            compiled_reference_model_filename=\
               options.compiled_reference_model_filename,
            verbose=options.verbose,
            output_times=options.output_times,
            single_precision_storage=options.single_precision_storage)

#
# The main initialization / runner routine.
#

def exec_pysp_compile_scenario_tree(options):
    import pyomo.environ

    start_time = time.time()

    try:

        run_pysp_compile_scenario_tree(options)

    finally:
        # if an exception is triggered, and we're running with pyro,
        # shut down everything - not doing so is annoying, and leads
        # to a lot of wasted compute time. but don't do this if the
        # shutdown-pyro option is disabled => the user wanted
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
    options = ConfigBlock()
    pysp_compile_scenario_tree_register_options(options)

    try:
        ap = argparse.ArgumentParser(prog='pysp_compile_scenario_tree')
        options.initialize_argparse(ap)
        options.import_argparse(ap.parse_args(args=args))
    except SystemExit as _exc:
        # the parser throws a system exit if "-h" is specified
        # - catch it to exit gracefully.
        return _exc.code

    return launch_command(exec_pysp_compile_scenario_tree,
                          options,
                          error_label="pysp_compile_scenario_tree: ",
                          disable_gc=options.disable_gc,
                          profile_count=options.profile,
                          traceback=options.traceback)

@pyomo_command('pysp_compile_scenario_tree',
               "Compile all scenarios on a PySP scenario tree into a more "
               "memory efficient form")
def pysp_compile_scenario_tree_main(args=None):
    return main(args=args)

if __name__ == "__main__":
    main(args=sys.argv[1:])
