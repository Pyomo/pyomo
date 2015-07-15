#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

import os
import gc
import sys
import time
import contextlib
import copy
import argparse
try:
    from guppy import hpy
    guppy_available = True
except ImportError:
    guppy_available = False


from pyutilib.misc import PauseGC
from pyutilib.misc.config import (ConfigValue,
                                  ConfigBlock)
from pyutilib.pyro import shutdown_pyro_components

from pyomo.util import pyomo_command
from pyomo.core.base import maximize, minimize

from pyomo.pysp.util.config import (safe_register_common_option,
                                    safe_register_unique_option,
                                    _domain_must_be_str)
from pyomo.pysp.scenariotree.scenariotreemanager import \
    (ScenarioTreeManagerSerial,
     ScenarioTreeManagerSPPyro)
import pyomo.pysp.scenariotree.scenariotreeserverutils as \
    scenariotreeserverutils
from pyomo.pysp.util.misc import launch_command

def pysp2smps_register_options(options):
    safe_register_common_option(options, "disable_gc")
    safe_register_common_option(options, "profile")
    safe_register_common_option(options, "traceback")
    safe_register_common_option(options, "verbose")
    safe_register_common_option(options, "output_times")
    safe_register_common_option(options,
                                "symbolic_solver_labels")
    safe_register_common_option(options,
                                "file_determinism")
    safe_register_unique_option(
        options,
        "implicit",
        ConfigValue(
            False,
            domain=bool,
            description=(
                "Generate SMPS files using implicit parameter "
                "distributions."
            ),
            doc=None,
            visibility=0))
    safe_register_unique_option(
        options,
        "explicit",
        ConfigValue(
            False,
            domain=bool,
            description=(
                "Generate SMPS files using explicit scenarios "
                "(or bundles)."
            ),
            doc=None,
            visibility=0))
    safe_register_unique_option(
        options,
        "output_directory",
        ConfigValue(
            ".",
            domain=_domain_must_be_str,
            description=(
                "The directory in which all SMPS related output files "
                "will be stored. Default is '.'."
            ),
            doc=None,
            visibility=0))
    safe_register_unique_option(
        options,
        "basename",
        ConfigValue(
            None,
            domain=_domain_must_be_str,
            description=(
                "The basename to use for all SMPS related output "
                "files. ** Required **"
            ),
            doc=None,
            visibility=0))
    ScenarioTreeManagerSerial.register_options(options)
    ScenarioTreeManagerSPPyro.register_options(options)

#
# Convert a PySP scenario tree formulation to SMPS input files
#

def run_pysp2smps(options):
    import smpsutils

    if (options.basename is None):
        raise ValueError("Output basename is required. "
                         "Use the --basename command-line option")

    if not os.path.exists(options.output_directory):
        os.makedirs(options.output_directory)

    start_time = time.time()

    io_options = {'symbolic_solver_labels':
                  options.symbolic_solver_labels,
                  'file_determinism':
                  options.file_determinism}

    if not (options.implicit or options.explicit):
        raise ValueError(
            "Requires at least one of --implicit or "
            "--explicit command-line flags be set")

    if options.implicit:

        if options.verbose:
            print("Importing model and scenario tree files")

        with ScenarioTreeInstanceFactory(
                options.model_directory,
                options.instance_directory,
                options.verbose) as scenario_instance_factory:

            if options.verbose or options.output_times:
                print("Time to import model and scenario tree "
                      "structure files=%.2f seconds"
                      %(time.time() - start_time))

            smpsutils.convert_implicit(options.output_directory,
                                       options.basename,
                                       scenario_instance_factory,
                                       io_options=io_options)

    if options.explicit:

        ScenarioTreeManager_class = None
        if options.scenario_tree_manager == 'serial':
            ScenarioTreeManager_class = ScenarioTreeManagerSerial
        elif options.scenario_tree_manager == 'sppyro':
            ScenarioTreeManager_class = ScenarioTreeManagerSPPyro

        with ScenarioTreeManager_class(options) \
             as scenario_tree_manager:
            smpsutils.convert_explicit(options.output_directory,
                                       options.basename,
                                       scenario_tree_manager,
                                       io_options=io_options)

    end_time = time.time()

    print("")
    print("Total conversion execution time=%.2f seconds"
          % (end_time - start_time))

#
# The main initialization / runner routine.
#

def exec_pysp2smps(options):
    import pyomo.environ

    start_time = time.time()

    try:

        run_pysp2smps(options)

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

#
# the main driver routine for the pysp2smps script.
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
    pysp2smps_register_options(options)

    try:
        ap = argparse.ArgumentParser(prog='pysp2smps')
        options.initialize_argparse(ap)
        options.import_argparse(ap.parse_args(args=args))
    except SystemExit as _exc:
        # the parser throws a system exit if "-h" is specified
        # - catch it to exit gracefully.
        return _exc.code

    return launch_command('exec_pysp2smps(options)',
                          globals(),
                          locals(),
                          error_label="pysp2smps: ",
                          disable_gc=options.disable_gc,
                          profile_count=options.profile,
                          traceback=options.traceback)

    return rc

@pyomo_command('pysp2smps',
               'Convert a PySP Scenario Tree Formulation to SMPS input format')
def pysp2smps_main(args=None):
    return main(args=args)

if __name__ == "__main__":
    main(args=sys.argv[1:])
