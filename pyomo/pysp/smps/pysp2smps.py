#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

import os
import logging
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

from pyomo.util import pyomo_command
from pyomo.core.base import maximize, minimize

from pyomo.pysp.util.config import (PySPConfigValue,
                                    PySPConfigBlock,
                                    safe_register_common_option,
                                    safe_register_unique_option,
                                    _domain_must_be_str)
from pyomo.pysp.scenariotree.instance_factory import \
    ScenarioTreeInstanceFactory
from pyomo.pysp.scenariotree.manager import \
    (ScenarioTreeManagerClientSerial,
     ScenarioTreeManagerClientPyro)
from pyomo.pysp.util.misc import launch_command
import pyomo.pysp.smps.smpsutils

logger = logging.getLogger('pyomo.pysp')

def pysp2smps_register_options(options=None):
    if options is None:
        options = PySPConfigBlock()
    safe_register_common_option(options, "disable_gc")
    safe_register_common_option(options, "profile")
    safe_register_common_option(options, "traceback")
    safe_register_common_option(options, "verbose")
    safe_register_common_option(options, "symbolic_solver_labels")
    safe_register_common_option(options, "file_determinism")
    safe_register_unique_option(
        options,
        "explicit",
        PySPConfigValue(
            False,
            domain=bool,
            description=(
                "Generate SMPS files using explicit scenarios "
                "(or bundles). ** This option is deprecated. It is "
                "the default behavior. ** "
            ),
            doc=None,
            visibility=0))
    safe_register_unique_option(
        options,
        "core_format",
        PySPConfigValue(
            "mps",
            domain=_domain_must_be_str,
            description=(
                "The format used to generate the core SMPS problem file. "
                "Choices are: [mps, lp]. The default format is MPS."
            ),
            doc=None,
            visibility=0),
        ap_kwds={'choices': ['mps','lp']})
    safe_register_unique_option(
        options,
        "output_directory",
        PySPConfigValue(
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
        PySPConfigValue(
            None,
            domain=_domain_must_be_str,
            description=(
                "The basename to use for all SMPS related output "
                "files. ** Required **"
            ),
            doc=None,
            visibility=0))
    safe_register_unique_option(
        options,
        "disable_consistency_checks",
        PySPConfigValue(
            False,
            domain=bool,
            description=(
                "Disables consistency checks that attempt to find issues "
                "with the SMPS conversion. By default, these checks are run "
                "after conversion takes place and leave behind a temporary "
                "directory with per-scenario output files if the checks fail. "
                "This option is not recommended, but can be used if the "
                "consistency checks are prohibitively slow."
            ),
            doc=None,
            visibility=0))
    safe_register_unique_option(
        options,
        "keep_scenario_files",
        PySPConfigValue(
            False,
            domain=bool,
            description=(
                "Keeps around the per-scenario SMPS files created for testing "
                "whether a conversion is valid (whether or not the validation "
                "checks are performed). These files can be useful for "
                "debugging purposes."
            ),
            doc=None,
            visibility=0))
    safe_register_unique_option(
        options,
        "keep_auxiliary_files",
        PySPConfigValue(
            False,
            domain=bool,
            description=(
                "Keep auxiliary files for the template scenario that are normally "
                "used for testing the validity of the SMPS conversion. "
                "These include the .row, .col, .sto.struct, and .[mps,lp].det files."
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

def run_pysp2smps(options):
    import pyomo.environ

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

    if options.compile_scenario_instances:
        raise ValueError("The pysp2smps script does not allow the "
                         "compile_scenario_instances option to be "
                         "set to True.")

    if options.explicit:
        logger.warn("DEPRECATED: The use of the --explicit option "
                    "is no longer necessary. It is the default behavior")

    manager_class = None
    if options.scenario_tree_manager == 'serial':
        manager_class = ScenarioTreeManagerClientSerial
    elif options.scenario_tree_manager == 'pyro':
        manager_class = ScenarioTreeManagerClientPyro

    with manager_class(options) as scenario_tree_manager:
        scenario_tree_manager.initialize()
        pyomo.pysp.smps.smpsutils.\
            convert_explicit(
                options.output_directory,
                options.basename,
                scenario_tree_manager,
                core_format=options.core_format,
                io_options=io_options,
                disable_consistency_checks=options.disable_consistency_checks,
                keep_scenario_files=options.keep_scenario_files,
                keep_auxiliary_files=options.keep_auxiliary_files,
                verbose=options.verbose)

    end_time = time.time()

    print("")
    print("Total execution time=%.2f seconds"
          % (end_time - start_time))

#
# the main driver routine for the pysp2smps script.
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
    options = PySPConfigBlock()
    pysp2smps_register_options(options)

    #
    # Prevent the compile_scenario_instances option from
    # appearing on the command line. This script relies on
    # the original constraints being present on the model
    #
    argparse_val = options.get('compile_scenario_instances')._argparse
    options.get('compile_scenario_instances')._argparse = None

    try:
        ap = argparse.ArgumentParser(prog='pysp2smps')
        options.initialize_argparse(ap)

        # restore the option so the class validation does not
        # raise an exception
        options.get('compile_scenario_instances')._argparse = argparse_val

        options.import_argparse(ap.parse_args(args=args))
    except SystemExit as _exc:
        # the parser throws a system exit if "-h" is specified
        # - catch it to exit gracefully.
        return _exc.code

    return launch_command(run_pysp2smps,
                          options,
                          error_label="pysp2smps: ",
                          disable_gc=options.disable_gc,
                          profile_count=options.profile,
                          traceback=options.traceback)

@pyomo_command('pysp2smps',
               "Convert a PySP Scenario Tree Formulation to SMPS "
               "input format")
def pysp2smps_main(args=None):
    return main(args=args)

if __name__ == "__main__":
    main(args=sys.argv[1:])
