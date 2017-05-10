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

from pyutilib.services import TempfileManager
from pyomo.pysp.util.config import (PySPConfigValue,
                                    PySPConfigBlock,
                                    safe_register_common_option,
                                    safe_register_unique_option,
                                    _domain_must_be_str)
from pyomo.pysp.util.misc import (parse_command_line,
                                  launch_command)
from pyomo.pysp.scenariotree.instance_factory import \
    ScenarioTreeInstanceFactory

def generate_scenario_tree_image(options):
    with ScenarioTreeInstanceFactory(
            options.model_location,
            options.scenario_tree_location) as factory:

        scenario_tree = factory.generate_scenario_tree(
            downsample_fraction=options.scenario_tree_downsample_fraction,
            bundles=options.scenario_bundle_specification,
            random_bundles=options.create_random_bundles,
            random_seed=options.scenario_tree_random_seed,
            verbose=options.verbose)

        with TempfileManager.push():
            tmpdotfile = TempfileManager.create_tempfile(suffix=".dot")
            scenario_tree.save_to_dot(tmpdotfile)
            os.system('dot -Tpdf -o %s %s' % (options.output_file,
                                              tmpdotfile))
            print("Output Saved To: %s" % (options.output_file))

def generate_scenario_tree_image_register_options(options=None):
    if options is None:
        options = PySPConfigBlock()
    safe_register_unique_option(
        options,
        "output_file",
        PySPConfigValue(
            "ScenarioStructure.pdf",
            domain=_domain_must_be_str,
            description=(
                "The name of the file in which to store the scenario "
                "tree image. Default is ScenarioStructure.pdf."
            ),
            doc=None,
            visibility=0))
    safe_register_common_option(options,
                                "verbose")
    safe_register_common_option(options,
                               "model_location")
    safe_register_common_option(options,
                               "scenario_tree_location")
    safe_register_common_option(options,
                                "scenario_tree_random_seed")
    safe_register_common_option(options,
                                "scenario_tree_downsample_fraction")
    safe_register_common_option(options,
                                "scenario_bundle_specification")
    safe_register_common_option(options,
                                "create_random_bundles")
    safe_register_common_option(options,
                               "disable_gc")
    safe_register_common_option(options,
                               "profile")
    safe_register_common_option(options,
                               "traceback")
    return options

#
# the main driver routine for the scenario_tree_image script.
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
            generate_scenario_tree_image_register_options,
            prog='scenario_tree_image',
            description=(
"""Generate a pdf image of the scenario tree."""
            ))

    except SystemExit as _exc:
        # the parser throws a system exit if "-h" is specified
        # - catch it to exit gracefully.
        return _exc.code

    return launch_command(generate_scenario_tree_image,
                          options,
                          error_label="scenario_tree_image: ",
                          disable_gc=options.disable_gc,
                          profile_count=options.profile,
                          traceback=options.traceback)

def scenario_tree_image_main(args=None):
    return main(args=args)

if __name__ == "__main__":
    sys.exit(main())
