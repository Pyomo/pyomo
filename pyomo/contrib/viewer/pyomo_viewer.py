#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  ___________________________________________________________________________
#
#  This module was originally developed as part of the IDAES PSE Framework
#
#  Institute for the Design of Advanced Energy Systems Process Systems
#  Engineering Framework (IDAES PSE Framework) Copyright (c) 2018-2019, by the
#  software owners: The Regents of the University of California, through
#  Lawrence Berkeley National Laboratory,  National Technology & Engineering
#  Solutions of Sandia, LLC, Carnegie Mellon University, West Virginia
#  University Research Corporation, et al. All rights reserved.
#
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.common.dependencies import attempt_import
from pyomo.common.deprecation import relocated_module_attribute
from pyomo.scripting.pyomo_parser import add_subparser

relocated_module_attribute(
    'QtApp', 'pyomo.contrib.viewer.pyomo_qtapp.QtApp', version='6.9.3'
)

qtconsole_app, qtconsole_available = attempt_import(
    "qtconsole.qtconsoleapp", defer_import=False
)


def main(*args):
    # Import the Qt infrastructure (if it exists)
    import pyomo.contrib.viewer.qt as myqt

    if not myqt.available or not qtconsole_available:
        errors = list(myqt.import_errors)
        if not qtconsole_available:
            errors.append(qtconsole_app._moduleunavailable_message())
        print("qt not available\n    " + "\n    ".join(errors))
        return

    # Ensure that all Pyomo plugins have been registered
    import pyomo.environ

    # Import & run the Qt application
    from pyomo.contrib.viewer.pyomo_qtapp import QtApp

    QtApp.launch_instance()


# Add a subparser for the model-viewer command
add_subparser(
    "model-viewer",
    func=main,
    help="Run the Pyomo model viewer",
    add_help=False,
    description="This runs the Pyomo model viewer",
)

if __name__ == '__main__':
    main()
