#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import code
import sys
import subprocess

from pyomo.common._command import pyomo_command
from pyomo.common.deprecation import deprecated
import pyomo.scripting.pyomo_parser


@pyomo_command('pyomo_python', "Launch script using Pyomo's python installation")
@deprecated(
    msg="The 'pyomo_python' command has been deprecated and will be removed",
    version="6.0",
)
def pyomo_python(args=None):
    if args is None:
        args = sys.argv[1:]
    if args is None or len(args) == 0:
        console = code.InteractiveConsole()
        console.interact('Pyomo Python Console\n' + sys.version)
    else:
        cmd = sys.executable + ' ' + ' '.join(args)
        subprocess.run(cmd)


@pyomo_command('pyomo', "The main command interface for Pyomo")
def pyomo(args=None):
    parser = pyomo.scripting.pyomo_parser.get_parser()
    if args is None:
        ret = parser.parse_args()
    else:
        ret = parser.parse_args(args)
    ret.func(ret)
