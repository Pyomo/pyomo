#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

from pyomo.scripting.pyomo_parser import add_subparser
from pyomo.scripting.pyomo_command import create_parser

def solve_exec(args=None):
    import pyomo.scripting.util
    from pyomo.scripting.pyomo_command import run_pyomo
    return pyomo.scripting.util.run_command(command=run_pyomo,
                                            parser=solve_parser,
                                            args=args,
                                            name='solve')

#
# Add a subparser for the solve command
#
solve_parser = create_parser(add_subparser('solve',
    func=solve_exec,
    help='Optimize a model',
    description='This pyomo subcommand is used to analyze optimization models.'))

