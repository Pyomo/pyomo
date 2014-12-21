#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

import argparse
from pyutilib.misc import Options
from pyomo.opt import ProblemFormat
import pyomo.scripting.pyomo_parser

def create_parser(parser=None, cmd=None):
    import pyomo.scripting.pyomo_command
    #
    # Setup command-line options
    #
    if parser is None:
        parser = argparse.ArgumentParser(
                usage = '%s [options] <model_file> [<data_files>]' % cmd
                )
    group = parser.add_argument_group('Target Format')
    group.add_argument('--lp',
        help='Generate a LP file',
        action='store_true',
        dest='lp_format',
        default=False)
    group.add_argument('--osil',
        help='Generate a OSIL file',
        action='store_true',
        dest='osil_format',
        default=False)
    group.add_argument('--dakota',
        help='Generate a DAKOTA file',
        action='store_true',
        dest='dakota_format',
        default=False)
    group.add_argument('--nl',
        help='Generate a NL file',
        action='store_true',
        dest='nl_format',
        default=False)
    pyomo.scripting.pyomo_command.add_model_group(parser)
    pyomo.scripting.pyomo_command.add_logging_group(parser)
    pyomo.scripting.pyomo_command.add_misc_group(parser)
    parser.add_argument('model_file', action='store', nargs='?', default='', help='A Python module that defines a Pyomo model')
    parser.add_argument('data_files', action='store', nargs='*', default=[], help='Pyomo data files that defined data used to create a model instance')
    return parser


def run_convert(options=Options(), parser=None):
    from pyomo.scripting.convert import convert, convert_dakota
    if options.dakota_format:    
        dakota_convert(options, parser)
    elif options.lp_format:
        convert(options, parser, ProblemFormat.cpxlp)
    elif options.nl_format:
        convert(options, parser, ProblemFormat.nl)
    elif options.osil_format:
        convert(options, parser, ProblemFormat.osil)
    else:
        raise RuntimeError("Unspecified target conversion format!")


def convert_exec(args=None):
    import pyomo.scripting.util
    return pyomo.scripting.util.run_command(command=run_convert, parser=convert_parser, args=args, name='convert')

#
# Add a subparser for the pyomo command
#
convert_parser = create_parser(
    parser=pyomo.scripting.pyomo_parser.add_subparser('convert',
        func=convert_exec,
        help='Convert a Pyomo model to another format',
        description='This pyomo subcommand is used to create a new model file in a specified format from a specified Pyomo model.'
        ))

