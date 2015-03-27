#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

import json
import sys
import argparse

from pyomo.opt import SolverFactory
from pyomo.scripting.pyomo_parser import add_subparser, CustomHelpFormatter


def create_parser(parser=None):
    #
    # Setup command-line options.  The '--solver' option creates
    # all subsequent options...
    #
    if parser is None:
        parser = argparse.ArgumentParser(
                usage = '%(prog)s [options] <model_or_config_file> [<data_files>]'
                )
    parser.add_argument('--solver',
        action='store',
        dest='solver',
        default=None)
    parser.add_argument('--generate-config-template',
        action='store',
        dest='template',
        default=None)
    return parser


def create_temporary_parser(solver=False, generate=False):
    #
    # We create a dummy parser and subparser, to help make the 'help'
    # output seem sensible.
    #
    parser = argparse.ArgumentParser(formatter_class=CustomHelpFormatter)
    _subparsers = parser.add_subparsers()
    _parser = _subparsers.add_parser('solve')
    _parser.formatter_class=CustomHelpFormatter
    if generate:
        #
        # Adding documentation about the two options that are
        # defined in the initial parser.
        #
        _parser.add_argument('--generate-config-template',
            action='store',
            dest='template',
            default=None,
            help='Create a configuration template file in YAML or JSON and exit.')
    if solver:
        _parser.add_argument('--solver',
            action='store',
            dest='solver',
            default=None,
            help='Solver name.  This option is required unless the solver name is specified in a configuration file.')
        _parser.usage = '%(prog)s [options] <model_or_config_file> [<data_files>]'
        _parser.epilog = """
Description:

The 'pyomo solve' subcommand optimizes a Pyomo model.  The --solver option
is required because the specific steps executed are solver dependent.
The standard steps executed by this subcommand are:

  - Apply pre-solve operations (e.g. import Python packages)
  - Create the model
  - Apply model transformations
  - Perform optimization
  - Apply post-solve operations (e.g. save optimal solutions)


This subcommand can be executed with or without a configuration file.
The command-line options can be used to perform simple optimization steps.
For example:

  pyomo solve --solver=glpk model.py model.dat

This uses the 'glpk' solver to optimize the model define in 'model.py'
using the Pyomo data file 'pyomo.dat'.  Solver-specific command-line
options can be listed by specifying the '--solver' option and the '--help'
(or '-h') option:

  pyomo solve --solver=glpk --help


A yaml or json configuration file can also be used to specify
options used by the solver.  For example:

  pyomo solve --solver=glpk config.yaml

No other command-line options are required!  Further, the '--solver'
option can be omitted if the solver name is specified in the configuration
file.

Configuration options are also solver-specific; a template configuration
file can be generated with the '--generate-config-template' option:

  pyomo solve --solver=glpk --generate-config-template=template.yaml
  pyomo solve --solver=glpk --generate-config-template=template.json

Note that yaml template files contain comments that describe the
configuration options.  Also, configuration files will generally support
more configuration options than are available with command-line options.

"""
    #
    _parser.add_argument('model_or_config_file',
        action='store',
        nargs='?',
        default='',
        help="A Python module that defines a Pyomo model, or a configuration file that defines options for 'pyomo solve' (in either YAML or JSON format)")
    _parser.add_argument('data_files',
        action='store',
        nargs='*',
        default=[],
        help='Pyomo data files that defined data used to initialize the model (specified in the first argument)')
    #
    return _parser


def solve_exec(args, unparsed):
    import pyomo.scripting.util
    #
    solver = getattr(args,'solver',None)
    if solver is None:
        #
        # Get configuration values if no solver has been specified
        #
        try:
            val = pyomo.scripting.util.get_config_values(unparsed[-1])
        except IndexError:
            val = None
        except IOError:
            val = None
        #
        # Try to get the solver name
        #
        if not val is None:
            try:
                solver = val['solvers'][0]['solver name']
            except:
                solver = None
        #
        # If no luck, then generate an error
        #
        if solver is None:
            if not ('-h' in unparsed or '--help' in unparsed):
                print("ERROR: No solver specified!")
                print("")
            parser = create_temporary_parser(solver=True, generate=True)
            parser.parse_args(args=unparsed+['-h'])
            sys.exit(1)

    config = None
    with SolverFactory(solver) as opt:
        if opt is None:
            print("ERROR: Unknown solver '%s'!" % solver)
            sys.exit(1)
        #
        # Generate a template file
        #
        if not args.template is None:
            config = opt.config_block(init=True)
            OUTPUT = open(args.template, 'w')
            if args.template.endswith('json'):
                OUTPUT.write(json.dumps(config.value(), indent=2))
            else:
                OUTPUT.write(config.generate_yaml_template())
            OUTPUT.close()
            print("  Created template file '%s'" % args.template)
            sys.exit(0)
        #
        # Parse previously unparsed options
        #
        config = opt.config_block()
        if '-h' in unparsed or '--help' in unparsed:
            _parser = create_temporary_parser(generate=True)
        else:
            _parser = create_temporary_parser()
        config.initialize_argparse(_parser)
        _parser.usage = '%(prog)s [options] <model_or_config_file> [<data_files>]'
        _options = _parser.parse_args(args=unparsed)
        #
        # Import the parsed values into the config block, then
        # create an Options object
        #
        config.import_argparse(_options)
        config.solvers[0].solver_name = getattr(args, 'solver', None)
        if _options.model_or_config_file.endswith('.py'):
            config.model.filename = _options.model_or_config_file
            config.data.files = _options.data_files
        else:
            val = pyomo.scripting.util.get_config_values(_options.model_or_config_file)
            config.set_value(val)

    if config is None:
        raise RuntimeError("Failed to create config object")

    from pyomo.scripting.pyomo_command import run_pyomo
    #
    # Note that we pass-in pre-parsed options.  The run_command()
    # function knows to not perform a parse, but instead to simply
    # used these parsed values.
    #
    return pyomo.scripting.util.run_command(command=run_pyomo,
                                            parser=_parser,
                                            options=config,
                                            name='pyomo solve')

#
# Add a subparser for the solve command
#
solve_parser = create_parser(add_subparser('solve',
    func=solve_exec,
    help='Optimize a model',
    add_help=False,
    description='This pyomo subcommand is used to analyze optimization models.'
    ))

