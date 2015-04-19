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

from pyutilib.misc import Options
from pyomo.opt import ProblemFormat, ProblemConfigFactory
from pyomo.scripting.pyomo_parser import add_subparser, CustomHelpFormatter


def create_parser(parser=None):
    #
    # Setup command-line options.
    #
    if parser is None:
        parser = argparse.ArgumentParser(
                usage = '%(prog)s [options] <model_or_config_file> [<data_files>]'
                )
    parser.add_argument('--output',
        action='store',
        dest='filename',
        help="Output file name. This option is required unless the file name is specified in a       configuration file.",
        default=None)
    parser.add_argument('--format',
        action='store',
        dest='format',
        help="Output format",
        default=None)
    parser.add_argument('--generate-config-template',
        action='store',
        dest='template',
        default=None)
    return parser


def run_convert(options=Options(), parser=None):
    from pyomo.scripting.convert import convert, convert_dakota
    if options.model.save_format is None and not options.model.save_file is None and '.' in options.model.save_file:
        options.model.save_format = options.model.save_file.split('.')[-1]
    #
    if options.model.save_format == 'dakota':
        return convert_dakota(options, parser)
    elif options.model.save_format == 'lp':
        return convert(options, parser, ProblemFormat.cpxlp)
    elif options.model.save_format == 'nl':
        return convert(options, parser, ProblemFormat.nl)
    elif options.model.save_format == 'osil':
        return convert(options, parser, ProblemFormat.osil)
    else:
        raise RuntimeError("Unspecified target conversion format!")


def convert_exec(args, unparsed):
    #
    import pyomo.scripting.util
    #
    # Generate a template file
    #
    if not args.template is None:
        config, blocks = ProblemConfigFactory('default').config_block(True)
        OUTPUT = open(args.template, 'w')
        if args.template.endswith('json'):
            OUTPUT.write(json.dumps(config.value(), indent=2))
        else:
            OUTPUT.write(config.generate_yaml_template())
        OUTPUT.close()
        print("  Created template file '%s'" % args.template)
        sys.exit(0)
    #
    save_filename = getattr(args,'filename',None)
    if save_filename is None:
        save_format = getattr(args,'format',None)
        if not save_format is None:
            save_filename = 'unknown.'+save_format
    if save_filename is None:
        #
        # Get configuration values if no model file has been specified
        #
        try:
            val = pyomo.scripting.util.get_config_values(unparsed[-1])
        except IndexError:
            val = None
        except IOError:
            val = None
        #
        # Try to get the model file
        #
        if not val is None:
            try:
                save_filename = val['model']['save file']
            except:
                pass
            if save_filename is None:
                try:
                    save_filename = 'unknown.'+str(val['model']['save format'])
                except:
                    pass
    #
    # Help options or no filename have been specified, then print help ...
    #
    if save_filename is None or '-h' in unparsed or '--help' in unparsed:
        if not ('-h' in unparsed or '--help' in unparsed):
            print("ERROR: No output file or format specified!")
            print("")
        config, blocks = ProblemConfigFactory('default').config_block()
        parser = create_temporary_parser(output=True, generate=True)
        config.initialize_argparse(parser)
        parser.parse_args(args=unparsed+['-h'])
        sys.exit(1)
    #
    # Parse previously unparsed options
    #
    config, blocks = ProblemConfigFactory('default').config_block()
    _parser = create_temporary_parser()
    config.initialize_argparse(_parser)
    _options = _parser.parse_args(args=unparsed)
    #
    # Import the parsed values into the config block, then
    # create an Options object
    #
    config.import_argparse(_options)
    config.model.save_file = getattr(args, 'filename', None)
    config.model.save_format = getattr(args, 'format', None)
    if _options.model_or_config_file.endswith('.py'):
        config.model.filename = _options.model_or_config_file
        config.data.files = _options.data_files
    else:
        val = pyomo.scripting.util.get_config_values(_options.model_or_config_file)
        config.set_value( val )
    #
    # Note that we pass-in pre-parsed options.  The run_command()
    # function knows to not perform a parse, but instead to simply
    # used these parsed values.
    #
    return pyomo.scripting.util.run_command(command=run_convert, parser=convert_parser, options=config, name='convert')

#
# Add a subparser for the pyomo command
#
convert_parser = create_parser(add_subparser('convert',
        func=convert_exec,
        help='Convert a Pyomo model to another format',
        add_help=False,
        description='This pyomo subcommand is used to create a new model file in a specified format from a Pyomo model.'
        ))



def create_temporary_parser(output=False, generate=False):
    #
    # We create a dummy parser and subparser, to help make the 'help'
    # output seem sensible.
    #
    parser = argparse.ArgumentParser(formatter_class=CustomHelpFormatter)
    _subparsers = parser.add_subparsers()
    _parser = _subparsers.add_parser('convert')
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
    if output:
        parser.add_argument('--output',
            action='store',
            dest='filename',
            help="Output file name. This option is required unless the file name is specified in a       configuration file.",
            default=None)
        parser.add_argument('--format',
            action='store',
            dest='format',
            help="Output format",
            default=None)
        _parser.usage = '%(prog)s [options] <model_or_config_file> [<data_files>]'
        _parser.epilog = """
Description:

The 'pyomo convert' subcommand converts a Pyomo model into a specified
format.  The --output option is used to specify an output file.
If only the --format option is specified, then the output filename is
unknown.<format>.  The standard steps executed by this subcommand are:

  - Apply pre-solve operations (e.g. import Python packages)
  - Create the model
  - Apply model transformations


This subcommand can be executed with or without a configuration file.
For example:

  pyomo convert --output=model.lp model.py model.dat

This creates the file 'model.lp' with format 'lp'.


A yaml or json configuration file can also be used to specify conversion
options.  For example:

  pyomo convert config.yaml

No other command-line options are required!

A template configuration file can be generated with the
'--generate-config-template' option:

  pyomo convert --generate-config-template=template.yaml
  pyomo convert --generate-config-template=template.json

Note that yaml template files contain comments that describe the
configuration options.  Also, configuration files will generally support
more configuration options than are available with command-line options.

"""
    #
    _parser.add_argument('--output',
        action='store',
        dest='filename',
        help="Output file name. This option is required unless the file name is specified in a       configuration file.",
        default=None)
    _parser.add_argument('--format',
        action='store',
        dest='format',
        help="Output format",
        default=None)
    _parser.add_argument('model_or_config_file', 
        action='store', 
        nargs='?', 
        default='',
        help="A Python module that defines a Pyomo model, or a configuration file that defines options for 'pyomo convert' (in either YAML or JSON format)")
    _parser.add_argument('data_files', 
        action='store', 
        nargs='*', 
        default=[],
        help='Pyomo data files that defined data used to initialize the model (specified in the first argument)')
    #
    return _parser
