
__all__ = ['add_subparser', 'get_parser', 'subparsers']

import argparse
import warnings
import sys

#
# Sort sub_parser names, since these are inserted throughout Pyomo
#
# NOTE: This may not be robust to different versions of argparse.  We're
# mucking with a non-public API here ...
#
class CustomHelpFormatter(argparse.RawDescriptionHelpFormatter):

    def _metavar_formatter(self, action, default_metavar):
        if action.metavar is not None:
            result = action.metavar
        elif action.choices is not None:
            choice_strs = sorted(str(choice) for choice in action.choices)
            result = '{%s}' % ','.join(choice_strs)
        else:
            result = default_metavar

        def format(tuple_size):
            if isinstance(result, tuple):
                return result
            else:
                return (result, ) * tuple_size
        return format

    def _iter_indented_subactions(self, action):
        try:
            get_subactions = action._get_subactions
        except AttributeError:
            pass
        else:
            self._indent()
            if isinstance(action, argparse._SubParsersAction):
                for subaction in sorted(get_subactions(), key=lambda x: x.dest):
                    yield subaction
            else:
                for subaction in get_subactions():
                    yield subaction
            self._dedent()

def get_version():
    from pyomo.version import version
    import platform
    return "Pyomo %s (%s %s on %s %s)" % (
            version,
            platform.python_implementation(),
            '.'.join( str(x) for x in sys.version_info[:3] ),
            platform.system(),
            platform.release() )

#
# `BaseException.message` is deprecated as of Python 2.6, its usage triggers
# a `DeprecationWarning`.  As `ArgumentError` derives indirectly from
# `BaseException`, `ArgumentError.message` triggers this warning too
#
if sys.version_info[:2] == (2,6):
    warnings.filterwarnings(
        'ignore',
        message='BaseException.message has been deprecated as of Python 2.6',
        category=DeprecationWarning,
        module='argparse')

#
# Create the argparse parser for Pyomo
#
doc="The 'pyomo' command is the top-level command for the Pyomo optimization software."
epilog="""
-------------------------------------------------------------------------
Pyomo supports a variety of modeling and optimization capabilities,
which are executed either as subcommands of 'pyomo' or as separate
commands.  Use the 'help' command to get information about the
capabilities installed with Pyomo.  Additionally, each subcommand
supports independent command-line options.  Use the -h option to
print details for a subcommand.  For example, type

   pyomo check -h

to print information about the `check` subcommand.
"""
pyomo_parser = argparse.ArgumentParser(description=doc, epilog=epilog, formatter_class=CustomHelpFormatter)
pyomo_parser.add_argument("--version", action="version", version=get_version())
pyomo_subparsers = pyomo_parser.add_subparsers(dest='subparser_name', title='subcommands')

subparsers = []

def add_subparser(name, **args):
    """
    Add a subparser to the 'pyomo' command.
    """
    global subparsers
    func = None
    if 'func' in args:
        func = args['func']
        del args['func']
    parser = pyomo_subparsers.add_parser(name, **args)
    subparsers.append(name)
    if not func is None:
        parser.set_defaults(func=func)
    return parser

def get_parser():
    """
    Return the parser used by the 'pyomo' commmand.
    """
    return pyomo_parser

