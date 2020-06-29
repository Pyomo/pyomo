#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import sys
import copy

try:
    import pkg_resources
    pyomo_commands = pkg_resources.iter_entry_points('pyomo.command')
except:
    pyomo_commands = []
#
# Load modules associated with Plugins that are defined in
# EGG files.
#
for entrypoint in pyomo_commands:
    try:
        plugin_class = entrypoint.load()
    except Exception:
        exctype, err, tb = sys.exc_info()  # BUG?
        import traceback
        msg = "Error loading pyomo.command entry point %s:\nOriginal %s: %s\n"\
              "Traceback:\n%s" \
              % (entrypoint, exctype.__name__, err,
                 ''.join(traceback.format_tb(tb)),)
        # clear local variables to remove circular references
        exctype = err = tb = None
        # TODO: Should this just log an error and re-raise the original
        # exception?
        raise ImportError(msg)


def main(args=None):
    #
    # Load subcommands
    #
    from pyomo.scripting import pyomo_parser
    import pyomo.environ
    #
    # Parse the arguments
    #
    parser = pyomo_parser.get_parser()
    if args is None:
        args = copy.copy(sys.argv[1:])
    #
    # This is a hack to convert a command-line to a 'solve' subcommand
    #
    if not args:
        args.append('-h')
    # FIXME: This should use the logger and not print()
    if args[0][0] == '-':
        if args[0] not in ['-h', '--help', '--version']:
            print("WARNING: converting to the 'pyomo solve' subcommand")
            args = ['solve'] + args[0:]
    elif args[0] not in pyomo_parser.subparsers:
        print("WARNING: converting to the 'pyomo solve' subcommand")
        args = ['solve'] + args[0:]
    #
    # Process arguments
    #
    _options, _unparsed = parser.parse_known_args(args)
    #
    # Process the results
    #
    if _options.func.__code__.co_argcount == 1:
        #
        # If the execution function only accepts one argument, then we
        # create an exception if there are unparsed arguments.
        #
        if len(_unparsed) > 0:
            #
            # Re-parse the command-line to create an exception
            #
            parser.parse_args(_unparsed)
        retval = _options.func(_options)
    else:
        retval = _options.func(_options, _unparsed)
    return retval


def main_console_script():
    "This is the entry point for the main Pyomo script"
    # Note that we eat the retval data structure and only return the
    # process return code
    ans = main()
    try:
        return ans.errorcode
    except AttributeError:
        return ans

if __name__ == '__main__':
    sys.exit(main_console_script())
