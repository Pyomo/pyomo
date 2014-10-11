
import sys

try:
    import pkg_resources
    #
    # Load modules associated with Plugins that are defined in
    # EGG files.
    #
    for entrypoint in pkg_resources.iter_entry_points('pyomo.command'):
        plugin_class = entrypoint.load()
except Exception:
    err = sys.exc_info()[1]
    sys.stderr.write( "Error loading pyomo.command entry points: %s  entrypoint='%s'\n" % (err, entrypoint) )


import pyomo.util.pyomo_parser


def main(args=None):
    #
    # Load subcommands
    #
    import pyomo.modeling
    #
    # Parse the arguments
    #
    parser = pyomo.util.pyomo_parser.get_parser()
    if args is None:
        ret = parser.parse_args()
    else:
        ret = parser.parse_args(args)
    #
    # Process the results
    #
    ret.func(ret)

