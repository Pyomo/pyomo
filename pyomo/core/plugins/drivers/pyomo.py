from coopr.misc import coopr_parser
from coopr.pyomo.scripting.pyomo import create_parser


def pyomo_exec(args=None):
    import coopr.pyomo.scripting.util
    from coopr.pyomo.scripting.pyomo import run_pyomo
    return coopr.pyomo.scripting.util.run_command(command=run_pyomo, parser=pyomo_parser, args=args, name='pyomo')

#
# Add a subparser for the coopr command
#
pyomo_parser = create_parser(coopr_parser.add_subparser('pyomo',
        func=pyomo_exec,
        help='Analyze a generic optimation model',
        description='This coopr subcommand is used to analyze optimization models.'
        ))

