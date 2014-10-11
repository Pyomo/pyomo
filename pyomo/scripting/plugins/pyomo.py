from pyomo.misc import pyomo_parser
from pyomo.core.scripting.pyomo import create_parser


def pyomo_exec(args=None):
    import pyomo.core.scripting.util
    from pyomo.core.scripting.pyomo import run_pyomo
    return pyomo.core.scripting.util.run_command(command=run_pyomo, parser=pyomo_parser, args=args, name='pyomo')

#
# Add a subparser for the pyomo command
#
pyomo_parser = create_parser(pyomo_parser.add_subparser('pyomo',
        func=pyomo_exec,
        help='Analyze a generic optimation model',
        description='This pyomo subcommand is used to analyze optimization models.'
        ))

