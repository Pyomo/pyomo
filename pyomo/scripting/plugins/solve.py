from pyomo.scripting import pyomo_parser
from pyomo.core.scripting.pyomo_command import create_parser


def solve_exec(args=None):
    import pyomo.core.scripting.util
    from pyomo.core.scripting.pyomo_command import run_pyomo
    return pyomo.core.scripting.util.run_command(command=run_pyomo, parser=pyomo_parser, args=args, name='solve')

#
# Add a subparser for the solve command
#
solve_parser = create_parser(solve_parser.add_subparser('solve',
        func=solve_exec,
        help='Analyze a generic optimation model',
        description='This pyomo subcommand is used to analyze optimization models.'
        ))

