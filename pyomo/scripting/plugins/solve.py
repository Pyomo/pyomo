from pyomo.scripting import pyomo_parser
from pyomo.scripting.pyomo_command import create_parser


def solve_exec(args=None):
    import pyomo.scripting.util
    from pyomo.scripting.pyomo_command import run_pyomo
    return pyomo.scripting.util.run_command(command=run_pyomo, parser=pyomo_parser, args=args, name='solve')

#
# Add a subparser for the solve command
#
solve_parser = create_parser(solve_parser.add_subparser('solve',
        func=solve_exec,
        help='Analyze a generic optimation model',
        description='This pyomo subcommand is used to analyze optimization models.'
        ))

