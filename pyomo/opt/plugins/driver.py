#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import argparse
import logging

import pyomo.scripting.pyomo_parser

logger = logging.getLogger('pyomo.solvers')


def setup_test_parser(parser):
    parser.add_argument('--csv-file', '--csv', action='store', dest='csv', default=None,
                        help='Save test results to this file in a CSV format')
    parser.add_argument("-d", "--debug", action="store_true", dest="debug", default=False,
                        help="Show debugging information and text generated during tests.")
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose", default=False,
                        help="Show verbose results output.")
    parser.add_argument("solver", metavar="SOLVER", default=None, nargs='*',
                        help="a solver name")

def test_exec(options):
    import pyomo.solvers.tests.testcases
    pyomo.solvers.tests.testcases.run_test_scenarios(options)
    
    
#
# Add a subparser for the pyomo command
#
setup_test_parser(
    pyomo.scripting.pyomo_parser.add_subparser('test-solvers',
        func=test_exec,
        help='Test Pyomo solvers',
        description='This pyomo subcommand is used to run tests on installed solvers.',
        epilog="""
This Pyomo subcommand executes solvers on a variety of test problems that
are defined in the pyomo.data.pyomo package.  The default behavior is to
test all available solvers, but the testing can be limited by explicitly
specifying the solvers that are tested.  For example:

  pyomo test-solvers glpk cplex

will test only the glpk and cplex solvers.

The default summary is a simple table that describes the percentage of
checks that passed.  The '-v' option can be used to provide a summary
of all checks that failed, which is generally useful for evaluating
solvers.  The '-d' option provides additional detail about all checks
performed (both passed and failed checks).  Additionally, this option
prints information about the optimization process, such as the pyomo
command-line that was executed.""",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
)

