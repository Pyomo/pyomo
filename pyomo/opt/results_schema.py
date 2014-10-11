#! /usr/bin/env python
#
# results_schema
#

import sys
from pyutilib.misc import Options
from pyomo.opt import SolverResults
from pyomo.util import pyomo_command


@pyomo_command('results_schema', "Print the predefined schema for a results object")
def main():
    if len(sys.argv) > 1:
        print("results_schema  - Print the predefined schema in a SolverResults object")
    options = Options(schema=True)
    r=SolverResults()
    repn = r._repn_(options)
    r.pprint(sys.stdout, options, repn=repn)
