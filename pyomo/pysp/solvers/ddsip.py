#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

# TODO: Expose DDSIP options by declaring them on the solver config object

import os
import sys
import time

import pyutilib.subprocess
import pyutilib.services

from pyomo.util import pyomo_command
from pyomo.core import maximize
from pyomo.pysp.util.configured_object import PySPConfiguredObject
from pyomo.pysp.util.config import (PySPConfigValue,
                                    PySPConfigBlock,
                                    safe_register_common_option,
                                    safe_register_unique_option,
                                    safe_declare_common_option,
                                    safe_declare_unique_option,
                                    _domain_positive,
                                    _domain_positive_integer,
                                    _domain_nonnegative_integer,
                                    _domain_must_be_str)
from pyomo.pysp.util.misc import (parse_command_line,
                                  launch_command)
from pyomo.pysp.scenariotree.manager_solver import \
    (ScenarioTreeManagerSolver,
     ScenarioTreeManagerFactory)
import pyomo.pysp.smps.smpsutils
from pyomo.pysp.embeddedsp import EmbeddedSP
from pyomo.pysp.solvers.spsolver import (SPSolver,
                                         SPSolverResults,
                                         SPSolverFactory)

_ddsip_group_label = "DDSIP Options"

class DDSIPSolver(SPSolver, PySPConfiguredObject):

    @classmethod
    def _declare_options(cls, options=None):
        if options is None:
            options = PySPConfigBlock()
        safe_register_unique_option(
            options,
            "executable",
            PySPConfigValue(
                "ddsip",
                domain=_domain_must_be_str,
                description=(
                    "Name of the executable used when launching the "
                    "DDSIP solver. The default is 'ddsip'. This "
                    "option can be set to an absolute or relative path. "
                    "Otherwise, it is assumed that the named executable "
                    "will be found in the shell's search path."
                ),
                doc=None,
                visibility=0),
            ap_group=_ddsip_group_label)
        return options

    def __init__(self, *args, **kwds):
        super(DDSIPSolver, self).__init__(*args, **kwds)
        self._name = "ddsip"

        self._firststage_var_suffix = '__DDSIP_FIRSTSTAGE'

    def _solve_impl(self,
                    sp,
                    keep_solver_files=False,
                    output_solver_log=False,
                    symbolic_solver_labels=False):

        if len(sp.scenario_tree.stages) > 2:
            raise ValueError("DDSIP solver does not handle more "
                             "than 2 time-stages")

        if sp.objective_sense == maximize:
            raise ValueError("DDSIP solver does not yet handle "
                             "maximization problems")

        pyutilib.services.TempfileManager.push()
        try:

            #
            # Setup the DDSIP working directory
            #

            working_directory = pyutilib.services.\
                                TempfileManager.create_tempdir()
            config_filename = os.path.join(working_directory,
                                           "config.ddsip")
            input_directory = os.path.join(working_directory,
                                           "sipin")
            output_directory = os.path.join(working_directory,
                                            "sipout")
            logfile = os.path.join(working_directory, "ddsip.log")

            os.makedirs(input_directory)
            assert os.path.exists(input_directory)
            assert not os.path.exists(output_directory)
            solution_filename = os.path.join(output_directory,
                                             "solution.out")

            #
            # Create the DDSIP input files
            #

            io_options = {'symbolic_solver_labels':
                          symbolic_solver_labels}

            symbol_map = pyomo.pysp.smps.smpsutils.\
                         convert_explicit(
                             input_directory,
                             "pysp_model",
                             sp,
                             core_format='lp',
                             io_options=io_options)

            self._write_config(config_filename)

            #
            # Launch DDSIP
            #

            if keep_solver_files:
                print("Solver working directory: '%s'"
                      % (working_directory))
                print("Solver log file: '%s'"
                      % (logfile))

            start = time.time()
            rc, log = pyutilib.subprocess.run(
                self.get_option("executable"),
                cwd=working_directory,
                stdin="pysp_model",
                outfile=logfile,
                tee=output_solver_log)
            stop = time.time()
            assert os.path.exists(output_directory)

            #
            # Parse the DDSIP solution
            #

            xhat, results = self._read_solution(solution_filename)

            results.solver_time = stop - start

            if symbol_map is not None:
                # load the first stage variable solution into
                # the reference model
                for symbol, varvalue in xhat.items():
                    symbol_map.bySymbol[symbol]().value = varvalue
            else:
                results.xhat = xhat

        finally:

            #
            # cleanup
            #
            pyutilib.services.TempfileManager.pop(
                remove=not keep_solver_files)

        return results

    def _write_config(self, filename):
        """ Writes an DDSIP config file """
        with open(filename, "w") as f:
            f.write('BEGIN \n\n\n')
            f.write('FIRSTCON '+str(NumberOfFirstStageConstraints)+'\n')
            f.write('FIRSTVAR '+str(NumberOfFirstStageVars)+'\n')
            f.write('SECCON '+str(NumberOfSecondStageConstraints)+'\n')
            f.write('SECVAR '+str(NumberOfSecondStageVars)+'\n')
            f.write('POSTFIX '+self._firststage_var_suffix+'\n')
            f.write('SCENAR '+str(NumberOfScenarios)+'\n')

            f.write('STOCRHS '+str(NumberOfStochasticRHS)+'\n')
            f.write('STOCCOST '+str(NumberOfStochasticCosts)+'\n')
            f.write('STOCMAT '+str(NumberOfStochasticMatrixEntries)+'\n')

            f.write("\n\nCPLEXBEGIN\n")
            f.write('1035 0 * Output on screen indicator\n')
            f.write('2008 0.001 * Absolute Gap\n')
            f.write('2009 0.001 * Relative Gap\n')
            f.write('1039 1200 * Time limit\n')
            f.write('1016 1e-9 * simplex feasibility tolerance\n')
            f.write('1014 1e-9 * simplex optimality tolerance\n')
            f.write('1065 40000 * Memory available for working storage\n')
            f.write('2010 1e-20 * integrality tolerance\n')
            f.write('2008 0 * Absolute gap\n')
            f.write('2020 0 * Priority order\n')
            f.write('2012 4 * MIP display level\n')
            f.write('2053 2 * disjunctive cuts\n')
            f.write('2040 2 * flow cover cuts\n')
            f.write('2060 3 *DiveType mip strategy dive (probe=3)\n')
            f.write('CPLEXEND\n\n')

            f.write('MAXINHERIT 15\n')
            f.write('OUTLEV 5 * Debugging\n')
            f.write('OUTFIL 2\n')
            f.write('STARTI 0  * (1 to use the starting values from PH)\n')
            f.write('NODELI 2000 * Sipdual node limit\n')
            f.write('TIMELIMIT 964000 * Sipdual time limit\n')
            f.write('HEURISTIC 99 3 7 * Heuristics: Down, Up, Near, Common, Byaverage ...(12)\n')
            f.write('ABSOLUTEGAP 0.001 * Absolute duality gap allowed in DD\n')
            f.write('EEVPROB 1\n')
            f.write('RELATIVEGAP 0.01 * (0.02) Relative duality gap allowed in DD\n')
            f.write('BRADIRECTION -1 * Branching direction in DD\n')
            f.write('BRASTRATEGY 1 * Branching strategy in DD (1 = unsolved nodes first, 0 = best bound)\n')
            f.write('EPSILON 1e-13 * Branch epsilon for cont. var.\n')
            f.write('ACCURACY 5e-16 * Accuracy\n')
            f.write('BOUSTRATEGY 1 * Bounding strategy in DD\n')
            f.write('NULLDISP 1e-16\n')
            f.write('RELAXF 0\n')
            f.write('INTFIRST 0 * Branch first on integer\n')
            f.write('HOTSTART 4 * use previous solution as integer starting info\n')

            f.write('\n\nRISKMO 0 * Risk Model\n')
            f.write('RISKALG 1\n')
            f.write('WEIGHT 1\n')
            f.write('TARGET 54 * target if needed\n')
            f.write('PROBLEV .8 * probability level\n')
            f.write('RISKBM 11000000 * big M in \n')

            f.write('\n\nCBFREQ 0 * (50) Conic Bundle in every ith node\n')
            f.write('CBITLIM 20 * (10) Descent iteration limit for conic bundle method\n')
            f.write('CBTOTITLIM 50 * (1000) Total iteration limit for conic bundle method\n')
            f.write('NONANT 1 * Non-anticipativity representation\n')
            f.write('DETEQU 1 * Write Deterministic Equivalent\n')
            f.write('\n\nEND\n')

    def _read_solution(self, filename):
        """ Parses an DDSIP solution file """

        results = SPSolverResults()
        xhat = {}
        with open(filename, 'r') as f:

            #
            # Status
            #
            results.solver_status = None

            #
            # Xhat
            #

            line = f.readline()
            while line.strip() != "1. Best Solution":
                line = f.readline()
            line = f.readline()
            assert line.startswith("Variable name                Value")
            line = f.readline()
            assert line.startswith("-----------------------------------")
            # TODO
            xhat = {}

            #
            # Objective and Bound
            #

            line = f.readline()
            while line.strip() != "2. Bounds":
                line = f.readline()
            line = f.readline()
            assert line.startswith("Scenario    Lower Bound (root)    Upper Bound")
            line = f.readline()
            assert line.startswith("------------------------------------------------")
            # TODO
            results.objective = None
            results.bound = None

        return xhat, results

def runddsip_register_options(options=None):
    if options is None:
        options = PySPConfigBlock()
    safe_register_common_option(options,
                               "verbose")
    safe_register_common_option(options,
                               "disable_gc")
    safe_register_common_option(options,
                               "profile")
    safe_register_common_option(options,
                               "traceback")
    safe_register_common_option(options,
                                "output_scenario_tree_solution")
    ScenarioTreeManagerFactory.register_options(options)
    DDSIPSolver.register_options(options)

    return options

def runddsip(options):
    """
    Construct a senario tree manager and solve it
    with the DDSIP solver.
    """
    start_time = time.time()
    with ScenarioTreeManagerFactory(options) \
         as manager:
        manager.initialize()
        print("")
        print("Running DDSIP solver for stochastic "
              "programming problems")
        ddsip = DDSIPSolver(options)
        results = ddsip.solve(manager)
        print(results)

        if options.output_scenario_tree_solution:
            print("Final solution (scenario tree format):")
            manager.scenario_tree.snapshotSolutionFromScenarios()
            manager.scenario_tree.pprintSolution()

    print("")
    print("Total execution time=%.2f seconds"
          % (time.time() - start_time))
    return 0

#
# the main driver routine
#

def main(args=None):
    #
    # Top-level command that executes everything
    #

    #
    # Import plugins
    #
    import pyomo.environ

    #
    # Parse command-line options.
    #
    try:
        options = parse_command_line(
            args,
            runddsip_register_options,
            prog='runddsip',
            description=(
"""Optimize a stochastic program using the DDSIP solver."""
            ))

    except SystemExit as _exc:
        # the parser throws a system exit if "-h" is specified
        # - catch it to exit gracefully.
        return _exc.code

    return launch_command(runddsip,
                          options,
                          error_label="runddsip: ",
                          disable_gc=options.disable_gc,
                          profile_count=options.profile,
                          traceback=options.traceback)

@pyomo_command('runddsip', 'Run the DDSIP solver')
def RunDDSIP_main(args=None):
    return main(args=args)

SPSolverFactory.register_solver("ddsip", DDSIPSolver)

if __name__ == "__main__":
    sys.exit(RunDDSIP_main())
