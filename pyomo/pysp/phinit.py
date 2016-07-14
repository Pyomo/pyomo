#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

import gc
import sys
import time
import contextlib
import random
import argparse
try:
    from guppy import hpy
    guppy_available = True
except ImportError:
    guppy_available = False
try:
    from pympler.muppy import muppy
    from pympler.muppy import summary
    from pympler.muppy import tracker
    from pympler.asizeof import *
    pympler_available = True
except ImportError:
    pympler_available = False
except AttributeError:
    pympler_available = False

from pyutilib.pyro import shutdown_pyro_components
from pyutilib.misc import import_file

from pyomo.util import pyomo_command
from pyomo.util.plugin import ExtensionPoint
from pyomo.core.base import maximize, minimize, Var, Suffix
from pyomo.opt.base import SolverFactory
from pyomo.opt.parallel import SolverManagerFactory
from pyomo.opt import undefined
from pyomo.pysp.phextension import IPHExtension
from pyomo.pysp.ef_writer_script import ExtensiveFormAlgorithm
from pyomo.pysp.ph import ProgressiveHedging
from pyomo.pysp.phutils import (reset_nonconverged_variables,
                                reset_stage_cost_variables,
                                _OLD_OUTPUT)
from pyomo.pysp.scenariotree.instance_factory import \
    ScenarioTreeInstanceFactory
from pyomo.pysp.solutionwriter import ISolutionWriterExtension
from pyomo.pysp.util.misc import (launch_command,
                                  load_extensions)
import pyomo.pysp.phsolverserverutils

#
# utility method to construct an option parser for ph arguments,
# to be supplied as an argument to the runph method.
#

def construct_ph_options_parser(usage_string):

    solver_list = SolverFactory.services()
    solver_list = sorted( filter(lambda x: '_' != x[0], solver_list) )
    solver_help = \
    "Specify the solver with which to solve scenario sub-problems.  The "      \
    "following solver types are currently supported: %s; Default: cplex"
    solver_help %= ', '.join( solver_list )

    parser = argparse.ArgumentParser()
    parser.usage = usage_string

    # NOTE: these groups should eventually be queried from
    # the PH, scenario tree, etc. classes (to facilitate re-use)
    inputOpts        = parser.add_argument_group('Input Options')
    scenarioTreeOpts = parser.add_argument_group('Scenario Tree Options')
    phOpts           = parser.add_argument_group('PH Options')
    solverOpts       = parser.add_argument_group('Solver Options')
    postprocessOpts  = parser.add_argument_group('Postprocessing Options')
    outputOpts       = parser.add_argument_group('Output Options')
    otherOpts        = parser.add_argument_group('Other Options')

    inputOpts.add_argument('-m','--model-directory',
      help='The directory in which all model (reference and scenario) definitions are stored. Default is ".".',
      action="store",
      dest="model_directory",
      type=str,
      default=".")
    inputOpts.add_argument('-i','--instance-directory',
      help='The directory in which all instance (reference and scenario) definitions are stored. This option is required if no callback is found in the model file.',
      action="store",
      dest="instance_directory",
      type=str,
      default=None)

    def _objective_sense_type(val):
        if val in ('min','minimize',minimize):
            return minimize
        elif val in ('max','maximize',maximize):
            return maximize
        else:
            return None
    inputOpts.add_argument('-o','--objective-sense-stage-based',
      help='The objective sense to use for the auto-generated scenario instance objective, which is equal to the '
           'sum of the scenario-tree stage costs. Default is None, indicating an Objective has been declared on the '
           'reference model.',
      dest="objective_sense",
      type=_objective_sense_type,
      choices=[maximize,'max','maximize',minimize,'min','minimize',None],
      default=None)
    inputOpts.add_argument('--ph-warmstart-file',
      help="Disable iteration 0 solves and warmstarts rho, weight, and xbar parameters from solution file.",
      action="store",
      dest="ph_warmstart_file",
      type=str,
      default=None)
    inputOpts.add_argument('--ph-warmstart-index',
      help="Indicates the index (ph iteration) of the warmstart that should be loaded from a ph history file.",
      action="store",
      dest="ph_warmstart_index",
      type=str,
      default=None)
    inputOpts.add_argument('--bounds-cfgfile',
      help="The name of python script containing a ph_boundsetter_callback function to compute and update scenario variable bounds. Default is None.",
      action="store",
      dest="bounds_cfgfile",
      default=None)

    scenarioTreeOpts.add_argument('--scenario-tree-seed',
      help="The random seed associated with manipulation operations on the scenario tree (e.g., down-sampling or bundle creation). Default is None, indicating unassigned.",
      action="store",
      dest="scenario_tree_random_seed",
      type=int,
      default=random.getrandbits(100))
    scenarioTreeOpts.add_argument('--scenario-tree-downsample-fraction',
      help="The proportion of the scenarios in the scenario tree that are actually used. Specific scenarios are selected at random. Default is 1.0, indicating no down-sampling.",
      action="store",
      dest="scenario_tree_downsample_fraction",
      type=float,
      default=1.0)
    scenarioTreeOpts.add_argument('--scenario-bundle-specification',
      help="The name of the scenario bundling specification to be used when executing Progressive Hedging. Default is None, indicating no bundling is employed. If the specified name ends with a .dat suffix, the argument is interpreted as a filename. Otherwise, the name is interpreted as a file in the instance directory, constructed by adding the .dat suffix automatically",
      action="store",
      dest="scenario_bundle_specification",
      default=None)
    scenarioTreeOpts.add_argument('--create-random-bundles',
      help="Specification to create the indicated number of random, equally-sized (to the degree possible) scenario bundles. Default is 0, indicating disabled.",
      action="store",
      dest="create_random_bundles",
      type=int,
      default=None)

    phOpts.add_argument('-r','--default-rho',
      help="The default (global) rho for all blended variables. *** Required ***",
      action="store",
      dest="default_rho",
      type=str,
      default="")
    phOpts.add_argument("--xhat-method",
      help="Specify the method used to compute a bounding solution at PH termination. Defaults to 'closest-scenario'. Other variants are: 'voting' and 'rounding'",
      action="store",
      dest="xhat_method",
      type=str,
      default="closest-scenario")
    phOpts.add_argument("--disable-xhat-computation",
      help="Disable computation of xhat at the conclusion of a PH run. Useful *only* when diagnosing PH convergence, as disabling means the solution at converence is not a non-anticipative solution.",
      action="store_true",
      dest="disable_xhat_computation",
      default=False)
    phOpts.add_argument("--overrelax",
      help="Compute weight updates using combination of previous and current variable averages",
      action="store_true",
      dest="overrelax",
      default=False)
    phOpts.add_argument("--nu",
      action="store",
      dest="nu",
      type=float,
      default=1.5)
    phOpts.add_argument("--async",
      help="Run PH in asychronous mode after iteration 0. Default is False.",
      action="store_true",
      dest="async",
      default=False)
    phOpts.add_argument("--async-buffer-length",
      help="Number of scenarios to collect, if in async mode, before doing statistics and weight updates. Default is 1.",
      action="store",
      dest="async_buffer_length",
      type=int,
      default=1)
    phOpts.add_argument('--rho-cfgfile',
      help="The name of python script containing a ph_rhosetter_callback function to compute and update PH rho values. Default is None.",
      action="store",
      dest="rho_cfgfile",
      type=str,
      default=None)
    phOpts.add_argument('--aggregate-cfgfile',
      help="The name of python script containing a ph_aggregategetter_callback function to collect and store aggregate scenario data on PH. Default is None.",
      action="store",
      dest="aggregate_cfgfile",
      type=str,
      default=None)
    phOpts.add_argument('--max-iterations',
      help="The maximum number of PH iterations. Default is 100.",
      action="store",
      dest="max_iterations",
      type=int,
      default=100)
    phOpts.add_argument('--termdiff-threshold',
      help="The convergence threshold used in the term-diff and normalized term-diff convergence criteria. Default is 0.0001.",
      action="store",
      dest="termdiff_threshold",
      type=float,
      default=0.0001)
    phOpts.add_argument('--enable-free-discrete-count-convergence',
      help="Terminate PH based on the free discrete variable count convergence metric. Default is False.",
      action="store_true",
      dest="enable_free_discrete_count_convergence",
      default=False)
    phOpts.add_argument('--free-discrete-count-threshold',
      help="The convergence threshold used in the criterion based on when the free discrete variable count convergence criterion. Default is 20.",
      action="store",
      dest="free_discrete_count_threshold",
      type=float,
      default=20)
    phOpts.add_argument('--enable-normalized-termdiff-convergence',
      help="Terminate PH based on the normalized termdiff convergence metric. Default is True.",
      action="store_true",
      dest="enable_normalized_termdiff_convergence",
      default=True)
    phOpts.add_argument('--enable-termdiff-convergence',
      help="Terminate PH based on the termdiff convergence metric. Default is False.",
      action="store_true",
      dest="enable_termdiff_convergence",
      default=False)
    phOpts.add_argument('--enable-outer-bound-convergence',
      help="Terminate PH based on the outer bound convergence metric. Default is False.",
      action="store_true",
      dest="enable_outer_bound_convergence",
      default=False)
    phOpts.add_argument('--enable-primal-dual-residual-convergence',
      help="Terminate PH based on the primal-dual residual convergence. Default is False.",
      action="store_true",
      dest="enable_primal_dual_residual_convergence",
      default=False)
    phOpts.add_argument('--outer-bound-convergence-threshold',
      help="The convergence threshold used in the outer bound convergerence criterion. Default is None, indicating unassigned",
      action="store",
      dest="outer_bound_convergence_threshold",
      type=float,
      default=None)
    phOpts.add_argument('--primal-dual-residual-convergence-threshold',
      help="The convergence threshold used in the primal-dual residual convergerence criterion. Default is 0.0001.",
      action="store",
      dest="primal_dual_residual_convergence_threshold",
      type=float,
      default=0.0001)
    phOpts.add_argument('--linearize-nonbinary-penalty-terms',
      help="Approximate the PH quadratic term for non-binary variables with a piece-wise linear function, using the supplied number of equal-length pieces from each bound to the average",
      action="store",
      dest="linearize_nonbinary_penalty_terms",
      type=int,
      default=0)
    phOpts.add_argument('--breakpoint-strategy',
      help="Specify the strategy to distribute breakpoints on the [lb, ub] interval of each variable when linearizing. 0 indicates uniform distribution. 1 indicates breakpoints at the node min and max, uniformly in-between. 2 indicates more aggressive concentration of breakpoints near the observed node min/max.",
      action="store",
      dest="breakpoint_strategy",
      type=int,
      default=0)
    phOpts.add_argument('--retain-quadratic-binary-terms',
      help="Do not linearize PH objective terms involving binary decision variables",
      action="store_true",
      dest="retain_quadratic_binary_terms",
      default=False)
    phOpts.add_argument('--drop-proximal-terms',
      help="Eliminate proximal terms (i.e., the quadratic penalty terms) from the weighted PH objective. Default is False.",
      action="store_true",
      dest="drop_proximal_terms",
      default=False)
    phOpts.add_argument('--enable-ww-extensions',
      help="Enable the Watson-Woodruff PH extensions plugin. Default is False.",
      action="store_true",
      dest="enable_ww_extensions",
      default=False)
    phOpts.add_argument('--ww-extension-cfgfile',
      help="The name of a configuration file for the Watson-Woodruff PH extensions plugin.",
      action="store",
      dest="ww_extension_cfgfile",
      type=str,
      default="")
    phOpts.add_argument('--ww-extension-suffixfile',
      help="The name of a variable suffix file for the Watson-Woodruff PH extensions plugin.",
      action="store",
      dest="ww_extension_suffixfile",
      type=str,
      default="")
    phOpts.add_argument('--ww-extension-annotationfile',
      help="The name of a variable annotation file for the Watson-Woodruff PH extensions plugin.",
      action="store",
      dest="ww_extension_annotationfile",
      type=str,
      default="")
    phOpts.add_argument('--user-defined-extension',
      help="The name of a python module specifying a user-defined PH extension plugin.",
      action="append",
      dest="user_defined_extensions",
      type=str,
      default=[])
    phOpts.add_argument('--preprocess-fixed-variables',
      help="Preprocess fixed/freed variables in scenario instances, rather than write them to solver plugins. Default is False.",
      action="store_false",
      dest="write_fixed_variables",
      default=True)

    solverOpts.add_argument('--scenario-mipgap',
      help="Specifies the mipgap for all PH scenario sub-problems",
      action="store",
      dest="scenario_mipgap",
      type=float,
      default=None)
    solverOpts.add_argument('--scenario-solver-options',
      help="Solver options for all PH scenario sub-problems",
      action="append",
      dest="scenario_solver_options",
      default=[])
    solverOpts.add_argument('--solver',
      help=solver_help,
      action="store",
      dest="solver_type",
      type=str,
      default="cplex")
    solverOpts.add_argument('--solver-io',
      help='The type of IO used to execute the solver.  Different solvers support different types of IO, but the following are common options: lp - generate LP files, nl - generate NL files, python - direct Python interface, os - generate OSiL XML files.',
      action='store',
      dest='solver_io',
      default=None)
    solverOpts.add_argument('--solver-manager',
      help="The type of solver manager used to coordinate scenario sub-problem solves. Default is serial.",
      action="store",
      dest="solver_manager_type",
      type=str,
      default="serial")
    solverOpts.add_argument('--pyro-host',
      help="The hostname to bind on when searching for a Pyro nameserver.",
      action="store",
      dest="pyro_host",
      default=None)
    solverOpts.add_argument('--pyro-port',
      help="The port to bind on when searching for a Pyro nameserver.",
      action="store",
      dest="pyro_port",
      type=int,
      default=None)
    solverOpts.add_argument('--handshake-with-phpyro',
      help="When updating weights, xbars, and rhos across the PHPyro solver manager, it is often expedient to ignore the simple acknowledgement results returned by PH solver servers. Enabling this option instead enables hand-shaking, to ensure message receipt. Clearly only makes sense if the PHPyro solver manager is selected",
      action="store_true",
      dest="handshake_with_phpyro",
      default=False)
    solverOpts.add_argument('--phpyro-required-workers',
      help="Set the number of idle phsolverserver worker processes expected to be available when the PHPyro solver manager is selected. This option should be used when the number of worker threads is less than the total number of scenarios (or bundles). When this option is not used, PH will attempt to assign each scenario (or bundle) to a single phsolverserver until the timeout indicated by the --phpyro-workers-timeout option occurs.",
      action="store",
      type=int,
      dest="phpyro_required_workers",
      default=None)
    solverOpts.add_argument('--phpyro-workers-timeout',
     help="Set the time limit (seconds) for finding idle phsolverserver worker processes to be used when the PHPyro solver manager is selected. This option is ignored when --phpyro-required-workers is set manually. Default is 30.",
      action="store",
      type=float,
      dest="phpyro_workers_timeout",
      default=30)
    solverOpts.add_argument('--phpyro-transmit-leaf-stage-variable-solution',
      help="By default, when running PH using the PHPyro solver manager, leaf-stage variable solutions are not transmitted back to the master PH instance during intermediate PH iterations. This flag will override that behavior for the rare cases where these values are needed. Using this option will possibly have a negative impact on runtime for PH iterations. When PH exits, variable values are collected from all stages whether or not this option was used. Also, note that PH extensions have the ability to override this flag at runtime.",
      action="store_true",
      dest="phpyro_transmit_leaf_stage_solution",
      default=False)
    solverOpts.add_argument('--disable-warmstarts',
      help="Disable warm-start of scenario sub-problem solves in PH iterations >= 1. Default is False.",
      action="store_true",
      dest="disable_warmstarts",
      default=False)
    solverOpts.add_argument('--shutdown-pyro',
      help="Shut down all Pyro-related components associated with the Pyro and PH Pyro solver managers (if specified), including the dispatch server, name server, and any solver servers. Default is False.",
      action="store_true",
      dest="shutdown_pyro",
      default=False)
    solverOpts.add_argument('--shutdown-pyro-workers',
      help="Shut down PH solver servers on exit, leaving dispatcher and nameserver running. Default is False.",
      action="store_true",
      dest="shutdown_pyro_workers",
      default=False)

    ef_options = ExtensiveFormAlgorithm.register_options(prefix="ef_")
    ef_options.initialize_argparse(parser)
    # temporary hack
    parser._ef_options = ef_options
    postprocessOpts.add_argument('--ef-output-file',
      help=("The name of the extensive form output file "
            "(currently LP, MPS, and NL file formats are "
            "supported). If the option value does not end "
            "in '.lp', '.mps', or '.nl', then the output format "
            "will be inferred from the settings for the --solver "
            "and --solver-io options, and the appropriate suffix "
            "will be appended to the name. Default is 'efout'."),
      action="store",
      dest="ef_output_file",
      type=str,
      default="efout")
    postprocessOpts.add_argument('--write-ef',
      help="Upon termination, create the extensive-form model and write it - accounting for all fixed variables. See --ef-output-file",
      action="store_true",
      dest="write_ef",
      default=False)
    postprocessOpts.add_argument('--solve-ef',
      help="Upon termination, create the extensive-form model and solve it - accounting for all fixed variables.",
      action="store_true",
      dest="solve_ef",
      default=False)
    postprocessOpts.add_argument('--ef-solution-writer',
      help="The plugin invoked to write the scenario tree solution following the EF solve. If specified, overrides the runph option of the same name; otherwise, the runph option value will be used.",
      action="append",
      dest="ef_solution_writer",
      type=str,
      default = [])


    outputOpts.add_argument('--output-scenario-tree-solution',
      help="If a feasible solution can be found, report it (even leaves) in scenario tree format upon termination. Default is False.",
      action="store_true",
      dest="output_scenario_tree_solution",
      default=False)
    outputOpts.add_argument('--output-solver-logs',
      help="Output solver logs during scenario sub-problem solves",
      action="store_true",
      dest="output_solver_log",
      default=False)
    outputOpts.add_argument('--symbolic-solver-labels',
      help='When interfacing with the solver, use symbol names derived from the model. For example, \"my_special_variable[1_2_3]\" instead of \"v1\". Useful for debugging. When using the ASL interface (--solver-io=nl), generates corresponding .row (constraints) and .col (variables) files. The ordering in these files provides a mapping from ASL index to symbolic model names.',
      action='store_true',
      dest='symbolic_solver_labels',
      default=False)
    outputOpts.add_argument('--output-solver-results',
      help="Output solutions obtained after each scenario sub-problem solve",
      action="store_true",
      dest="output_solver_results",
      default=False)
    outputOpts.add_argument('--output-times',
      help="Output timing statistics for various PH components",
      action="store_true",
      dest="output_times",
      default=False)
    outputOpts.add_argument('--output-instance-construction-time',
      help="Output timing statistics for instance construction (client-side only when using PHPyro",
      action="store_true",
      dest="output_instance_construction_time",
      default=False)
    outputOpts.add_argument('--report-only-statistics',
      help="When reporting solutions (if enabled), only output per-variable statistics - not the individual scenario values. Default is False.",
      action="store_true",
      dest="report_only_statistics",
      default=False)
    outputOpts.add_argument('--report-solutions',
      help="Always report PH solutions after each iteration. Enabled if --verbose is enabled. Default is False.",
      action="store_true",
      dest="report_solutions",
      default=False)
    outputOpts.add_argument('--report-weights',
      help="Always report PH weights prior to each iteration. Enabled if --verbose is enabled. Default is False.",
      action="store_true",
      dest="report_weights",
      default=False)
    outputOpts.add_argument('--report-rhos-all-iterations',
      help="Always report PH rhos prior to each iteration. Default is False.",
      action="store_true",
      dest="report_rhos_each_iteration",
      default=False)
    outputOpts.add_argument('--report-rhos-first-iterations',
      help="Report rhos prior to PH iteration 1. Enabled if --verbose is enabled. Default is False.",
      action="store_true",
      dest="report_rhos_first_iteration",
      default=False)
    outputOpts.add_argument('--report-for-zero-variable-values',
      help="Report statistics (variables and weights) for all variables, not just those with values differing from 0. Default is False.",
      action="store_true",
      dest="report_for_zero_variable_values",
      default=False)
    outputOpts.add_argument('--report-only-nonconverged-variables',
      help="Report statistics (variables and weights) only for non-converged variables. Default is False.",
      action="store_true",
      dest="report_only_nonconverged_variables",
      default=False)
    outputOpts.add_argument('--solution-writer',
      help="The plugin invoked to write the scenario tree solution. Defaults to the empty list.",
      action="append",
      dest="solution_writer",
      type=str,
      default = [])
    outputOpts.add_argument('--suppress-continuous-variable-output',
      help="Eliminate PH-related output involving continuous variables.",
      action="store_true",
      dest="suppress_continuous_variable_output",
      default=False)
    outputOpts.add_argument('--verbose',
      help="Generate verbose output for both initialization and execution. Default is False.",
      action="store_true",
      dest="verbose",
      default=False)

    otherOpts.add_argument('--disable-gc',
      help="Disable the python garbage collecter. Default is False.",
      action="store_true",
      dest="disable_gc",
      default=False)
    if pympler_available or guppy_available:
        otherOpts.add_argument("--profile-memory",
                             help="If Pympler or Guppy is available (installed), report memory usage statistics for objects created after each PH iteration. A value of 0 indicates disabled. A value of 1 forces summary output after each PH iteration >= 1. Values greater than 2 are currently not supported.",
                             action="store",
                             dest="profile_memory",
                             type=int,
                             default=0)
    otherOpts.add_argument('-k','--keep-solver-files',
      help="Retain temporary input and output files for scenario sub-problem solves",
      action="store_true",
      dest="keep_solver_files",
      default=False)
    otherOpts.add_argument('--profile',
      help="Enable profiling of Python code.  The value of this option is the number of functions that are summarized.",
      action="store",
      dest="profile",
      type=int,
      default=0)
    otherOpts.add_argument('--traceback',
      help="When an exception is thrown, show the entire call stack. Ignored if profiling is enabled. Default is False.",
      action="store_true",
      dest="traceback",
      default=False)
    otherOpts.add_argument('--compile-scenario-instances',
      help="Replace all linear constraints on scenario instances with a more memory efficient sparse matrix representation. Default is False.",
      action="store_true",
      dest="compile_scenario_instances",
      default=False)

    #
    # Hacks to register plugin options until things move over to the
    # new scripting interface
    #
    otherOpts.add_argument('--activate-jsonio-solution-saver',
                         help=("Activate the jsonio IPySPSolutionSaverExtension. Stores "
                               "scenario tree node solution in form that can be reloaded "
                               "for evaluation on other scenario trees"),
                         action='store_true',
                         dest='activate_jsonio_solution_saver',
                         default=False)
    otherOpts.add_argument('--jsonsaver-output-name',
                         help=("The directory or filename where the scenario tree solution "
                               "should be saved to."),
                         action='store',
                         dest='jsonsaver_output_name',
                         type=str,
                         default=None)
    otherOpts.add_argument('--jsonsaver-save-stages',
                         help=("The number of scenario tree stages to store for the solution. "
                               "The default value of 0 indicates that all stages should be stored."),
                         action='store',
                         dest='jsonsaver_save_stages',
                         type=int,
                         default=0)

    return parser


def PH_DefaultOptions():
    parser = construct_ph_options_parser("")
    options = parser.parse_args([''])
    # temporary hack
    options._ef_options = parser._ef_options
    options._ef_options.import_argparse(options)
    return options

#
# Import the scenario tree and model data using a
# PH options dictionary
#

def GenerateScenarioTreeForPH(options,
                              scenario_instance_factory,
                              include_scenarios=None):

    scenario_tree = scenario_instance_factory.generate_scenario_tree(
        include_scenarios=include_scenarios,
        downsample_fraction=options.scenario_tree_downsample_fraction,
        bundles=options.scenario_bundle_specification,
        random_bundles=options.create_random_bundles,
        random_seed=options.scenario_tree_random_seed,
        verbose=options.verbose)

    #
    # print the input tree for validation/information purposes.
    #
    if options.verbose:
        scenario_tree.pprint()

    #
    # validate the tree prior to doing anything serious
    #
    if not scenario_tree.validate():
        raise RuntimeError("Scenario tree is invalid")
    else:
        if options.verbose:
            print("Scenario tree is valid!")

    if options.solver_manager_type != "phpyro":

        start_time = time.time()

        if not _OLD_OUTPUT:
            print("Constructing scenario tree instances")
        instance_dictionary = \
            scenario_instance_factory.construct_instances_for_scenario_tree(
                scenario_tree,
                output_instance_construction_time=options.output_instance_construction_time,
                compile_scenario_instances=options.compile_scenario_instances,
                verbose=options.verbose)

        if options.verbose or options.output_times:
            print("Time to construct scenario instances=%.2f seconds"
                  % (time.time() - start_time))

        if not _OLD_OUTPUT:
            print("Linking instances into scenario tree")
        start_time = time.time()

        # with the scenario instances now available, link the
        # referenced objects directly into the scenario tree.
        scenario_tree.linkInInstances(instance_dictionary,
                                      objective_sense=options.objective_sense,
                                      create_variable_ids=True)

        if options.verbose or options.output_times:
            print("Time link scenario tree with instances=%.2f seconds"
                  % (time.time() - start_time))

    return scenario_tree

#
# Create a PH object from scratch using
# the options object.
#

def PHAlgorithmBuilder(options, scenario_tree):

    import pyomo.environ
    import pyomo.solvers.plugins.smanager.phpyro
    import pyomo.solvers.plugins.smanager.pyro

    solution_writer_plugins = ExtensionPoint(ISolutionWriterExtension)
    for plugin in solution_writer_plugins:
        plugin.disable()

    solution_plugins = []
    if len(options.solution_writer) > 0:
        for this_extension in options.solution_writer:
            if this_extension in sys.modules:
                print("User-defined PH solution writer module="
                      +this_extension+" already imported - skipping")
            else:
                print("Trying to import user-defined PH "
                      "solution writer module="+this_extension)
                # make sure "." is in the PATH.
                original_path = list(sys.path)
                sys.path.insert(0,'.')
                import_file(this_extension)
                print("Module successfully loaded")
                sys.path[:] = original_path # restore to what it was

            # now that we're sure the module is loaded, re-enable this
            # specific plugin.  recall that all plugins are disabled
            # by default in phinit.py, for various reasons. if we want
            # them to be picked up, we need to enable them explicitly.
            import inspect
            module_to_find = this_extension
            if module_to_find.rfind(".py"):
                module_to_find = module_to_find.rstrip(".py")
            if module_to_find.find("/") != -1:
                module_to_find = module_to_find.split("/")[-1]

            for name, obj in inspect.getmembers(sys.modules[module_to_find],
                                                inspect.isclass):
                import pyomo.util
                # the second condition gets around goofyness related
                # to issubclass returning True when the obj is the
                # same as the test class.
                if issubclass(obj, pyomo.util.plugin.SingletonPlugin) and name != "SingletonPlugin":
                    for plugin in solution_writer_plugins(all=True):
                        if isinstance(plugin, obj):
                            plugin.enable()
                            solution_plugins.append(plugin)

    #
    # if any of the ww extension configuration options are specified
    # without the ww extension itself being enabled, halt and warn the
    # user - this has led to confusion in the past, and will save user
    # support time.
    #
    if (len(options.ww_extension_cfgfile) > 0) and \
       (options.enable_ww_extensions is False):
        raise ValueError("A configuration file was specified "
                         "for the WW extension module, but the WW extensions "
                         "are not enabled!")

    if (len(options.ww_extension_suffixfile) > 0) and \
       (options.enable_ww_extensions is False):
        raise ValueError("A suffix file was specified for the WW "
                         "extension module, but the WW extensions are not "
                         "enabled!")

    if (len(options.ww_extension_annotationfile) > 0) and \
       (options.enable_ww_extensions is False):
        raise ValueError("A annotation file was specified for the "
                         "WW extension module, but the WW extensions are not "
                         "enabled!")

    #
    # disable all plugins up-front. then, enable them on an as-needed
    # basis later in this function. the reason that plugins should be
    # disabled is that they may have been programmatically enabled in
    # a previous run of PH, and we want to start from a clean slate.
    #
    ph_extension_point = ExtensionPoint(IPHExtension)

    for plugin in ph_extension_point:
        plugin.disable()

    ph_plugins = []
    #
    # deal with any plugins. ww extension comes first currently,
    # followed by an option user-defined plugin.  order only matters
    # if both are specified.
    #
    if options.enable_ww_extensions:

        import pyomo.pysp.plugins.wwphextension

        # explicitly enable the WW extension plugin - it may have been
        # previously loaded and/or enabled.
        ph_extension_point = ExtensionPoint(IPHExtension)

        for plugin in ph_extension_point(all=True):
           if isinstance(plugin,
                         pyomo.pysp.plugins.wwphextension.wwphextension):

              plugin.enable()
              ph_plugins.append(plugin)

              # there is no reset-style method for plugins in general,
              # or the ww ph extension in plugin in particular. if no
              # configuration or suffix filename is specified, set to
              # None so that remnants from the previous use of the
              # plugin aren't picked up.
              if len(options.ww_extension_cfgfile) > 0:
                 plugin._configuration_filename = options.ww_extension_cfgfile
              else:
                 plugin._configuration_filename = None
              if len(options.ww_extension_suffixfile) > 0:
                 plugin._suffix_filename = options.ww_extension_suffixfile
              else:
                 plugin._suffix_filename = None
              if len(options.ww_extension_annotationfile) > 0:
                 plugin._annotation_filename = options.ww_extension_annotationfile
              else:
                 plugin._annotation_filename = None

    if len(options.user_defined_extensions) > 0:
        for this_extension in options.user_defined_extensions:
            if this_extension in sys.modules:
                print("User-defined PH extension module="
                      +this_extension+" already imported - skipping")
            else:
                print("Trying to import user-defined PH extension module="
                      +this_extension)
                # make sure "." is in the PATH.
                original_path = list(sys.path)
                sys.path.insert(0,'.')
                import_file(this_extension)
                print("Module successfully loaded")
                # restore to what it was
                sys.path[:] = original_path

            # now that we're sure the module is loaded, re-enable this
            # specific plugin.  recall that all plugins are disabled
            # by default in phinit.py, for various reasons. if we want
            # them to be picked up, we need to enable them explicitly.
            import inspect
            module_to_find = this_extension
            if module_to_find.rfind(".py"):
                module_to_find = module_to_find.rstrip(".py")
            if module_to_find.find("/") != -1:
                module_to_find = module_to_find.split("/")[-1]

            for name, obj in inspect.getmembers(sys.modules[module_to_find],
                                                inspect.isclass):
                import pyomo.util
                # the second condition gets around goofyness related
                # to issubclass returning True when the obj is the
                # same as the test class.
                if issubclass(obj, pyomo.util.plugin.SingletonPlugin) and name != "SingletonPlugin":
                    ph_extension_point = ExtensionPoint(IPHExtension)
                    for plugin in ph_extension_point(all=True):
                        if isinstance(plugin, obj):
                            plugin.enable()
                            ph_plugins.append(plugin)

    ph = None
    solver_manager = None
    try:

        # construct the solver manager.
        if options.verbose:
            print("Constructing solver manager of type="
                  +options.solver_manager_type)
        solver_manager = SolverManagerFactory(
            options.solver_manager_type,
            host=options.pyro_host,
            port=options.pyro_port)

        if solver_manager is None:
            raise ValueError("Failed to create solver manager of "
                             "type="+options.solver_manager_type+
                         " specified in call to PH constructor")

        ph = ProgressiveHedging(options)

        if isinstance(solver_manager,
                      pyomo.solvers.plugins.smanager.phpyro.SolverManager_PHPyro):

            if scenario_tree.contains_bundles():
                num_jobs = len(scenario_tree._scenario_bundles)
                if not _OLD_OUTPUT:
                    print("Bundle solver jobs available: "+str(num_jobs))
            else:
                num_jobs = len(scenario_tree._scenarios)
                if not _OLD_OUTPUT:
                    print("Scenario solver jobs available: "+str(num_jobs))

            servers_expected = options.phpyro_required_workers
            if (servers_expected is None):
                servers_expected = num_jobs

            timeout = options.phpyro_workers_timeout if \
                      (options.phpyro_required_workers is None) else \
                      None

            solver_manager.acquire_servers(servers_expected,
                                           timeout)

        ph.initialize(scenario_tree=scenario_tree,
                      solver_manager=solver_manager,
                      ph_plugins=ph_plugins,
                      solution_plugins=solution_plugins)

    except:

        if ph is not None:

            ph.release_components()

        if solver_manager is not None:

            if isinstance(solver_manager,
                          pyomo.solvers.plugins.smanager.phpyro.SolverManager_PHPyro):
                solver_manager.release_servers(shutdown=ph._shutdown_pyro_workers)
            elif isinstance(solver_manager,
                          pyomo.solvers.plugins.smanager.pyro.SolverManager_Pyro):
                if ph._shutdown_pyro_workers:
                    solver_manager.shutdown_workers()
            solver_manager.deactivate()

        print("Failed to initialize progressive hedging algorithm")
        raise

    return ph

def PHFromScratch(options):
    start_time = time.time()
    if options.verbose:
        print("Importing model and scenario tree files")

    scenario_instance_factory = \
        ScenarioTreeInstanceFactory(options.model_directory,
                                    options.instance_directory)

    if options.verbose or options.output_times:
        print("Time to import model and scenario tree "
              "structure files=%.2f seconds"
              %(time.time() - start_time))

    try:

        scenario_tree = \
            GenerateScenarioTreeForPH(options,
                                      scenario_instance_factory)

    except:
        print("Failed to initialize model and/or scenario tree data")
        scenario_instance_factory.close()
        raise

    ph = None
    try:
        ph = PHAlgorithmBuilder(options, scenario_tree)
    except:

        print("A failure occurred in PHAlgorithmBuilder. Cleaning up...")
        if ph is not None:
            ph.release_components()
        scenario_instance_factory.close()
        raise

    return ph

#
# There is alot of cleanup that should be be done before a ph object
# goes out of scope (e.g.  releasing PHPyro workers, closing file
# archives, etc.). However, many of the objects requiring context
# management serve a purpose beyond the lifetime of the
# ProgressiveHedging object that references them. This function
# assumes the user does not care about this and performs all necessary
# cleanup when we exit the scope of the 'with' block. Example:
#
# with PHFromScratchManagedContext(options) as ph:
#    ph.run()
#

@contextlib.contextmanager
def PHFromScratchManagedContext(options):

    ph = None
    try:
        ph = PHFromScratch(options)
        yield ph

    finally:

        PHCleanup(ph)

def PHCleanup(ph):

    if ph is None:

        return

    ph.release_components()

    if ph._solver_manager is not None:

        import pyomo.environ
        import pyomo.solvers.plugins.smanager.phpyro
        import pyomo.solvers.plugins.smanager.pyro

        if isinstance(ph._solver_manager,
                      pyomo.solvers.plugins.smanager.phpyro.SolverManager_PHPyro):
            ph._solver_manager.release_servers(shutdown=ph._shutdown_pyro_workers)
        elif isinstance(ph._solver_manager,
                        pyomo.solvers.plugins.smanager.pyro.SolverManager_Pyro):
            if ph._shutdown_pyro_workers:
                ph._solver_manager.shutdown_workers()
        ph._solver_manager.deactivate()

    if ph._scenario_tree is not None:

        if ph._scenario_tree._scenario_instance_factory is not None:

            ph._scenario_tree._scenario_instance_factory.close()

#
# Given a PH object, execute it and optionally solve the EF at the
# end.
#

def run_ph(options, ph):

    import pyomo.environ
    import pyomo.solvers.plugins.smanager.phpyro
    import pyomo.solvers.plugins.smanager.pyro

    start_time = time.time()

    #
    # kick off the solve
    #
    retval = ph.solve()
    if retval is not None:
        # assume something else wrote out the list of scenarios
        raise RuntimeError("Failure Encountered")

    end_ph_time = time.time()

    print("")
    print("Total PH execution time=%.2f seconds" %(end_ph_time - start_time))
    print("")
    if options.output_times:
        ph.print_time_stats()

    ph.save_solution()

    # Another hack to execute the jsonio saver plugin until
    # we move over to the new scripting interface
    if options.activate_jsonio_solution_saver:
        print("Executing jsonio solution saver extension")
        import pyomo.pysp.plugins.jsonio
        jsonsaver = pyomo.pysp.plugins.jsonio.JSONSolutionSaverExtension()
        jsonsaver_options = jsonsaver.register_options()
        jsonsaver_options.jsonsaver_output_name = \
            options.jsonsaver_output_name
        jsonsaver_options.jsonsaver_save_stages = \
            options.jsonsaver_save_stages
        jsonsaver.set_options(jsonsaver_options)
        jsonsaver.save(ph)

    ef_solution_writers = ()
    if len(options.ef_solution_writer) > 0:
        ef_solution_writers = load_extensions(
            options.ef_solution_writer,
            ISolutionWriterExtension)
    else:
        # inherit the PH solution writer(s)
        ef_solution_writers = ph._solution_plugins

    #
    # create the extensive form binding instance, so that we can
    # either write or solve it (if specified).
    #
    ef = None
    if (options.write_ef) or (options.solve_ef):

        if not isinstance(ph._solver_manager,
                          pyomo.solvers.plugins.smanager.\
                          phpyro.SolverManager_PHPyro):

            # The instances are about to be added as sublocks to the
            # extensive form instance. If bundles exist, we must
            # distroy them to avoid errors
            ph._destory_bundle_binding_instances()

        else:

            print("Constructing scenario instances for extensive form solve")

            instances = ph._scenario_tree._scenario_instance_factory.\
                        construct_instances_for_scenario_tree(
                            ph._scenario_tree,
                            output_instance_construction_time=\
                              ph._output_instance_construction_time,
                            verbose=options.verbose)

            ph._scenario_tree.linkInInstances(
                instances,
                create_variable_ids=False,
                master_scenario_tree=ph._scenario_tree,
                initialize_solution_data=False)

            ph._setup_scenario_instances()

            if options.verbose:
                print("Creating deterministic SymbolMaps for scenario instances")
            scenario_ph_symbol_maps_start_time = time.time()
            # Define for what components we generate symbols
            symbol_ctypes = (Var, Suffix)
            ph._create_instance_symbol_maps(symbol_ctypes)
            scenario_ph_symbol_maps_end_time = time.time()
            if options.output_times:
                print("PH SymbolMap creation time=%.2f seconds"
                      % (scenario_ph_symbol_maps_end_time - \
                         scenario_ph_symbol_maps_start_time))

            # if specified, run the user script to initialize variable
            # bounds at their whim.
            if ph._bound_setter is not None:

                print("Executing user bound setter callback function")
                for scenario in ph._scenario_tree._scenarios:
                    ph._callback_function[ph._bound_setter](
                        ph,
                        ph._scenario_tree,
                        scenario)

            pyomo.pysp.phsolverserverutils.\
                warmstart_scenario_instances(ph)

            ph._preprocess_scenario_instances()

        ph_solver_manager = ph._solver_manager
        ph._solver_manager = None
        try:
            # The post-solve plugins may have done more variable
            # fixing. These should be pushed to the instance at this
            # point.
            print("Pushing fixed variable statuses to scenario instances")
            ph._push_all_node_fixed_to_instances()
            total_fixed_discrete_vars, total_fixed_continuous_vars = \
                ph.compute_fixed_variable_counts()
            print("Number of discrete variables fixed "
                  "prior to ef creation="
                  +str(total_fixed_discrete_vars)+
                  " (total="+str(ph._total_discrete_vars)+")")
            print("Number of continuous variables fixed "
                  "prior to ef creation="
                  +str(total_fixed_continuous_vars)+
                  " (total="+str(ph._total_continuous_vars)+")")
        finally:
            ph._solver_manager = ph_solver_manager

        ef_options = options._ef_options
        # Have any matching ef options that are not explicitly
        # set by the user inherit from the "PH" values on the
        # argparse object. The user has the option of overriding
        # these values by setting the --ef-* versions via the
        # command-line
        ExtensiveFormAlgorithm.update_options_from_argparse(
            ef_options,
            options,
            prefix="ef_",
            srcprefix="",
            skip_userset=True,
            error_if_missing=False)

        if _OLD_OUTPUT:
            print("Creating extensive form for remainder problem")

        ef = ExtensiveFormAlgorithm(ph, ef_options, prefix="ef_")
        ef.build_ef()

        # set the value of each non-converged, non-final-stage
        # variable to None - this will avoid infeasible warm-stats.
        reset_nonconverged_variables(ph._scenario_tree, ph._instances)
        reset_stage_cost_variables(ph._scenario_tree, ph._instances)

    # The EFAlgorithm will handle its own preprocessing, so
    # be sure to remove any flags that hack preprocessing
    # behavior from the instances. This also releases
    # any PHPyro workers
    ph.release_components()

    #
    # solve the extensive form and load the solution back into the PH
    # scenario tree. Contents from the PH solve will obviously be
    # over-written!
    #
    if options.write_ef:

        print("Writing extensive form")
        ef.write(options.ef_output_file)

    if options.solve_ef:

        print("Solving extensive form")
        failed = ef.solve(exception_on_failure=False)

        if failed:
            print("EF solve failed solution status check:\n"
                  "Solver Status: %s\n"
                  "Termination Condition: %s\n"
                  "Solution Status: %s\n"
                  % (ef.solver_status,
                     ef.termination_condition,
                     ef.solution_status))
        else:

            print("EF solve completed and solution status is %s"
                  % ef.solution_status)
            print("EF solve termination condition is %s"
                  % ef.termination_condition)
            print("EF objective: %12.5f" % ef.objective)
            if ef.gap is not undefined:
                print("EF gap:       %12.5f" % ef.gap)
                print("EF bound:     %12.5f" % ef.bound)
            else:
                assert ef.bound is undefined
                print("EF gap:       <unknown>")
                print("EF bound:     <unknown>")

            ph.update_variable_statistics()

            # handle output of solution from the scenario tree.
            print("")
            print("Extensive form solution:")
            ph.scenario_tree.pprintSolution()
            print("")
            print("Extensive form costs:")
            ph.scenario_tree.pprintCosts()

            for plugin in ef_solution_writers:
                plugin.write(ph.scenario_tree, "postphef")

            # Another hack to execute the jsonio saver plugin until we move over
            # to the new scripting interface
            if options.activate_jsonio_solution_saver:
                print("Executing jsonio solution saver extension")
                import pyomo.pysp.plugins.jsonio
                jsonsaver = pyomo.pysp.plugins.jsonio.JSONSolutionSaverExtension()
                jsonsaver_options = jsonsaver.register_options()
                jsonsaver_options.jsonsaver_output_name = \
                    options.jsonsaver_output_name
                jsonsaver_options.jsonsaver_save_stages = \
                    options.jsonsaver_save_stages
                jsonsaver.set_options(jsonsaver_options)
                jsonsaver.save(ph)

    if ef is not None:
        ef.close()

#
# The main PH initialization / runner routine.
#

def exec_runph(options):

    import pyomo.environ

    start_time = time.time()

    try:

        # This context manages releasing pyro workers and
        # closing file archives
        with PHFromScratchManagedContext(options) as ph:
            run_ph(options, ph)

    # This context will shutdown the pyro nameserver if requested.
    # Ideally, pyro workers can be reused without restarting the
    # nameserver
    finally:
        # if an exception is triggered, and we're running with
        # pyro, shut down everything - not doing so is
        # annoying, and leads to a lot of wasted compute
        # time. but don't do this if the shutdown-pyro option
        # is disabled => the user wanted
        if ((options.solver_manager_type == "pyro") or \
            (options.solver_manager_type == "phpyro")) and \
            options.shutdown_pyro:
            print("\n")
            print("Shutting down Pyro solver components.")
            shutdown_pyro_components(host=options.pyro_host,
                                     port=options.pyro_port,
                                     num_retries=0)

    print("")
    print("Total execution time=%.2f seconds"
          % (time.time() - start_time))

    return 0

#
# the main driver routine for the runph script.
#

def main(args=None):
    #
    # Top-level command that executes the runph command
    #

    #
    # Import plugins
    #
    import pyomo.environ

    #
    # Parse command-line options.
    #
    try:
        ph_options_parser = \
            construct_ph_options_parser("runph [options]")
        options = ph_options_parser.parse_args(args=args)
        # temporary hack
        options._ef_options = ph_options_parser._ef_options
        options._ef_options.import_argparse(options)
    except SystemExit as _exc:
        # the parser throws a system exit if "-h" is specified
        # - catch it to exit gracefully.
        return _exc.code

    return launch_command(exec_runph,
                          options,
                          error_label="runph: ",
                          disable_gc=options.disable_gc,
                          profile_count=options.profile,
                          traceback=options.traceback)

@pyomo_command('runph', 'Optimize with the PH solver (primal search)')
def PH_main(args=None):
    return main(args=args)
