### Ideas
# - Should be easy to warm start the benders script
#   from a history file, so one wouldn't need to start
#   from scratch
# - Do some exception/signal handling to cleanly exit
#   (and save history if possible)
# -

### Users should be able to provide
# - A poole xbars
# - A set of cuts
# - A history file

### Big Improvement Ideas
# Separate this into another module so that PH can use it as well
# Initialization Options for LB0
# - User specififed
# - Iter 0 PH
#   - One subproblem per scenario (ignore bundles) - compute xbar
#   - Respect bundles - solve - compute xbars
#   - Independent (Larger?) bundles for the initial compute xbar solves
# - Relaxed MIP (and/or combine this with bundles)

### Lower Priority TODOs:
# - feasibility cuts
# - FirstStageDerived variables
# - Piecewise (e.g., transformation variables added after construction)
# - relaxed master iterations


import time
import os
import math

thisfile = os.path.abspath(__file__)
thisfile.replace(".pyc","").replace(".py","")

from pyomo.core import *
from pyomo.core.base.var import _VarDataWithDomain
from pyomo.opt import SolverFactory, SolverManagerFactory
from pyomo.solvers.plugins.smanager.phpyro import SolverManager_PHPyro
from pyomo.pysp.plugins.phhistoryextension import load_history
from pyomo.pysp.phinit import *
from pyomo.pysp.plugins.phhistoryextension import phhistoryextension

try:
    import pstats
    pstats_available=True
except ImportError:
    pstats_available=False

from optparse import OptionParser, OptionGroup, SUPPRESS_HELP

# for profiling
try:
    import cProfile as profile
except ImportError:
    import profile

#
# utility method to construct an option parser for benders arguments.
#

def construct_benders_options_parser(usage_string):

    solver_list = SolverFactory.services()
    solver_list = sorted( filter(lambda x: '_' != x[0], solver_list) )
    solver_help = \
    "Specify the solver with which to solve scenario sub-problems.  The "      \
    "following solver types are currently supported: %s; Default: cplex"
    solver_help %= ', '.join( solver_list )

    master_solver_help = ("Specify the solver with which to solve the master benders problem. "
                          "The following solver types are currently supported: %s; Default: cplex")
    master_solver_help %= ', '.join( solver_list )

    parser = OptionParser()
    parser.usage = usage_string

    # NOTE: these groups should eventually be queried from the PH, scenario tree, etc. classes (to facilitate re-use).
    inputOpts        = OptionGroup( parser, 'Input Options' )
    scenarioTreeOpts = OptionGroup( parser, 'Scenario Tree Options' )
    bOpts            = OptionGroup( parser, 'Benders Options' )
    msolverOpts     = OptionGroup( parser, 'Master Solver Options' )
    ssolverOpts      = OptionGroup( parser, 'Subproblem Solver Options' )
    outputOpts       = OptionGroup( parser, 'Output Options' )
    otherOpts        = OptionGroup( parser, 'Other Options' )

    parser.add_option_group( inputOpts )
    parser.add_option_group( scenarioTreeOpts )
    parser.add_option_group( bOpts )
    parser.add_option_group( msolverOpts )
    parser.add_option_group( ssolverOpts )
    parser.add_option_group( outputOpts )
    parser.add_option_group( otherOpts )

    inputOpts.add_option('-m','--model-directory',
      help='The directory in which all model (reference and scenario) definitions are stored. Default is ".".',
      action="store",
      dest="model_directory",
      type="string",
      default=".")
    inputOpts.add_option('-i','--instance-directory',
      help='The directory in which all instance (reference and scenario) definitions are stored. Default is ".".',
      action="store",
      dest="instance_directory",
      type="string",
      default=".")
    def objective_sense_callback(option, opt_str, value, parser):
        if value in ('min','minimize',minimize):
            parser.values.objective_sense = minimize
        elif value in ('max','maximize',maximize):
            parser.values.objective_sense = maximize
        else:
            parser.values.objective_sense = None
    inputOpts.add_option('-o','--objective-sense-stage-based',
      help='The objective sense to use for the auto-generated scenario instance objective, which is equal to the '
           'sum of the scenario-tree stage costs. Default is None, indicating an Objective has been declared on the '
           'reference model.',
      action="callback",
      dest="objective_sense",
      type="choice",
      choices=[maximize,'max','maximize',minimize,'min','minimize',None],
      default=None,
      callback=objective_sense_callback)

    scenarioTreeOpts.add_option('--scenario-tree-seed',
      help="The random seed associated with manipulation operations on the scenario tree (e.g., down-sampling or bundle creation). Default is 0, indicating unassigned.",
      action="store",
      dest="scenario_tree_random_seed",
      type="int",
      default=0)
    scenarioTreeOpts.add_option('--scenario-tree-downsample-fraction',
      help="The proportion of the scenarios in the scenario tree that are actually used. Specific scenarios are selected at random. Default is 1.0, indicating no down-sampling.",
      action="store",
      dest="scenario_tree_downsample_fraction",
      type="float",
      default=1.0)
    scenarioTreeOpts.add_option('--scenario-bundle-specification',
      help="The name of the scenario bundling specification to be used when executing Progressive Hedging. Default is None, indicating no bundling is employed. If the specified name ends with a .dat suffix, the argument is interpreted as a filename. Otherwise, the name is interpreted as a file in the instance directory, constructed by adding the .dat suffix automatically",
      action="store",
      dest="scenario_bundle_specification",
      default=None)
    scenarioTreeOpts.add_option('--create-random-bundles',
      help="Specification to create the indicated number of random, equally-sized (to the degree possible) scenario bundles. Default is 0, indicating disabled.",
      action="store",
      dest="create_random_bundles",
      type="int",
      default=None)

    bOpts.add_option('--max-iterations',
      help="The maximal number of benders iterations. Default is 100.",
      action="store",
      dest="max_iterations",
      type="int",
      default=100)
    bOpts.add_option('--percent-gap',
      help="Percent optimality gap required for convergence. Default is 0.0001%.",
      action="store",
      dest="percent_gap",
      type="float",
      default=0.0001)
    bOpts.add_option('--multicut-level',
      help="The number of cut groups added to the master benders problem each iteration. Default is 1.",
      action="store",
      dest="multicuts",
      type="int",
      default=1)
    bOpts.add_option('--user-bound',
      help="A user provided best bound for the relaxed (master) problem. When provided, will be used in the optimality gap calculation if appropriate.",
      action="store",
      dest="user_bound",
      type="float",
      default=None)

    msolverOpts.add_option('--master-disable-warmstarts',
      help="Disable warm-start of the benders master problem solves. Default is False.",
      action="store_true",
      dest="master_disable_warmstart",
      default=False)
    msolverOpts.add_option('--master-solver',
      help=master_solver_help,
      action="store",
      dest="master_solver_type",
      type="string",
      default="cplex")
    msolverOpts.add_option('--master-solver-io',
      help='The type of IO used to execute the master solver.  Different solvers support different types of IO, but the following are common options: lp - generate LP files, nl - generate NL files, python - direct Python interface, os - generate OSiL XML files.',
      action='store',
      dest='master_solver_io',
      type='string',
      default=None)
    msolverOpts.add_option('--master-mipgap',
      help="Specifies the mipgap for the master benders solves.",
      action="store",
      dest="master_mipgap",
      type="float",
      default=None)
    msolverOpts.add_option('--master-solver-options',
      help="Solver options for the master benders problem.",
      action="append",
      dest="master_solver_options",
      type="string",
      default=[])
    msolverOpts.add_option('--master-output-solver-log',
      help="Output solver logs during master benders solves solves",
      action="store_true",
      dest="master_output_solver_log",
      default=False)
    msolverOpts.add_option('--master-keep-solver-files',
      help="Retain temporary input and output files for master benders solves",
      action="store_true",
      dest="master_keep_solver_files",
      default=False)
    msolverOpts.add_option('--master-symbolic-solver-labels',
       help='When interfacing with the solver, use symbol names derived from the model. For example, \"my_special_variable[1_2_3]\" instead of \"v1\". Useful for debugging. When using the ASL interface (--solver-io=nl), generates corresponding .row (constraints) and .col (variables) files. The ordering in these files provides a mapping from ASL index to symbolic model names.',
      action='store_true',
      dest='master_symbolic_solver_labels',
      default=False)

    ssolverOpts.add_option('--output-solver-logs',
      help="Output solver logs during scenario sub-problem solves",
      action="store_true",
      dest="output_solver_logs",
      default=False)
    ssolverOpts.add_option('--symbolic-solver-labels',
      help='When interfacing with the solver, use symbol names derived from the model. For example, \"my_special_variable[1_2_3]\" instead of \"v1\". Useful for debugging. When using the ASL interface (--solver-io=nl), generates corresponding .row (constraints) and .col (variables) files. The ordering in these files provides a mapping from ASL index to symbolic model names.',
      action='store_true',
      dest='symbolic_solver_labels',
      default=False)
    ssolverOpts.add_option('--scenario-mipgap',
      help="Specifies the mipgap for all sub-problems",
      action="store",
      dest="scenario_mipgap",
      type="float",
      default=None)
    ssolverOpts.add_option('--scenario-solver-options',
      help="Solver options for all sub-problems",
      action="append",
      dest="scenario_solver_options",
      type="string",
      default=[])
    ssolverOpts.add_option('--solver',
      help=solver_help,
      action="store",
      dest="solver_type",
      type="string",
      default="cplex")
    ssolverOpts.add_option('--solver-io',
      help='The type of IO used to execute the solver.  Different solvers support different types of IO, but the following are common options: lp - generate LP files, nl - generate NL files, python - direct Python interface, os - generate OSiL XML files.',
      action='store',
      dest='solver_io',
      default=None)
    ssolverOpts.add_option('--solver-manager',
      help="The type of solver manager used to coordinate scenario sub-problem solves. Default is serial.",
      action="store",
      dest="solver_manager_type",
      type="string",
      default="serial")
    ssolverOpts.add_option('--phpyro-required-workers',
      help="Set the number of idle phsolverserver worker processes expected to be available when the PHPyro solver manager is selected. This option should be used when the number of worker threads is less than the total number of scenarios (or bundles). When this option is not used, PH will attempt to assign each scenario (or bundle) to a single phsolverserver until the timeout indicated by the --phpyro-workers-timeout option occurs.",
      action="store",
      type=int,
      dest="phpyro_required_workers",
      default=None)
    ssolverOpts.add_option('--phpyro-workers-timeout',
     help="Set the time limit (seconds) for finding idle phsolverserver worker processes to be used when the PHPyro solver manager is selected. This option is ignored when --phpyro-required-workers is set manually. Default is 30.",
      action="store",
      type=float,
      dest="phpyro_workers_timeout",
      default=30)
    ssolverOpts.add_option('--disable-warmstarts',
      help="Disable warm-start of scenario sub-problem solves in iterations >= 1. Default is False.",
      action="store_true",
      dest="disable_warmstarts",
      default=False)
    ssolverOpts.add_option('--shutdown-pyro',
      help="Shut down all Pyro-related components associated with the Pyro and PH Pyro solver managers (if specified), including the dispatch server, name server, and any solver servers. Default is False.",
      action="store_true",
      dest="shutdown_pyro",
      default=False)


    outputOpts.add_option('--output-scenario-tree-solution',
      help="Report the full solution (even leaves) in scenario tree format upon termination. Values represent averages, so convergence is not an issue. Default is False.",
      action="store_true",
      dest="output_scenario_tree_solution",
      default=False)
    outputOpts.add_option('--output-solver-results',
      help="Output solutions obtained after each scenario sub-problem solve",
      action="store_true",
      dest="output_solver_results",
      default=False)
    outputOpts.add_option('--output-times',
      help="Output timing statistics for various components",
      action="store_true",
      dest="output_times",
      default=False)
    outputOpts.add_option('--output-instance-construction-times',
      help="Output timing statistics for instance construction timing statistics (client-side only when using PHPyro",
      action="store_true",
      dest="output_instance_construction_times",
      default=False)
    outputOpts.add_option('--verbose',
      help="Generate verbose output for both initialization and execution. Default is False.",
      action="store_true",
      dest="verbose",
      default=False)

    otherOpts.add_option('--disable-gc',
      help="Disable the python garbage collecter. Default is False.",
      action="store_true",
      dest="disable_gc",
      default=False)
    otherOpts.add_option('-k','--keep-solver-files',
      help="Retain temporary input and output files for scenario sub-problem solves",
      action="store_true",
      dest="keep_solver_files",
      default=False)
    otherOpts.add_option('--profile',
      help="Enable profiling of Python code.  The value of this option is the number of functions that are summarized.",
      action="store",
      dest="profile",
      type="int",
      default=0)
    otherOpts.add_option('--traceback',
      help="When an exception is thrown, show the entire call stack. Ignored if profiling is enabled. Default is False.",
      action="store_true",
      dest="traceback",
      default=False)


    # These options need to be here because we piggy back
    # off of PH for solving the subproblems (for now)
    # We hide them because they don't make sense for
    # this application
    otherOpts.add_option('--handshake-with-phpyro-but-do-not-use',
                          help=SUPPRESS_HELP,
                          dest="handshake_with_phpyro",
                          default=False)
    otherOpts.add_option('--bounds-cfgfile-but-do-not-use',
                         help=SUPPRESS_HELP,
                         dest="bounds_cfgfile",
                         default=None)
    otherOpts.add_option('-r','--default-rho-but-do-not-use',
                      help=SUPPRESS_HELP,
                      dest="default_rho",
                      default=1.0)
    otherOpts.add_option("--overrelax-but-do-not-use",
                      help=SUPPRESS_HELP,
                      dest="overrelax",
                      default=False)
    otherOpts.add_option("--nu-but-do-not-use",
                      help=SUPPRESS_HELP,
                      dest='nu',
                      default=1.5)
    otherOpts.add_option("--async-but-do-not-use",
                      help=SUPPRESS_HELP,
                      dest="async",
                      default=False)
    otherOpts.add_option("--async-buffer-len-but-do-not-use",
                      help=SUPPRESS_HELP,
                      dest="async_buffer_len",
                      default=1)
    otherOpts.add_option('--rho-cfgfile-but-do-not-use',
                      help=SUPPRESS_HELP,
                      dest="rho_cfgfile",
                      default=None)
    otherOpts.add_option('--aggregate-cfgfile-but-do-not-use',
                      help=SUPPRESS_HELP,
                      dest="aggregate_cfgfile",
                      default=None)
    otherOpts.add_option('--termdiff-threshold-but-do-not-use',
                      help=SUPPRESS_HELP,
                      dest="termdiff_threshold",
                      default=0.0001)
    otherOpts.add_option('--enable-free-discrete-count-convergence-but-do-not-use',
                      help=SUPPRESS_HELP,
                      dest="enable_free_discrete_count_convergence",
                      default=False)
    otherOpts.add_option('--enable-normalized-termdiff-convergence-but-do-not-use',
                      help=SUPPRESS_HELP,
                      dest="enable_normalized_termdiff_convergence",
                      default=True)
    otherOpts.add_option('--enable-termdiff-convergence-but-do-not-use',
                      help=SUPPRESS_HELP,
                      dest="enable_termdiff_convergence",
                      default=False)
    otherOpts.add_option('--free-discrete-count-threshold-but-do-not-use',
                      help=SUPPRESS_HELP,
                      dest="free_discrete_count_threshold",
                      default=20)
    otherOpts.add_option('--linearize-nonbinary-penalty-terms-but-do-not-use',
                      help=SUPPRESS_HELP,
                      dest="linearize_nonbinary_penalty_terms",
                      default=0)
    otherOpts.add_option('--breakpoint-strategy-but-do-not-use',
                      help=SUPPRESS_HELP,
                      dest="breakpoint_strategy",
                      default=0)
    otherOpts.add_option('--retain-quadratic-binary-terms-but-do-not-use',
                      help=SUPPRESS_HELP,
                      dest="retain_quadratic_binary_terms",
                      default=False)
    otherOpts.add_option('--drop-proximal-terms-but-do-not-use',
                      help=SUPPRESS_HELP,
                      dest="drop_proximal_terms",
                      default=False)
    otherOpts.add_option('--enable-ww-extensions-but-do-not-use',
                      help=SUPPRESS_HELP,
                      dest="enable_ww_extensions",
                      default=False)
    otherOpts.add_option('--ww-extension-cfgfile-but-do-not-use',
                      help=SUPPRESS_HELP,
                      dest="ww_extension_cfgfile",
                      default="")
    otherOpts.add_option('--ww-extension-suffixfile-but-do-not-use',
                      help=SUPPRESS_HELP,
                      dest="ww_extension_suffixfile",
                      default="")
    otherOpts.add_option('--ww-extension-annotationfile-but-do-not-use',
                      help=SUPPRESS_HELP,
                      dest="ww_extension_annotationfile",
                      default="")
    otherOpts.add_option('--user-defined-extension-but-do-not-use',
                      help=SUPPRESS_HELP,
                      dest="user_defined_extensions",
                      default=[])
    otherOpts.add_option("--flatten-expressions-but-do-not-use", "--linearize-expressions-but-do-not-use",
                      help=SUPPRESS_HELP,
                      dest="flatten_expressions",
                      default=False)
    otherOpts.add_option('--preprocess-fixed-variables-but-do-not-use',
                      help=SUPPRESS_HELP,
                      dest="write_fixed_variables",
                      default=True)
    otherOpts.add_option('--ef-disable-warmstarts-but-do-not-use',
                          help=SUPPRESS_HELP,
                          dest="ef_disable_warmstarts",
                          default=None)
    otherOpts.add_option('--ef-output-file-but-do-not-use',
                               help=SUPPRESS_HELP,
                               dest="ef_output_file",
                               default="efout")
    otherOpts.add_option('--solve-ef-but-do-not-use',
                               help=SUPPRESS_HELP,
                               dest="solve_ef",
                               default=False)
    otherOpts.add_option('--ef-solver-but-do-not-use',
                               help=SUPPRESS_HELP,
                               dest="ef_solver_type",
                               default=None)
    otherOpts.add_option('--ef-solution-writer-but-do-not-use',
                               help=SUPPRESS_HELP,
                               dest="ef_solution_writer",
                               default = [])
    otherOpts.add_option('--ef-solver-io-but-do-not-use',
                               help=SUPPRESS_HELP,
                               dest='ef_solver_io',
                               default=None)
    otherOpts.add_option('--ef-solver-manager-but-do-not-use',
                               help=SUPPRESS_HELP,
                               dest="ef_solver_manager_type",
                               default="serial")
    otherOpts.add_option('--ef-mipgap-but-do-not-use',
                               help=SUPPRESS_HELP,
                               dest="ef_mipgap",
                               default=None)
    otherOpts.add_option('--ef-disable-warmstart-but-do-not-use',
                               help=SUPPRESS_HELP,
                               dest="ef_disable_warmstart",
                               default=False)
    otherOpts.add_option('--ef-solver-options-but-do-not-use',
                               help=SUPPRESS_HELP,
                               dest="ef_solver_options",
                               default=[])
    otherOpts.add_option('--ef-output-solver-log-but-do-not-use',
                               help=SUPPRESS_HELP,
                               dest="ef_output_solver_log",
                               default=None)
    otherOpts.add_option('--ef-keep-solver-files-but-do-not-use',
                               help=SUPPRESS_HELP,
                               dest="ef_keep_solver_files",
                               default=None)
    otherOpts.add_option('--ef-symbolic-solver-labels-but-do-not-use',
                               help=SUPPRESS_HELP,
                               dest='ef_symbolic_solver_labels',
                               default=None)
    outputOpts.add_option('--report-only-statistics-but-do-not-use',
                          help=SUPPRESS_HELP,
                          dest="report_only_statistics",
                          default=False)
    outputOpts.add_option('--report-solutions-but-do-not-use',
                          help=SUPPRESS_HELP,
                          dest="report_solutions",
                          default=False)
    outputOpts.add_option('--report-weights-but-do-not-use',
                          help=SUPPRESS_HELP,
                          dest="report_weights",
                          default=False)
    outputOpts.add_option('--report-rhos-all-iterations-but-do-not-use',
                          help=SUPPRESS_HELP,
                          dest="report_rhos_each_iteration",
                          default=False)
    outputOpts.add_option('--report-rhos-first-iterations-but-do-not-use',
                          help=SUPPRESS_HELP,
                          dest="report_rhos_first_iteration",
                          default=False)
    outputOpts.add_option('--report-for-zero-variable-values-but-do-not-use',
                          help=SUPPRESS_HELP,
                          dest="report_for_zero_variable_values",
                          default=False)
    outputOpts.add_option('--report-only-nonconverged-variables-but-do-not-use',
                          help=SUPPRESS_HELP,
                          dest="report_only_nonconverged_variables",
                          default=False)
    outputOpts.add_option('--restore-from-checkpoint-but-do-not-use',
                          help=SUPPRESS_HELP,
                          dest="restore_from_checkpoint",
                          default=None)
    outputOpts.add_option('--solution-writer-but-do-not-use',
                          help=SUPPRESS_HELP,
                          dest="solution_writer",
                          default = [])
    outputOpts.add_option('--suppress-continuous-variable-output-but-do-not-use',
                          help=SUPPRESS_HELP,
                          dest="suppress_continuous_variable_output",
                          default=False)
    outputOpts.add_option('--write-ef-but-do-not-use',
                          help=SUPPRESS_HELP,
                          dest="write_ef",
                          default=False)
    otherOpts.add_option('--checkpoint-interval-but-do-not-use',
                         help=SUPPRESS_HELP,
                         dest="checkpoint_interval",
                         default=0)
    otherOpts.add_option('--phpyro-transmit-leaf-stage-variable-solution-but-do-not-use',
                          help=SUPPRESS_HELP,
                          dest="phpyro_transmit_leaf_stage_solution",
                          default=False)


    

    return parser


def Benders_DefaultOptions():
    parser = construct_benders_options_parser("")
    options, _ = parser.parse_args([''])
    return options

def collect_workers(solver_manager, scenario_tree, options):
    workers_expected = options.phpyro_required_workers
    timeout = options.phpyro_workers_timeout
    if scenario_tree.contains_bundles():
        num_jobs = len(scenario_tree._scenario_bundles)
        print("Bundle solver jobs available: "+str(num_jobs))
    else:
        num_jobs = len(scenario_tree._scenarios)
        print("Scenario solver jobs available: "+str(num_jobs))

    if (workers_expected is None):
        workers_expected = num_jobs
    else:
        timeout = None

    solver_manager.acquire_workers(workers_expected,
                                   timeout)

def EXTERNAL_deactivate_firststage_cost(ph,
                                        scenario_tree,
                                        scenario):
    assert len(ph._scenario_tree._stages) == 2
    assert scenario in ph._scenario_tree._scenarios
    firststage = ph._scenario_tree.findRootNode()._stage
    scenario._instance.find_component("PYSP_STAGE_COST_TERM_"+firststage._name).value = 0.0
    ph._problem_states.objective_updated[scenario._name] = True

def EXTERNAL_activate_firststage_cost(ph,
                                      scenario_tree,
                                      scenario):
    assert len(ph._scenario_tree._stages) == 2
    assert scenario in ph._scenario_tree._scenarios
    firststage = ph._scenario_tree.findRootNode._stage
    stagecost_var = instance.find_component(firststage._cost_variable[0])[firststage._cost_variable[1]]
    scenario._instance.find_component("PYSP_STAGE_COST_TERM_"+firststage._name).value = stagecost_var
    ph._problem_states.objective_updated[scenario._name] = True

def EXTERNAL_activate_fix_constraints(ph,
                                      scenario_tree,
                                      scenario):
    assert len(ph._scenario_tree._stages) == 2
    assert scenario in ph._scenario_tree._scenarios
    rootnode = ph._scenario_tree.findRootNode()
    scenario._instance.find_component("PYSP_BENDERS_FIX_"+str(rootnode._name)).activate()
    ph._problem_states.user_constraints_updated[scenario._name] = True

def EXTERNAL_deactivate_fix_constraints(ph,
                                        scenario_tree,
                                        scenario):
    assert len(ph._scenario_tree._stages) == 2
    assert scenario in ph._scenario_tree._scenarios
    rootnode = ph._scenario_tree.findRootNode()
    scenario._instance.find_component("PYSP_BENDERS_FIX_"+str(rootnode._name)).deactivate()
    ph._problem_states.user_constraints_updated[scenario._name] = True

def EXTERNAL_initialize_for_benders(ph,
                                    scenario_tree,
                                    scenario):
    assert len(ph._scenario_tree._stages) == 2
    assert scenario in ph._scenario_tree._scenarios

    rootnode = ph._scenario_tree.findRootNode()
    leafstage = scenario._leaf_node._stage
    instance = scenario._instance

    # disaggregate the objective into stage costs
    cost_terms = 0.0
    for node in scenario._node_list:
        stage = node._stage
        stagecost_var = instance.find_component(stage._cost_variable[0])[stage._cost_variable[1]]
        instance.add_component("PYSP_STAGE_COST_TERM_"+stage._name, Expression(initialize=stagecost_var))
        cost_terms += instance.find_component("PYSP_STAGE_COST_TERM_"+stage._name)
    scenario._instance_cost_expression.value = cost_terms

    # TODO: Remove first stage constraints?

    if scenario_tree.contains_bundles():
        found = 0
        for scenario_bundle in scenario_tree._scenario_bundles:
            if scenario._name in scenario_bundle._scenario_names:
                found += 1
                bundle_instance = ph._bundle_binding_instance_map[scenario_bundle._name]
                if not hasattr(bundle_instance,"dual"):
                    bundle_instance.dual = Suffix(direction=Suffix.IMPORT)
        assert found == 1
    else:
        instance.dual = Suffix(direction=Suffix.IMPORT)
    scenario_bySymbol = instance._ScenarioTreeSymbolMap.bySymbol

    for variable_id in rootnode._variable_ids:
        vardata = scenario_bySymbol[variable_id]
        if isinstance(vardata, _VarDataWithDomain):
            vardata.domain = Reals
        else:
            vardata.parent_component().domain = Reals

    nodal_index_set_name = "PHINDEX_"+str(rootnode._name)
    nodal_index_set = instance.find_component(nodal_index_set_name)

    fix_param_name = "PYSP_BENDERS_FIX_VALUE"+str(rootnode._name)
    instance.add_component(fix_param_name, Param(nodal_index_set, mutable=True, initialize=0.0))
    fix_param = instance.find_component(fix_param_name)

    def fix_rule(m,variable_id):
        return  scenario_bySymbol[variable_id] - fix_param[variable_id] == 0.0
    instance.add_component("PYSP_BENDERS_FIX_"+str(rootnode._name),
                           Constraint(nodal_index_set, rule=fix_rule))
    instance.find_component("PYSP_BENDERS_FIX_"+str(rootnode._name)).deactivate()

    ph._problem_states.user_constraints_updated[scenario._name] = True
    ph._problem_states.objective_updated[scenario._name] = True

def EXTERNAL_update_fix_constraints(ph,
                                    scenario_tree,
                                    scenario,
                                    fix_values):
    assert len(ph._scenario_tree._stages) == 2
    assert scenario in ph._scenario_tree._scenarios

    rootnode = ph._scenario_tree.findRootNode()
    instance = scenario._instance
    fix_param_name = "PYSP_BENDERS_FIX_VALUE"+str(rootnode._name)
    fix_param = instance.find_component(fix_param_name)
    fix_param.store_values(fix_values)
    ph._problem_states.user_constraints_updated[scenario._name] = True

def EXTERNAL_collect_cut_data(ph,
                              scenario_tree,
                              scenario):
    assert len(ph._scenario_tree._stages) == 2
    assert scenario in ph._scenario_tree._scenarios

    dual_suffix = None
    sum_probability_bundle = None
    if scenario_tree.contains_bundles():
        found = 0
        for scenario_bundle in scenario_tree._scenario_bundles:
            if scenario._name in scenario_bundle._scenario_names:
                found += 1
                dual_suffix = ph._bundle_binding_instance_map[scenario_bundle._name].dual
                sum_probability_bundle = scenario_bundle._probability
        assert found == 1

    else:
        dual_suffix  = scenario._instance.dual
        sum_probability_bundle = scenario._probability
    rootnode = ph._scenario_tree.findRootNode()
    scenario_results = {}
    scenario_results['SSC'] = scenario._stage_costs[scenario._leaf_node._stage._name]
    duals = scenario_results['duals'] = {}
    benders_fix_constraint = scenario._instance.find_component("PYSP_BENDERS_FIX_"+str(rootnode._name))
    for variable_id in rootnode._variable_ids:
        duals[variable_id] = dual_suffix[benders_fix_constraint[variable_id]] \
                             * sum_probability_bundle \
                             / scenario._probability
    return scenario_results

def solve_extensive_form_for_xbars(scenario_tree):

    rootnode = scenario_tree.findRootNode()
    binding_instance = create_ef_instance(scenario_tree)
    binding_instance.preprocess()
    binding_instance.load(master_solver.solve(binding_instance))
    scenario_tree.pullScenarioSolutionsFromInstances()
    print "Extensive Form objective:", scenario_tree.findRootNode().computeExpectedNodeCost()
    ef_var = binding_instance.find_component("MASTER_BLEND_VAR_"+str(rootnode._name))
    xbars = {}
    for variable_id in rootnode._variable_ids:
        xbars[variable_id] = value(ef_var[variable_id])

    return xbars, scenario_tree.findRootNode().computeExpectedNodeCost()

class BendersOptimalityCut(object):

    def __init__(self, xbars, ssc, duals):
        self.xbars = xbars
        self.ssc = ssc
        self.duals = duals

class BendersAlgorithm(object):

    def __init__(self, options):

        self._options = options

        # TODO: Do some options validation

        # The master (first-stage) benders instance
        self._master = None
        # The scenario tree object consisting of a single
        # scenario, most used to navigate the master in
        # terms of the scenario tree variable ids used
        # used by the true scenario subproblems
        self._master_tree = None
        # The ph object used to manage subproblem solves
        self._ph = None
        self._solver_manager = None
        self._master_solver = None

        # This number is cached after we initialize
        self._num_first_stage_constraints = None

        # This is created during initialization. It is
        # a bit of a hack used right now to track
        # benders history
        self._history_plugin = None

    def close(self):

        if self._ph is not None:

            self._ph.release_component()

    def initialize(self, options, scenario_tree, solver_manager, master_solver):

        self._solver_manager = solver_manager
        self._master_solver = master_solver

        history_plugin = self._history_plugin = phhistoryextension()

        print("")
        print("Initializing the Benders decomposition for stochastic problems (i.e., the L-shaped method)")

        ph = ProgressiveHedging(options)

        ph.initialize(scenario_tree=scenario_tree,
                      solver_manager=solver_manager)


        rootnode = ph._scenario_tree.findRootNode()
        firststage = rootnode._stage

        ph._total_fixed_continuous_vars = None
        ph._total_fixed_discrete_vars = None
        ph._total_continuous_vars = None
        ph._total_discrete_vars = None
        history_plugin.pre_ph_initialization(ph)
        history_plugin.post_ph_initialization(ph)

        objective_sense = ph._objective_sense

        # construct master problem
        master_singleton_tree = scenario_tree._scenario_instance_factory.generate_scenario_tree()
        master_singleton_tree.compress([master_singleton_tree._scenarios[0]._name])
        master_singleton_dict = master_singleton_tree._scenario_instance_factory.\
                                construct_instances_for_scenario_tree(master_singleton_tree)
        # with the scenario instances now available, link the
        # referenced objects directly into the scenario tree.
        master_singleton_tree.linkInInstances(master_singleton_dict,
                                              create_variable_ids=True)
        master_scenario_name = master_singleton_dict.keys()[0]
        master_scenario = master_singleton_tree.get_scenario(master_scenario_name)
        master = master_singleton_dict.values()[0]
        master_rootnode = master_singleton_tree.findRootNode()
        master_firststage = master_rootnode._stage
        # Deactivate second-stage constraints
        master.preprocess()
        num_first_stage_constraints = 0
        for block in master.all_blocks():
            canonical_repn = getattr(block,"canonical_repn",None)
            if canonical_repn is None:
                raise ValueError("Unable to find canonical_repn ComponentMap "
                                 "on block %s" % (block.cname(True)))
                for name, index, constraint_data in itertools.chain(block.active_component_data(SOSConstraint),
                                                                    block.active_component_data(Constraint)):
                    constraint = constraint_data.parent_component()
                    node = master_scenario.constraintNode(constraint, index, repn=canonical_repn)
                    if node._stage is not master_firststage:
                        constraint[index].deactivate()
                    else:
                        num_first_stage_constraints += 1

        self._num_first_stage_constraints = num_first_stage_constraints
        master.preprocess()
        # deactivate original objective
        find_active_objective(master,safety_checks=True).deactivate()
        # add cut variable(s)
        master.add_component("PYSP_BENDERS_ALPHA_"+str(rootnode._name),Var())
        master_alpha = master.find_component("PYSP_BENDERS_ALPHA_"+str(rootnode._name))
        master.add_component("PYSP_BENDERS_BUNDLE_ALPHA_"+str(rootnode._name)+"_index",RangeSet(0,self._options.multicuts-1))
        bundle_alpha_index = master.find_component("PYSP_BENDERS_BUNDLE_ALPHA_"+str(rootnode._name)+"_index")
        master.add_component("PYSP_BENDERS_BUNDLE_ALPHA_"+str(rootnode._name), Var(bundle_alpha_index))
        bundle_alpha = master.find_component("PYSP_BENDERS_BUNDLE_ALPHA_"+str(rootnode._name))
        bundles = [[] for i in xrange(self._options.multicuts)]
        assert 1 <= self._options.multicuts <= len(ph._scenario_tree._scenarios)
        # TODO: random shuffle of scenarios
        for cnt, scenario in enumerate(scenario_tree._scenarios):
            bundles[cnt % self._options.multicuts].append(scenario._name)
        setattr(master,"PYSP_BENDERS_CUT_BUNDLES"+str(rootnode._name),bundles)
        if objective_sense == minimize:
            master.add_component("PYSP_BUNDLE_AVERAGE_ALPHA_CUT_"+str(rootnode._name),
                                 Constraint(expr=master_alpha >= sum(bundle_alpha[i] for i in bundle_alpha_index)))
        else:
            master.add_component("PYSP_BUNDLE_AVERAGE_ALPHA_CUT_"+str(rootnode._name),
                                 Constraint(expr=master_alpha <= sum(bundle_alpha[i] for i in bundle_alpha_index)))
        #master_bundle_alpha
        # Fixing will disable any warmstart, and just use the masters
        # initial guess for xbar based on the first stage cost and constraints
        #master_alpha.fix(0)

        # add new objective
        master_firststage_cost_var = master.find_component(master_firststage._cost_variable[0])\
                                     [master_firststage._cost_variable[1]]
        master.add_component("PYSP_BENDERS_OBJECTIVE_"+str(rootnode._name),
                             Objective(expr=master_firststage_cost_var + master_alpha,
                                       sense=objective_sense))
        master_objective = master.find_component("PYSP_BENDERS_OBJECTIVE_"+str(rootnode._name))
        master.add_component("PYSP_BENDERS_CUTS_"+str(rootnode._name),
                             ConstraintList(noruleinit=True))

        self._master_tree = master_singleton_tree
        self._master = master
        self._ph = ph

    def deactivate_firststage_cost(self):
        ph = self._ph
        solver_manager = self._solver_manager
        if isinstance(solver_manager, SolverManager_PHPyro):

            ahs = []
            object_names = None
            if ph._scenario_tree.contains_bundles():

                object_names = [scenario_bundle._name for scenario_bundle \
                                in ph._scenario_tree._scenario_bundles]

            else:

                object_names = [scenario._name for scenario in ph._scenario_tree._scenarios]

            for object_name in object_names:
                ahs.append(
                    transmit_external_function_invocation_to_worker(
                        ph,
                        object_name,
                        thisfile,
                        "EXTERNAL_deactivate_firststage_cost",
                        invocation_type=InvocationType.PerScenarioInvocation,
                        return_action_handle=True))
            solver_manager.wait_all(ahs)

        else:

            for scenario in ph._scenario_tree._scenarios:

                EXTERNAL_deactivate_firststage_cost(ph, ph._scenario_tree, scenario)

    def activate_firststage_cost(self):
        ph = self._ph
        solver_manager = self._solver_manager
        if isinstance(solver_manager, SolverManager_PHPyro):

            ahs = []
            object_names = None
            if ph._scenario_tree.contains_bundles():

                object_names = [scenario_bundle._name for scenario_bundle \
                                in ph._scenario_tree._scenario_bundles]

            else:

                object_names = [scenario._name for scenario in ph._scenario_tree._scenarios]

            for object_name in object_names:
                ahs.append(
                    transmit_external_function_invocation_to_worker(
                        ph,
                        object_name,
                        thisfile,
                        "EXTERNAL_activate_firststage_cost",
                        invocation_type=InvocationType.PerScenarioInvocation,
                        return_action_handle=True))
            solver_manager.wait_all(ahs)

        else:

            for scenario in ph._scenario_tree._scenarios:

                EXTERNAL_activate_firststage_cost(ph, ph._scenario_tree, scenario)

    def activate_fix_constraints(self):
        ph = self._ph
        solver_manager = self._solver_manager
        if isinstance(solver_manager, SolverManager_PHPyro):

            ahs = []
            object_names = None
            if ph._scenario_tree.contains_bundles():

                object_names = [scenario_bundle._name for scenario_bundle \
                                in ph._scenario_tree._scenario_bundles]

            else:

                object_names = [scenario._name for scenario in ph._scenario_tree._scenarios]

            for object_name in object_names:
                ahs.append(
                    transmit_external_function_invocation_to_worker(
                        ph,
                        object_name,
                        thisfile,
                        "EXTERNAL_activate_fix_constraints",
                        invocation_type=InvocationType.PerScenarioInvocation,
                        return_action_handle=True))
            solver_manager.wait_all(ahs)

        else:

            for scenario in ph._scenario_tree._scenarios:

                EXTERNAL_activate_fix_constraints(ph, ph._scenario_tree, scenario)

    def deactivate_fix_constraints(self):
        ph = self._ph
        solver_manager = self._solver_manager
        if isinstance(solver_manager, SolverManager_PHPyro):

            ahs = []
            for scenario in ph._scenario_tree._scenarios:
                ahs.append(
                    transmit_external_function_invocation_to_worker(
                        ph,
                        scenario._name,
                        thisfile,
                        "EXTERNAL_deactivate_fix_constraints",
                        invocation_type=InvocationType.PerScenarioInvocation,
                        return_action_handle=True))
            solver_manager.wait_all(ahs)

        else:

            for scenario in ph._scenario_tree._scenarios:

                EXTERNAL_deactivate_fix_constraints(ph, ph._scenario_tree, scenario)

    def initialize_for_benders(self):
        ph = self._ph
        solver_manager = self._solver_manager
        if isinstance(solver_manager, SolverManager_PHPyro):

            ahs = []
            object_names = None
            if ph._scenario_tree.contains_bundles():

                object_names = [scenario_bundle._name for scenario_bundle \
                                in ph._scenario_tree._scenario_bundles]

            else:

                object_names = [scenario._name for scenario in ph._scenario_tree._scenarios]

            for object_name in object_names:
                ahs.append(
                    transmit_external_function_invocation_to_worker(
                        ph,
                        object_name,
                        thisfile,
                        "EXTERNAL_initialize_for_benders",
                        invocation_type=InvocationType.PerScenarioInvocation,
                        return_action_handle=True))
            solver_manager.wait_all(ahs)

        else:

            for scenario in ph._scenario_tree._scenarios:

                EXTERNAL_initialize_for_benders(ph, ph._scenario_tree, scenario)

    def update_fix_constraints(self, fix_values):
        ph = self._ph
        solver_manager = self._solver_manager
        if isinstance(solver_manager, SolverManager_PHPyro):

            ahs = []
            object_names = None
            if ph._scenario_tree.contains_bundles():

                object_names = [scenario_bundle._name for scenario_bundle \
                                in ph._scenario_tree._scenario_bundles]

            else:

                object_names = [scenario._name for scenario in ph._scenario_tree._scenarios]

            for object_name in object_names:
                ahs.append(
                    transmit_external_function_invocation_to_worker(
                        ph,
                        object_name,
                        thisfile,
                        "EXTERNAL_update_fix_constraints",
                        invocation_type=InvocationType.PerScenarioInvocation,
                        function_args=(fix_values,),
                        return_action_handle=True))

            solver_manager.wait_all(ahs)

        else:

            for scenario in ph._scenario_tree._scenarios:

                EXTERNAL_update_fix_constraints(ph, ph._scenario_tree, scenario, fix_values)

    def collect_cut_data(self):

        ph = self._ph
        solver_manager = self._solver_manager
        results = {}
        if isinstance(solver_manager, SolverManager_PHPyro):

            ahs = []
            ah_map = {}
            object_names = None
            bundling = ph._scenario_tree.contains_bundles()
            if bundling:

                object_names = [scenario_bundle._name for scenario_bundle \
                                in ph._scenario_tree._scenario_bundles]

            else:

                object_names = [scenario._name for scenario in ph._scenario_tree._scenarios]

            for object_name in object_names:

                ah = transmit_external_function_invocation_to_worker(
                    ph,
                    object_name,
                    thisfile,
                    "EXTERNAL_collect_cut_data",
                    invocation_type=InvocationType.PerScenarioInvocation,
                    return_action_handle=True)

                ah_map[ah] = object_name
                ahs.append(ah)

            num_so_far = 0
            while num_so_far < len(ahs):

                action_handle = solver_manager.wait_any()

                if action_handle not in ahs:
                    solver_manager.get_results(action_handle)
                    continue

                results.update(
                    solver_manager.get_results(action_handle))

                num_so_far += 1
        else:

            for scenario in ph._scenario_tree._scenarios:

                results[scenario._name] = \
                    EXTERNAL_collect_cut_data(ph, ph._scenario_tree, scenario)

        return results

    def generate_cut(self, xbars):

        ph = self._ph

        self.update_fix_constraints(xbars)

        ph.solve_subproblems()

        cut_data = self.collect_cut_data()
        benders_cut = BendersOptimalityCut(xbars,
                                           dict((name, cut_data[name]['SSC']) for name in cut_data),
                                           dict((name, cut_data[name]['duals']) for name in cut_data))
        return benders_cut

    def add_cut(self, benders_cut, per_bundle=False):

        master = self._master
        ph = self._ph
        master_bySymbol = master._ScenarioTreeSymbolMap.bySymbol
        rootnode = ph._scenario_tree.findRootNode()
        benders_cuts = master.find_component("PYSP_BENDERS_CUTS_"+str(rootnode._name))
        master_alpha = master.find_component("PYSP_BENDERS_ALPHA_"+str(rootnode._name))
        bundle_alpha = master.find_component("PYSP_BENDERS_BUNDLE_ALPHA_"+str(rootnode._name))

        xbars = benders_cut.xbars
        if per_bundle:

            for i, cut_scenarios in enumerate(getattr(master,"PYSP_BENDERS_CUT_BUNDLES"+str(rootnode._name))):

                cut_expression = 0.0

                for scenario_name in cut_scenarios:
                    scenario_duals = benders_cut.duals[scenario_name]
                    scenario_ssc = benders_cut.ssc[scenario_name]
                    scenario = ph._scenario_tree.get_scenario(scenario_name)
                    #print scenario_duals
                    cut_expression += scenario._probability * \
                                      (scenario_ssc + \
                                       sum(scenario_duals[variable_id]*(master_bySymbol[variable_id]-xbars[variable_id]) \
                                           for variable_id in rootnode._variable_ids))

                cut_expression -= bundle_alpha[i]

                if ph._objective_sense == minimize:

                    benders_cuts.add((None,cut_expression,0.0))

                else:

                    benders_cuts.add((0.0,cut_expression,None))

        else:

            cut_expression = 0.0

            for scenario in ph._scenario_tree._scenarios:
                scenario_name = scenario._name
                scenario_duals = benders_cut.duals[scenario_name]
                scenario_ssc = benders_cut.ssc[scenario_name]
                scenario = ph._scenario_tree.get_scenario(scenario_name)
                #print scenario_duals
                cut_expression += scenario._probability * \
                                  (scenario_ssc + \
                                   sum(scenario_duals[variable_id]*(master_bySymbol[variable_id]-xbars[variable_id]) \
                                       for variable_id in rootnode._variable_ids))

            cut_expression -= master_alpha

            if ph._objective_sense == minimize:

                benders_cuts.add((None,cut_expression,0.0))

            else:

                benders_cuts.add((0.0,cut_expression,None))

        master.preprocess()

    def extract_master_xbars(self):

        master = self._master
        ph = self._ph
        master_bySymbol = master._ScenarioTreeSymbolMap.bySymbol
        rootnode = ph._scenario_tree.findRootNode()
        return dict((variable_id, value(master_bySymbol[variable_id])) \
                    for variable_id in rootnode._variable_ids)

    def solve(self):

        start_time = time.time()

        history_plugin = self._history_plugin
        ph = self._ph
        master = self._master
        master_solver = self._master_solver


        objective_sense = ph._objective_sense
        rootnode = ph._scenario_tree.findRootNode()
        master_alpha = master.find_component("PYSP_BENDERS_ALPHA_"+str(rootnode._name))
        master_objective = master.find_component("PYSP_BENDERS_OBJECTIVE_"+str(rootnode._name))

        print("Determining trivial lower bound using perfect information (on LP relaxation)")
        ph._solver.options['relax_integrality'] = True
        ph.solve_subproblems()
        ph.update_variable_statistics()
        ph._solver.options['relax_integrality'] = False
        trivial_bound = sum(scenario._probability * scenario._objective for scenario in \
                            ph._scenario_tree._scenarios)

        print("Initializing subproblems for benders")
        self.initialize_for_benders()
        self.deactivate_firststage_cost()
        self.activate_fix_constraints()
        if not master_alpha.fixed:
            print("Determining initial alpha bound from scenario solves")

            benders_cut = self.generate_cut(rootnode._xbars)

            self.add_cut(benders_cut)

        MASTER_bound_history = {}
        OBJECTIVE_history = {}
        MASTER_bound_history[0] = trivial_bound
        MASTER_bound_history[-1] = self._options.user_bound if (self._options.user_bound is not None) else \
                                   (float('-inf') if (objective_sense is minimize) else float('inf'))
        first_master_bound = max(MASTER_bound_history) if (objective_sense is minimize) else min(MASTER_bound_history)
        incumbent_objective = float('inf') if (objective_sense is minimize) else float('-inf')
        new_xbars = None

        def print_dictionary(dictionary):
            #Find longest key
            longest_message = max(len(str(x[0])) for x in dictionary)

            #Find longest dictionary value
            longest_value = max(len(str(x[1])) for x in dictionary)
            for key, value in dictionary:
                print(('{:<'+str(longest_message)+'}' '{:^3}' '{:<'+str(longest_value)+'}').format(key,":",value))



        print("-"*20)
        print("Problem Statistics")
        print("-"*20)

        problem_statistics = []
        problem_statistics.append(("Number of first-stage variables"   , str(len(rootnode._variable_ids))+" ("+\
                                   str(len([variable_id for variable_id in rootnode._variable_ids \
                                            if rootnode.is_variable_discrete(variable_id)]))+" integer)"))
        problem_statistics.append(("Number of first-stage constraints" , self._num_first_stage_constraints))
        problem_statistics.append(("Number of scenarios"               , len(rootnode._scenarios)))
        problem_statistics.append(("Number of bundles"                 , len(ph._scenario_tree._scenario_bundles)))
        problem_statistics.append(("Maximum number of iterations"      , self._options.max_iterations))
        problem_statistics.append(("Benders decomposition convergence gap", self._options.percent_gap*100))
        problem_statistics.append(("Trivial Decomposition Bound"       , str(trivial_bound)+" (used for computing the optimality gap)"))
        problem_statistics.append(("User Provided Bound"       , str(self._options.user_bound)+" (used for computing the optimality gap)"))
        print_dictionary(problem_statistics)

        print("")
        width_log_table = 100
        print("-"*width_log_table)
        print("%6s %16s %16s %11s %30s" % ("Iter", "Master Bound", "Best Incumbent", "Gap", "Solution Times [s]"))
        print("%6s %16s %16s %11s %10s %10s %10s %10s" % ("", "", "", "", "Master", "Sub Min", "Sub Max", "Cumm"))
        print("-"*width_log_table)
        ph._current_iteration = 1
        for i in range(1,self._options.max_iterations+1):
            history_plugin.pre_iteration_k_solves(ph)

            ph._current_iteration += 1

            start_time_master =time.time()
            master.preprocess()
            common_kwds = {
                'tee':self._options.master_output_solver_log,
                'keepfiles':self._options.master_keep_solver_files,
                'symbolic_solver_labels':self._options.master_symbolic_solver_labels}
            master_solver.options.mipgap = self._options.master_mipgap
            if (not self._options.master_disable_warmstart) and (master_solver.warm_start_capable()):
                results_master = self._master_solver.solve(master, warmstart=True, **common_kwds)
            else:
                results_master = self._master_solver.solve(master, **common_kwds)

            if len(results_master.solution) == 0:
                raise RuntimeError("Solve failed for master; no solutions generated")
            master.load(results_master)
            stop_time_master = time.time()

            if master_alpha.fixed:
                assert i == 1
                master_alpha.free()

            current_master_bound = value(master_objective)
            solution0 = results_master.solution(0)
            if hasattr(solution0, "gap") and \
               (solution0.gap is not None):
                if objective_sense == minimize:
                    current_master_bound -= solution0.gap
                else:
                    current_master_bound += solution0.gap

            MASTER_bound_history[i] = current_master_bound

            new_xbars = self.extract_master_xbars()

            new_cut_info = self.generate_cut(new_xbars)

            mean    = sum(ph._solve_times.values()) / float(len(ph._solve_times.values()))
            std_dev = sqrt(sum(pow(x-mean,2.0) for x in ph._solve_times.values()) / float(len(ph._solve_times.values())))
            min_time_sub = min(ph._solve_times.values())
            max_time_sub = max(ph._solve_times.values())

            for scenario in ph._scenario_tree._scenarios:
                scenario._w[rootnode._name].update(new_cut_info.duals[scenario._name])

            current_objective = OBJECTIVE_history[i] = \
                current_master_bound - value(master_alpha) + \
                sum(scenario._probability * new_cut_info.ssc[scenario._name] \
                    for scenario in ph._scenario_tree._scenarios)

            incumbent_objective_prev = incumbent_objective
            best_master_bound = max(MASTER_bound_history.values()) if (objective_sense == minimize) else \
                                min(MASTER_bound_history.values())
            incumbent_objective = min(OBJECTIVE_history.values()) if (objective_sense == minimize) else \
                                  max(OBJECTIVE_history.values())
            if objective_sense == minimize:
                if incumbent_objective < incumbent_objective_prev:
                    ph.cacheSolutions(ph._incumbent_cache_id)
            else:
                if incumbent_objective > incumbent_objective_prev:
                    ph.cacheSolutions(ph._incumbent_cache_id)

            optimality_gap = abs(best_master_bound-incumbent_objective)/(1e-10+abs(incumbent_objective))
            print("%6d %16.4f %16.4f %10.2f%% %10.2f %10.2f %10.2f %10.2f"
                  % (i, current_master_bound, incumbent_objective,
                     optimality_gap*100, stop_time_master - start_time_master,
                     min_time_sub, max_time_sub, time.time()-start_time))

            #If the optimality gap is below the convergence threshold set
            #by the user, quit the loop
            if optimality_gap <= self._options.percent_gap:
                print("-"*width_log_table)
                print(" ")
                print("Benders decomposition converged")
                break
            #Else, add a cut to the master problem
            else:
                self.add_cut(new_cut_info)

        history_plugin.pre_iteration_k_solves(ph)

        ph.restoreCachedSolutions(ph._incumbent_cache_id)

        history_plugin.post_ph_execution(ph)

        
        print("")
        print("Restoring scenario tree solution to best incumbent"
              "to best incumbent solution.")
        if (ph._best_incumbent_key is not None) and \
           (ph._best_incumbent_key != ph._current_iteration):
            ph.restoreCachedSolutions(ph._incumbent_cache_id)
        if isinstance(self._solver_manager,
                      pyomo.solvers.plugins.smanager.phpyro.SolverManager_PHPyro):
            collect_full_results(self,
                                 TransmitType.all_stages | \
                                 TransmitType.blended | \
                                 TransmitType.derived | \
                                 TransmitType.fixed)

        print("")
        print("***********************************************************************************************")
        print(">>>THE EXPECTED SUM OF THE STAGE COST VARIABLES="+str(rootnode.computeExpectedNodeCost())+"<<<")
        print("***********************************************************************************************")

        if self._options.output_scenario_tree_solution:
            print("Final solution (scenario tree format):")
            ph._scenario_tree.pprintSolution()

def exec_benders(options):

    import pyomo.environ

    start_time = time.time()
    if options.verbose:
        print("Importing model and scenario tree files")

    scenario_factory = ScenarioTreeInstanceFactory(options.model_directory,
                                                   options.instance_directory,
                                                   options.verbose)

    if options.verbose or options.output_times:
        print("Time to import model and scenario tree structure files=%.2f seconds"
              %(time.time() - start_time))

    solver_manager = None
    try:

        master_solver = SolverFactory(options.master_solver_type)
        if len(options.master_solver_options):
            master_solver.set_options("".join(options.master_solver_options))

        scenario_tree = GenerateScenarioTreeForPH(options, scenario_factory)

        solver_manager = SolverManagerFactory(options.solver_manager_type)

        if isinstance(solver_manager, SolverManager_PHPyro):

            collect_workers(solver_manager, scenario_tree, options)

        benders = BendersAlgorithm(options)

        benders.initialize(options, scenario_tree, solver_manager, master_solver)

        benders.solve()

    finally:

        if master_solver is not None:

            master_solver.deactivate()

        if isinstance(solver_manager, SolverManager_PHPyro):

            solver_manager.release_workers()

        scenario_factory.close()

    print("")
    print("Total execution time=%.2f seconds" %(time.time() - start_time))

#
# the main driver routine for the runph script.
#

def main(args=None):

    #
    # Import plugins
    #
    import pyomo.environ
    #
    # Parse command-line options.
    #
    try:
        benders_options_parser = construct_benders_options_parser("runbenders [options]")
        (options, args) = benders_options_parser.parse_args(args=args)
    except SystemExit:
        # the parser throws a system exit if "-h" is specified - catch
        # it to exit gracefully.
        return

    #
    # Run Benders - precise invocation depends on whether we want profiling
    # output.
    #

    # if an exception is triggered and traceback is enabled, 'ans'
    # won't have a value and the return statement from this function
    # will flag an error, masking the stack trace that you really want
    # to see.
    ans = None

    if pstats_available and options.profile > 0:
        #
        # Call the main Benders routine with profiling.
        #
        tfile = TempfileManager.create_tempfile(suffix=".profile")
        tmp = profile.runctx('exec_benders(options)',globals(),locals(),tfile)
        p = pstats.Stats(tfile).strip_dirs()
        p.sort_stats('time', 'cumulative')
        p = p.print_stats(options.profile)
        p.print_callers(options.profile)
        p.print_callees(options.profile)
        p = p.sort_stats('cumulative','calls')
        p.print_stats(options.profile)
        p.print_callers(options.profile)
        p.print_callees(options.profile)
        p = p.sort_stats('calls')
        p.print_stats(options.profile)
        p.print_callers(options.profile)
        p.print_callees(options.profile)
        TempfileManager.clear_tempfiles()
        ans = [tmp, None]
    else:
        #
        # Call the main Benders routine without profiling.
        #

        if options.traceback:
            ans = exec_benders(options)
        else:
            try:
                try:
                    ans = exec_benders(options)
                except ValueError:
                    str = sys.exc_info()[1]
                    print("VALUE ERROR:")
                    print(str)
                    raise
                except KeyError:
                    str = sys.exc_info()[1]
                    print("KEY ERROR:")
                    print(str)
                    raise
                except TypeError:
                    str = sys.exc_info()[1]
                    print("TYPE ERROR:")
                    print(str)
                    raise
                except NameError:
                    str = sys.exc_info()[1]
                    print("NAME ERROR:")
                    print(str)
                    raise
                except IOError:
                    str = sys.exc_info()[1]
                    print("IO ERROR:")
                    print(str)
                    raise
                except pyutilib.common.ApplicationError:
                    str = sys.exc_info()[1]
                    print("APPLICATION ERROR:")
                    print(str)
                    raise
                except RuntimeError:
                    str = sys.exc_info()[1]
                    print("RUN-TIME ERROR:")
                    print(str)
                    raise
                except:
                    print("Encountered unhandled exception")
                    traceback.print_exc()
                    raise
            except:
                print("\n")
                print("To obtain further information regarding the "
                      "source of the exception, use the --traceback option")

                # if an exception is triggered, and we're running with
                # pyro, shut down everything - not doing so is
                # annoying, and leads to a lot of wasted compute
                # time. but don't do this if the shutdown-pyro option
                # is disabled => the user wanted
                if ((options.solver_manager_type == "pyro") or \
                    (options.solver_manager_type == "phpyro")) and \
                    options.shutdown_pyro:
                    print("\n")
                    print("Shutting down Pyro solver components, "
                          "following exception trigger")
                    shutDownPyroComponents()

    gc.enable()

    return ans


@pyomo_command('runbenders', 'Optimize with the Benders solver')
def Benders_main(args=None):
    return main(args=args)
