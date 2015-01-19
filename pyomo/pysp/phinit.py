#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

import gc      # garbage collection control.
import os
import pickle  # for serializing
import sys
import tempfile
import shutil
import string
import time
import contextlib
try:
    import pstats
    pstats_available=True
except ImportError:
    pstats_available=False

from optparse import OptionParser, OptionGroup

# for profiling
try:
    import cProfile as profile
except ImportError:
    import profile

try:
    from pympler.muppy import muppy
    from pympler.muppy import summary
    from pympler.muppy import tracker
    from pympler.asizeof import *
    pympler_available = True
except ImportError:
    pympler_available = False

from pyutilib.pyro import shutdown_pyro_components
from pyomo.util import pyomo_command
from pyomo.util.plugin import ExtensionPoint
from pyutilib.misc import import_file
from pyutilib.services import TempfileManager
from pyutilib.misc import ArchiveReaderFactory, ArchiveReader

from pyomo.opt.base import SolverFactory
from pyomo.opt.parallel import SolverManagerFactory
from pyomo.pysp.ef_writer_script import EF_DefaultOptions, EFAlgorithmBuilder
from pyomo.pysp.ph import *
from pyomo.pysp.phutils import reset_nonconverged_variables, reset_stage_cost_variables
from pyomo.pysp.scenariotree import *
from pyomo.pysp.solutionwriter import ISolutionWriterExtension
from pyomo.solvers.plugins.smanager.phpyro import SolverManager_PHPyro
from pyomo.solvers.plugins.smanager.pyro import SolverManager_Pyro

#
# utility method to construct an option parser for ph arguments,
# to be supplied as an argument to the runph method.
#

from pyomo.pysp.ph import _OLD_OUTPUT

def construct_ph_options_parser(usage_string):

    solver_list = SolverFactory.services()
    solver_list = sorted( filter(lambda x: '_' != x[0], solver_list) )
    solver_help = \
    "Specify the solver with which to solve scenario sub-problems.  The "      \
    "following solver types are currently supported: %s; Default: cplex"
    solver_help %= ', '.join( solver_list )

    parser = OptionParser()
    parser.usage = usage_string

    # NOTE: these groups should eventually be queried from the PH, scenario tree, etc. classes (to facilitate re-use).
    inputOpts        = OptionGroup( parser, 'Input Options' )
    scenarioTreeOpts = OptionGroup( parser, 'Scenario Tree Options' )
    phOpts           = OptionGroup( parser, 'PH Options' )
    solverOpts       = OptionGroup( parser, 'Solver Options' )
    postprocessOpts  = OptionGroup( parser, 'Postprocessing Options' )
    outputOpts       = OptionGroup( parser, 'Output Options' )
    otherOpts        = OptionGroup( parser, 'Other Options' )

    parser.add_option_group( inputOpts )
    parser.add_option_group( scenarioTreeOpts )
    parser.add_option_group( phOpts )
    parser.add_option_group( solverOpts )
    parser.add_option_group( postprocessOpts )
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
    inputOpts.add_option('-p','--ph-warmstart-file',
      help="Disable iteration 0 solves and warmstarts rho, weight, and xbar parameters from solution file.",
      action="store",
      dest="ph_warmstart_file",
      type=str,
      default=None)
    inputOpts.add_option('--bounds-cfgfile',
      help="The name of python script containing a ph_boundsetter_callback function to compute and update scenario variable bounds. Default is None.",
      action="store",
      dest="bounds_cfgfile",
      default=None)

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

    phOpts.add_option('-r','--default-rho',
      help="The default (global) rho for all blended variables. *** Required ***",
      action="store",
      dest="default_rho",
      type=str,
      default="")
    phOpts.add_option("--overrelax",
      help="Compute weight updates using combination of previous and current variable averages",
      action="store_true",
      dest="overrelax",
      default=False)
    phOpts.add_option("--nu",
      action="store",
      dest="nu",
      type="float",
      default=1.5)
    phOpts.add_option("--async",
      help="Run PH in asychronous mode after iteration 0. Default is False.",
      action="store_true",
      dest="async",
      default=False)
    phOpts.add_option("--async-buffer-len",
      help="Number of scenarios to collect, if in async mode, before doing statistics and weight updates. Default is 1.",
      action="store",
      dest="async_buffer_len",
      type = "int",
      default=1)
    phOpts.add_option('--rho-cfgfile',
      help="The name of python script containing a ph_rhosetter_callback function to compute and update PH rho values. Default is None.",
      action="store",
      dest="rho_cfgfile",
      type="string",
      default=None)
    phOpts.add_option('--aggregate-cfgfile',
      help="The name of python script containing a ph_aggregategetter_callback function to collect and store aggregate scenario data on PH. Default is None.",
      action="store",
      dest="aggregate_cfgfile",
      type="string",
      default=None)
    phOpts.add_option('--max-iterations',
      help="The maximal number of PH iterations. Default is 100.",
      action="store",
      dest="max_iterations",
      type="int",
      default=100)
    phOpts.add_option('--termdiff-threshold',
      help="The convergence threshold used in the term-diff and normalized term-diff convergence criteria. Default is 0.0001.",
      action="store",
      dest="termdiff_threshold",
      type="float",
      default=0.0001)
    phOpts.add_option('--enable-free-discrete-count-convergence',
      help="Terminate PH based on the free discrete variable count convergence metric. Default is False.",
      action="store_true",
      dest="enable_free_discrete_count_convergence",
      default=False)
    phOpts.add_option('--enable-normalized-termdiff-convergence',
      help="Terminate PH based on the normalized termdiff convergence metric. Default is True.",
      action="store_true",
      dest="enable_normalized_termdiff_convergence",
      default=True)
    phOpts.add_option('--enable-termdiff-convergence',
      help="Terminate PH based on the termdiff convergence metric. Default is False.",
      action="store_true",
      dest="enable_termdiff_convergence",
      default=False)
    phOpts.add_option('--free-discrete-count-threshold',
      help="The convergence threshold used in the criterion based on when the free discrete variable count convergence criterion. Default is 20.",
      action="store",
      dest="free_discrete_count_threshold",
      type="float",
      default=20)
    phOpts.add_option('--linearize-nonbinary-penalty-terms',
      help="Approximate the PH quadratic term for non-binary variables with a piece-wise linear function, using the supplied number of equal-length pieces from each bound to the average",
      action="store",
      dest="linearize_nonbinary_penalty_terms",
      type="int",
      default=0)
    phOpts.add_option('--breakpoint-strategy',
      help="Specify the strategy to distribute breakpoints on the [lb, ub] interval of each variable when linearizing. 0 indicates uniform distribution. 1 indicates breakpoints at the node min and max, uniformly in-between. 2 indicates more aggressive concentration of breakpoints near the observed node min/max.",
      action="store",
      dest="breakpoint_strategy",
      type="int",
      default=0)
    phOpts.add_option('--retain-quadratic-binary-terms',
      help="Do not linearize PH objective terms involving binary decision variables",
      action="store_true",
      dest="retain_quadratic_binary_terms",
      default=False)
    phOpts.add_option('--drop-proximal-terms',
      help="Eliminate proximal terms (i.e., the quadratic penalty terms) from the weighted PH objective. Default is False.",
      action="store_true",
      dest="drop_proximal_terms",
      default=False)
    phOpts.add_option('--enable-ww-extensions',
      help="Enable the Watson-Woodruff PH extensions plugin. Default is False.",
      action="store_true",
      dest="enable_ww_extensions",
      default=False)
    phOpts.add_option('--ww-extension-cfgfile',
      help="The name of a configuration file for the Watson-Woodruff PH extensions plugin.",
      action="store",
      dest="ww_extension_cfgfile",
      type="string",
      default="")
    phOpts.add_option('--ww-extension-suffixfile',
      help="The name of a variable suffix file for the Watson-Woodruff PH extensions plugin.",
      action="store",
      dest="ww_extension_suffixfile",
      type="string",
      default="")
    phOpts.add_option('--ww-extension-annotationfile',
      help="The name of a variable annotation file for the Watson-Woodruff PH extensions plugin.",
      action="store",
      dest="ww_extension_annotationfile",
      type="string",
      default="")
    phOpts.add_option('--user-defined-extension',
      help="The name of a python module specifying a user-defined PH extension plugin.",
      action="append",
      dest="user_defined_extensions",
      type="string",
      default=[])
    phOpts.add_option("--flatten-expressions", "--linearize-expressions",
      help="EXPERIMENTAL: An option intended for use on linear or mixed-integer models " \
           "in which expression trees in a model (constraints or objectives) are compacted " \
           "into a more memory-efficient and concise form. The trees themselves are eliminated. ",
      action="store_true",
      dest="flatten_expressions",
      default=False)
    phOpts.add_option('--preprocess-fixed-variables',
      help="Preprocess fixed/freed variables in scenario instances, rather than write them to solver plugins. Default is False.",
      action="store_false",
      dest="write_fixed_variables",
      default=True)

    solverOpts.add_option('--scenario-mipgap',
      help="Specifies the mipgap for all PH scenario sub-problems",
      action="store",
      dest="scenario_mipgap",
      type="float",
      default=None)
    solverOpts.add_option('--scenario-solver-options',
      help="Solver options for all PH scenario sub-problems",
      action="append",
      dest="scenario_solver_options",
      type="string",
      default=[])
    solverOpts.add_option('--solver',
      help=solver_help,
      action="store",
      dest="solver_type",
      type="string",
      default="cplex")
    solverOpts.add_option('--solver-io',
      help='The type of IO used to execute the solver.  Different solvers support different types of IO, but the following are common options: lp - generate LP files, nl - generate NL files, python - direct Python interface, os - generate OSiL XML files.',
      action='store',
      dest='solver_io',
      default=None)
    solverOpts.add_option('--solver-manager',
      help="The type of solver manager used to coordinate scenario sub-problem solves. Default is serial.",
      action="store",
      dest="solver_manager_type",
      type="string",
      default="serial")
    solverOpts.add_option('--pyro-hostname',
      help="The hostname to bind on. By default, the first dispatcher found will be used. This option can also help speed up initialization time if the hostname is known (e.g., localhost)",
      action="store",
      dest="pyro_hostname",
      default=None)
    solverOpts.add_option('--handshake-with-phpyro',
      help="When updating weights, xbars, and rhos across the PHPyro solver manager, it is often expedient to ignore the simple acknowledgement results returned by PH solver servers. Enabling this option instead enables hand-shaking, to ensure message receipt. Clearly only makes sense if the PHPyro solver manager is selected",
      action="store_true",
      dest="handshake_with_phpyro",
      default=False)
    solverOpts.add_option('--phpyro-required-workers',
      help="Set the number of idle phsolverserver worker processes expected to be available when the PHPyro solver manager is selected. This option should be used when the number of worker threads is less than the total number of scenarios (or bundles). When this option is not used, PH will attempt to assign each scenario (or bundle) to a single phsolverserver until the timeout indicated by the --phpyro-workers-timeout option occurs.",
      action="store",
      type=int,
      dest="phpyro_required_workers",
      default=None)
    solverOpts.add_option('--phpyro-workers-timeout',
     help="Set the time limit (seconds) for finding idle phsolverserver worker processes to be used when the PHPyro solver manager is selected. This option is ignored when --phpyro-required-workers is set manually. Default is 30.",
      action="store",
      type=float,
      dest="phpyro_workers_timeout",
      default=30)
    solverOpts.add_option('--phpyro-transmit-leaf-stage-variable-solution',
      help="By default, when running PH using the PHPyro solver manager, leaf-stage variable solutions are not transmitted back to the master PH instance during intermediate PH iterations. This flag will override that behavior for the rare cases where these values are needed. Using this option will possibly have a negative impact on runtime for PH iterations. When PH exits, variable values are collected from all stages whether or not this option was used. Also, note that PH extensions have the ability to override this flag at runtime.",
      action="store_true",
      dest="phpyro_transmit_leaf_stage_solution",
      default=False)
    solverOpts.add_option('--disable-warmstarts',
      help="Disable warm-start of scenario sub-problem solves in PH iterations >= 1. Default is False.",
      action="store_true",
      dest="disable_warmstarts",
      default=False)
    solverOpts.add_option('--shutdown-pyro',
      help="Shut down all Pyro-related components associated with the Pyro and PH Pyro solver managers (if specified), including the dispatch server, name server, and any solver servers. Default is False.",
      action="store_true",
      dest="shutdown_pyro",
      default=False)

    solverOpts.add_option('--ef-disable-warmstarts',
      help="Override the runph option of the same name during the EF solve.",
      action="store_true",
      dest="ef_disable_warmstarts",
      default=None)
    postprocessOpts.add_option('--ef-output-file',
      help="The basename of the extensive form output file (currently only LP and NL formats are supported), if writing or solving of the extensive form is enabled. The full output filename will be of the form '<basename>.{lp,nl}', where the suffix type is determined by the value of the --ef-solver-io or --solver-io option. Default is 'efout'.",
      action="store",
      dest="ef_output_file",
      type="string",
      default="efout")
    postprocessOpts.add_option('--solve-ef',
      help="Upon termination, create the extensive-form model and solve it - accounting for all fixed variables.",
      action="store_true",
      dest="solve_ef",
      default=False)
    postprocessOpts.add_option('--ef-solver',
      help="Override the runph option of the same name during the EF solve.",
      action="store",
      dest="ef_solver_type",
      type="string",
      default=None)
    postprocessOpts.add_option('--ef-solution-writer',
      help="The plugin invoked to write the scenario tree solution following the EF solve. If specified, overrides the runph option of the same name; otherwise, the runph option value will be used.",
      action="append",
      dest="ef_solution_writer",
      type="string",
      default = [])
    postprocessOpts.add_option('--ef-solver-io',
      help="Override the runph option of the same name during the EF solve.",
      action='store',
      dest='ef_solver_io',
      type='string',
      default=None)
    postprocessOpts.add_option('--ef-solver-manager',
      help="The type of solver manager used to execute the extensive form solve. Default is serial. This option is not inherited from the runph scenario-based option.",
      action="store",
      dest="ef_solver_manager_type",
      type="string",
      default="serial")
    postprocessOpts.add_option('--ef-mipgap',
      help="Specifies the mipgap for the EF solve. This option is not inherited from the runph scenario-based option.",
      action="store",
      dest="ef_mipgap",
      type="float",
      default=None)
    postprocessOpts.add_option('--ef-disable-warmstart',
      help="Disable warm-start of the post-PH EF solve. Default is False. This option is not inherited from the runph scenario-based option.",
      action="store_true",
      dest="ef_disable_warmstart",
      default=False)
    postprocessOpts.add_option('--ef-solver-options',
      help="Solver options for the EF problem. This option is not inherited from the runph scenario-based option.",
      action="append",
      dest="ef_solver_options",
      type="string",
      default=[])
    postprocessOpts.add_option('--ef-output-solver-log',
      help="Override the runph option of the same name during the EF solve.",
      action="store_true",
      dest="ef_output_solver_log",
      default=None)
    postprocessOpts.add_option('--ef-keep-solver-files',
      help="Override the runph option of the same name during the EF solve.",
      action="store_true",
      dest="ef_keep_solver_files",
      default=None)
    postprocessOpts.add_option('--ef-symbolic-solver-labels',
      help='Override the runph option of the same name during the EF solve.',
      action='store_true',
      dest='ef_symbolic_solver_labels',
      default=None)

    outputOpts.add_option('--output-scenario-tree-solution',
      help="Report the full solution (even leaves) in scenario tree format upon termination. Values represent averages, so convergence is not an issue. Default is False.",
      action="store_true",
      dest="output_scenario_tree_solution",
      default=False)
    outputOpts.add_option('--output-solver-logs',
      help="Output solver logs during scenario sub-problem solves",
      action="store_true",
      dest="output_solver_logs",
      default=False)
    outputOpts.add_option('--symbolic-solver-labels',
      help='When interfacing with the solver, use symbol names derived from the model. For example, \"my_special_variable[1_2_3]\" instead of \"v1\". Useful for debugging. When using the ASL interface (--solver-io=nl), generates corresponding .row (constraints) and .col (variables) files. The ordering in these files provides a mapping from ASL index to symbolic model names.',
      action='store_true',
      dest='symbolic_solver_labels',
      default=False)
    outputOpts.add_option('--output-solver-results',
      help="Output solutions obtained after each scenario sub-problem solve",
      action="store_true",
      dest="output_solver_results",
      default=False)
    outputOpts.add_option('--output-times',
      help="Output timing statistics for various PH components",
      action="store_true",
      dest="output_times",
      default=False)
    outputOpts.add_option('--output-instance-construction-times',
      help="Output timing statistics for instance construction timing statistics (client-side only when using PHPyro",
      action="store_true",
      dest="output_instance_construction_times",
      default=False)
    outputOpts.add_option('--report-only-statistics',
      help="When reporting solutions (if enabled), only output per-variable statistics - not the individual scenario values. Default is False.",
      action="store_true",
      dest="report_only_statistics",
      default=False)
    outputOpts.add_option('--report-solutions',
      help="Always report PH solutions after each iteration. Enabled if --verbose is enabled. Default is False.",
      action="store_true",
      dest="report_solutions",
      default=False)
    outputOpts.add_option('--report-weights',
      help="Always report PH weights prior to each iteration. Enabled if --verbose is enabled. Default is False.",
      action="store_true",
      dest="report_weights",
      default=False)
    outputOpts.add_option('--report-rhos-all-iterations',
      help="Always report PH rhos prior to each iteration. Default is False.",
      action="store_true",
      dest="report_rhos_each_iteration",
      default=False)
    outputOpts.add_option('--report-rhos-first-iterations',
      help="Report rhos prior to PH iteration 1. Enabled if --verbose is enabled. Default is False.",
      action="store_true",
      dest="report_rhos_first_iteration",
      default=False)
    outputOpts.add_option('--report-for-zero-variable-values',
      help="Report statistics (variables and weights) for all variables, not just those with values differing from 0. Default is False.",
      action="store_true",
      dest="report_for_zero_variable_values",
      default=False)
    outputOpts.add_option('--report-only-nonconverged-variables',
      help="Report statistics (variables and weights) only for non-converged variables. Default is False.",
      action="store_true",
      dest="report_only_nonconverged_variables",
      default=False)
    outputOpts.add_option('--restore-from-checkpoint',
      help="The name of the checkpoint file from which PH should be initialized. Default is None",
      action="store",
      dest="restore_from_checkpoint",
      default=None)
    outputOpts.add_option('--solution-writer',
      help="The plugin invoked to write the scenario tree solution. Defaults to the empty list.",
      action="append",
      dest="solution_writer",
      type="string",
      default = [])
    outputOpts.add_option('--suppress-continuous-variable-output',
      help="Eliminate PH-related output involving continuous variables.",
      action="store_true",
      dest="suppress_continuous_variable_output",
      default=False)
    outputOpts.add_option('--verbose',
      help="Generate verbose output for both initialization and execution. Default is False.",
      action="store_true",
      dest="verbose",
      default=False)
    outputOpts.add_option('--write-ef',
      help="Upon termination, create the extensive-form model and write it to a file - accounting for all fixed variables.",
      action="store_true",
      dest="write_ef",
      default=False)

    otherOpts.add_option('--disable-gc',
      help="Disable the python garbage collecter. Default is False.",
      action="store_true",
      dest="disable_gc",
      default=False)
    if pympler_available:
        otherOpts.add_option("--profile-memory",
                             help="If Pympler is available (installed), report memory usage statistics for objects created after each PH iteration. A value of 0 indicates disabled. A value of 1 forces summary output after each PH iteration >= 1. Values greater than 2 are currently not supported.",
                             action="store",
                             dest="profile_memory",
                             type=int,
                             default=0)
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
    otherOpts.add_option('--checkpoint-interval',
      help="The number of iterations between writing of a checkpoint file. Default is 0, indicating never.",
      action="store",
      dest="checkpoint_interval",
      type="int",
      default=0)
    otherOpts.add_option('--traceback',
      help="When an exception is thrown, show the entire call stack. Ignored if profiling is enabled. Default is False.",
      action="store_true",
      dest="traceback",
      default=False)

    return parser


def PH_DefaultOptions():
    parser = construct_ph_options_parser("")
    options, _ = parser.parse_args([''])
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
        bundles_file=options.scenario_bundle_specification,
        random_bundles=options.create_random_bundles,
        random_seed=options.scenario_tree_random_seed)

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
                flatten_expressions=options.flatten_expressions,
                report_timing=options.output_times,
                preprocess=False)

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
                module_to_find = string.split(module_to_find,"/")[-1]

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

        from pyomo.pysp.plugins import wwphextension

        # explicitly enable the WW extension plugin - it may have been
        # previously loaded and/or enabled.
        ph_extension_point = ExtensionPoint(IPHExtension)

        for plugin in ph_extension_point(all=True):
           if isinstance(plugin, wwphextension.wwphextension):
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
                module_to_find = string.split(module_to_find,"/")[-1]

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
        solver_manager = SolverManagerFactory(options.solver_manager_type,
                                              host=options.pyro_hostname)
        if solver_manager is None:
            raise ValueError("Failed to create solver manager of "
                             "type="+options.solver_manager_type+
                         " specified in call to PH constructor")

        ph = ProgressiveHedging(options)

        if isinstance(solver_manager, SolverManager_PHPyro):

            if scenario_tree.contains_bundles():
                num_jobs = len(scenario_tree._scenario_bundles)
                if not _OLD_OUTPUT:
                    print("Bundle solver jobs available: "+str(num_jobs))
            else:
                num_jobs = len(scenario_tree._scenarios)
                if not _OLD_OUTPUT:
                    print("Scenario solver jobs available: "+str(num_jobs))

            workers_expected = options.phpyro_required_workers
            if (workers_expected is None):
                workers_expected = num_jobs

            timeout = options.phpyro_workers_timeout if \
                      (options.phpyro_required_workers is None) else \
                      None

            solver_manager.acquire_workers(workers_expected,
                                           timeout)

        ph.initialize(scenario_tree=scenario_tree,
                      solver_manager=solver_manager,
                      ph_plugins=ph_plugins,
                      solution_plugins=solution_plugins)

    except:

        if ph is not None:

            ph.release_components()

        if solver_manager is not None:

            if isinstance(solver_manager, SolverManager_PHPyro):
                solver_manager.release_workers()

            solver_manager.deactivate()

        print("Failed to initialize PH Algorithm")
        raise

    return ph

def PHFromScratch(options):

    start_time = time.time()
    if options.verbose:
        print("Importing model and scenario tree files")

    scenario_instance_factory = \
        ScenarioTreeInstanceFactory(options.model_directory,
                                    options.instance_directory,
                                    options.verbose)

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

        print("Failed to initialize ProgessiveHedging algorithm instance")
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
        
        if isinstance(ph._solver_manager,
                      SolverManager_PHPyro):

            ph._solver_manager.release_workers()

        ph._solver_manager.deactivate()

    if ph._scenario_tree is not None:

        if ph._scenario_tree._scenario_instance_factory is not None:

            ph._scenario_tree._scenario_instance_factory.close()

#
# Given a PH object, execute it and optionally solve the EF at the
# end.
#

def run_ph(options, ph):

    start_time = time.time()

    #
    # kick off the solve
    #
    retval = ph.solve()
    if retval is not None:
        raise RuntimeError("No solution was obtained for scenario: "+retval)

    end_ph_time = time.time()

    print("")
    print("Total PH execution time=%.2f seconds" %(end_ph_time - start_time))
    print("")
    if options.output_times:
        ph.print_time_stats()

    ph.save_solution()

    #
    # create the extensive form binding instance, so that we can
    # either write or solve it (if specified).
    #
    if (options.write_ef) or (options.solve_ef):

        if not isinstance(ph._solver_manager, SolverManager_PHPyro):

            # The instances are about to be added as sublocks to the
            # extensive form instance. If bundles exist, we must
            # distroy them to avoid errors
            ph._destory_bundle_binding_instances()

        else:

            print("Constructing scenario instances for extensive form solve")

            instances = ph._scenario_tree._scenario_instance_factory.\
                        construct_instances_for_scenario_tree(
                            ph._scenario_tree,
                            flatten_expressions=options.flatten_expressions,
                            report_timing=options.output_times,
                            preprocess=False)

            ph._scenario_tree.linkInInstances(
                instances,
                create_variable_ids=False,
                master_scenario_tree=ph._scenario_tree,
                initialize_solution_data=False)

            ph._setup_scenario_instances()
            
            # if specified, run the user script to initialize variable
            # bounds at their whim.
            if ph._bound_setter is not None:

                print("Executing user bound setter callback function")
                for scenario in ph._scenario_tree._scenarios:
                    ph._callback_function[ph._bound_setter](
                        ph,
                        ph._scenario_tree,
                        scenario)

            # warm start the instances
            for scenario in ph._scenario_tree._scenarios:
                scenario.push_solution_to_instance()

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

        # TODO: There _is_ a better way to push runph "ef" options
        #       onto a runef options object
        ef_options = EF_DefaultOptions()
        ef_options.verbose = options.verbose
        ef_options.output_times = options.output_times
        ef_options.output_file = options.ef_output_file
        ef_options.solve_ef = options.solve_ef
        ef_options.solver_manager_type = options.ef_solver_manager_type
        if ef_options.solver_manager_type == "phpyro":
            print("*** WARNING ***: PHPyro is not a supported solver "
                  "manager type for the extensive-form solver. "
                  "Falling back to serial.")
            ef_options.solver_manager_type = 'serial'
        ef_options.mipgap = options.ef_mipgap
        ef_options.solver_options = options.ef_solver_options
        ef_options.disable_warmstart = options.ef_disable_warmstart
        #
        # The following options will inherit the runph option if not
        # specified
        #
        if options.ef_disable_warmstarts is not None:
            ef_options.disable_warmstarts = options.ef_disable_warmstarts
        else:
            ef_options.disable_warmstarts = options.disable_warmstarts
        if len(options.ef_solution_writer) > 0:
            ef_options.solution_writer = options.ef_solution_writer
        else:
            ef_options.solution_writer = options.solution_writer
        if options.ef_solver_io is not None:
            ef_options.solver_io = options.ef_solver_io
        else:
            ef_options.solver_io = options.solver_io
        if options.ef_solver_type is not None:
            ef_options.solver_type = options.ef_solver_type
        else:
            ef_options.solver_type = options.solver_type
        if options.ef_output_solver_log is not None:
            ef_options.output_solver_log = options.ef_output_solver_log
        else:
            ef_options.output_solver_log = options.output_solver_logs
        if options.ef_keep_solver_files is not None:
            ef_options.keep_solver_files = options.ef_keep_solver_files
        else:
            ef_options.keep_solver_files = options.keep_solver_files
        if options.ef_symbolic_solver_labels is not None:
            ef_options.symbolic_solver_labels = options.ef_symbolic_solver_labels
        else:
            ef_options.symbolic_solver_labels = options.symbolic_solver_labels

        if _OLD_OUTPUT:
            print("Creating extensive form for remainder problem")

        ef = EFAlgorithmBuilder(ef_options, ph._scenario_tree)

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

        ef.write()

    if options.solve_ef:

        ef.solve()
        # This is a hack. I think this method should be on the scenario tree

        ph.update_variable_statistics()
        # 
        ef.save_solution(label="postphef")

#
# The main PH initialization / runner routine. Really only branches
# based on the construction source - a checkpoint or from scratch.
#

def exec_ph(options):

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
            shutdown_pyro_components()


    print("")
    print("Total execution time=%.2f seconds"
          % (time.time() - start_time))

#
# the main driver routine for the runph script.
#

def main(args=None):
    #
    # Top-level command that executes the extensive form writer.
    # This is segregated from run_ef_writer to enable profiling.
    #

    #
    # Import plugins
    #
    import pyomo.environ
    #
    # Parse command-line options.
    #
    try:
        ph_options_parser = construct_ph_options_parser("runph [options]")
        (options, args) = ph_options_parser.parse_args(args=args)
    except SystemExit:
        # the parser throws a system exit if "-h" is specified - catch
        # it to exit gracefully.
        return
    #
    # Control the garbage collector - more critical than I would like
    # at the moment.
    #

    if options.disable_gc:
        gc.disable()
    else:
        gc.enable()

    #
    # Run PH - precise invocation depends on whether we want profiling
    # output.
    #

    # if an exception is triggered and traceback is enabled, 'ans'
    # won't have a value and the return statement from this function
    # will flag an error, masking the stack trace that you really want
    # to see.
    ans = None

    if pstats_available and options.profile > 0:
        #
        # Call the main PH routine with profiling.
        #
        tfile = TempfileManager.create_tempfile(suffix=".profile")
        tmp = profile.runctx('exec_ph(options)',globals(),locals(),tfile)
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
        # Call the main PH routine without profiling.
        #

        if options.traceback:
            ans = exec_ph(options)
        else:
            try:
                try:
                    ans = exec_ph(options)
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

    gc.enable()

    return ans

@pyomo_command('runph', 'Optimize with the PH solver (primal search)')
def PH_main(args=None):
    return main(args=args)


