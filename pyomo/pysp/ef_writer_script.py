#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2009 Sandia Corporation.
#  This software is distributed under the BSD License.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  For more information, see the Pyomo README.txt file.
#  _________________________________________________________________________

import gc
import os
import sys
import string
import traceback
import shutil
try:
    import pstats
    pstats_available=True
except ImportError:
    pstats_available=False

from optparse import OptionParser, OptionGroup

try:
    import cProfile as profile
except ImportError:
    import profile

from pyomo.util import pyomo_command
from pyomo.util.plugin import ExtensionPoint
from pyutilib.services import TempfileManager
from pyomo.opt.base import SolverFactory, ConverterError, ProblemFormat
from pyomo.opt.base.solvers import UnknownSolver
from pyomo.opt.parallel import SolverManagerFactory
from pyomo.pysp.ef import *
from pyomo.pysp.solutionwriter import ISolutionWriterExtension
from pyomo.solvers.plugins.smanager.phpyro import SolverManager_PHPyro
from pyomo.solvers.plugins.smanager.pyro import SolverManager_Pyro

from six import iteritems

#
# utility method to construct an option parser for ef writer arguments
#

def construct_ef_writer_options_parser(usage_string):

    solver_list = SolverFactory.services()
    solver_list = sorted( filter(lambda x: '_' != x[0], solver_list) )
    solver_help = \
    "Specify the solver with which to solve the extensive form.  The "      \
    "following solver types are currently supported: %s; Default: cplex"
    solver_help %= ', '.join( solver_list )

    parser = OptionParser()
    parser.usage=usage_string

    inputOpts        = OptionGroup( parser, 'Input Options' )
    scenarioTreeOpts = OptionGroup( parser, 'Scenario Tree Options' )
    efOpts           = OptionGroup( parser, 'EF Options' )
    ccOpts           = OptionGroup( parser, 'Chance Constraint Options' )
    solverOpts       = OptionGroup( parser, 'Solver Options' )
    outputOpts       = OptionGroup( parser, 'Output Options' )
    otherOpts        = OptionGroup( parser, 'Other Options' )
    parser.add_option_group( inputOpts )
    parser.add_option_group( scenarioTreeOpts )
    parser.add_option_group( efOpts )
    parser.add_option_group( ccOpts )
    parser.add_option_group( solverOpts )
    parser.add_option_group( outputOpts )
    parser.add_option_group( otherOpts )

    inputOpts.add_option('-i','--instance-directory',
      help='The directory in which all instance (reference and scenario) definitions are stored. Default is ".".',
      action='store',
      dest='instance_directory',
      type='string',
      default='.')
    inputOpts.add_option('-m','--model-directory',
      help='The directory in which all model (reference and scenario) definitions are stored. Default is ".".',
      action='store',
      dest='model_directory',
      type='string',
      default='.')
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
      help="The random seed associated with manipulation operations on the scenario tree (e.g., down-sampling). Default is 0, indicating unassigned.",
      action="store",
      dest="scenario_tree_random_seed",
      type="int",
      default=None)
    scenarioTreeOpts.add_option('--scenario-tree-downsample-fraction',
      help="The proportion of the scenarios in the scenario tree that are actually used. Specific scenarios are selected at random. Default is 1.0, indicating no down-sampling.",
      action="store",
      dest="scenario_tree_downsample_fraction",
      type="float",
      default=1.0)

    efOpts.add_option('--cvar-weight',
      help='The weight associated with the CVaR term in the risk-weighted objective formulation. Default is 1.0. If the weight is 0, then *only* a non-weighted CVaR cost will appear in the EF objective - the expected cost component will be dropped.',
      action='store',
      dest='cvar_weight',
      type='float',
      default=1.0)
    efOpts.add_option('--generate-weighted-cvar',
      help='Add a weighted CVaR term to the primary objective',
      action='store_true',
      dest='generate_weighted_cvar',
      default=False)
    efOpts.add_option('--risk-alpha',
      help='The probability threshold associated with cvar (or any future) risk-oriented performance metrics. Default is 0.95.',
      action='store',
      dest='risk_alpha',
      type='float',
      default=0.95)
    efOpts.add_option("--flatten-expressions", "--linearize-expressions",
      help="EXPERIMENTAL: An option intended for use on linear or mixed-integer models " \
           "in which expression trees in a model (constraints or objectives) are compacted " \
           "into a more memory-efficient and concise form. The trees themselves are eliminated. ",
      action="store_true",
      dest="flatten_expressions",
      default=False)

    ccOpts.add_option('--cc-alpha',
      help='The probability threshold associated with a chance constraint. The RHS will be one minus this value. Default is 0.',
      action='store',
      dest='cc_alpha',
      type='float',
      default=0.0)

    ccOpts.add_option('--cc-indicator-var',
      help='The name of the binary variable to be used to construct a chance constraint. Default is None, which indicates no chance constraint.',
      action='store',
      dest='cc_indicator_var',
      type='string',
      default=None)

    solverOpts.add_option('--mipgap',
      help='Specifies the mipgap for the EF solve.',
      action='store',
      dest='mipgap',
      type='float',
      default=None)
    solverOpts.add_option('--solve',
      help='Following write of the extensive form model, solve it.',
      action='store_true',
      dest='solve_ef',
      default=False)
    solverOpts.add_option('--solver',
      help=solver_help,
      action='store',
      dest='solver_type',
      type='string',
      default='cplex')
    solverOpts.add_option('--solver-io',
      help='The type of IO used to execute the solver.  Different solvers support different types of IO, but the following are common options: lp - generate LP files, nl - generate NL files, python - direct Python interface, os - generate OSiL XML files.',
      action='store',
      dest='solver_io',
      default=None)
    solverOpts.add_option('--solver-manager',
      help='The type of solver manager used to coordinate scenario sub-problem solves. Default is serial.',
      action='store',
      dest='solver_manager_type',
      type='string',
      default='serial')
    solverOpts.add_option('--solver-options',
      help='Solver options for the extensive form problem.',
      action='append',
      dest='solver_options',
      type='string',
      default=[])
    solverOpts.add_option('--disable-warmstarts',
      help="Disable warm-starts of EF solves. Default is False.",
      action="store_true",
      dest="disable_warmstarts",
      default=False)
    solverOpts.add_option('--shutdown-pyro',
      help="Shut down all Pyro-related components associated with the Pyro solver manager (if specified), including the dispatch server, name server, and any mip servers. Default is False.",
      action="store_true",
      dest="shutdown_pyro",
      default=False)

    outputOpts.add_option('--output-file',
      help="The name of the extensive form output file (currently only LP and NL file formats are supported). If the option name does not end in '.lp' or '.nl', then the output format will be determined by the value of the --solver-io option, and the appropriate ending suffix will be added to the name. Default is 'efout'.",
      action='store',
      dest='output_file',
      type='string',
      default="efout")
    outputOpts.add_option('--symbolic-solver-labels',
      help='When interfacing with the solver, use symbol names derived from the model. For example, \"my_special_variable[1_2_3]\" instead of \"v1\". Useful for debugging. When using the ASL interface (--solver-io=nl), generates corresponding .row (constraints) and .col (variables) files. The ordering in these files provides a mapping from ASL index to symbolic model names.',
      action='store_true',
      dest='symbolic_solver_labels',
      default=False)
    outputOpts.add_option('--output-solver-log',
      help='Output solver log during the extensive form solve.',
      action='store_true',
      dest='output_solver_log',
      default=False)
    outputOpts.add_option('--solution-writer',
      help='The plugin invoked to write the scenario tree solution. Defaults to the empty list.',
      action='append',
      dest='solution_writer',
      type='string',
      default = [])
    outputOpts.add_option('--verbose',
      help='Generate verbose output, beyond the usual status output. Default is False.',
      action='store_true',
      dest='verbose',
      default=False)
    outputOpts.add_option('--output-times',
      help="Output timing statistics for various EF components",
      action="store_true",
      dest="output_times",
      default=False)
    otherOpts.add_option('--disable-gc',
      help='Disable the python garbage collecter. Default is False.',
      action='store_true',
      dest='disable_gc',
      default=False)
    otherOpts.add_option('-k','--keep-solver-files',
      help='Retain temporary input and output files for solve.',
      action='store_true',
      dest='keep_solver_files',
      default=False)
    otherOpts.add_option('--profile',
      help='Enable profiling of Python code.  The value of this option is the number of functions that are summarized.',
      action='store',
      dest='profile',
      default=0)
    otherOpts.add_option('--traceback',
      help='When an exception is thrown, show the entire call stack. Ignored if profiling is enabled. Default is False.',
      action='store_true',
      dest='traceback',
      default=False)

    return parser

def EF_DefaultOptions():
    parser = construct_ef_writer_options_parser("")
    options, _ = parser.parse_args([''])
    return options

def GenerateScenarioTreeForEF(options,
                              scenario_instance_factory,
                              include_scenarios=None):

    from pyomo.pysp.ph import _OLD_OUTPUT

    try:

        scenario_tree = scenario_instance_factory.generate_scenario_tree(
            include_scenarios=include_scenarios,
            downsample_fraction=options.scenario_tree_downsample_fraction,
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
            if not _OLD_OUTPUT:
                print("Time link scenario tree with instances=%.2f seconds"
                      % (time.time() - start_time))

    except:
        if scenario_instance_factory is not None:
            scenario_instance_factory.close()
        print("Failed to initialize model and/or scenario tree data")
        raise

    return scenario_tree

def CreateExtensiveFormInstance(options, scenario_tree):

    start_time = time.time()
    from pyomo.pysp.ph import _OLD_OUTPUT
    if not _OLD_OUTPUT:
        print("Starting to build extensive form instance")
    else:
        print("Creating extensive form binding instance")

    # then validate the associated parameters.
    generate_weighted_cvar = False
    cvar_weight = None
    risk_alpha = None
    if options.generate_weighted_cvar is True:
        generate_weighted_cvar = True
        cvar_weight = options.cvar_weight
        risk_alpha = options.risk_alpha

    binding_instance = create_ef_instance(scenario_tree,
                                          verbose_output=options.verbose,
                                          generate_weighted_cvar=generate_weighted_cvar,
                                          cvar_weight=cvar_weight,
                                          risk_alpha=risk_alpha,
                                          cc_indicator_var_name=options.cc_indicator_var,
                                          cc_alpha=options.cc_alpha)

    if options.verbose or options.output_times:
        print("Time to construct extensive form instance=%.2f seconds"
              %(time.time() - start_time))

    return binding_instance

class ExtensiveFormAlgorithm(object):

    def __init__(self,
                 options,
                 binding_instance,
                 scenario_tree,
                 solver_manager,
                 solver,
                 solution_plugins=None):

        self._options = options
        self._binding_instance = binding_instance
        self._scenario_tree = scenario_tree
        self._solver_manager = solver_manager
        self._solver = solver
        self._solution_plugins = solution_plugins

    def write(self):

        start_time = time.time()

        output_filename = os.path.expanduser(self._options.output_file)
        suf = os.path.splitext(output_filename)[1]
        if suf not in ['.nl','.lp']:
            if self._solver.problem_format() == ProblemFormat.cpxlp:
                output_filename += '.lp'
            elif self._solver.problem_format() == ProblemFormat.nl:
                output_filename += '.nl'
            else:
                raise ValueError("Could not determine output file format. "
                                 "No recognized ending suffix was provided "
                                 "and no format was indicated was by the "
                                 "--solver-io option.")

        start_time = time.time()
        if self._options.verbose:
            print("Starting to write extensive form")

        # Do the preprocessing as necessary
        if not output_filename.endswith(".nl"):
            self._binding_instance.preprocess()

        symbol_map = write_ef(self._binding_instance,
                              output_filename,
                              self._options.symbolic_solver_labels)

        print("Extensive form written to file="+output_filename)
        if self._options.verbose or self._options.output_times:
            print("Time to write output file=%.2f seconds"
                  % (time.time() - start_time))

        return output_filename, symbol_map


    def solve(self):

        start_time = time.time()
        print("Queuing extensive form solve")

        # Do the preprocessing as necessary
        if self._solver.problem_format() != ProblemFormat.nl:
            self._binding_instance.preprocess()

        if (not self._options.disable_warmstarts) and \
           (self._solver.warm_start_capable()):
            action_handle = self._solver_manager.queue(self._binding_instance,
                                                       opt=self._solver,
                                                       tee=self._options.output_solver_log,
                                                       keepfiles=self._options.keep_solver_files,
                                                       symbolic_solver_labels=self._options.symbolic_solver_labels,
                                                       warmstart=True)
        else:
            action_handle = self._solver_manager.queue(self._binding_instance,
                                                       opt=self._solver,
                                                       tee=self._options.output_solver_log,
                                                       keepfiles=self._options.keep_solver_files,
                                                       symbolic_solver_labels=self._options.symbolic_solver_labels)
        print("Waiting for extensive form solve")
        results = self._solver_manager.wait_for(action_handle)

        if len(results.solution) == 0:
            results.write()
            raise RuntimeError("Solve failed; no solutions generated")

        # a temporary hack - if results come back from Pyro, they
        # won't have a symbol map attached. so create one.
        if results._symbol_map is None:
           results._symbol_map = symbol_map_from_instance(self._binding_instance)

        from pyomo.pysp.ph import _OLD_OUTPUT
        if not _OLD_OUTPUT:
            print("Done with extensive form solve - loading results")
        self._binding_instance.load(results)

        if not _OLD_OUTPUT:
            print("Storing solution in scenario tree")
        self._scenario_tree.pullScenarioSolutionsFromInstances()
        self._scenario_tree.snapshotSolutionFromScenarios()
        # TODO
        #self._scenario_tree.update_variable_statistics()

        if self._options.verbose or self._options.output_times:
            print("Time to solve and load results for the "
                  "extensive form=%.2f seconds"
                  % (time.time()-start_time))

        # print *the* metric of interest.
        root_node = self._scenario_tree._stages[0]._tree_nodes[0]
        print("")
        print("********************************"
              "********************************"
              "*******************************")
        print(">>>THE EXPECTED SUM OF THE STAGE COST VARIABLES="
              +str(root_node.computeExpectedNodeCost())+"<<<")
        print("********************************"
              "********************************"
              "*******************************")

        # handle output of solution from the scenario tree.
        print("")
        print("Extensive form solution:")
        self._scenario_tree.pprintSolution()
        print("")
        print("Extensive form costs:")
        self._scenario_tree.pprintCosts()

    def save_solution(self, label="ef"):

        if self._solution_plugins is not None:

            for plugin in self._solution_plugins:

                plugin.write(self._scenario_tree, label)

def EFAlgorithmBuilder(options, scenario_tree):

    solution_writer_plugins = ExtensionPoint(ISolutionWriterExtension)
    for plugin in solution_writer_plugins:
        plugin.disable()

    solution_plugins = []
    if len(options.solution_writer) > 0:
        for this_extension in options.solution_writer:
            if this_extension in sys.modules:
                print("User-defined EF solution writer module="
                      +this_extension+" already imported - skipping")
            else:
                print("Trying to import user-defined EF "
                      "solution writer module="+this_extension)
                # make sure "." is in the PATH.
                original_path = list(sys.path)
                sys.path.insert(0,'.')
                pyutilib.misc.import_file(this_extension)
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

            for name, obj in inspect.getmembers(sys.modules[module_to_find], inspect.isclass):
                import pyomo.util
                # the second condition gets around goofyness related to issubclass returning
                # True when the obj is the same as the test class.
                if issubclass(obj, pyomo.util.plugin.SingletonPlugin) and name != "SingletonPlugin":
                    for plugin in solution_writer_plugins(all=True):
                        if isinstance(plugin, obj):
                            plugin.enable()
                            solution_plugins.append(plugin)

    ef_solver = SolverFactory(options.solver_type,
                              solver_io=options.solver_io)
    if isinstance(ef_solver, UnknownSolver):
        raise ValueError("Failed to create solver of type="+
                         options.solver_type+
                         " for use in extensive form solve")
    if len(options.solver_options) > 0:
        print("Initializing ef solver with options="
              +str(options.solver_options))
        ef_solver.set_options("".join(options.solver_options))
    if options.mipgap is not None:
        if (options.mipgap < 0.0) or (options.mipgap > 1.0):
            raise ValueError("Value of the mipgap parameter for the EF "
                             "solve must be on the unit interval; "
                             "value specified="+str(options.mipgap))
        ef_solver.options.mipgap = float(options.mipgap)

    ef_solver_manager = SolverManagerFactory(options.solver_manager_type)
    if ef_solver_manager is None:
        raise ValueError("Failed to create solver manager of type="
                         +options.solver_type+
                         " for use in extensive form solve")

    ef_solver.symbolic_solver_labels = options.symbolic_solver_labels

    binding_instance = CreateExtensiveFormInstance(options, scenario_tree)

    ef = ExtensiveFormAlgorithm(options,
                                binding_instance,
                                scenario_tree,
                                ef_solver_manager,
                                ef_solver,
                                solution_plugins=solution_plugins)

    return ef

def run_ef(options, ef):

    if options.solve_ef:
        retval = ef.solve()
        ef.save_solution()
    else:
        retval = ef.write()

    return retval

def exec_ef(options):
    #
    # Import plugins
    #
    import pyomo.environ

    start_time = time.time()

    if options.verbose:
        from pyomo.pysp.ph import _OLD_OUTPUT
        if not _OLD_OUTPUT:
            print("Importing model and scenario tree files")
        else:
            print("Loading scenario and instance data")
            if options.verbose:
                print("Constructing reference model and instance")
                print("Constructing scenario tree instance")
                print("Constructing scenario tree object")

    scenario_instance_factory = ScenarioTreeInstanceFactory(options.model_directory,
                                                            options.instance_directory,
                                                            options.verbose)

    if options.verbose or options.output_times:
        from pyomo.pysp.ph import _OLD_OUTPUT
        if not _OLD_OUTPUT:
            print("Time to import model and scenario tree structure files=%.2f seconds"
                  %(time.time() - start_time))

    ef = None
    try:

        scenario_tree = GenerateScenarioTreeForEF(options,
                                                  scenario_instance_factory)



        ef = EFAlgorithmBuilder(options, scenario_tree)

        run_ef(options, ef)

    finally:

        if ef is not None:
            if ef._solver_manager is not None:
                ef._solver_manager.deactivate()
            if ef._solver is not None:
                ef._solver.deactivate()

            if (isinstance(ef._solver_manager, SolverManager_Pyro) or \
                isinstance(ef._solver_manager, SolverManager_PHPyro)) and \
                (options.shutdown_pyro):
                print("Shutting down Pyro solver components")
                shutdown_pyro_components()

        if scenario_instance_factory is not None:
            scenario_instance_factory.close()

    print("")
    print("Total EF execution time=%.2f seconds" %(time.time() - start_time))
    print("")

def main(args=None):

    #
    # Top-level command that executes the extensive form writer/solver.
    # This is segregated from run_ef to enable profiling.
    #

    #
    # Parse command-line options.
    #
    try:
        options_parser = construct_ef_writer_options_parser("runef [options]")
        (options, args) = options_parser.parse_args(args=args)
    except SystemExit:
        # the parser throws a system exit if "-h" is specified - catch
        # it to exit gracefully.
        return

    if options.disable_gc is True:
        gc.disable()
    else:
        gc.enable()

    # if an exception is triggered and traceback is enabled, 'ans' won't
    # have a value and the return statement from this function will flag
    # an error, masking the stack trace that you really want to see.
    ans = None

    if pstats_available and options.profile > 0:
        #
        # Call the main ef writer with profiling.
        #
        tfile = TempfileManager.create_tempfile(suffix=".profile")
        tmp = profile.runctx('exec_ef(options)',globals(),locals(),tfile)
        p = pstats.Stats(tfile).strip_dirs()
        p.sort_stats('time', 'cumulative')
        options.profile = eval(options.profile)
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
        # Call the main EF writer without profiling.
        #
        if options.traceback is True:
            ans = exec_ef(options)
        else:
            errmsg = None
            try:
                ans = exec_ef(options)
            except ValueError:
                err = sys.exc_info()[1]
                errmsg = 'VALUE ERROR: %s' % err
            except KeyError:
                err = sys.exc_info()[1]
                errmsg = 'KEY ERROR: %s' % err
            except TypeError:
                err = sys.exc_info()[1]
                errmsg = 'TYPE ERROR: %s' % err
            except NameError:
                err = sys.exc_info()[1]
                errmsg = 'NAME ERROR: %s' % err
            except IOError:
                err = sys.exc_info()[1]
                errmsg = 'I/O ERROR: %s' % err
            except ConverterError:
                err = sys.exc_info()[1]
                errmsg = 'CONVERSION ERROR: %s' % err
            except RuntimeError:
                err = sys.exc_info()[1]
                errmsg = 'RUN-TIME ERROR: %s' % err
            except pyutilib.common.ApplicationError:
                err = sys.exc_info()[1]
                errmsg = 'APPLICATION ERROR: %s' % err
            except Exception:
                err = sys.exc_info()[1]
                errmsg = 'UNKNOWN ERROR: %s' % err
                traceback.print_exc()

            if errmsg is not None:
                sys.stderr.write(errmsg+'\n')

    gc.enable()

    return ans

@pyomo_command('runef', 'Convert a SP to extensive form and optimize')
def EF_main(args=None):
    return main(args=args)
