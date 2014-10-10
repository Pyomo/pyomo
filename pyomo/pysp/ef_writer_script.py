#  _________________________________________________________________________
#
#  Pyomo: A COmmon Optimization Python Repository
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

from pyomo.misc import pyomo_command
from pyomo.misc.plugin import ExtensionPoint
from pyutilib.services import TempfileManager
from pyomo.opt.base import SolverFactory, ConverterError, ProblemFormat
from pyomo.opt.base.solvers import UnknownSolver
from pyomo.opt.parallel import SolverManagerFactory
from pyomo.pysp.ef import *
from pyomo.pysp.solutionwriter import ISolutionWriterExtension
import pyomo.solvers.plugins.smanager.pyro

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
      dest="linearize_expressions",
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
    solverOpts.add_option('--shutdown-pyro',
      help="Shut down all Pyro-related components associated with the Pyro solver manager (if specified), including the dispatch server, name server, and any mip servers. Default is False.",
      action="store_true",
      dest="shutdown_pyro",
      default=False)    

    outputOpts.add_option('--output-file',
      help='Specify the name of the extensive form output file',
      action='store',
      dest='output_file',
      type='string',
      default=None)
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


@pyomo_command('runef', 'Convert a SP to extensive form and optimize')
def run_ef_writer(options, args):
    #
    # Import plugins
    #
    import pyomo.environ

    start_time = time.time()    

    solution_writer_plugins = ExtensionPoint(ISolutionWriterExtension)
    for plugin in solution_writer_plugins:
       plugin.disable()

    if len(options.solution_writer) > 0:
        for this_extension in options.solution_writer:
            if this_extension in sys.modules:
                print("User-defined EF solution writer module="+this_extension+" already imported - skipping")
            else:
                print("Trying to import user-defined EF solution writer module="+this_extension)
                # make sure "." is in the PATH.
                original_path = list(sys.path)
                sys.path.insert(0,'.')
                pyutilib.misc.import_file(this_extension)
                print("Module successfully loaded")
                sys.path[:] = original_path # restore to what it was

            # now that we're sure the module is loaded, re-enable this specific plugin.
            # recall that all plugins are disabled by default in phinit.py, for various
            # reasons. if we want them to be picked up, we need to enable them explicitly.
            import inspect
            module_to_find = this_extension
            if module_to_find.rfind(".py"):
                module_to_find = module_to_find.rstrip(".py")
            if module_to_find.find("/") != -1:
                module_to_find = string.split(module_to_find,"/")[-1]

            for name, obj in inspect.getmembers(sys.modules[module_to_find], inspect.isclass):
                import pyomo.misc
                # the second condition gets around goofyness related to issubclass returning 
                # True when the obj is the same as the test class.
                if issubclass(obj, pyomo.misc.plugin.SingletonPlugin) and name != "SingletonPlugin":
                    for plugin in solution_writer_plugins(all=True):
                        if isinstance(plugin, obj):
                            plugin.enable()


    # if the user enabled the addition of the weighted cvar term to the objective,
    # then validate the associated parameters.
    generate_weighted_cvar = False
    cvar_weight = None
    risk_alpha = None

    if options.generate_weighted_cvar is True:

        generate_weighted_cvar = True
        cvar_weight = options.cvar_weight
        risk_alpha = options.risk_alpha

    # validate the solution writer plugin exists, to avoid a lot of wasted work.
    for solution_writer_name in options.solution_writer:
        print("Trying to import solution writer="+solution_writer_name)
        pyutilib.misc.import_file(solution_writer_name) 
        print("Module successfully loaded")

    # if the user hasn't requested that the extensive form be solved,
    # the solution writers are a waste of time!!!
    if (len(options.solution_writer) > 0) and (options.solve_ef is False):
        raise RuntimeError("Solution writers were specified, but there was no request to solve the extensive form.")

    output_file = options.output_file
    if output_file is not None:
        output_file = os.path.expanduser(output_file)

    if not options.solve_ef:
        
        if output_file is None:
            output_file = 'efout.lp'
        
        scenario_tree, binding_instance, scenario_instances, _ = write_ef_from_scratch(options.model_directory,
                                                                                       options.instance_directory,
                                                                                       options.objective_sense,
                                                                                       output_file,
                                                                                       options.symbolic_solver_labels,
                                                                                       options.verbose,
                                                                                       options.linearize_expressions,
                                                                                       options.scenario_tree_downsample_fraction,
                                                                                       options.scenario_tree_random_seed,
                                                                                       generate_weighted_cvar,
                                                                                       cvar_weight,
                                                                                       risk_alpha,
                                                                                       options.cc_indicator_var,
                                                                                       options.cc_alpha)

        if (scenario_tree is None) or (binding_instance is None) or (scenario_instances is None):
            raise RuntimeError("Failed to write extensive form.")

        end_write_time = time.time()
        print("Time to create and write the extensive form=%.2f seconds" %(end_write_time - start_time))

    else:

        ef_solver = SolverFactory(options.solver_type, solver_io=options.solver_io)
        if isinstance(ef_solver, UnknownSolver):
            raise ValueError("Failed to create solver of type="+options.solver_type+" for use in extensive form solve")
        if len(options.solver_options) > 0:
            print("Initializing ef solver with options="+str(options.solver_options))
            ef_solver.set_options("".join(options.solver_options))
        if options.mipgap is not None:
            if (options.mipgap < 0.0) or (options.mipgap > 1.0):
                raise ValueError("Value of the mipgap parameter for the EF solve must be on the unit interval; value specified="+str(options.mipgap))
            ef_solver.options.mipgap = float(options.mipgap)

        if (options.keep_solver_files is True) or (output_file is not None):
            ef_solver.keepfiles = True

        ef_solver_manager = SolverManagerFactory(options.solver_manager_type)
        if ef_solver is None:
            raise ValueError("Failed to create solver manager of type="+options.solver_type+" for use in extensive form solve")

        ef_solver.symbolic_solver_labels = options.symbolic_solver_labels

        scenario_tree, binding_instance, scenario_instances = create_ef_from_scratch(options.model_directory,
                                                                                     options.instance_directory,
                                                                                     options.objective_sense,
                                                                                     options.verbose,
                                                                                     options.linearize_expressions,
                                                                                     options.scenario_tree_downsample_fraction,
                                                                                     options.scenario_tree_random_seed,
                                                                                     generate_weighted_cvar,
                                                                                     cvar_weight,
                                                                                     risk_alpha,
                                                                                     options.cc_indicator_var,
                                                                                     options.cc_alpha,
                                                                                     skip_canonical=((ef_solver.problem_format() == ProblemFormat.nl)))

        if (scenario_tree is None) or (binding_instance is None) or (scenario_instances is None):
            raise RuntimeError("Failed to create extensive form.")

        end_write_time = time.time()
        print("Time to create the extensive form=%.2f seconds" %(end_write_time - start_time))

        print("Queuing extensive form solve")
        ef_action_handle = ef_solver_manager.queue(binding_instance, opt=ef_solver, tee=options.output_solver_log)
        print("Waiting for extensive form solve")
        ef_results = ef_solver_manager.wait_for(ef_action_handle)

        if output_file is not None:
            if ef_solver._problem_files is not None:
                if os.path.splitext(output_file)[1] != \
                   os.path.splitext(ef_solver._problem_files[0])[1]:
                    output_file += os.path.splitext(ef_solver._problem_files[0])[1]
                shutil.copy(ef_solver._problem_files[0],output_file)
                print("Output file written to file= "+output_file)
            else:
                print("No output file could be created with the chosen solver")

        if len(ef_results.solution) == 0:
            ef_results.write()
            raise RuntimeError("Solve failed; no solutions generated")

        binding_instance.load(ef_results)

        scenario_tree.pullScenarioSolutionsFromInstances()
        scenario_tree.snapshotSolutionFromScenarios()
        # TODO:
        # scenario_tree.update_variable_statistic()

        end_solve_time = time.time()
        print("Time to solve and load results for the extensive form=%.2f seconds" %(end_solve_time - end_write_time))

        # print *the* metric of interest.
        root_node = scenario_tree._stages[0]._tree_nodes[0]              
        print("")
        print("***********************************************************************************************")
        print(">>>THE EXPECTED SUM OF THE STAGE COST VARIABLES="+str(root_node.computeExpectedNodeCost())+"<<<")
        print("***********************************************************************************************")

        # handle output of solution from the scenario tree.
        print("")
        print("Extensive form solution:")
        scenario_tree.pprintSolution()
        print("")
        print("Extensive form costs:")
        scenario_tree.pprintCosts()

        solution_writer_plugins = ExtensionPoint(ISolutionWriterExtension)
        for plugin in solution_writer_plugins:
            plugin.write(scenario_tree, scenario_instances, "ef")

        if isinstance(ef_solver_manager,pyomo.solvers.plugins.smanager.pyro.SolverManager_Pyro) and (options.shutdown_pyro is True):
           print("Shutting down Pyro solver components")
           shutDownPyroComponents()            

    overall_end_time = time.time()
    print("Total execution time=%.2f seconds" %(overall_end_time - start_time))

def main(args=None):

    #
    # Top-level command that executes the extensive form writer.
    # This is segregated from run_ef_writer to enable profiling.
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
        tmp = profile.runctx('run_ef_writer(options,args)',globals(),locals(),tfile)
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
            ans = run_ef_writer(options, args)
        else:
            errmsg = None
            try:
                ans = run_ef_writer(options, args)
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
