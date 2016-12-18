#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

import sys
import argparse
try:
    from pympler import muppy
    from pympler.muppy import summary
    from pympler import tracker
    from pympler.asizeof import *
    pympler_available = True
except:
    pympler_available = False

from pyutilib.misc import Options, Container

from pyomo.util import pyomo_command
import pyomo.scripting.util
from pyomo.core import ConcreteModel

def add_model_group(parser):
    raise IOError
    group = parser.add_argument_group('Model Options')

    group.add_argument('--preprocess', '--load',
        help='Specify a Python module that gets immediately executed (before '\
             'the optimization model is setup).  If this option is specified '\
             'multiple times, then the modules are executed in the specified '\
             'order.',
        action='append',
        dest='preprocess',
        default=[])
    group.add_argument('--transform',
        help='Specify a list of transformations that are applied before the model '\
             'is optimized.',
        action='append',
        dest='transformations',
        default=[])
    group.add_argument('--model-name',
        help='The name of the model object that is created in the specified.' \
             'Pyomo module',
        action='store',
        metavar='NAME',
        dest='model_name',
        default=None)
    group.add_argument('--model-options',
        help='Options passed into a create_model() function to construct the '\
             'model.',
        action='append',
        dest='model_options',
        default=[])
    group.add_argument('--save-model',
        help='Specify the filename to which the model is saved.  The suffix ' \
             'of this filename specifies the file format.  If debugging is '  \
             "on, then this defaults to writing the file 'unknown.lp'.",
        action='store',
        dest='save_model',
        default=None)
    group.add_argument('--symbolic-solver-labels',
        help='When interfacing with the solver, use symbol names derived from the model. For example, \"my_special_variable[1_2_3]\" instead of \"v1\". Useful for debugging. When using the ASL interface (--solver-io=nl), generates corresponding .row (constraints) and .col (variables) files. The ordering in these files provides a mapping from ASL index to symbolic model names.',
        action='store_true',
        dest='symbolic_solver_labels',
        default=False)
    group.add_argument('--file-determinism',
        help='When interfacing with a solver using file based I/O, set the effort level for ensuring the file creation process is determistic. The default (1) sorts the index of components when transforming the model. Anything less than 1 disables index sorting. Anything greater than 1 additionaly sorts by component name to override declartion order.',
        action='store',
        type=int,
        dest='file_determinism',
        default=1)
    group.add_argument('--ns', '--namespace',
        help='Specify a namespace that is used to select data in Pyomo data files.  If this is specified multiple '\
            'times then the union of the data in these namespaces is used to create the model instance.',
        action='append',
        dest='namespaces',
        default=[])
    return group


def add_logging_group(parser):
    group = parser.add_argument_group('Logging Options')

    group.add_argument('-q','--quiet',
        help='Disable all log messages except for those that refer to errors.',
        action='store_true',
        dest='quiet',
        default=False)
    group.add_argument('-w','--warning',
        help='Enable warning log messages for pyomo and pyutilib [default behavior].',
        action='store_true',
        dest='warning',
        default=False)
    group.add_argument('-i','--info',
        help='Enable informative log messages for pyomo and pyutilib.',
        action='store_true',
        dest='info',
        default=False)
    group.add_argument('-v','--verbose',
        help="Indicate that debugging log messages should be printed.  This option can be specified multiple times to add log messages for other parts of pyomo and pyutilib.",
        action='count',
        dest='verbose',
        default=0)
    group.add_argument('-d', '--debug',
        help='This option indicates that debugging is performed.  This implies the verbose flag, but it also allows exceptions to trigger a failure in which the program stack is printed.',
        action='store_true',
        dest='debug',
        default=False)
    #group.add_argument('--logfile',
        #help='Print log messages to the specified file.',
        #action='store',
        #dest='logfile',
        #default=None)
    return group


def add_solver_group(parser):
    solver_help=\
        "This option specifies the type of solver that is used "\
        "to optimize the Pyomo model instance.  Run the 'pyomo help --solvers' "\
        "command to get detailed information concerning how solvers are "\
        "executed."

    group = parser.add_argument_group('Solver Options')

    group.add_argument('--solver',
        help=solver_help,
        action='store',
        dest='solver',
        #choices=solver_list,
        default='glpk')
    group.add_argument('--solver-io',
        help='The type of IO used to execute the solver.  Different solvers support different types of IO, but the following are common options: lp - generate LP files, nl - generate NL files, python - direct Python interface, os - generate OSiL XML files.',
        action='store',
        dest='solver_io',
        #choices=['lp', 'nl', 'python', 'os'],
        default=None)
    group.add_argument('--solver-manager',
        help='Specify the technique that is used to manage solver executions.',
        action='store',
        dest='smanager_type',
        #choices=SolverManagerFactory.services(),
        default='serial')
    group.add_argument('--solver-manager-pyro-host',
      help="The hostname to bind on when searching for a Pyro nameserver.",
      action="store",
      dest="pyro_host",
      default=None)
    group.add_argument('--solver-manager-pyro-port',
      help="The port to bind on when searching for a Pyro nameserver.",
      action="store",
      dest="pyro_port",
      type="int",
      default=None)
    group.add_argument('--solver-options',
        help='Options passed into the solver.',
        action='append',
        dest='solver_options',
        default=[])
    group.add_argument('--solver-suffixes',
        help='One or more solution suffixes to be extracted by the solver (e.g., rc, dual, or slack). Multiple options are specified by supplying the keyword options multiple times. The use of this option is not required when a suffix has been declared on the model using Pyomo\'s Suffix component.',
        action='append',
        dest='solver_suffixes',
        default=[])
    group.add_argument('--timelimit',
        help='Limit to the number of seconds that the solver is run.',
        action='store',
        dest='timelimit',
        type=int,
        default=0)
    group.add_argument('--stream-solver','--stream-output',
        help='Stream the solver output to provide information about the '     \
             "solver's progress.",
        action='store_true',
        dest='tee',
        default=False)
    return group


def add_postsolve_group(parser):
    group = parser.add_argument_group('Post-Solve Options')

    group.add_argument('--postprocess',
        help='Specify a Python module that gets executed after optimization. ' \
             'If this option is specified multiple times, then the modules '  \
             'are executed in the specified order.',
        action='append',
        dest='postprocess',
        default=[])
    group.add_argument('-l','--log',
        help='Print the solver logfile after performing optimization.',
        action='store_true',
        dest='log',
        default=False)
    group.add_argument('--save-results',
        help='Specify the filename to which the results are saved.',
        action='store',
        dest='save_results',
        default=None)
    group.add_argument('--show-results',
        help='Print the results object after optimization.',
        action='store_true',
        dest='show_results',
        default=False)
    group.add_argument('--json',
        help='Print results in JSON format.  The default is YAML, if the PyYAML utility is available',
        action='store_true',
        dest='json',
        default=False)
    group.add_argument('-s','--summary',
        help='Summarize the final solution after performing optimization.',
        action='store_true',
        dest='summary',
        default=False)
    return group


def add_misc_group(parser):
    group = parser.add_argument_group('Miscellaneous Options')

    group.add_argument('--output',
        help='Redirect output to the specified file.',
        action='store',
        dest='output',
        default=None)
    group.add_argument('-c', '--catch-errors',
        help='This option allows exceptions to trigger a failure in which the program stack is printed.',
        action='store_true',
        dest='catch',
        default=False)
    #group.add_argument('--logfile',
    group.add_argument('--disable-gc',
        help='Disable the garbage collecter.',
        action='store_true',
        dest='disable_gc',
        default=False)
    group.add_argument('--interactive',
        help='After executing Pyomo, launch an interactive Python shell.  If IPython is installed, this shell is an IPython shell.',
        action='store_true',
        dest='interactive',
        default=False)
    group.add_argument('-k','--keepfiles',
        help='Keep temporary files.',
        action='store_true',
        dest='keepfiles',
        default=False)
    group.add_argument('--path',
        help='Give a path that is used to find the Pyomo python files.',
        action='store',
        dest='path',
        default='.')
    group.add_argument('--profile',
        help='Enable profiling of Python code.  The value of this option is ' \
             'the number of functions that are summarized.',
        action='store',
        dest='profile',
        type=int,
        default=0)
    if pympler_available is True:
        group.add_argument("--profile-memory",
                             help="If Pympler is available, report memory usage statistics for the generated instance and any associated processing steps. A value of 0 indicates disabled. A value of 1 forces the print of the total memory after major stages of the pyomo script. A value of 2 forces summary memory statistics after major stages of the pyomo script. A value of 3 forces detailed memory statistics during instance creation and various steps of preprocessing. Values equal to 4 and higher currently provide no additional information. Higher values automatically enable all functionality associated with lower values, e.g., 3 turns on detailed and summary statistics.",
                             action="store",
                             dest="profile_memory",
                             type=int,
                             default=0)
    group.add_argument('--report-timing',
        help='Report various timing statistics during model construction. Defaults to disabled.',
        action='store_true',
        dest='report_timing',
        default=False)
    group.add_argument('--tempdir',
        help='Specify the directory where temporary files are generated.',
        action='store',
        dest='tempdir',
        default=None)
    group.add_argument('--version',
        help='Display main Pyomo version and exit',
        action='store_true',
        dest='version',
        default=False)
    return group


def run_pyomo(options=Options(), parser=None):
    data = Options(options=options)

    if options.model.filename == '':
        parser.print_help()
        return Container()

    try:
        pyomo.scripting.util.setup_environment(data)

        pyomo.scripting.util.apply_preprocessing(data,
                                                 parser=parser)
    except:
        # TBD: I should be able to call this function in the case of
        #      an exception to perform cleanup. However, as it stands
        #      calling finalize with its default keyword value for
        #      model(=None) results in an a different error related to
        #      task port values.  Not sure how to interpret that.
        pyomo.scripting.util.finalize(data,
                                      model=ConcreteModel(),
                                      instance=None,
                                      results=None)
        raise
    else:
        if data.error:
            # TBD: I should be able to call this function in the case of
            #      an exception to perform cleanup. However, as it stands
            #      calling finalize with its default keyword value for
            #      model(=None) results in an a different error related to
            #      task port values.  Not sure how to interpret that.
            pyomo.scripting.util.finalize(data,
                                          model=ConcretModel(),
                                          instance=None,
                                          results=None)
            return Container()                                   #pragma:nocover

    try:
        model_data = pyomo.scripting.util.create_model(data)
    except:
        # TBD: I should be able to call this function in the case of
        #      an exception to perform cleanup. However, as it stands
        #      calling finalize with its default keyword value for
        #      model(=None) results in an a different error related to
        #      task port values.  Not sure how to interpret that.
        pyomo.scripting.util.finalize(data,
                                      model=ConcreteModel(),
                                      instance=None,
                                      results=None)
        raise
    else:
        if (((not options.runtime.logging == 'debug') and \
             options.model.save_file) or \
            options.runtime.only_instance):
            pyomo.scripting.util.finalize(data,
                                          model=model_data.model,
                                          instance=model_data.instance,
                                          results=None)
            return Container(instance=model_data.instance)

    try:
        opt_data = pyomo.scripting.util.apply_optimizer(data,
                                                        instance=model_data.instance)

        pyomo.scripting.util.process_results(data,
                                             instance=model_data.instance,
                                             results=opt_data.results,
                                             opt=opt_data.opt)

        pyomo.scripting.util.apply_postprocessing(data,
                                                  instance=model_data.instance,
                                                  results=opt_data.results)
    except:
        # TBD: I should be able to call this function in the case of
        #      an exception to perform cleanup. However, as it stands
        #      calling finalize with its default keyword value for
        #      model(=None) results in an a different error related to
        #      task port values.  Not sure how to interpret that.
        pyomo.scripting.util.finalize(data,
                                      model=ConcreteModel(),
                                      instance=None,
                                      results=None)
        raise
    else:
        pyomo.scripting.util.finalize(data,
                                      model=model_data.model,
                                      instance=model_data.instance,
                                      results=opt_data.results)

        return Container(options=options,
                         instance=model_data.instance,
                         results=opt_data.results,
                         local=opt_data.local)

def run(args=None):
    from pyomo.scripting.pyomo_main import main
    if args is None:
        return main()
    else:
        return main(['solve']+args, get_return=True)

