#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import os
import os.path
import sys
import glob
import datetime
import textwrap
import logging
import argparse

import pyutilib.subprocess
from pyutilib.misc import Options

from pyomo.common import get_pyomo_commands
import pyomo.scripting.pyomo_parser

logger = logging.getLogger('pyomo.solvers')


#--------------------------------------------------
# install-extras
#   -v
#--------------------------------------------------

def setup_install_extras_parser(parser):
    parser.add_argument("--pip-args", dest="arg", nargs="*", help="Arguments that are passed to the 'pip' command when installing packages")
    parser.add_argument("-q", dest="tee", action="store_false", default=True, help="Suppress the default verbose output")

def install_extras_exec(options):
    cmddir = os.path.dirname(os.path.abspath(sys.executable))+os.sep
    if options.arg is None:
        command = [cmddir+'get_pyomo_extras']
    else:
        command = [cmddir+'get_pyomo_extras'] + options.arg
    if not options.tee:
        command.append('-q')
    pyutilib.subprocess.run(command, tee=True)

#
# Add a subparser for the pyomo install-extras subcommand
#
setup_install_extras_parser(
    pyomo.scripting.pyomo_parser.add_subparser('install-extras',
        func=install_extras_exec,
        help='Install "extra" packages that Pyomo can leverage.',
        description="""
This pyomo subcommand is used to install packages that Pyomo could
leverage.  The installation of some packages may fail, but this
subcommand ignore these failures and provides a summary describing
which packages were installed.
""",
        epilog="""
Since pip options begin with a dash, the --pip-args option can only
be used with the equals syntax.  For example:\n\n
    pyomo install-extras --pip-args="--upgrade"
""",
        formatter_class=pyomo.scripting.pyomo_parser.CustomHelpFormatter
        ))


#--------------------------------------------------
# run
#   --list
#--------------------------------------------------

def setup_command_parser(parser):
    parser.add_argument("--list", dest="summary", action='store_true', default=False,
                        help="List the commands that are installed with Pyomo")
    parser.add_argument("command", nargs='*', help="The command and command-line options")

def command_exec(options):
    cmddir = os.path.dirname(os.path.abspath(sys.executable))+os.sep
    if options.summary:
        print("")
        print("The following commands are installed in the Pyomo bin directory:")
        print("----------------------------------------------------------------")
        for file in sorted(glob.glob(cmddir+'*')):
            print(" "+os.path.basename(file))
        print("")
        if len(options.command) > 0:
            print("WARNING: ignoring command specification")
        return
    if len(options.command) == 0:
        print("  ERROR: no command specified")
        return
    if not os.path.exists(cmddir+options.command[0]):
        print("  ERROR: the command '%s' does not exist" % (cmddir+options.command[0]))
        return
    pyutilib.subprocess.run(cmddir+' '.join(options.command), tee=True)

#
# Add a subparser for the pyomo command
#
setup_command_parser(
    pyomo.scripting.pyomo_parser.add_subparser('run',
        func=command_exec,
        help='Execute a command from the Pyomo bin (or Scripts) directory.',
        description='This pyomo subcommand is used to execute commands installed with Pyomo.',
        epilog="""
This subcommand can execute any command from the bin (or Script)
directory that is created when Pyomo is installed.  Note that this
includes any commands that are installed by other Python packages
that are installed with Pyomo.  Thus, if Pyomo is installed in the
Python system directories, then this command executes any command
included with Python.
"""
        ))

#--------------------------------------------------
# help
#   --components
#   --command
#   --api
#   --transformations
#   --solvers
#--------------------------------------------------

def help_commands():
    print("")
    print("The following commands are installed with Pyomo:")
    print("-"*75)
    registry = get_pyomo_commands()
    d = max(len(key) for key in registry)
    fmt = "%%-%ds  %%s" % d
    for key in sorted(registry.keys(), key=lambda v: v.upper()):
        print(fmt % (key, registry[key]))
    print("")

def help_writers():
    import pyomo.environ
    from pyomo.opt.base import WriterFactory
    wrapper = textwrap.TextWrapper()
    wrapper.initial_indent = '      '
    wrapper.subsequent_indent = '      '
    print("")
    print("Pyomo Problem Writers")
    print("---------------------")
    for writer in sorted(WriterFactory):
        print("  "+writer)
        print(wrapper.fill(WriterFactory.doc(writer)))

def help_checkers():
    import pyomo.environ
    import pyomo.common.plugin
    from pyomo.checker import IModelChecker
    wrapper = textwrap.TextWrapper()
    wrapper.initial_indent = '      '
    wrapper.subsequent_indent = '      '
    print("")
    print("Pyomo Model Checkers")
    print("--------------------")
    ep = pyomo.common.plugin.ExtensionPoint(IModelChecker)
    tmp = {}
    for checker in ep.extensions():
        for alias in getattr(checker, '_factory_aliases', set()):
            tmp[alias[0]] = alias[1]
    for key in sorted(tmp.keys()):
        print("  "+key)
        print(wrapper.fill(tmp[key]))

def help_datamanagers(options):
    import pyomo.environ
    from pyomo.dataportal import DataManagerFactory
    wrapper = textwrap.TextWrapper()
    wrapper.initial_indent = '      '
    wrapper.subsequent_indent = '      '
    print("")
    print("Pyomo Data Managers")
    print("-------------------")
    for xform in sorted(DataManagerFactory):
        print("  "+xform)
        print(wrapper.fill(DataManagerFactory.doc(xform)))

def help_api(options):
    import pyomo.common
    services = pyomo.common.PyomoAPIFactory.services()
    #
    f = {}
    for name in services:
        f[name] = pyomo.common.PyomoAPIFactory(name)
    #
    ns = {}
    for name in services:
        ns_set = ns.setdefault(f[name].__namespace__, set())
        ns_set.add(name)
    #
    if options.asciidoc:
        print("//")
        print("// Pyomo Library API Documentation")
        print("//")
        print("// Generated with 'pyomo api' on ",datetime.date.today())
        print("//")
        print("")
        print("== Pyomo Functor API ==")
        for ns_ in sorted(ns.keys()):
            print("")
            level = ns_+" Functors"
            print('=== %s ===' % level)
            for name in sorted(ns[ns_]):
                if ns_ != '':
                    tname = name[len(ns_)+1:]
                else:
                    tname = name
                print("")
                print('==== %s ====' % tname)
                print(f[name].__short_doc__)
                if f[name].__long_doc__ != '':
                    print("")
                    print(f[name].__long_doc__)
                print("")
                flag=False
                print("- [underline]#Required Keyword Arguments:#")
                for port in sorted(f[name].inputs):
                    if f[name].inputs[port].optional:
                        flag=True
                        continue
                    print("")
                    print('*%s*::\n %s' % (port, f[name].inputs[port].doc))
                if flag:
                    # A function may not have optional arguments
                    print("")
                    print("- [underline]#Optional Keyword Arguments:#")
                    for port in sorted(f[name].inputs):
                        if not f[name].inputs[port].optional:
                            continue
                        print("")
                        print('*%s*::\n %s' % (port, f[name].inputs[port].doc))
                print("")
                print("- [underline]#Return Values:#")
                for port in sorted(f[name].outputs):
                    print("")
                    print('*%s*::\n %s' % (port, f[name].outputs[port].doc))
                print("")
    else:
        print("")
        print("Pyomo Functor API")
        print("-----------------")
        wrapper = textwrap.TextWrapper(subsequent_indent='')
        print(wrapper.fill("The Pyomo library contains a set of functors that define operations that are likely to be major steps in Pyomo scripts.  This API is defined with functors to ensure a consistent function syntax.  Additionally, these functors can be accessed with a factory, thereby avoiding the need to import modules throughout Pyomo."))
        print("")
        for ns_ in sorted(ns.keys()):
            print("")
            level = ns_+" Functors"
            print("-"*len(level))
            print(level)
            print("-"*len(level))
            for name in sorted(ns[ns_]):
                if ns_ != '':
                    tname = name[len(ns_)+1:]
                else:
                    tname = name
                print(tname+':')
                for line in f[name].__short_doc__.split('\n'):
                    print("    "+line)

def help_environment():
    cmddir = os.path.dirname(os.path.abspath(sys.executable))+os.sep
    info = Options()
    #
    info.python = Options()
    info.python.version = '%d.%d.%d' % sys.version_info[:3]
    info.python.executable = sys.executable
    info.python.platform = sys.platform
    try:
        packages = []
        import pip
        for package in pip.get_installed_distributions():
            packages.append( Options(name=package.project_name, version=package.version) )
        info.python.packages = packages
    except:
        pass
    #
    info.environment = Options()
    path = os.environ.get('PATH', None)
    if not path is None:
        info.environment['shell path'] = path.split(os.pathsep)
    info.environment['python path'] = sys.path
    #
    print('#')
    print('# Information About the Python and Shell Environment')
    print('#')
    print(str(info))

def help_transformations():
    import pyomo.environ
    from pyomo.core import TransformationFactory
    wrapper = textwrap.TextWrapper()
    wrapper.initial_indent = '      '
    wrapper.subsequent_indent = '      '
    print("")
    print("Pyomo Model Transformations")
    print("---------------------------")
    for xform in sorted(TransformationFactory):
        print("  "+xform)
        print(wrapper.fill(TransformationFactory.doc(xform)))

def help_solvers():
    import pyomo.environ
    wrapper = textwrap.TextWrapper(replace_whitespace=False)
    print("")
    print("Pyomo Solvers and Solver Managers")
    print("---------------------------------")

    print(wrapper.fill("Pyomo uses 'solver managers' to execute 'solvers' that perform optimization and other forms of model analysis.  A solver directly executes an optimizer, typically using an executable found on the user's PATH environment.  Solver managers support a flexible mechanism for asyncronously executing solvers either locally or remotely.  The following solver managers are available in Pyomo:"))
    print("")
    solvermgr_list = list(pyomo.opt.SolverManagerFactory)
    solvermgr_list = sorted( filter(lambda x: '_' != x[0], solvermgr_list) )
    n = max(map(len, solvermgr_list))
    wrapper = textwrap.TextWrapper(subsequent_indent=' '*(n+9))
    for s in solvermgr_list:
        format = '    %-'+str(n)+'s     %s'
        print(wrapper.fill(format % (s , pyomo.opt.SolverManagerFactory.doc(s))))
    print("")
    wrapper = textwrap.TextWrapper(subsequent_indent='')
    print(wrapper.fill("If no solver manager is specified, Pyomo uses the serial solver manager to execute solvers locally.  The pyro and phpyro solver managers require the installation and configuration of the pyro software.  The neos solver manager is used to execute solvers on the NEOS optimization server."))
    print("")

    print("")
    print("Serial Solver Interfaces")
    print("------------------------")
    print(wrapper.fill("The serial, pyro and phpyro solver managers support the following solver interfaces:"))
    print("")
    solver_list = list(pyomo.opt.SolverFactory)
    solver_list = sorted( filter(lambda x: '_' != x[0], solver_list) )
    n = max(map(len, solver_list))
    wrapper = textwrap.TextWrapper(subsequent_indent=' '*(n+9))
    try:
        # Disable warnings
        logging.disable(logging.WARNING)
        for s in solver_list:
            # Create a solver, and see if it is available
            with pyomo.opt.SolverFactory(s) as opt:
                if s == 'py' or (hasattr(opt, "_metasolver") and opt._metasolver):
                    # py is a metasolver, but since we don't specify a subsolver
                    # for this test, opt is actually an UnknownSolver, so we
                    # can't try to get the _metasolver attribute from it.
                    # Also, default to False if the attribute isn't implemented
                    msg = '    %-'+str(n)+'s   + %s'
                elif opt.available(False):
                    msg = '    %-'+str(n)+'s   * %s'
                else:
                    msg = '    %-'+str(n)+'s     %s'
                print(wrapper.fill(msg % (s, pyomo.opt.SolverFactory.doc(s))))
    finally:
        # Reset logging level
        logging.disable(logging.NOTSET)
    print("")
    wrapper = textwrap.TextWrapper(subsequent_indent='')
    print(wrapper.fill("An asterisk indicates solvers that are currently available to be run from Pyomo with the serial solver manager. A plus indicates meta-solvers, that are always available."))
    print('')
    print(wrapper.fill('Pyomo also supports solver interfaces that are wrappers around third-party solver interfaces. These interfaces require a subsolver specification that indicates the solver being executed.  For example, the following indicates that the ipopt solver will be used:'))
    print('')
    print('   asl:ipopt')
    print('')
    print(wrapper.fill('The asl interface provides a generic wrapper for all solvers that use the AMPL Solver Library.'))
    print('')
    print(wrapper.fill('Note that subsolvers can not be enumerated automatically for these interfaces.  However, if a solver is specified that is not found, Pyomo assumes that the asl solver interface is being used.  Thus the following solver name will launch ipopt if the \'ipopt\' executable is on the user\'s path:'))
    print('')
    print('   ipopt')
    print('')
    try:
        logging.disable(logging.WARNING)
        import pyomo.neos.kestrel
        kestrel = pyomo.neos.kestrel.kestrelAMPL()
        #print "HERE", solver_list
        solver_list = list(set([name[:-5].lower() for name in kestrel.solvers() if name.endswith('AMPL')]))
        #print "HERE", solver_list
        if len(solver_list) > 0:
            print("")
            print("NEOS Solver Interfaces")
            print("----------------------")
            print(wrapper.fill("The neos solver manager supports solver interfaces that can be executed remotely on the NEOS optimization server.  The following solver interfaces are available with your current system configuration:"))
            print("")
            solver_list = sorted(solver_list)
            n = max(map(len, solver_list))
            format = '    %-'+str(n)+'s     %s'
            for name in solver_list:
                print(wrapper.fill(format % (name , pyomo.neos.doc.get(name,'Unexpected NEOS solver'))))
            print("")
        else:
            print("")
            print("NEOS Solver Interfaces")
            print("----------------------")
            print(wrapper.fill("The neos solver manager supports solver interfaces that can be executed remotely on the NEOS optimization server.  This server is not available with your current system configuration."))
            print("")
    except ImportError:
        pass
    finally:
        logging.disable(logging.NOTSET)

def print_components(data):
    """
    Print information about modeling components supported by Pyomo.
    """
    print("")
    print("----------------------------------------------------------------")
    print("Pyomo Model Components:")
    print("----------------------------------------------------------------")
    components = pyomo.core.base._pyomo.model_components()
    index = pyutilib.misc.sort_index(components)
    for i in index:
        print("")
        print(" "+components[i][0])
        for line in textwrap.wrap(components[i][1], 59):
            print("    "+line)
    print("")
    print("----------------------------------------------------------------")
    print("Pyomo Virtual Sets:")
    print("----------------------------------------------------------------")
    pyomo_sets = pyomo.core.base._pyomo.predefined_sets()
    index = pyutilib.misc.sort_index(pyomo_sets)
    for i in index:
        print("")
        print(" "+pyomo_sets[i][0])
        print("    "+pyomo_sets[i][1])

def help_exec(options):
    flag=False
    if options.commands:
        if options.asciidoc:
            print("The '--commands' help information is not printed in an asciidoc format.")
        flag=True
        help_commands()
    if options.components:
        if options.asciidoc:
            print("The '--components' help information is not printed in an asciidoc format.")
        flag=True
        print_components(None)
    if options.api:
        flag=True
        help_api(options)
    if options.datamanager:
        flag=True
        help_datamanagers(options)
    if options.environment:
        flag=True
        help_environment()
    if options.transformations:
        if options.asciidoc:
            print("The '--transformations' help information is not printed in an asciidoc format.")
        flag=True
        help_transformations()
    if options.solvers:
        if options.asciidoc:
            print("The '--solvers' help information is not printed in an asciidoc format.")
        flag=True
        help_solvers()
    if options.writers:
        flag=True
        if options.asciidoc:
            print("The '--writers' help information is not printed in an asciidoc format.")
        help_writers()
    if options.checkers:
        flag=True
        if options.asciidoc:
            print("The '--checkers' help information is not printed in an asciidoc format.")
        help_checkers()
    if not flag:
        help_parser.print_help()

#
# Add a subparser for the pyomo command
#
def setup_help_parser(parser):
    parser.add_argument("-a", "--api", dest="api", action='store_true', default=False,
                        help="Print a summary of the Pyomo Library API")
    parser.add_argument("--asciidoc", dest="asciidoc", action='store_true', default=False,
                        help="Generate output that is compatible with asciidoc's markup language")
    parser.add_argument("--checkers", dest="checkers", action='store_true', default=False,
                        help="List the available model checkers")
    parser.add_argument("-c", "--commands", dest="commands", action='store_true', default=False,
                        help="List the commands that are installed with Pyomo")
    parser.add_argument("--components", dest="components", action='store_true', default=False,
                        help="List the components that are available in Pyomo's modeling environment")
    parser.add_argument("-d", "--data-managers", dest="datamanager", action='store_true', default=False,
                        help="Print a summary of the data managers in Pyomo")
    parser.add_argument("-i", "--info", dest="environment", action='store_true', default=False,
                        help="Summarize the environment and Python installation")
    parser.add_argument("-s", "--solvers", dest="solvers", action='store_true', default=False,
                        help="Summarize the available solvers and solver interfaces")
    parser.add_argument("-t", "--transformations", dest="transformations", action='store_true', default=False,
                        help="List the available model transformations")
    parser.add_argument("-w", "--writers", dest="writers", action='store_true', default=False,
                        help="List the available problem writers")
    return parser

help_parser = setup_help_parser(
  pyomo.scripting.pyomo_parser.add_subparser('help',
        func=help_exec,
        help='Print help information.',
        description="This pyomo subcommand is used to print information about Pyomo's subcommands and installed Pyomo services."
        ))
