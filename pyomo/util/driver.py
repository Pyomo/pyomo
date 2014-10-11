
import argparse
import os
import os.path
import sys
import glob
import datetime
import textwrap
import logging

import pyutilib.subprocess

from pyomo.util import pyomo_parser
from pyomo.util import get_pyomo_commands

logger = logging.getLogger('pyomo.solvers')


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
        print("ERROR: no command specified")
        return
    if not os.path.exists(cmddir+options.command[0]):
        print("ERROR: the command '%s' does not exist" % (cmddir+options.command[0]))
        return
    pyutilib.subprocess.run(cmddir+' '.join(options.command), tee=True)

#
# Add a subparser for the pyomo command
#
setup_command_parser(
    pyomo_parser.add_subparser('run',
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

def help_api(options):
    import pyomo.util
    services = pyomo.util.PyomoAPIFactory.services()
    #
    f = {}
    for name in services:
        f[name] = pyomo.util.PyomoAPIFactory(name)
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

def help_transformations():
    import pyomo.environ
    from pyomo.core import TransformationFactory
    wrapper = textwrap.TextWrapper()
    wrapper.initial_indent = '      '
    wrapper.subsequent_indent = '      '
    print("")
    for xform in sorted(TransformationFactory.services()):
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
    solvermgr_list = pyomo.opt.SolverManagerFactory.services()
    solvermgr_list = sorted( filter(lambda x: '_' != x[0], solvermgr_list) )
    n = max(map(len, solvermgr_list))
    wrapper = textwrap.TextWrapper(subsequent_indent=' '*(n+9))
    for s in solvermgr_list:
        # Disable warnings
        _level = logger.getEffectiveLevel()
        logger.setLevel(logging.ERROR)
        format = '    %-'+str(n)+'s     %s'
        # Reset logging level
        logger.setLevel(level=_level)
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
    solver_list = pyomo.opt.SolverFactory.services()
    solver_list = sorted( filter(lambda x: '_' != x[0], solver_list) )
    n = max(map(len, solver_list))
    wrapper = textwrap.TextWrapper(subsequent_indent=' '*(n+9))
    for s in solver_list:
        # Disable warnings
        _level = logger.getEffectiveLevel()
        logger.setLevel(logging.ERROR)
        # Create a solver, and see if it is available
        opt = pyomo.opt.SolverFactory(s)
        if s == 'asl' or s == 'py' or opt.available(False):
            format = '    %-'+str(n)+'s   * %s'
        else:
            format = '    %-'+str(n)+'s     %s'
        # Reset logging level
        logger.setLevel(level=_level)
        print(wrapper.fill(format % (s , pyomo.opt.SolverFactory.doc(s))))
    print("")
    wrapper = textwrap.TextWrapper(subsequent_indent='')
    print(wrapper.fill("An asterisk indicates that this solver is currently available to be run from Pyomo with the serial solver manager."))
    print('')
    print(wrapper.fill('Several solver interfaces are wrappers around third-party solver interfaces:  asl, openopt and os.  These interfaces require a subsolver specification that indicates the solver being executed.  For example, the following indicates that the OpenOpt pswarm solver is being used:'))
    print('')
    print('   openopt:pswarm')
    print('')
    print(wrapper.fill('The OpenOpt optimization package will launch the pswarm solver to perform optimization.  Similarly, the following indicates that the ipopt solver will be used:'))
    print('')
    print('   asl:ipopt')
    print('')
    print(wrapper.fill('The asl interface provides a generic wrapper for all solvers that use the AMPL Solver Library.'))
    print('')
    print(wrapper.fill('Note that subsolvers can not be enumerated automatically for these interfaces.  However, if a solver is specified that is not found, Pyomo assumes that the asl solver interface is being used.  Thus the following solver name will launch ipopt if the \'ipopt\' executable is on the user\'s path:'))
    print('')
    print('   ipopt')
    print('')
    _level = logger.getEffectiveLevel()
    logger.setLevel(logging.ERROR)
    try:
        #logger.setLevel(logging.WARNING)
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
    logger.setLevel(level=_level)

def help_exec(options):
    flag=False
    if options.commands:
        if options.asciidoc:
            print("The '--commands' help information is not printed in an asciidoc format.")
        flag=True
        help_commands()
    if options.api:
        flag=True
        help_api(options)
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
    if not flag:
        help_parser.print_help()

#
# Add a subparser for the pyomo command
#
def setup_help_parser(parser):
    parser.add_argument("-c", "--commands", dest="commands", action='store_true', default=False,
                        help="List the commands that are installed with Pyomo")
    parser.add_argument("-a", "--api", dest="api", action='store_true', default=False,
                        help="Print a summary of the Pyomo Library API")
    parser.add_argument("--asciidoc", dest="asciidoc", action='store_true', default=False,
                        help="Generate output that is compatible with asciidoc's markup language")
    parser.add_argument("-t", "--transformations", dest="transformations", action='store_true', default=False,
                        help="List the available model transformations")
    parser.add_argument("-s", "--solvers", dest="solvers", action='store_true', default=False,
                        help="Summarize the available solvers and solver interfaces")
    return parser

help_parser = setup_help_parser(
  pyomo_parser.add_subparser('help',
        func=help_exec, 
        help='Print help information.',
        description="This pyomo subcommand is used to print information about Pyomo's subcommands and installed Pyomo services."
        ))


