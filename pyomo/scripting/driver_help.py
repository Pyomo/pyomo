#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import os
import os.path
import sys
import glob
import textwrap
import logging
import socket
import subprocess

import pyomo.common
from pyomo.common.collections import Bunch
from pyomo.common.tee import capture_output
import pyomo.scripting.pyomo_parser

logger = logging.getLogger('pyomo.solvers')

# --------------------------------------------------
# run
#   --list
# --------------------------------------------------


def setup_command_parser(parser):
    parser.add_argument(
        "--list",
        dest="summary",
        action='store_true',
        default=False,
        help="List the commands that are installed with Pyomo",
    )
    parser.add_argument(
        "command", nargs='*', help="The command and command-line options"
    )


def command_exec(options):
    cmddir = os.path.dirname(os.path.abspath(sys.executable)) + os.sep
    if options.summary:
        print("")
        print("The following commands are installed in the Pyomo bin directory:")
        print("----------------------------------------------------------------")
        for file in sorted(glob.glob(cmddir + '*')):
            print(" " + os.path.basename(file))
        print("")
        if len(options.command) > 0:
            print("WARNING: ignoring command specification")
        return
    if len(options.command) == 0:
        print("  ERROR: no command specified")
        return 1
    if not os.path.exists(cmddir + options.command[0]):
        print(
            "  ERROR: the command '%s' does not exist" % (cmddir + options.command[0])
        )
        return 1
    return subprocess.run(
        [cmddir] + options.command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    ).returncode


#
# Add a subparser for the pyomo command
#
setup_command_parser(
    pyomo.scripting.pyomo_parser.add_subparser(
        'run',
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
""",
    )
)

# --------------------------------------------------
# help
#   --components
#   --command
#   --transformations
#   --solvers
# --------------------------------------------------


def help_commands():
    print("")
    print("The following commands are installed with Pyomo:")
    print("-" * 75)
    registry = pyomo.common.get_pyomo_commands()
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
        print("  " + writer)
        print(wrapper.fill(WriterFactory.doc(writer)))


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
        print("  " + xform)
        print(wrapper.fill(DataManagerFactory.doc(xform)))


def help_environment():
    info = Bunch()
    #
    info.python = Bunch()
    info.python.version = '%d.%d.%d' % sys.version_info[:3]
    info.python.executable = sys.executable
    info.python.platform = sys.platform
    try:
        packages = []
        import pip

        for package in pip.get_installed_distributions():
            packages.append(Bunch(name=package.project_name, version=package.version))
        info.python.packages = packages
    except:
        pass
    #
    info.environment = Bunch()
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
        print("  " + xform)
        _doc = TransformationFactory.doc(xform) or ""
        # Ideally, the Factory would ensure that the doc string
        # indicated deprecation, but as @deprecated() is Pyomo
        # functionality and the Factory comes directly from PyUtilib,
        # PyUtilib probably shouldn't contain Pyomo-specific processing.
        # The next best thing is to ensure that the deprecation status
        # is indicated here.
        _init_doc = TransformationFactory.get_class(xform).__init__.__doc__ or ""
        if _init_doc.strip().startswith('DEPRECATED') and 'DEPRECATE' not in _doc:
            _doc = ' '.join(('[DEPRECATED]', _doc))
        if _doc:
            print(wrapper.fill(_doc))


def help_solvers():
    import pyomo.environ

    wrapper = textwrap.TextWrapper(replace_whitespace=False)
    print("")
    print("Pyomo Solvers and Solver Managers")
    print("---------------------------------")

    print(
        wrapper.fill(
            "Pyomo uses 'solver managers' to execute 'solvers' that perform optimization and other forms of model analysis.  A solver directly executes an optimizer, typically using an executable found on the user's PATH environment.  Solver managers support a flexible mechanism for asynchronously executing solvers either locally or remotely.  The following solver managers are available in Pyomo:"
        )
    )
    print("")
    solvermgr_list = list(pyomo.opt.SolverManagerFactory)
    solvermgr_list = sorted(filter(lambda x: '_' != x[0], solvermgr_list))
    n = max(map(len, solvermgr_list))
    wrapper = textwrap.TextWrapper(subsequent_indent=' ' * (n + 9))
    for s in solvermgr_list:
        format = '    %-' + str(n) + 's     %s'
        print(wrapper.fill(format % (s, pyomo.opt.SolverManagerFactory.doc(s))))
    print("")
    wrapper = textwrap.TextWrapper(subsequent_indent='')
    print(
        wrapper.fill(
            "If no solver manager is specified, Pyomo uses the serial solver manager to execute solvers locally.  The neos solver manager is used to execute solvers on the NEOS optimization server."
        )
    )
    print("")

    print("")
    print("Serial Solver Interfaces")
    print("------------------------")
    print(wrapper.fill("The serial manager supports the following solver interfaces:"))
    print("")
    solver_list = list(pyomo.opt.SolverFactory)
    solver_list = sorted(filter(lambda x: '_' != x[0], solver_list))
    _data = []
    try:
        # Disable warnings
        logging.disable(logging.WARNING)
        # suppress ALL output
        with capture_output(capture_fd=True):
            for s in solver_list:
                # Create a solver, and see if it is available
                with pyomo.opt.SolverFactory(s) as opt:
                    ver = ''
                    if opt.available(False):
                        avail = '-'
                        if opt.license_is_valid():
                            avail = '+'
                        try:
                            ver = opt.version()
                            if isinstance(ver, str):
                                pass
                            elif ver:
                                while len(ver) > 2 and ver[-1] == 0:
                                    ver = ver[:-1]
                                ver = '.'.join(str(v) for v in ver)
                            else:
                                ver = ''
                        except (AttributeError, NameError):
                            pass
                    elif s == 'py':
                        # py is a metasolver, but since we don't specify a subsolver
                        # for this test, opt is actually an UnknownSolver, so we
                        # can't try to get the _metasolver attribute from it.
                        avail = '*'
                    elif isinstance(s, pyomo.opt.solvers.UnknownSolver):
                        # We can get here if creating a registered
                        # solver failed (i.e., an exception was raised
                        # in __init__)
                        avail = ''
                    elif getattr(opt, "_metasolver", False):
                        # Note: default to False if the attribute isn't implemented
                        avail = '*'
                    else:
                        avail = ''
                    _data.append((avail, s, ver, pyomo.opt.SolverFactory.doc(s)))
    finally:
        # Reset logging level
        logging.disable(logging.NOTSET)
    nameFieldLen = max(len(line[1]) for line in _data)
    verFieldLen = max(len(line[2]) for line in _data)
    fmt = '   %%1s%%-%ds %%-%ds %%s' % (nameFieldLen, verFieldLen)
    wrapper = textwrap.TextWrapper(
        subsequent_indent=' ' * (nameFieldLen + verFieldLen + 6)
    )
    for _line in _data:
        print(wrapper.fill(fmt % _line))

    print("")
    wrapper = textwrap.TextWrapper(subsequent_indent='')
    print(
        wrapper.fill(
            """The leading symbol (one of *, -, +) indicates the current solver availability.  A plus (+) indicates the solver is currently available to be run from Pyomo with the serial solver manager, and (if applicable) has a valid license.  A minus (-) indicates the solver executables are available but do not report having a valid license.  The solver may still be usable in an unlicensed or "demo" mode for limited problem sizes. An asterisk (*) indicates meta-solvers or generic interfaces, which are always available."""
        )
    )
    print('')
    print(
        wrapper.fill(
            'Pyomo also supports solver interfaces that are wrappers around third-party solver interfaces. These interfaces require a subsolver specification that indicates the solver being executed.  For example, the following indicates that the ipopt solver will be used:'
        )
    )
    print('')
    print('   asl:ipopt')
    print('')
    print(
        wrapper.fill(
            'The asl interface provides a generic wrapper for all solvers that use the AMPL Solver Library.'
        )
    )
    print('')
    print(
        wrapper.fill(
            'Note that subsolvers can not be enumerated automatically for these interfaces.  However, if a solver is specified that is not found, Pyomo assumes that the asl solver interface is being used.  Thus the following solver name will launch ipopt if the \'ipopt\' executable is on the user\'s path:'
        )
    )
    print('')
    print('   ipopt')
    print('')
    try:
        logging.disable(logging.WARNING)
        socket.setdefaulttimeout(10)
        import pyomo.neos.kestrel

        kestrel = pyomo.neos.kestrel.kestrelAMPL()

        solver_list = list(
            set(
                [
                    name[:-5].lower()
                    for name in kestrel.solvers()
                    if name.endswith('AMPL')
                ]
            )
        )

        print("")
        print("NEOS Solver Interfaces")
        print("----------------------")
        if len(solver_list) > 0:
            print(
                wrapper.fill(
                    "The neos solver manager supports solver interfaces that can "
                    "be executed remotely on the NEOS optimization server.  The "
                    "following solver interfaces are available with your current "
                    "system configuration:"
                )
            )
            print("")
            solver_list = sorted(solver_list)
            n = max(map(len, solver_list))
            format = '    %-' + str(n) + 's     %s'
            for name in solver_list:
                print(
                    wrapper.fill(
                        format
                        % (name, pyomo.neos.doc.get(name, 'Unexpected NEOS solver'))
                    )
                )
            print("")
        else:

            def _specific_neos_error(err):
                import socket, xmlrpc.client

                if isinstance(err, NotImplementedError):
                    return (
                        "your Python was compiled without SSL support; "
                        "HTTPS connections (required by NEOS) cannot be made."
                    )
                if isinstance(err, socket.timeout):
                    return "connection to neos-server.org timed out."
                if isinstance(err, socket.gaierror):
                    return "neos-server.org could not be resolved (DNS failure)."
                if isinstance(err, xmlrpc.client.ProtocolError):
                    return f"server responded with HTTP {err.errcode}."
                return str(err)

            reason = _specific_neos_error(getattr(kestrel, "connect_error", None))

            print(
                wrapper.fill(
                    "The neos solver manager executes solvers on the remote "
                    "NEOS optimization server, but that server is currently "
                    f"unavailable on this system.  Reason: {reason}"
                )
            )
            print("")
    except ImportError:
        pass
    finally:
        logging.disable(logging.NOTSET)
        socket.setdefaulttimeout(None)


def print_components(data):
    """
    Print information about modeling components supported by Pyomo.
    """
    from pyomo.core.base.component import ModelComponentFactory, GlobalSets

    print("")
    print("----------------------------------------------------------------")
    print("Pyomo Model Components:")
    print("----------------------------------------------------------------")
    for name in sorted(ModelComponentFactory):
        print("")
        print(" " + name)
        for line in textwrap.wrap(ModelComponentFactory.doc(name), 59):
            print("    " + line)
    print("")
    print("----------------------------------------------------------------")
    print("Pyomo Virtual Sets:")
    print("----------------------------------------------------------------")
    for name, obj in sorted(GlobalSets.items()):
        print("")
        print(" " + name)
        print("    " + obj.doc)


def help_exec(options):
    flag = False
    if options.commands:
        if options.asciidoc:
            print(
                "The '--commands' help information is not printed in an asciidoc format."
            )
        flag = True
        help_commands()
    if options.components:
        if options.asciidoc:
            print(
                "The '--components' help information is not printed in an asciidoc format."
            )
        flag = True
        print_components(None)
    if options.datamanager:
        flag = True
        help_datamanagers(options)
    if options.environment:
        flag = True
        help_environment()
    if options.transformations:
        if options.asciidoc:
            print(
                "The '--transformations' help information is not printed in an asciidoc format."
            )
        flag = True
        help_transformations()
    if options.solvers:
        if options.asciidoc:
            print(
                "The '--solvers' help information is not printed in an asciidoc format."
            )
        flag = True
        help_solvers()
    if options.writers:
        flag = True
        if options.asciidoc:
            print(
                "The '--writers' help information is not printed in an asciidoc format."
            )
        help_writers()
    if not flag:
        help_parser.print_help()


#
# Add a subparser for the pyomo command
#
def setup_help_parser(parser):
    parser.add_argument(
        "--asciidoc",
        dest="asciidoc",
        action='store_true',
        default=False,
        help="Generate output that is compatible with asciidoc's markup language",
    )
    parser.add_argument(
        "-c",
        "--commands",
        dest="commands",
        action='store_true',
        default=False,
        help="List the commands that are installed with Pyomo",
    )
    parser.add_argument(
        "--components",
        dest="components",
        action='store_true',
        default=False,
        help="List the components that are available in Pyomo's modeling environment",
    )
    parser.add_argument(
        "-d",
        "--data-managers",
        dest="datamanager",
        action='store_true',
        default=False,
        help="Print a summary of the data managers in Pyomo",
    )
    parser.add_argument(
        "-i",
        "--info",
        dest="environment",
        action='store_true',
        default=False,
        help="Summarize the environment and Python installation",
    )
    parser.add_argument(
        "-s",
        "--solvers",
        dest="solvers",
        action='store_true',
        default=False,
        help="Summarize the available solvers and solver interfaces",
    )
    parser.add_argument(
        "-t",
        "--transformations",
        dest="transformations",
        action='store_true',
        default=False,
        help="List the available model transformations",
    )
    parser.add_argument(
        "-w",
        "--writers",
        dest="writers",
        action='store_true',
        default=False,
        help="List the available problem writers",
    )
    return parser


help_parser = setup_help_parser(
    pyomo.scripting.pyomo_parser.add_subparser(
        'help',
        func=help_exec,
        help='Print help information.',
        description="This pyomo subcommand is used to print information about Pyomo's subcommands and installed Pyomo services.",
    )
)
