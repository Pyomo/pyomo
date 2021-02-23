#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import six
from pyomo.scripting.pyomo_parser import add_subparser, CustomHelpFormatter

from pyomo.common.deprecation import deprecated

def get_packages():
    packages = [
        'sympy', 
        'xlrd', 
        'openpyxl', 
        #('suds-jurko', 'suds'),
        ('PyYAML', 'yaml'),
        'pypyodbc', 
        'pymysql', 
        #'openopt', 
        #'FuncDesigner', 
        #'DerApproximator', 
        ('ipython[notebook]', 'IPython'),
        ('pyro4', 'Pyro4'),
    ]
    if six.PY2:
        packages.append(('pyro','Pyro'))
    return packages

@deprecated(
        "Use of the pyomo install-extras is deprecated."
        "The current recommended course of action is to manually install "
        "optional dependencies as needed.",
        version='5.7.1')
def install_extras(args=[], quiet=False):
    #
    # Verify that pip is installed
    #
    try:
        import pip
        pip_version = pip.__version__.split('.')
        for i,s in enumerate(pip_version):
            try:
                pip_version[i] = int(s)
            except:
                pass
        pip_version = tuple(pip_version)
    except ImportError:
        print("You must have 'pip' installed to run this script.")
        raise SystemExit

    cmd = ['--disable-pip-version-check', 'install','--upgrade']
    # Disable the PIP download cache
    if pip_version[0] >= 6:
        cmd.append('--no-cache-dir')
    else:
        cmd.append('--download-cache')
        cmd.append('')

    if not quiet:
        print(' ')
        print('-'*60)
        print("Installation Output Logs")
        print("  (A summary will be printed below)")
        print('-'*60)
        print(' ')

    results = {}
    for package in get_packages():
        if type(package) is tuple:
            package, pkg_import = package
        else:
            pkg_import = package
        try:
            # Allow the user to provide extra options
            pip.main(cmd + args + [package])
            __import__(pkg_import)
            results[package] = True
        except:
            results[package] = False
        try:
            pip.logger.consumers = []
        except AttributeError:
            # old pip versions (prior to 6.0~104^2)
            pip.log.consumers = []

    if not quiet:
        print(' ')
        print(' ')
    print('-'*60)
    print("Installation Summary")
    print('-'*60)
    print(' ')
    for package, result in sorted(six.iteritems(results)):
        if result:
            print("YES %s" % package)
        else:
            print("NO  %s" % package)


def pyomo_subcommand(options):
    return install_extras(options.args, quiet=options.quiet)


_parser = add_subparser(
    'install-extras',
    func=pyomo_subcommand,
    help='Install "extra" packages that Pyomo can leverage.',
    description="""
This pyomo subcommand uses PIP to install optional third-party Python
packages that Pyomo could leverage from PyPI.  The installation of some
packages may fail, but this subcommand ignore these failures and
provides a summary describing which packages were installed.
""",
    epilog="""
Since pip options begin with a dash, the --pip-args option can only be
used with the equals syntax.  --pip-args may appear multiple times on
the command line.  For example:\n\n
    pyomo install-extras --pip-args="--upgrade"
""",
    formatter_class=CustomHelpFormatter,
)

_parser.add_argument(
    '-q', '--quiet',
    action='store_true',
    dest='quiet',
    default=False,
    help="Suppress some terminal output",
)
_parser.add_argument(
    "--pip-args",
    dest="args",
    action="append",
    help=("Arguments that are passed to the 'pip' command when "
          "installing packages"),
)

