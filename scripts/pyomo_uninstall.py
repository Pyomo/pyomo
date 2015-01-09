#
# A script to uninstall Coopr, Pyomo and PyUtilib.
#
try:
    import pip
except ImportError:
    print("You must have 'pip' installed to run this script.")
    raise SystemExit


packages = [
'Pyomo',
'Coopr',
'PyUtilib',
'coopr.age',
'coopr.bilevel',
'coopr.core',
'coopr.dae',
'coopr.environ',
'coopr.gdp',
'coopr.misc',
'coopr.mpec',
'coopr.neos',
'coopr.openopt',
'coopr.opt',
'coopr.os',
'coopr.pyomo',
'coopr.pysos',
'coopr.pysp',
'coopr.solvers',
'coopr.sucasa',
'pyutilib.R',
'pyutilib.autotest',
'pyutilib.common',
'pyutilib.component.app',
'pyutilib.component.config',
'pyutilib.component.core',
'pyutilib.component.executables',
'pyutilib.component.loader',
'pyutilib.dev',
'pyutilib.enum',
'pyutilib.excel',
'pyutilib.math',
'pyutilib.misc',
'pyutilib.ply',
'pyutilib.pyro',
'pyutilib.services',
'pyutilib.subprocess',
'pyutilib.svn',
'pyutilib.th',
'pyutilib.virtualenv',
'pyutilib.workflow',
]

print("Uninstalling...")
for package in packages:
    try:
        pip.main(['uninstall','-y',package])
    except:
        pass
    #
    # See https://github.com/pypa/pip/issues/1618 for an 
    # explanation of this hack.  This reset's the logger used by
    # pip.
    #
    pip.logger.consumers = []

