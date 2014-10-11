
import sys
if sys.version_info > (3,0):
    import importlib

#
# These packages contain plugins that need to be loaded
#
packages = [ 'pyomo.opt', 'pyomo.core', 'pyomo.checker', 'pyomo.repn', 
             'pyomo.os', 'pyomo.pysp', 'pyomo.neos',
             'pyomo.openopt', 'pyomo.solvers', 'pyomo.gdp', 'pyomo.mpec', 
             'pyomo.dae', 'pyomo.bilevel']
# 
# These packages are under development, or they may be omitted in a Pyomo installation
#
optional_packages = set([])


def do_import(pname):
    if sys.version_info > (3,0):
        importlib.import_module(pname)
    else:
        __import__(pname, globals(), locals(), [], -1)


def import_packages():
    for name in packages:
        pname = name+'.plugins'
        imported = False
        if name in optional_packages:
            try:
                do_import(pname)
                imported = True
            except ImportError:
                pass
        else:
            try:
                do_import(pname)
            except ImportError:
                exctype, err, tb = sys.exc_info()[0:3] # BUG?
                import traceback
                msg = "pyomo.environ failed to import %s:\nOriginal %s: %s\nTraceback:\n%s" \
                    % ( pname, type(err).__name__, err, 
                        #''.join(traceback.format_stack(f=tb.tb_frame.f_back)),
                        ''.join(traceback.format_tb(tb)), )
                # clear local variables to remove circular references
                exctype = err = tb = None
                raise ImportError(msg)
            imported = True
        if imported:
            pkg = sys.modules[pname]
            pkg.load()

import_packages()

