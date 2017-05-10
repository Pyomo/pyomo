#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

#
# Expose the symbols from pyomo.core
#
from pyomo.core import *
from pyomo.opt import SolverFactory, SolverManagerFactory, UnknownSolver
from pyomo.util.plugin import PluginGlobals as _PG

import sys as _sys
if _sys.version_info[0] >= 3:
    import importlib

    def _do_import(pkg_name):
        importlib.import_module(pkg_name)
else:
    def _do_import(pkg_name):
        __import__(pkg_name, globals(), locals(), [], -1)

#
# These packages contain plugins that need to be loaded
#
_packages = ['pyomo.opt', 'pyomo.core', 'pyomo.checker', 'pyomo.repn',
             'pyomo.pysp', 'pyomo.neos',
             'pyomo.solvers', 'pyomo.gdp', 'pyomo.mpec',
             'pyomo.dae', 'pyomo.bilevel', 'pyomo.scripting']
#
# These packages are under development, or they may be omitted in a
# Pyomo installation; silently ignore any import errors.
#
_optional_packages = set([])


def _import_packages():
    for name in _packages:
        pname = name+'.plugins'
        try:
            _do_import(pname)
        except ImportError:
            if name in _optional_packages:
                # Note: Continue to avoid the package load().
                #
                # TODO: Should this generate a warning if the package exists
                # on the file system?
                continue
            else:
                exctype, err, tb = _sys.exc_info()  # BUG?
                import traceback
                msg = "pyomo.environ failed to import %s:\nOriginal %s: %s\n"\
                      "Traceback:\n%s" \
                      % (pname, exctype.__name__, err,
                         ''.join(traceback.format_tb(tb)),)
                # clear local variables to remove circular references
                exctype = err = tb = None
                # TODO: Should this just log an error and re-raise the
                # original exception?
                raise ImportError(msg)

        pkg = _sys.modules[pname]
        pkg.load()

_PG.add_env("pyomo")
_import_packages()
_PG.pop_env()

# The following sets up the ctype properties on the
# components in core/kernel. It is a temporary hack
def _setup_kernel():
    from pyomo.core.base import (Var,
                                 Constraint,
                                 Objective,
                                 Block,
                                 Param,
                                 Expression,
                                 SOSConstraint,
                                 Suffix)

    from pyomo.core.kernel.component_block import (block,
                                                   tiny_block,
                                                   block_tuple,
                                                   block_list,
                                                   block_dict)
    from pyomo.core.kernel.component_variable import (variable,
                                                      variable_tuple,
                                                      variable_list,
                                                      variable_dict)
    from pyomo.core.kernel.component_constraint import (constraint,
                                                        linear_constraint,
                                                        constraint_tuple,
                                                        constraint_list,
                                                        constraint_dict)
    from pyomo.core.kernel.component_parameter import (parameter,
                                                       parameter_tuple,
                                                       parameter_list,
                                                       parameter_dict)
    from pyomo.core.kernel.component_expression import (expression,
                                                        data_expression,
                                                        expression_tuple,
                                                        expression_list,
                                                        expression_dict)
    from pyomo.core.kernel.component_objective import (objective,
                                                       objective_tuple,
                                                       objective_list,
                                                       objective_dict)
    from pyomo.core.kernel.component_sos import (sos,
                                                 sos1,
                                                 sos2,
                                                 sos_tuple,
                                                 sos_list,
                                                 sos_dict)
    from pyomo.core.kernel.component_suffix import (suffix,
                                                    export_suffix_generator,
                                                    import_suffix_generator,
                                                    local_suffix_generator,
                                                    suffix_generator)

    #
    # setup ctypes
    #

    variable._ctype = Var
    variable_tuple._ctype = Var
    variable_list._ctype = Var
    variable_dict._ctype = Var

    block._ctype = Block
    block_tuple._ctype = Block
    block_list._ctype = Block
    block_dict._ctype = Block
    tiny_block._ctype = Block

    constraint._ctype = Constraint
    linear_constraint._ctype = Constraint
    constraint_tuple._ctype = Constraint
    constraint_list._ctype = Constraint
    constraint_dict._ctype = Constraint

    parameter._ctype = Param
    parameter_tuple._ctype = Param
    parameter_list._ctype = Param
    parameter_dict._ctype = Param

    objective._ctype = Objective
    objective_tuple._ctype = Objective
    objective_list._ctype = Objective
    objective_dict._ctype = Objective

    expression._ctype = Expression
    expression_tuple._ctype = Expression
    expression_list._ctype = Expression
    expression_dict._ctype = Expression

    sos._ctype = SOSConstraint
    sos_tuple._ctype = SOSConstraint
    sos_list._ctype = SOSConstraint
    sos_dict._ctype = SOSConstraint

    suffix._ctype = Suffix

_setup_kernel()
del _setup_kernel
