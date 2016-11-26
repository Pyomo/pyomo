#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

__all__ = ['test_solvers', 'test_solver_cases']

import os
import six
import logging

from pyutilib.misc import Options
from pyomo.opt import SolverFactory
from pyomo.opt.base.solvers import UnknownSolver

import pyomo.environ
from pyomo.solvers.plugins.solvers.GUROBI import GUROBISHELL
from pyomo.solvers.plugins.solvers.BARON import BARONSHELL

logger_solvers = logging.getLogger('pyomo.solvers')
_level_solvers = logger_solvers.getEffectiveLevel()
logger_solvers.setLevel( logging.ERROR )

logger_opt = logging.getLogger('pyomo.opt')
_level_opt = logger_opt.getEffectiveLevel()
logger_opt.setLevel( logging.ERROR )


# ----------------------------------------------------------------

_test_solvers = {}

def test_solvers(arg=None):
    if arg is None:
        return _test_solvers.keys()
    return _test_solvers[arg]


_test_solver_cases = {}

def test_solver_cases(*args):
    if len(args) == 0:
        return _test_solver_cases.keys()
    return _test_solver_cases[args]


# ----------------------------------------------------------------

#
# NOTE: we initialize the test case, since different
# interfaces may be used for the same "solver"
#
def initialize(**kwds):
    obj = Options(**kwds)
    #
    # Set obj.available
    #
    opt = None
    try:
        opt = SolverFactory(obj.name, solver_io=obj.io)
    except:
        pass
    if opt is None or isinstance(opt, UnknownSolver):
        obj.available = False
    elif (obj.name == "gurobi") and \
       (not GUROBISHELL.license_is_valid()):
        obj.available = False
    elif (obj.name == "baron") and \
       (not BARONSHELL.license_is_valid()):
        obj.available = False
    else:
        obj.available = \
            (opt.available(exception_flag=False)) and \
            ((not hasattr(opt,'executable')) or \
            (opt.executable() is not None))
    #
    # Check capabilities
    #
    if obj.available:
        for _c in _test_solvers[obj.name].capabilities:
            if not _c in opt._capabilities:
                raise ValueError("Solver %s does not support capability %s!" % (obj.name, _c))
        #
        # Get version
        #
        obj.version = opt.version()
    return obj


#
# CPLEX
#

_test_solvers['cplex'] = Options(
        name='cplex', 
        capabilities= set(['linear',
                       'integer',
                       'quadratic_objective',
                       'quadratic_constraint',
                       'sos1',
                       'sos2']))

_test_solver_cases['cplex', 'lp'] = initialize(
        name='cplex',
        io='lp',
        import_suffixes=['slack','dual','rc'])

_test_solver_cases['cplex', 'mps'] = initialize(
        name='cplex',
        io='mps',
        import_suffixes=['slack','dual','rc'])

_test_solver_cases['cplex', 'nl'] = initialize(
        name='cplex',
        io='nl',
        import_suffixes=['dual'])

_test_solver_cases['cplex', 'python'] = initialize(
        name='cplex',
        io='python',
        import_suffixes=['slack','dual','rc'])

#
# CPLEX PERSISTENT
#

_test_solvers['_cplex_persistent'] = Options(
        name='_cplex_persistent', 
        capabilities= set(['linear',
                       'integer',
                       'quadratic_objective',
                       'quadratic_constraint',
                       'sos1',
                       'sos2']))

_test_solver_cases['_cplex_persistent', 'python'] = initialize(
        name='_cplex_persistent',
        io='python',
        import_suffixes=['slack','dual','rc'])

#
# GUROBI
#
# **NOTE: Gurobi does not handle quadratic constraints before
#         Major Version 5
#
_test_solvers['gurobi'] = Options(
        name='gurobi', 
        capabilities= set(['linear',
                       'integer',
                       'quadratic_objective',
                       'quadratic_constraint',
                       'sos1',
                       'sos2']))

_test_solver_cases['gurobi', 'lp'] = initialize(
        name='gurobi',
        io='lp',
        import_suffixes=['slack','dual','rc'])

_test_solver_cases['gurobi', 'mps'] = initialize(
        name='gurobi',
        io='mps',
        import_suffixes=['slack','dual','rc'])

_test_solver_cases['gurobi', 'nl'] = initialize(
        name='gurobi',
        io='nl',
        options={'qcpdual':1,'simplex':1},
        import_suffixes=['dual'])

_test_solver_cases['gurobi', 'python'] = initialize(
        name='gurobi',
        io='python',
        import_suffixes=['slack','dual','rc'])

#
# GLPK
#
_test_solvers['glpk'] = Options(
        name='glpk', 
        capabilities= set(['linear',
                       'integer']))

if 'GLPKSHELL_old' in str(pyomo.solvers.plugins.solvers.GLPK.GLPK().__class__):
    glpk_import_suffixes = ['dual']
else:
    glpk_import_suffixes = ['rc','dual']

_test_solver_cases['glpk', 'lp'] = initialize(
        name='glpk',
        io='lp',
        import_suffixes=glpk_import_suffixes)

_test_solver_cases['glpk', 'mps'] = initialize(
        name='glpk',
        io='mps',
        import_suffixes=glpk_import_suffixes,
        io_options={"skip_objective_sense": True})

_test_solver_cases['glpk', 'python'] = initialize(
        name='glpk',
        io='python',
        import_suffixes=[])

#
# CBC
#
_test_solvers['cbc'] = Options(
        name='cbc', 
        capabilities = set(['linear',
                        'integer',
                        'sos1',
                        'sos2']))

_test_solver_cases['cbc', 'lp'] = initialize(
        name='cbc',
        io='lp',
        import_suffixes=['dual','rc'])

_test_solver_cases['cbc', 'nl'] = initialize(
        name='cbc',
        io='nl',
        import_suffixes=['dual'])

#
# PICO
#
_test_solvers['pico'] = Options(
        name='pico', 
        capabilities = set(['linear',
                        'integer']))

_test_solver_cases['pico', 'lp'] = initialize(
        name='pico',
        io='lp',
        import_suffixes=['dual'])

_test_solver_cases['pico', 'nl'] = initialize(
        name='pico',
        io='nl',
        import_suffixes=['dual'])

#
# XPRESS
#
_test_solvers['xpress'] = Options(
        name='xpress', 
        capabilities= set(['linear',
                       'integer',
                       'quadratic_objective',
                       'quadratic_constraint',
                       'sos1',
                       'sos2']))

_test_solver_cases['xpress', 'lp'] = initialize(
        name='xpress',
        io='lp',
        import_suffixes=['dual','rc','slack'])

_test_solver_cases['xpress', 'mps'] = initialize(
        name='xpress',
        io='mps',
        import_suffixes=['dual','rc','slack'])

_test_solver_cases['xpress', 'nl'] = initialize(
        name='xpress',
        io='nl',
        import_suffixes=['dual'])

#
# IPOPT
#
_test_solvers['ipopt'] = Options(
        name='ipopt', 
        capabilities= set(['linear',
                       'quadratic_objective',
                       'quadratic_constraint']))

_test_solver_cases['ipopt', 'nl'] = initialize(
        name='ipopt',
        io='nl',
        import_suffixes=['dual'])

#
# SCIP
#
_test_solvers['scip'] = Options(
        name='scip', 
        capabilities= set(['linear',
                       'integer',
                       'quadratic_objective',
                       'quadratic_constraint',
                       'sos1',
                       'sos2']))

_test_solver_cases['scip', 'nl'] = initialize(
        name='scip',
        io='nl',
        import_suffixes=[])

#
# BARON
#
_test_solvers['baron'] = Options(
        name='baron', 
        capabilities= set(['linear',
                       'integer',
                       'quadratic_objective',
                       'quadratic_constraint']))

_test_solver_cases['baron', 'bar'] = initialize(
        name='baron',
        io='bar',
        import_suffixes=['rc','dual'])

#
# KNITROAMPL
#
_test_solvers['knitroampl'] = Options(
        name='knitroampl', 
        capabilities= set(['linear',
                       'integer',
                       'quadratic_objective',
                       'quadratic_constraint']))

_test_solver_cases['knitroampl', 'nl'] = initialize(
        name='knitroampl',
        io='nl',
        import_suffixes=['dual'])


logger_opt.setLevel( _level_opt )
logger_solvers.setLevel( _level_solvers )

#
# Error Checks
#
for solver in six.itervalues(_test_solvers):
    if solver.capabilities is None:
        solver.capabilities = set([])
    assert (solver.name is not None) and (type(solver.name) is str)
    assert type(solver.capabilities) is set
    for tag in solver.capabilities:
        assert type(tag) is str

for sc in six.itervalues(_test_solver_cases):
    if sc.export_suffixes is None:
        sc.export_suffixes = []
    if sc.import_suffixes is None:
        sc.import_suffixes = []
    if sc.options is None:
        sc.options = {}
    if sc.io_options is None:
        sc.io_options = {}
    assert (sc.io is not None) and (type(sc.io) is str)
    assert type(sc.export_suffixes) in [list,tuple]
    assert type(sc.import_suffixes) in [list,tuple]
    assert type(sc.options) is dict
    for tag in sc.export_suffixes:
        assert type(tag) is str
    for tag in sc.import_suffixes:
        assert type(tag) is str

