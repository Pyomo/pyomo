#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

__all__ = ['test_solver_cases', 'available_solvers']

import six
import logging

from pyomo.common.collections import Options
from pyomo.opt import SolverFactory
from pyomo.opt.base.solvers import UnknownSolver

import pyomo.environ
from pyomo.solvers.plugins.solvers.GUROBI import GUROBISHELL
from pyomo.solvers.plugins.solvers.BARON import BARONSHELL
from pyomo.solvers.plugins.solvers.mosek_direct import MOSEKDirect

# ----------------------------------------------------------------

_test_solver_cases = {}


# ----------------------------------------------------------------

licensed_solvers_with_demo_mode = {'baron',}

#
# NOTE: we initialize the test case, since different
# interfaces may be used for the same "solver"
#
def initialize(**kwds):
    obj = Options(**kwds)
    #
    # Set obj.available
    #
    try:
        opt = SolverFactory(obj.name, solver_io=obj.io)
    except:
        opt = None

    if opt is None or isinstance(opt, UnknownSolver):
        obj.available = False
    elif not opt.available(exception_flag=False):
        obj.available = False
    elif hasattr(opt, 'executable') and opt.executable() is None:
        obj.available = False
    elif not opt.license_is_valid() \
         and obj.name not in licensed_solvers_with_demo_mode:
        obj.available = False
    else:
        obj.available = True
    #
    # Set the limits for the solver's "demo" (unlicensed) mode:
    #   ( nVars, nCons, nNonZeros )
    obj.demo_limits = (None, None, None)
    if obj.available:
        if obj.name == "baron" and not opt.license_is_valid():
            obj.demo_limits = (10, 10, 50)
    #
    # Check capabilities, even if the solver is not available
    #
    if not (opt is None or isinstance(opt, UnknownSolver)):
        for _c in obj.capabilities:
            if not _c in opt._capabilities:
                raise ValueError("Solver %s does not support capability %s!" % (obj.name, _c))
    #
    # Get version
    #
    if obj.available:
        obj.version = opt.version()
    return obj


def test_solver_cases(*args):
    """
    A function for accessing _test_solver_casess as global state
    """
    if len(_test_solver_cases) == 0:
        logging.disable(logging.WARNING)


        #
        # MOSEK
        #

        _mosek_capabilities = set(['linear',
                                   'integer',
                                   'quadratic_objective',
                                   'quadratic_constraint',
                                   'conic_constraints'])
    
        _test_solver_cases['mosek', 'python'] = initialize(
            name='mosek_direct',
            io='python',
            capabilities=_mosek_capabilities,
            import_suffixes=['dual', 'rc', 'slack'])

        #
        # MOSEK Persistent
        #
        _test_solver_cases['mosek_persistent','python'] = initialize(
                name = 'mosek_persistent',
                io = 'python',
                capabilities=_mosek_capabilities,
                import_suffixes=['dual','rc','slack'])

        #
        # CPLEX
        #

        _cplex_capabilities= set(['linear',
                                  'integer',
                                  'quadratic_objective',
                                  'quadratic_constraint',
                                  'sos1',
                                  'sos2'])

        _test_solver_cases['cplex', 'lp'] = initialize(
            name='cplex',
            io='lp',
            capabilities=_cplex_capabilities,
            import_suffixes=['slack','dual','rc'])

        _test_solver_cases['cplex', 'mps'] = initialize(
            name='cplex',
            io='mps',
            capabilities=_cplex_capabilities,
            import_suffixes=['slack','dual','rc'])

        _test_solver_cases['cplex', 'nl'] = initialize(
            name='cplex',
            io='nl',
            capabilities=_cplex_capabilities,
            import_suffixes=['dual'])

        _test_solver_cases['cplex', 'python'] = initialize(
            name='cplex',
            io='python',
            capabilities=_cplex_capabilities,
            import_suffixes=['slack','dual','rc'])

        #
        # CPLEX PERSISTENT
        #

        _test_solver_cases['cplex_persistent', 'python'] = initialize(
            name='cplex_persistent',
            io='python',
            capabilities=_cplex_capabilities,
            import_suffixes=['slack','dual','rc'])

        #
        # GAMS
        #

        _gams_capabilities= set(['linear',
                                 'integer',
                                 'quadratic_objective',
                                 'quadratic_constraint'])

        _test_solver_cases['gams', 'gms'] = initialize(
            name='gams',
            io='gms',
            capabilities=_gams_capabilities,
            import_suffixes=['dual','rc'])

        _test_solver_cases['gams', 'python'] = initialize(
            name='gams',
            io='python',
            capabilities=_gams_capabilities,
            import_suffixes=['dual','rc'])

        #
        # GUROBI
        #
        # **NOTE: Gurobi does not handle quadratic constraints before
        #         Major Version 5
        #
        _gurobi_capabilities= set(['linear',
                                   'integer',
                                   'quadratic_objective',
                                   'quadratic_constraint',
                                   'sos1',
                                   'sos2'])

        _test_solver_cases['gurobi', 'lp'] = initialize(
            name='gurobi',
            io='lp',
            capabilities=_gurobi_capabilities,
            import_suffixes=['slack','dual','rc'])

        _test_solver_cases['gurobi', 'mps'] = initialize(
            name='gurobi',
            io='mps',
            capabilities=_gurobi_capabilities,
            import_suffixes=['slack','dual','rc'])

        _test_solver_cases['gurobi', 'nl'] = initialize(
            name='gurobi',
            io='nl',
            capabilities=_gurobi_capabilities,
            options={'qcpdual':1,'simplex':1},
            import_suffixes=['dual'])

        _test_solver_cases['gurobi', 'python'] = initialize(
            name='gurobi',
            io='python',
            capabilities=_gurobi_capabilities,
            import_suffixes=['slack','dual','rc'])

        #
        # Gurobi PERSISTENT
        #

        _test_solver_cases['gurobi_persistent', 'python'] = initialize(
            name='gurobi_persistent',
            io='python',
            capabilities=_gurobi_capabilities,
            import_suffixes=['slack', 'dual', 'rc'])

        #
        # GLPK
        #
        _glpk_capabilities= set(['linear',
                                 'integer'])

        if 'GLPKSHELL_old' in str(pyomo.solvers.plugins.solvers.GLPK.GLPK().__class__):
            glpk_import_suffixes = ['dual']
        else:
            glpk_import_suffixes = ['rc','dual']

        _test_solver_cases['glpk', 'lp'] = initialize(
            name='glpk',
            io='lp',
            capabilities=_glpk_capabilities,
            import_suffixes=glpk_import_suffixes)

        _test_solver_cases['glpk', 'mps'] = initialize(
            name='glpk',
            io='mps',
            capabilities=_glpk_capabilities,
            import_suffixes=glpk_import_suffixes,
            io_options={"skip_objective_sense": True})

        _test_solver_cases['glpk', 'python'] = initialize(
            name='glpk',
            io='python',
            capabilities=_glpk_capabilities,
            import_suffixes=[])

        #
        # CBC
        #
        _cbc_lp_capabilities = set(['linear', 'integer'])

        _test_solver_cases['cbc', 'lp'] = initialize(
            name='cbc',
            io='lp',
            capabilities=_cbc_lp_capabilities,
            import_suffixes=['dual','rc'])

        _cbc_nl_capabilities = set(['linear', 'integer', 'sos1', 'sos2'])

        _test_solver_cases['cbc', 'nl'] = initialize(
            name='cbc',
            io='nl',
            capabilities=_cbc_nl_capabilities,
            import_suffixes=['dual'])

        #_cbc_mps_capabilities = set(['linear', 'integer', 'sos1', 'sos2'])

        #_test_solver_cases['cbc', 'mps'] = initialize(
            #name='cbc',
            #io='mps',
            #capabilities=_cbc_mps_capabilities,
            #import_suffixes=['dual', 'rc'])

        #
        # PICO
        #
        _pico_capabilities = set(['linear', 'integer'])

        _test_solver_cases['pico', 'lp'] = initialize(
            name='pico',
            io='lp',
            capabilities=_pico_capabilities,
            import_suffixes=['dual'])

        _test_solver_cases['pico', 'nl'] = initialize(
            name='pico',
            io='nl',
            capabilities=_pico_capabilities,
            import_suffixes=['dual'])

        #
        # XPRESS
        #
        _xpress_capabilities= set(['linear',
                                   'integer',
                                   'quadratic_objective',
                                   'quadratic_constraint',
                                   'sos1',
                                   'sos2'])

        _test_solver_cases['xpress', 'lp'] = initialize(
            name='xpress',
            io='lp',
            capabilities=_xpress_capabilities,
            import_suffixes=['dual','rc','slack'],
            options={'bargapstop':1e-9,})

        _test_solver_cases['xpress', 'mps'] = initialize(
            name='xpress',
            io='mps',
            capabilities=_xpress_capabilities,
            import_suffixes=['dual','rc','slack'],
            options={'bargapstop':1e-9,})

        _test_solver_cases['xpress', 'nl'] = initialize(
            name='xpress',
            io='nl',
            capabilities=_xpress_capabilities,
            import_suffixes=['dual'],
            options={'bargapstop':1e-9,})

        _test_solver_cases['xpress', 'python'] = initialize(
            name='xpress',
            io='python',
            capabilities=_xpress_capabilities,
            import_suffixes=['dual','rc','slack'],
            options={'bargapstop':1e-9,})

        #
        # XPRESS PERSISTENT 
        #

        _test_solver_cases['xpress_persistent', 'python'] = initialize(
            name='xpress_persistent',
            io='python',
            capabilities=_xpress_capabilities,
            import_suffixes=['slack', 'dual', 'rc'],
            options={'bargapstop':1e-9,})

        #
        # IPOPT
        #
        _ipopt_capabilities= set(['linear',
                                  'quadratic_objective',
                                  'quadratic_constraint'])

        _test_solver_cases['ipopt', 'nl'] = initialize(
            name='ipopt',
            io='nl',
            capabilities=_ipopt_capabilities,
            import_suffixes=['dual'])

        #
        # SCIP
        #
        _scip_capabilities= set(['linear',
                                 'integer',
                                 'quadratic_objective',
                                 'quadratic_constraint',
                                 'sos1',
                                 'sos2'])

        _test_solver_cases['scip', 'nl'] = initialize(
            name='scip',
            io='nl',
            capabilities=_scip_capabilities,
            import_suffixes=[])

        #
        # CONOPT
        #
        _conopt_capabilities= set(['linear',
                                   'integer',
                                   'quadratic_objective',
                                   'quadratic_constraint',
                                   'sos1',
                                   'sos2'])
        _test_solver_cases['conopt', 'nl'] = initialize(
            name='conopt',
            io='nl',
            capabilities=_conopt_capabilities,
            import_suffixes=[])

        #
        # BARON
        #
        _baron_capabilities= set(['linear',
                                  'integer',
                                  'quadratic_objective',
                                  'quadratic_constraint'])

        _test_solver_cases['baron', 'bar'] = initialize(
            name='baron',
            io='bar',
            capabilities=_baron_capabilities,
            import_suffixes=['rc','dual'])

        #
        # KNITROAMPL
        #
        _knitroampl_capabilities= set(['linear',
                                       'integer',
                                       'quadratic_objective',
                                       'quadratic_constraint'])

        _test_solver_cases['knitroampl', 'nl'] = initialize(
            name='knitroampl',
            io='nl',
            capabilities=_knitroampl_capabilities,
            import_suffixes=['dual'])


        logging.disable(logging.NOTSET)

        #
        # Error Checks
        #
        for sc in six.itervalues(_test_solver_cases):
            if sc.capabilities is None:
                sc.capabilities = set([])
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
            assert type(sc.capabilities) is set
            for tag in sc.capabilities:
                assert type(tag) is str

    if len(args) == 0:
        return _test_solver_cases.keys()
    return _test_solver_cases[args]

