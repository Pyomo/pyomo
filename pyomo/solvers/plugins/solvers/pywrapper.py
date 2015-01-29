#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________


import logging

import pyomo.util.plugin
from pyomo.opt import OptSolver, SolverFactory

logger = logging.getLogger('pyomo.solvers')

class pywrapper(OptSolver):
    """Direct python solver interface
    """

    pyomo.util.plugin.alias('py', doc='Direct python solver interfaces')

    def __new__(cls, *args, **kwds):
        mode = kwds.get('solver_io', 'python')
        if mode is None:
            mode = 'python'
        if mode != 'python':
            logging.getLogger('pyomo.solvers').error("Cannot specify IO mode '%s' for direct python solver interface" % mode)
            return None
        #
        if not 'solver' in kwds:
            logging.getLogger('pyomo.solvers').warning("No solver specified for direct python solver interface")
            return None
        kwds['solver_io'] = 'python'
        solver = kwds['solver']
        del kwds['solver']
        return SolverFactory(solver, **kwds)

