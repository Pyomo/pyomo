#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________


import logging

from pyomo.opt import OptSolver, SolverFactory

logger = logging.getLogger('pyomo.solvers')


@SolverFactory.register('py', doc='Direct python solver interfaces')
class pywrapper(OptSolver):
    """Direct python solver interface
    """

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

