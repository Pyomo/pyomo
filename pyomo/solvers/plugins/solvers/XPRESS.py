from pyomo.opt.base import OptSolver
from pyomo.opt.base.solvers import SolverFactory
import logging


logger = logging.getLogger('pyomo.solvers')


@SolverFactory.register('xpress', doc='The XPRESS LP/MIP solver')
class XPRESS(OptSolver):
    """The XPRESS LP/MIP solver
    """

    def __new__(cls, *args, **kwds):
        mode = kwds.pop('solver_io', 'python')
        if mode is None:
            mode = 'python'

        if mode not in {'python', 'direct', 'persistent'}:
            logger.error('Pyomo currently only supports a Python interface to XPRESS. '
                         'Please use one of python, direct, or persistent for solver_io.')
            return
        if mode in ['python', 'direct']:
            opt = SolverFactory('xpress_direct', **kwds)
        elif mode == 'persistent':
            opt = SolverFactory('xpress_persistent', **kwds)
        if opt is None:
            logger.error('Python API for XPRESS is not installed')
        return opt
