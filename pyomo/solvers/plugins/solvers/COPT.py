#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import logging

from pyomo.opt.base import OptSolver
from pyomo.opt.base.solvers import SolverFactory

logger = logging.getLogger('pyomo.solvers')


@SolverFactory.register('copt', doc='The COPT solver')
class COPT(OptSolver):
    """
    The COPT LP/MIP solver
    """

    def __new__(cls, *args, **kwds):
        mode = kwds.pop('solver_io', 'python')
        if mode is None:
            mode = 'python'
        if mode == 'lp' or mode == 'mps':
            # COPT provides command line tool 'copt_cmd' but not integrated into Pyomo
            logger.error('LP/MPS mode not supported by Pyomo interface of COPT')
            return
        if mode in ['python', 'direct']:
            opt = SolverFactory('copt_direct', **kwds)
            if opt is None:
                logger.error('Python API for COPT is not installed')
                return
            return opt
        if mode == 'persistent':
            opt = SolverFactory('copt_persistent', **kwds)
            if opt is None:
                logger.error('Python API for COPT is not installed')
                return
            return opt

        if mode == 'os':
            opt = SolverFactory('_ossolver', **kwds)
        elif mode == 'nl':
            opt = SolverFactory('asl', **kwds)
        else:
            logger.error('Unknown IO type: %s' % mode)
            return
        opt.set_options('solver=coptampl')
        return opt
