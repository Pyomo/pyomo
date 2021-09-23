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

from pyomo.opt.base import SolverFactory, OptSolver
from pyomo.core import Var, value
from pyomo.contrib.trustregion.TRF import TRF
from pyomo.contrib.trustregion_new.config_options import get_TRF_config

logger = logging.getLogger('pyomo.contrib.trustregion')

__version__ = (0, 2, 0)

@SolverFactory.register(
        'trustregion',
        doc='Trust region filter method for black box/glass box optimization.'
    )
class TrustRegionSolver(OptSolver):
    """
    A trust region filter method for black box / glass box optimization
    Solves nonlinear optimization problems containing external function calls
    through automatic construction of reduced models (RM), also known as
    surrogate models.

    Adapted from Yoshio, Biegler (2020) AIChE Journal.

    """

    CONFIG = get_TRF_config()

    def __init__(self, **kwds):
        # set persistent config options
        tmp_kwds = {'type':kwds.pop('type', 'trustregion')}
        self.config = self.CONFIG(kwds, preserve_implicit=True)
        #
        # Call base class constructor
        #
        tmp_kwds['solver'] = self.config.solver
        OptSolver.__init__(self, **tmp_kwds)

    def available(self, exception_flag=True):
        """Check if solver is available.
        """
        return True

    def version(self):
        """Return a 3-tuple describing the solver version."""
        return __version__

    def license_is_valid(self):
        """License for using TRF"""
        return True

    # The Pyomo solver API expects that solvers support the context
    # manager API
    def __enter__(self):
        return self

    def __exit__(self, et, ev, tb):
        pass

    def solve(self, model, eflist, **kwds):
        # set customized config parameters
        self._local_config = self.config(kwds, preserve_implicit=True)

        # first store all data we will need to change in original model as a tuple
        # [0]=Var component, [1]=external function list, [2]=config block
        model._tmp_trf_data = (list(model.component_data_objects(Var)), eflist, self._local_config)
        # now clone the model
        inst = model.clone()

        # call TRF on cloned model
        TRF(inst, inst._tmp_trf_data[1], inst._tmp_trf_data[2])

        # copy potentially changed variable values back to original model and return
        for inst_var, orig_var in zip(inst._tmp_trf_data[0], model._tmp_trf_data[0]):
            orig_var.set_value(value(inst_var))
