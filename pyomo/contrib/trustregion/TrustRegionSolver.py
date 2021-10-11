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

from pyomo.opt.base import OptSolver
from pyomo.core import Var, value
from pyomo.contrib.trustregion.TrustRegionMethod import TrustRegionMethod
from pyomo.contrib.trustregion.config_options import get_TRF_config

logger = logging.getLogger('pyomo.contrib.trustregion')

__version__ = (0, 2, 0)


class TrustRegionSolver(OptSolver):
    """
    A trust region filter method for black box / glass box optimization
    Solves nonlinear optimization problems containing external function calls
    through automatic construction of reduced models (RM), also known as
    surrogate models.

    Adapted from Yoshio, Biegler (2020) AIChE Journal.
    """

    CONFIG = get_TRF_config()

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
        config = self.CONFIG(kwds.pop('options', {}))

        # first store all data we will need to change in original model as a tuple
        # [0]=Var component, [1]=external function list, [2]=config block
        model._tmp_TRF_data = (list(model.component_data_objects(Var)), eflist, config)
        # now clone the model
        inst = model.clone()

        # call TRF on cloned model
        TrustRegionMethod(inst, inst._tmp_TRF_data[1], inst._tmp_TRF_data[2])

        # copy potentially changed variable values back to original model and return
        for inst_var, orig_var in zip(inst._tmp_TRF_data[0], model._tmp_TRF_data[0]):
            orig_var.set_value(value(inst_var))
