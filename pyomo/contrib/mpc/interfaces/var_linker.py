#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.contrib.mpc.interfaces.copy_values import copy_values_at_time


class DynamicVarLinker(object):
    """
    The purpose of this class is so that we do not have
    to call find_component or construct ComponentUIDs in a loop
    when transferring values between two different dynamic models.
    It also allows us to transfer values between variables that
    have different names in different models.

    """

    def __init__(
        self, source_variables, target_variables, source_time=None, target_time=None
    ):
        # Right now all the transfers I can think of only happen
        # in one direction
        if len(source_variables) != len(target_variables):
            raise ValueError(
                "%s must be provided two lists of time-indexed variables "
                "of equal length. Got lengths %s and %s"
                % (type(self), len(source_variables), len(target_variables))
            )
        self._source_variables = source_variables
        self._target_variables = target_variables
        self._source_time = source_time
        self._target_time = target_time

    def transfer(self, t_source=None, t_target=None):
        if t_source is None and self._source_time is None:
            raise RuntimeError(
                "Source time points were not provided in the transfer method "
                "or in the constructor."
            )
        elif t_source is None:
            t_source = self._source_time
        if t_target is None and self._target_time is None:
            raise RuntimeError(
                "Target time points were not provided in the transfer method "
                "or in the constructor."
            )
        elif t_target is None:
            t_target = self._target_time
        copy_values_at_time(
            self._source_variables, self._target_variables, t_source, t_target
        )
