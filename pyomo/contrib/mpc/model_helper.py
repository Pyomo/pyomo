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

from pyomo.dae.flatten import flatten_dae_components
from pyomo.common.modeling import NOTSET
from pyomo.core.base.var import Var
from pyomo.core.base.expression import Expression
from pyomo.core.base.componentuid import ComponentUID
from pyomo.core.expr.numeric_expr import value as pyo_value

from pyomo.contrib.mpc.model_linker import copy_values_at_time
from pyomo.contrib.mpc.dynamic_data.series_data import TimeSeriesData
from pyomo.contrib.mpc.dynamic_data.scalar_data import ScalarData
from pyomo.contrib.mpc.cost_expressions import (
    get_tracking_cost_from_constant_setpoint,
)

iterable_scalars = (str, bytes)


def _to_iterable(item):
    if hasattr(item, "__iter__"):
        if isinstance(item, iterable_scalars):
            yield item
        else:
            for obj in item:
                yield obj
    else:
        yield item


class DynamicModelHelper(object):
    """
    Somewhat like a serializer, with the ability to generate and load
    different types of data from a dynamic model. Also somewhat like a
    wrapper around the flattened indexed variables, with the ability
    to shift values between different points in time.

    """

    def __init__(self, model, time, context=NOTSET):
        """
        Construct with a model and a set. We will flatten the model
        with respect to this set and generate CUIDs with wildcards.
        """
        scalar_vars, dae_vars = flatten_dae_components(model, time, Var)
        scalar_expr, dae_expr = flatten_dae_components(model, time, Expression)
        self.model = model
        self.time = time
        self.scalar_vars = scalar_vars
        self.dae_vars = dae_vars
        self.scalar_expr = scalar_expr
        self.dae_expr = dae_expr

        if context is NOTSET:
            context = model

        # Use buffer to reduce repeated work during name/cuid generation
        cuid_buffer = {}
        self.scalar_var_cuids = [
            ComponentUID(var, cuid_buffer=cuid_buffer, context=context)
            for var in self.scalar_vars
        ]
        self.dae_var_cuids = [
            ComponentUID(var.referent, cuid_buffer=cuid_buffer, context=context)
            for var in self.dae_vars
        ]

        self.dae_expr_cuids = [
            ComponentUID(expr.referent, cuid_buffer=cuid_buffer, context=context)
            for expr in self.dae_expr
        ]

    def get_scalar_variable_data(self):
        """
        Get data corresponding to non-time-indexed variables.

        Returns
        -------
        dict
            Maps CUIDs of non-time-indexed variables to the value of these
            variables.
        """
        return {
            cuid: var.value
            for cuid, var in zip(self.scalar_var_cuids, self.scalar_vars)
        }

    def get_data_at_time(self, time=None, include_expr=False):
        """
        Gets data at a single time point or set of time point. Note that
        the returned type changes depending on whether a scalar or iterable
        is supplied.
        """
        if time is None:
            # TODO: Default should be entire time set?
            # What about for steady models? Should this branch on length
            # of time?
            time = self.time if len(self.time) > 1 else self.time.at(1)
        try:
            # Assume time is iterable
            time_list = list(time)
            data = {
                cuid: [var[t].value for t in time]
                for cuid, var in zip(self.dae_var_cuids, self.dae_vars)
            }
            if include_expr:
                data.update({
                    cuid: [pyo_value(expr[t]) for t in time]
                    for cuid, expr in zip(self.dae_expr_cuids, self.dae_expr)
                })
            # Return a TimeSeriesData object, as this is more convenient
            # for the calling code.
            return TimeSeriesData(data, time_list, time_set=self.time)
        except TypeError:
            # time is a scalar
            # Maybe checking if time is an instance of numeric_types would
            # be better.
            # Return a dict mapping CUIDs to values. Should I have a similar
            # class for "scalar data"?
            data = {
                cuid: var[time].value
                for cuid, var in zip(self.dae_var_cuids, self.dae_vars)
            }
            if include_expr:
                data.update({
                    cuid: pyo_value(expr[time])
                    for cuid, expr in zip(self.dae_expr_cuids, self.dae_expr)
                })
            return ScalarData(data)

    # TODO: A unified load_data method would be nice:
    # - Branch on type (ScalarData and SeriesData handled)
    # - Branch on key (find_component -> is_indexed)
    # - Branch on iterable value. Iterable => load at all time
    # This is something to do once I have tests.

    def load_scalar_data(self, data):
        """
        Expects a dict mapping CUIDs (or strings) to values. Keys can
        correspond to time-indexed or non-time-indexed variables.
        """
        for cuid, val in data.items():
            var = self.model.find_component(cuid)
            var_iter = (var,) if not var.is_indexed() else var.values()
            for var in var_iter:
                var.set_value(val)

    def load_data_at_time(self, data, time_points=None):
        """
        Expects a dict mapping CUIDs to values, except this time
        we assume that the variables are indexed. Should this be
        combined with the above method (which could then check
        if the variable is indexed).
        """
        if time_points is None:
            time_points = self.time
        else:
            time_points = list(_to_iterable(time_points))
        if isinstance(data, ScalarData):
            data = data.get_data()
        else:
            # This processes keys in the incoming data dictionary
            # so they don't necessarily have to be CUIDs.
            data = ScalarData(data, time_set=self.time).get_data()
        for cuid, val in data.items():
            var = self.model.find_component(cuid)
            for t in time_points:
                var[t].set_value(val)

    def copy_values_at_time(self, source_time=None, target_time=None):
        """
        Copy values of all time-indexed variables from source time point
        to target time points.

        Parameters
        ----------
        source_time: Float
            Time point from which to copy values.
        target_time: Float or iterable
            Time point or points to which to copy values.

        """
        if source_time is None:
            source_time = self.time.first()
        if target_time is None:
            target_time = self.time
        copy_values_at_time(
            self.dae_vars,
            self.dae_vars,
            source_time,
            target_time,
        )

    def shift_values_by_time(self, dt):
        """
        Shift values in time indexed variables by a specified time offset.
        """
        seen = set()
        t0 = self.time.first()
        tf = self.time.last()
        time_map = {}
        for var in self.dae_vars:
            if id(var[tf]) in seen:
                # Assume that if var[tf] has been encountered, this is a
                # reference to a "variable" we have already processed.
                continue
            else:
                seen.add(id(var[tf]))
            new_values = []
            for t in self.time:
                if t not in time_map:
                    # Build up a map from target to source time points,
                    # as I don't want to call find_nearest_index
                    # more frequently than I have to.
                    t_new = t + dt
                    idx = self.time.find_nearest_index(t_new, tolerance=None)
                    # TODO: What if t_new is not a valid time point?
                    # Right now we just proceed with the closest valid time
                    # point. We're relying on the fact that indices of t0 or
                    # tf are returned if t_new is outside the bounds of the
                    # time set.
                    t_new = self.time.at(idx)
                    time_map[t] = t_new
                t_new = time_map[t]
                new_values.append(var[t_new].value)
            for i, t in enumerate(self.time):
                var[t].set_value(new_values[i])

    def get_tracking_cost_from_constant_setpoint(
        self, setpoint_data, time=None, variables=None, weight_data=None
    ):
        if not isinstance(setpoint_data, ScalarData):
            setpoint_data = ScalarData(setpoint_data)
        if time is None:
            time = self.time
        if variables is None:
            # Use variables provided by the setpoint.
            # NOTE: Nondeterministic order in Python < 3.7
            variables = [
                self.model.find_component(key)
                for key in setpoint_data.get_data().keys()
            ]
        else:
            # Variables were provided. These could be anything. Process them
            # to get time-indexed variables on the model.
            variables = [
                self.model.find_component(
                    get_time_indexed_cuid(var, (self.time,))
                ) for var in variables
            ]
        return get_tracking_cost_from_constant_setpoint(
            variables, time, setpoint_data, weight_data=weight_data
        )
