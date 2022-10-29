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

from pyomo.contrib.mpc.interfaces.load_data import (
    load_data_from_scalar,
    load_data_from_series,
    load_data_from_interval,
)
from pyomo.contrib.mpc.interfaces.copy_values import copy_values_at_time
from pyomo.contrib.mpc.data.find_nearest_index import find_nearest_index
from pyomo.contrib.mpc.data.get_cuid import get_indexed_cuid
from pyomo.contrib.mpc.data.dynamic_data_base import _is_iterable
from pyomo.contrib.mpc.data.series_data import TimeSeriesData
from pyomo.contrib.mpc.data.interval_data import IntervalData
from pyomo.contrib.mpc.data.scalar_data import ScalarData
from pyomo.contrib.mpc.data.convert import _process_to_dynamic_data
from pyomo.contrib.mpc.modeling.cost_expressions import (
    get_penalty_from_constant_target,
    get_penalty_from_target,
)
from pyomo.contrib.mpc.modeling.constraints import (
    get_piecewise_constant_constraints,
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


class DynamicModelInterface(object):
    """A helper class for working with dynamic models, e.g. those where
    many components are indexed by some ordered set referred to as "time."

    This class provides methods for interacting with time-indexed
    components, for instance, loading and extracting data or shifting
    values by some time offset. It also provides methods for constructing
    components useful for dynamic optimization.

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
        self._scalar_vars = scalar_vars
        self._dae_vars = dae_vars
        self._scalar_expr = scalar_expr
        self._dae_expr = dae_expr

        if context is NOTSET:
            context = model

        # Use buffer to reduce repeated work during name/cuid generation
        cuid_buffer = {}
        self._scalar_var_cuids = [
            ComponentUID(var, cuid_buffer=cuid_buffer, context=context)
            for var in self._scalar_vars
        ]
        self._dae_var_cuids = [
            ComponentUID(var.referent, cuid_buffer=cuid_buffer, context=context)
            for var in self._dae_vars
        ]
        self._dae_expr_cuids = [
            ComponentUID(expr.referent, cuid_buffer=cuid_buffer, context=context)
            for expr in self._dae_expr
        ]

    def get_scalar_variables(self):
        return self._scalar_vars

    def get_indexed_variables(self):
        return self._dae_vars

    def get_scalar_expressions(self):
        return self._scalar_expr

    def get_indexed_expressions(self):
        return self._dae_expr

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
            for cuid, var in zip(self._scalar_var_cuids, self._scalar_vars)
        }

    def get_data_at_time(self, time=None, include_expr=False):
        """
        Gets data at a single time point or set of time point. Note that
        the returned type changes depending on whether a scalar or iterable
        is supplied.

        """
        if time is None:
            # Default is to use the entire time set, treating a singleton
            # as a scalar.
            time = self.time if len(self.time) > 1 else self.time.at(1)
        if _is_iterable(time):
            # Assume time is iterable
            time_list = list(time)
            data = {
                cuid: [var[t].value for t in time]
                for cuid, var in zip(self._dae_var_cuids, self._dae_vars)
            }
            if include_expr:
                data.update({
                    cuid: [pyo_value(expr[t]) for t in time]
                    for cuid, expr in zip(self._dae_expr_cuids, self._dae_expr)
                })
            # Return a TimeSeriesData object
            return TimeSeriesData(data, time_list, time_set=self.time)
        else:
            # time is a scalar
            data = {
                cuid: var[time].value
                for cuid, var in zip(self._dae_var_cuids, self._dae_vars)
            }
            if include_expr:
                data.update({
                    cuid: pyo_value(expr[time])
                    for cuid, expr in zip(self._dae_expr_cuids, self._dae_expr)
                })
            # Return ScalarData object
            return ScalarData(data)

    def load_data(
        self,
        data,
        time_points=None,
        tolerance=0.0,
        prefer_left=None,
        exclude_left_endpoint=None,
        exclude_right_endpoint=None,
    ):
        """Method to load data into the model.

        Loads data into indicated variables in the model, possibly
        at specified time points.

        Arguments
        ---------
        data: ScalarData, TimeSeriesData, or mapping
            If ScalarData, loads values into indicated variables at
            all (or specified) time points. If TimeSeriesData, loads
            lists of values into time points.
            If mapping, checks whether each variable and value is
            indexed or iterable and correspondingly loads data into
            variables.
        time_points: Iterable (optional)
            Subset of time points into which data should be loaded.
            Default of None corresponds to loading into all time points.

        """
        if time_points is None:
            time_points = self.time
        if isinstance(data, IntervalData):
            # Set default arguments to load from interval
            if prefer_left is None:
                prefer_left = True
            if exclude_left_endpoint is None:
                exclude_left_endpoint = prefer_left
            if exclude_right_endpoint is None:
                exclude_right_endpoint = not prefer_left
            load_data_from_interval(
                data,
                self.model,
                time_points,
                tolerance=tolerance,
                prefer_left=prefer_left,
                exclude_left_endpoint=exclude_left_endpoint,
                exclude_right_endpoint=exclude_right_endpoint,
            )
            return

        # Make sure these arguments are not set for non-interval data.
        if prefer_left is not None:
            raise RuntimeError(
                "Invalid argument prefer_left with data type %s"
                % IntervalData
            )
        if exclude_left_endpoint is not None:
            raise RuntimeError(
                "Invalid argument exclude_left_endpoint with data type %s"
                % IntervalData
            )
        if exclude_right_endpoint is not None:
            raise RuntimeError(
                "Invalid argument exclude_right_endpoint with data type %s"
                % IntervalData
            )

        if isinstance(data, ScalarData):
            load_data_from_scalar(data, self.model, time_points)
        elif isinstance(data, TimeSeriesData):
            load_data_from_series(data, self.model, time_points, tolerance=tolerance)
        else:
            # Attempt to load data by assuming it is a map from something
            # find_component-compatible to values.
            for cuid, vals in data.items():
                var = self.model.find_component(cuid)
                if var.is_indexed():
                    # Assume we are indexed by time.
                    if not _is_iterable(vals):
                        # Load value into all time points
                        for t in time_points:
                            var[t].set_value(vals)
                    else:
                        # Load values into corresponding time points
                        if len(time_points) != len(vals):
                            raise RuntimeError(
                                "Cannot load a different number of values"
                                " than we have time points"
                            )
                        for i, t in enumerate(time_points):
                            var[t].set_value(vals[i])
                else:
                    # Assume vals is a scalar
                    var.set_value(vals)

    def load_scalar_data(self, data):
        """
        Expects a dict mapping CUIDs (or strings) to values. Keys can
        correspond to time-indexed or non-time-indexed variables.
        """
        #for cuid, val in data.items():
        #    var = self.model.find_component(cuid)
        #    var_iter = (var,) if not var.is_indexed() else var.values()
        #    for var in var_iter:
        #        var.set_value(val)
        self.load_data(data)

    def load_data_at_time(self, data, time_points=None):
        """
        Expects a dict mapping CUIDs to values, except this time
        we assume that the variables are indexed. Should this be
        combined with the above method (which could then check
        if the variable is indexed).
        """
        #if time_points is None:
        #    time_points = self.time
        #else:
        #    time_points = list(_to_iterable(time_points))
        #if isinstance(data, ScalarData):
        #    data = data.get_data()
        #else:
        #    # This processes keys in the incoming data dictionary
        #    # so they don't necessarily have to be CUIDs.
        #    data = ScalarData(data, time_set=self.time).get_data()
        #for cuid, val in data.items():
        #    var = self.model.find_component(cuid)
        #    for t in time_points:
        #        var[t].set_value(val)
        self.load_data(data, time_points=time_points)

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
            self._dae_vars,
            self._dae_vars,
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
        time_list = list(self.time)
        for var in self._dae_vars:
            if id(var[tf]) in seen:
                # Assume that if var[tf] has been encountered, this is a
                # reference to a "variable" we have already processed.
                continue
            else:
                seen.add(id(var[tf]))
            new_values = []
            for t in time_list:
                if t not in time_map:
                    # Build up a map from target to source time points,
                    # as I don't want to call find_nearest_index more
                    # frequently than I have to.
                    t_new = t + dt
                    idx = find_nearest_index(time_list, t_new, tolerance=None)
                    # If t_new is not a valid time point, we proceed with the
                    # closest valid time point.
                    # We're relying on the fact that indices of t0 or tf are
                    # returned if t_new is outside the bounds of the time set.
                    t_new = time_list[idx]
                    time_map[t] = t_new
                t_new = time_map[t]
                new_values.append(var[t_new].value)
            for i, t in enumerate(self.time):
                var[t].set_value(new_values[i])

    def get_penalty_from_target(
        self,
        target_data,
        time=None,
        variables=None,
        weight_data=None,
        variable_set=None,
        tolerance=None,
        prefer_left=None,
    ):
        """A method to get a quadratic penalty expression from a provided
        setpoint data structure

        Parameters
        ----------
        target_data: ScalarData, TimeSeriesData, or IntervalData
            Holds target values for variables
        time: Set (optional)
            Points at which to apply the tracking cost. Default will use
            the model's time set.
        variables: List of Pyomo VarData (optional)
            Subset of variables supplied in setpoint_data to use in the
            tracking cost. Default is to use all variables supplied.
        weight_data: ScalarData (optional)
            Holds the weights to use in the tracking cost for each variable
        variable_set: Set (optional)
            A set indexing the list of provided variables, if one already
            exists. 
        tolerance: Float (optional)
            Tolerance for checking inclusion in an interval. Only may be
            provided if IntervalData is provided for target_data. In this
            case the default is 0.0.
        prefer_left: Bool (optional)
            Flag indicating whether the left end point of intervals should
            be preferred over the right end point. Only may be provided if
            IntervalData is provided for target_data. In this case the
            default is False.

        Returns
        -------
        Expression
            Expression indexed by indices into the the list of variables and
            time containing the weighted tracking cost for each variable
            at each point in time

        """
        if time is None:
            time = self.time
        target_data = _process_to_dynamic_data(target_data)
        if variables is None:
            # Use variables provided by the setpoint.
            # NOTE: Nondeterministic order in non-C Python < 3.7
            # Should these data structures use OrderedDicts internally
            # to enforce an order here?
            variables = [
                self.model.find_component(key)
                for key in target_data.get_data().keys()
            ]
        else:
            # Variables were provided. These could be anything. Process them
            # to get time-indexed variables on the model.
            variables = [
                self.model.find_component(get_indexed_cuid(var, (self.time,)))
                for var in variables
            ]
        return get_penalty_from_target(
            variables,
            time,
            target_data,
            weight_data=weight_data,
            variable_set=variable_set,
            tolerance=tolerance,
            prefer_left=prefer_left,
        )

    def get_piecewise_constant_constraints(
        self, variables, sample_points, use_next=True, tolerance=0.0
    ):
        """A method to get an indexed constraint ensuring that inputs
        are piecewise constant.

        Parameters
        ----------
        variables: List of Pyomo Vars
            Variables to enforce piecewise constant
        sample_points: List of floats
            Points marking the boundaries of intervals within which
            variables must be constant
        use_next: Bool (optional)
            Whether to enforce constancy by setting each variable equal
            to itself at the next point in time (as opposed to at the
            previous point in time). Default is True.
        tolerance: Float (optional)
            Absolute tolerance used to determine whether provided sample
            points are in the model's time set.

        Returns
        -------
        Tuple:
            First entry is a Set indexing the list of provided variables
            (with integers). Second entry is a constraint indexed by this
            set and time enforcing the piecewise constant condition via
            equality constraints.

        """
        cuids = [
            get_indexed_cuid(var, (self.time,))
            for var in variables
        ]
        variables = [self.model.find_component(cuid) for cuid in cuids]
        time_list = list(self.time)
        # Make sure that sample points exist (within tolerance) in the time
        # set.
        sample_point_indices = [
            find_nearest_index(time_list, t, tolerance=tolerance)
            for t in sample_points
        ]
        sample_points = [time_list[i] for i in sample_point_indices]
        return get_piecewise_constant_constraints(
            variables,
            self.time,
            sample_points,
            use_next=use_next,
        )
