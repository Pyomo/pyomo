from pyomo.core.expr.numvalue import value as pyo_value


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


def copy_values_at_time(
        source_vars,
        target_vars,
        source_time_points,
        target_time_points,
        ):
    # Process input arguments to wrap scalars in a list
    source_time_points = list(_to_iterable(source_time_points))
    target_time_points = list(_to_iterable(target_time_points))
    if (len(source_time_points) != len(target_time_points)
            and len(source_time_points) != 1):
        raise ValueError(
            "copy_values_at_time can only copy values when lists of time\n"
            "points have the same length or the source list has length one."
        )
    n_points = len(target_time_points)
    if len(source_time_points) == 1:
        source_time_points = source_time_points * n_points
    for s_var, t_var in zip(source_vars, target_vars):
        for s_t, t_t in zip(source_time_points, target_time_points):
            # Using the value function allows expressions to substitute
            # for variables. However, it raises an error if the expression
            # cannot be evaluated (e.g. has value None).
            #t_var[t_t].set_value(pyo_value(s_var[s_t]))
            t_var[t_t].set_value(s_var[s_t].value)


class DynamicVarLinker(object):
    """
    The purpose of this class is so that we do not have
    to call find_component or construct ComponentUIDs in a loop
    when transferring values between two different dynamic models.
    It also allows us to transfer values between variables that
    have different names in different models.

    """

    def __init__(self, 
            source_variables,
            target_variables,
            source_time=None,
            target_time=None,
            ):
        # Right now all the transfers I can think of only happen
        # in one direction
        if len(source_variables) != len(target_variables):
            raise ValueError(
                "%s must be provided two lists of time-indexed variables "
                "of equal length.\nGot lengths %s and %s"
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
            self._source_variables,
            self._target_variables,
            t_source,
            t_target,
        )
