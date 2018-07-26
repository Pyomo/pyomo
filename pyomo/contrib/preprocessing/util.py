import logging

from six import StringIO

from pyomo.common.log import LoggingIntercept


class SuppressConstantObjectiveWarning(LoggingIntercept):
    """Suppress the infeasible model warning message from solve().

    The "WARNING: Constant objective detected, replacing with a placeholder"
    warning message from calling solve() is often unwanted, but there is no
    clear way to suppress it.

    TODO need to fix this so that it only suppresses the desired message.

    """

    def __init__(self):
        super(SuppressConstantObjectiveWarning, self).__init__(
            StringIO(), 'pyomo.core', logging.WARNING)
