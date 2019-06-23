#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
#
# Utility classes for working with the logger
#

import logging
from pyutilib.misc import LogHandler

from pyomo.common.fileutils import PYOMO_ROOT_DIR

#
# Set up the root Pyomo namespace logger
#
_logger = logging.getLogger('pyomo')
_logger.addHandler( LogHandler(
    PYOMO_ROOT_DIR, verbosity=lambda: _logger.isEnabledFor(logging.DEBUG) ))
_logger.setLevel(logging.WARNING)


class LoggingIntercept(object):
    """Context manager for intercepting messages sent to a log stream

    This class is designed to enable easy testing of log messages.

    The LoggingIntercept context manager will intercept messages sent to
    a log stream matching a specified level and send the messages to the
    specified output stream.  Other handlers registered to the target
    logger will be temporarily removed and the logger will be set not to
    propagate messages up to higher-level loggers.

    Args:
        output (FILE): the file stream to send log messages to
        module (str): the target logger name to intercept
        level (int): the logging level to intercept

    Examples:
        >>> import six, logging
        >>> from pyomo.common.log import LoggingInercept
        >>> buf = six.StringIO()
        >>> with LoggingIntercept(buf, 'pyomo.core', logging.WARNING):
        ...     logging.getLogger('pyomo.core').warn('a simple message')
        >>> buf.getvalue()
    """

    def __init__(self, output, module=None, level=logging.WARNING):
        self.handler = logging.StreamHandler(output)
        self.handler.setFormatter(logging.Formatter('%(message)s'))
        self.handler.setLevel(level)
        self.module = module
        self._save = None

    def __enter__(self):
        logger = logging.getLogger(self.module)
        self._save = logger.level, logger.propagate, logger.handlers
        logger.handlers = []
        logger.propagate = 0
        logger.setLevel(self.handler.level)
        logger.addHandler(self.handler)

    def __exit__(self, et, ev, tb):
        logger = logging.getLogger(self.module)
        logger.removeHandler(self.handler)
        logger.setLevel(self._save[0])
        logger.propagate = self._save[1]
        for h in self._save[2]:
            logger.handlers.append(h)
