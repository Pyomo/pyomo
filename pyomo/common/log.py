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
#
#  This module was originally developed as part of the PyUtilib project
#  Copyright (c) 2008 Sandia Corporation.
#  This software is distributed under the BSD License.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  ___________________________________________________________________________
#
# Utility classes for working with the logger
#
import inspect
import io
import logging
import re
import sys
import textwrap

from pyomo.version.info import releaselevel
from pyomo.common.deprecation import deprecated
from pyomo.common.fileutils import PYOMO_ROOT_DIR
from pyomo.common.formatting import wrap_reStructuredText

_indentation_re = re.compile(r'\s*')

_RTD_URL = "https://pyomo.readthedocs.io/en/%s/errors.html" % (
    'stable'
    if (releaselevel == 'final' or 'sphinx' in sys.modules or 'Sphinx' in sys.modules)
    else 'latest'
)


def RTD(_id):
    _id = str(_id).lower()
    assert _id[0] in 'wex'
    return f"{_RTD_URL}#{_id}"


_DEBUG = logging.DEBUG
_NOTSET = logging.NOTSET
if not __debug__:

    def is_debug_set(logger):
        return False

elif hasattr(getattr(logging.getLogger(), 'manager', None), 'disable'):
    # This works for CPython and PyPy, but relies on a manager attribute
    # to get the current value of the logging.disabled() flag
    # (technically not included in the official logging documentation)
    def is_debug_set(logger):
        """A variant of Logger.isEnableFor that returns False if NOTSET

        The implementation of logging.Logger.isEnableFor() returns True
        if the effective level of the logger is NOTSET.  This variant
        only returns True if the effective level of the logger is NOTSET
        < level <= DEBUG.  This is used in Pyomo to detect if the user
        explicitly requested DEBUG output.

        This implementation mimics the core functionality of
        isEnabledFor() by directly querying the (undocumented) 'manager'
        attribute to get the current value for logging.disabled()

        """
        if logger.manager.disable >= _DEBUG:
            return False
        _level = logger.getEffectiveLevel()
        # Filter out NOTSET and higher levels
        return _NOTSET < _level <= _DEBUG

else:
    # This is inefficient (it indirectly checks effective level twice),
    # but is included for [as yet unknown] platforms that ONLY implement
    # the API documented in the logging library
    def is_debug_set(logger):
        if not logger.isEnabledFor(_DEBUG):
            return False
        return logger.getEffectiveLevel() > _NOTSET


class WrappingFormatter(logging.Formatter):
    _flag = "<<!MSG!>>"

    def __init__(self, **kwds):
        if 'fmt' not in kwds:
            if kwds.get('style', '%') == '%':
                kwds['fmt'] = '%(levelname)s: %(message)s'
            elif kwds['style'] == '{':
                kwds['fmt'] = '{levelname}: {message}'
            elif kwds['style'] == '$':
                kwds['fmt'] = '$levelname: $message'
            else:
                raise ValueError('unrecognized style flag "%s"' % (kwds['style'],))
        self._wrapper = textwrap.TextWrapper(width=kwds.pop('wrap', 78))
        self._wrapper.subsequent_indent = kwds.pop('hang', ' ' * 4)
        if not self._wrapper.subsequent_indent:
            self._wrapper.subsequent_indent = ''
        self.basepath = kwds.pop('base', None)
        super(WrappingFormatter, self).__init__(**kwds)

    def format(self, record):
        msg = record.getMessage()
        if record.msg.__class__ is not str and isinstance(record.msg, Preformatted):
            return msg

        _orig = {
            k: getattr(record, k) for k in ('msg', 'args', 'pathname', 'levelname')
        }
        _id = getattr(record, 'id', None)
        record.msg = self._flag
        record.args = None
        if _id:
            record.levelname += f" ({_id.upper()})"
        if self.basepath and record.pathname.startswith(self.basepath):
            record.pathname = '[base]' + record.pathname[len(self.basepath) :]
        try:
            raw_msg = super(WrappingFormatter, self).format(record)
        finally:
            for k, v in _orig.items():
                setattr(record, k, v)

        # We want to normalize the incoming message *before* we start
        # formatting (wrapping) paragraphs.
        #
        # Most of the messages are either unformatted long lines or
        # triple-quote blocks of text.  In the latter case, if the text
        # starts on the same line as the triple-quote, then it is almost
        # certainly NOT indented with the bulk of the text, which will
        # cause dedent to get confused and not strip any leading
        # whitespace.
        #
        # A standard approach is to use inspect.cleandoc, which
        # allows for the first line to have 0 indent.
        msg = inspect.cleandoc(msg)

        # Split the formatted log message (that currently has _flag in
        # lieu of the actual message content) into lines, then
        # recombine, substituting and wrapping any lines that contain
        # _flag.
        return '\n'.join(
            self._wrap_msg(line, msg, _id) if self._flag in line else line
            for line in raw_msg.splitlines()
        )

    def _wrap_msg(self, format_line, msg, _id):
        _init = self._wrapper.initial_indent, self._wrapper.subsequent_indent
        # We will honor the "hang" argument (for specifying a hanging
        # indent) unless the formatting line was indented (e.g. because
        # DEBUG was set), in which case we will use that for both the
        # first line and all subsequent lines.
        indent = _indentation_re.match(format_line).group()
        if indent:
            self._wrapper.initial_indent = self._wrapper.subsequent_indent = indent
        try:
            wrapped_msg = wrap_reStructuredText(
                format_line.strip().replace(self._flag, msg), self._wrapper
            )
        finally:
            # Restore the wrapper state
            self._wrapper.initial_indent, self._wrapper.subsequent_indent = _init
        if _id:
            wrapped_msg += f"\n{indent}{_init[1]}See also {RTD(_id)}"
        return wrapped_msg


class LegacyPyomoFormatter(logging.Formatter):
    """This mocks up the legacy Pyomo log formatting.

    This formatter takes a callback function (`verbosity`) that will be
    called for each message.  Based on the result, one of two formatting
    templates will be used.

    """

    def __init__(self, **kwds):
        if 'fmt' in kwds:
            raise ValueError("'fmt' is not a valid option for the LegacyFormatter")
        if 'style' in kwds:
            raise ValueError("'style' is not a valid option for the LegacyFormatter")

        self.verbosity = kwds.pop('verbosity', lambda: True)
        self.standard_formatter = WrappingFormatter(**kwds)
        self.verbose_formatter = WrappingFormatter(
            fmt='%(levelname)s: "%(pathname)s", %(lineno)d, %(funcName)s\n'
            '    %(message)s',
            hang=False,
            **kwds,
        )
        super(LegacyPyomoFormatter, self).__init__()

    def format(self, record):
        if self.verbosity():
            return self.verbose_formatter.format(record)
        else:
            return self.standard_formatter.format(record)


class StdoutHandler(logging.StreamHandler):
    """A logging handler that emits to the current value of sys.stdout"""

    def flush(self):
        self.stream = sys.stdout
        super(StdoutHandler, self).flush()

    def emit(self, record):
        self.stream = sys.stdout
        super(StdoutHandler, self).emit(record)


class Preformatted(object):
    __slots__ = ('msg',)

    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return str(self.msg)

    def __repr__(self):
        return f'Preformatted({self.msg!r})'


class _GlobalLogFilter(object):
    def __init__(self):
        self.logger = logging.getLogger()

    def filter(self, record):
        return not self.logger.handlers


# This mocks up the historical Pyomo logging system, which uses a
# different formatter based on if the main pyomo logger is enabled for
# debugging.  It has been updated to suppress output if any handlers
# have been defined at the root level.
pyomo_logger = logging.getLogger('pyomo')
pyomo_handler = StdoutHandler()
pyomo_formatter = LegacyPyomoFormatter(
    base=PYOMO_ROOT_DIR, verbosity=lambda: pyomo_logger.isEnabledFor(logging.DEBUG)
)
pyomo_handler.setFormatter(pyomo_formatter)
pyomo_handler.addFilter(_GlobalLogFilter())
pyomo_logger.addHandler(pyomo_handler)


@deprecated(
    'The pyomo.common.log.LogHandler class has been deprecated '
    'in favor of standard Handlers from the Python logging module '
    'combined with the pyomo.common.log.WrappingFormatter.',
    version='5.7.3',
)
class LogHandler(logging.StreamHandler):
    def __init__(self, base='', stream=None, level=logging.NOTSET, verbosity=None):
        super(LogHandler, self).__init__(stream)
        self.setLevel(level),
        if verbosity is None:
            verbosity = lambda: True
        self.setFormatter(LegacyPyomoFormatter(base=base, verbosity=verbosity))


class LoggingIntercept(object):
    """Context manager for intercepting messages sent to a log stream

    This class is designed to enable easy testing of log messages.

    The LoggingIntercept context manager will intercept messages sent to
    a log stream matching a specified level and send the messages to the
    specified output stream.  Other handlers registered to the target
    logger will be temporarily removed and the logger will be set not to
    propagate messages up to higher-level loggers.

    Parameters
    ----------
    output: io.TextIOBase
        the file stream to send log messages to
    module: str
        the target logger name to intercept
    level: int
        the logging level to intercept
    formatter: logging.Formatter
        the formatter to use when rendering the log messages.  If not
        specified, uses `'%(message)s'`

    Examples:
        >>> import io, logging
        >>> from pyomo.common.log import LoggingIntercept
        >>> buf = io.StringIO()
        >>> with LoggingIntercept(buf, 'pyomo.core', logging.WARNING):
        ...     logging.getLogger('pyomo.core').warning('a simple message')
        >>> buf.getvalue()

    """

    def __init__(self, output=None, module=None, level=logging.WARNING, formatter=None):
        self.handler = None
        self.output = output
        self.module = module
        self._level = level
        if formatter is None:
            formatter = logging.Formatter('%(message)s')
        self._formatter = formatter
        self._save = None

    def __enter__(self):
        # Set up the handler
        output = self.output
        if output is None:
            output = io.StringIO()
        assert self.handler is None
        self.handler = logging.StreamHandler(output)
        self.handler.setFormatter(self._formatter)
        self.handler.setLevel(self._level)
        # Register the handler with the appropriate module scope
        logger = logging.getLogger(self.module)
        self._save = logger.level, logger.propagate, logger.handlers
        logger.handlers = []
        logger.propagate = 0
        logger.setLevel(self.handler.level)
        logger.addHandler(self.handler)
        return output

    def __exit__(self, et, ev, tb):
        logger = logging.getLogger(self.module)
        logger.removeHandler(self.handler)
        self.handler = None
        logger.setLevel(self._save[0])
        logger.propagate = self._save[1]
        for h in self._save[2]:
            logger.handlers.append(h)


class LogStream(io.TextIOBase):
    """
    This class logs whatever gets sent to the write method.
    This is useful for logging solver output (a LogStream
    instance can be handed to TeeStream from pyomo.common.tee).
    """

    def __init__(self, level, logger):
        self._level = level
        self._logger = logger
        self._buffer = ''

    def write(self, s: str) -> int:
        res = len(s)
        if self._buffer:
            s = self._buffer + s
        lines = s.split('\n')
        for line in lines[:-1]:
            self._logger.log(self._level, line)
        self._buffer = lines[-1]
        return res

    def flush(self):
        if self._buffer:
            self.write('\n')
