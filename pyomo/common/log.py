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
import os
import re
import sys
import textwrap

from pyomo.version.info import releaselevel
from pyomo.common.deprecation import deprecated
from pyomo.common.fileutils import PYOMO_ROOT_DIR
from pyomo.common.flags import in_testing_environment, building_documentation
from pyomo.common.formatting import wrap_reStructuredText

_indentation_re = re.compile(r'\s*')

_RTD_URL = "https://pyomo.readthedocs.io/en/%s/errors.html"


def RTD(_id):
    _id = str(_id).lower()
    _release = (
        'stable' if releaselevel == 'final' or in_testing_environment() else 'latest'
    )
    assert _id[0] in 'wex'
    return (_RTD_URL % (_release,)) + f"#{_id}"


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
        # Filter out NOTSET and higher levels
        return _NOTSET < logger.getEffectiveLevel() <= _DEBUG

elif sys.version_info[:3] < (3, 13, 4):
    # This is inefficient (it indirectly checks effective level twice),
    # but is included for [as yet unknown] platforms that ONLY implement
    # the API documented in the logging library
    def is_debug_set(logger):
        if not logger.isEnabledFor(_DEBUG):
            return False
        return logger.getEffectiveLevel() > _NOTSET

else:
    # Python 3.14 (and backported to python 3.13.4) changed the behavior
    # of isEnabledFor() so that it always returns False when called
    # while a log record is in flight (learned this from
    # https://github.com/hynek/structlog/pull/723).  In newer versions
    # of Python, we will only rely on getEffectiveLevel().
    def is_debug_set(logger):
        return _NOTSET < logger.getEffectiveLevel() <= _DEBUG


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
        if getattr(record, 'cleandoc', True):
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

    def __init__(self):
        super().__init__()
        self.stream = None

    def flush(self):
        try:
            orig = self.stream
            self.stream = sys.stdout
            super(StdoutHandler, self).flush()
        finally:
            self.stream = orig

    def emit(self, record):
        try:
            orig = self.stream
            self.stream = sys.stdout
            super(StdoutHandler, self).emit(record)
        finally:
            self.stream = orig


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
        # We will not emit messages using the default Pyomo log handler
        # if someone has registered a global handler.  However, we will
        # ignore this if we are building documentation
        # (sphinx.ext.doctest adds a handler, but we want to ignore that
        # handler when we are testing our documentation!)
        return not self.logger.handlers or building_documentation()


# This mocks up the historical Pyomo logging system, which uses a
# different formatter based on if the main pyomo logger is enabled for
# debugging.  It has been updated to suppress output if any handlers
# have been defined at the root level.
pyomo_logger = logging.getLogger('pyomo')
pyomo_handler = logging.StreamHandler(sys.stdout)
pyomo_formatter = LegacyPyomoFormatter(
    base=PYOMO_ROOT_DIR, verbosity=lambda: is_debug_set(pyomo_logger)
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
    r"""Context manager for intercepting messages sent to a log stream

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
        the target logger name to intercept. `logger` and `module` are
        mutually exclusive.

    level: int
        the logging level to intercept

    formatter: logging.Formatter
        the formatter to use when rendering the log messages.  If not
        specified, uses `'%(message)s'`

    logger: logging.Logger
        the target logger to intercept. `logger` and `module` are
        mutually exclusive.

    Examples
    --------
    >>> import io, logging
    >>> from pyomo.common.log import LoggingIntercept
    >>> buf = io.StringIO()
    >>> with LoggingIntercept(buf, 'pyomo.core', logging.WARNING):
    ...     logging.getLogger('pyomo.core').warning('a simple message')
    >>> buf.getvalue()
    'a simple message\n'

    """

    def __init__(
        self,
        output=None,
        module=None,
        level=logging.WARNING,
        formatter=None,
        logger=None,
    ):
        self.handler = None
        self.output = output
        if logger is not None:
            if module is not None:
                raise ValueError(
                    "LoggingIntercept: only one of 'module' and 'logger' is allowed"
                )
            self._logger = logger
        else:
            self._logger = logging.getLogger(module)
        self._level = level
        if formatter is None:
            formatter = logging.Formatter('%(message)s')
        self._formatter = formatter
        self._save = None

    def __enter__(self):
        # Get the logger for the scope we will be overriding
        logger = self._logger
        self._save = logger.level, logger.propagate, logger.handlers
        if self._level is None:
            self._level = logger.getEffectiveLevel()
        # Set up the handler
        output = self.output
        if output is None:
            output = io.StringIO()
        assert self.handler is None
        self.handler = logging.StreamHandler(output)
        self.handler.setFormatter(self._formatter)
        self.handler.setLevel(self._level)
        # Register the handler with the appropriate module scope
        logger.handlers = []
        logger.propagate = False
        logger.setLevel(self.handler.level)
        logger.addHandler(self.handler)
        return output

    def __exit__(self, et, ev, tb):
        logger = self._logger
        logger.removeHandler(self.handler)
        self.handler = None
        logger.setLevel(self._save[0])
        logger.propagate = self._save[1]
        assert not logger.handlers
        logger.handlers.extend(self._save[2])

    @property
    def module(self):
        return self._logger.name


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
        if not s:
            return 0
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

    def redirect_streams(self, redirects):
        """Redirect StreamHandler objects to the original file descriptors

        This utility method for py:class:`~pyomo.common.tee.capture_output`
        locates any StreamHandlers that would process messages from the
        logger assigned to this :py:class:`LogStream` that would write
        to the file descriptors redirected by `capture_output` and
        yields context managers that will redirect those StreamHandlers
        back to duplicates of the original file descriptors.

        """
        found = 0
        logger = self._logger
        while isinstance(logger, logging.LoggerAdapter):
            logger = logger.logger
        while logger:
            for handler in logger.handlers:
                found += 1
                if not isinstance(handler, logging.StreamHandler):
                    continue
                try:
                    fd = handler.stream.fileno()
                except (AttributeError, OSError):
                    fd = None
                if fd not in redirects:
                    continue
                yield _StreamRedirector(handler, redirects[fd].original_fd)
            if not logger.propagate:
                break
            else:
                logger = logger.parent
        if not found:
            fd = logging.lastResort.stream.fileno()
            if not redirects:
                yield _LastResortRedirector(fd)
            elif fd in redirects:
                yield _LastResortRedirector(redirects[fd].original_fd)


class _StreamRedirector(object):
    def __init__(self, handler, fd):
        self.handler = handler
        self.fd = fd
        self.local_fd = None
        self.orig_stream = None

    def __enter__(self):
        assert self.local_fd is None
        self.orig_stream = self.handler.stream
        # Note: ideally, we would use closefd=True and let Python handle
        # closing the local file descriptor that we are about to create.
        # However, it appears that closefd is ignored on Windows (see
        # #3587), so we will just handle it explicitly ourselves.
        self.local_fd = os.dup(self.fd)
        self.handler.stream = os.fdopen(
            self.local_fd, mode="w", closefd=False
        ).__enter__()

    def __exit__(self, et, ev, tb):
        try:
            self.handler.stream.__exit__(et, ev, tb)
            os.close(self.local_fd)
        finally:
            self.handler.stream = self.orig_stream


class _LastResortRedirector(object):
    def __init__(self, fd):
        self.fd = fd
        self.local_fd = None
        self.orig_stream = None

    def __enter__(self):
        assert self.local_fd is None
        self.orig = logging.lastResort
        # Note: ideally, we would use closefd=True and let Python handle
        # closing the local file descriptor that we are about to create.
        # However, it appears that closefd is ignored on Windows (see
        # #3587), so we will just handle it explicitly ourselves.
        self.local_fd = os.dup(self.fd)
        logging.lastResort = logging.StreamHandler(
            os.fdopen(self.local_fd, mode="w", closefd=False).__enter__()
        )

    def __exit__(self, et, ev, tb):
        try:
            logging.lastResort.stream.close()
            os.close(self.local_fd)
        finally:
            logging.lastResort = self.orig
