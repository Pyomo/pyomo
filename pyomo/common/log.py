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
#  This module was originally developed as part of the PyUtilib project
#  Copyright (c) 2008 Sandia Corporation.
#  This software is distributed under the BSD License.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  ___________________________________________________________________________
#
# Utility classes for working with the logger
#

import logging
import re
import sys
import textwrap

from pyomo.common.deprecation import deprecated
from pyomo.common.fileutils import PYOMO_ROOT_DIR

_indentation_re = re.compile(r'\s*')
_bullet_re = re.compile(r'(?:[-*] +)|(\[\s*[A-Za-z0-9\.]+\s*\] +)')
_bullet_char = '-*['

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
                raise ValueError('unrecognized style flag "%s"'
                                 % (kwds['style'],))
        self._wrapper = textwrap.TextWrapper(width=kwds.pop('wrap', 78))
        self.hang = kwds.pop('hang', ' '*4)
        self.basepath = kwds.pop('base', None)
        super(WrappingFormatter, self).__init__(**kwds)

    def format(self, record):
        _orig = {k:getattr(record, k) for k in ('msg', 'args', 'pathname')}
        msg = record.getMessage()
        record.msg = self._flag
        record.args = None
        if self.basepath and record.pathname.startswith(self.basepath):
            record.pathname = '[base]' + record.pathname[len(self.basepath):]
        try:
            raw_msg = super(WrappingFormatter, self).format(record)
        finally:
            for k,v in _orig.items():
                setattr(record, k, v)

        # Most of the messages are either unformatted long lines or
        # triple-quote blocks of text.  In the latter case, if the text
        # starts on the same line as the triple-quote, then it is almost
        # certainly NOT indented with the bulk of the text, which will
        # cause dedent to get confused and not strip any leading
        # whitespace.  This attempts to work around that case:
        #
        #if not (_msg.startswith('\n') or _indentation_re.match(_msg).group()):
        #    # copy the indention for the second line to the first:
        #    lines = _msg.splitlines()
        #    if len(lines) > 1:
        #        _msg = _indentation_re.match(lines[1]).group() + _msg
        #
        # The problem with the above logic is that users may want a
        # simple introductory line followed by an intented line (our
        # tests did this!), and cannot specify it without adding an
        # extra blank line to the output.  In contrast, it is possible
        # for the user to fix the scenario above that motivated this
        # code by just indenting their first line correctly.
        msg = textwrap.dedent(msg).strip()

        # Split the formatted log message (that currently has _flag in
        # lieu of the actual message content) into lines, then
        # recombine, substituting and wrapping any lines that contain
        # _flag.
        return '\n'.join(
            self._wrap_msg(l, msg) if self._flag in l else l
            for l in raw_msg.splitlines()
        )

    def _wrap_msg(self, l, msg):
        indent = _indentation_re.match(l).group()
        return self._wrap(l.strip().replace(self._flag, msg), indent)

    def _wrap(self, msg, base_indent):
        # As textwrap only works on single paragraphs, we need to break
        # up the incoming message into paragraphs before we pass it to
        # textwrap.
        paragraphs = []
        verbatim = False
        for line in msg.rstrip().splitlines():
            leading = _indentation_re.match(line).group()
            content = line.strip()
            if not content:
                paragraphs.append((None, None))
            elif content == '```':
                verbatim ^= True
            elif verbatim:
                paragraphs.append((None, line))
            else:
                matchBullet = _bullet_re.match(content)
                if matchBullet:
                    paragraphs.append(
                        (leading + ' '*len(matchBullet.group()), [content]))
                elif paragraphs and paragraphs[-1][0] == leading:
                    paragraphs[-1][1].append( content )
                else:
                    paragraphs.append((leading, [content]))

        base_indent = (self.hang or '') + base_indent

        for i, (indent, par) in enumerate(paragraphs):
            if indent is None:
                if par is None:
                    paragraphs[i] = ''
                else:
                    paragraphs[i] = base_indent + par
                continue

            par_indent = base_indent + indent
            self._wrapper.subsequent_indent = par_indent
            if not i and self.hang:
                self._wrapper.initial_indent = par_indent[len(self.hang):]
            else:
                self._wrapper.initial_indent = par_indent

            # Bulleted lists get indented with a hanging indent
            bullet = _bullet_re.match(par[0])
            if bullet:
                self._wrapper.initial_indent = par_indent[:-len(bullet.group())]

            paragraphs[i] = self._wrapper.fill(' '.join(par))
        return '\n'.join(paragraphs)


class LegacyPyomoFormatter(logging.Formatter):
    """This mocks up the legacy Pyomo log formating.

    This formatter takes a callback function (`verbosity`) that will be
    called for each message.  Based on the result, one of two formatting
    templates will be used.

    """
    def __init__(self, **kwds):
        if 'fmt' in kwds:
            raise ValueError(
                "'fmt' is not a valid option for the LegacyFormatter")
        if 'style' in kwds:
            raise ValueError(
                "'style' is not a valid option for the LegacyFormatter")

        self.verbosity = kwds.pop('verbosity', lambda: True)
        self.standard_formatter = WrappingFormatter(**kwds)
        self.verbose_formatter = WrappingFormatter(
            fmt='%(levelname)s: "%(pathname)s", %(lineno)d, %(funcName)s\n'
            '    %(message)s',
            hang=False,
            **kwds
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


class _GlobalLogFilter(object):
    def __init__(self):
        self.logger = logging.getLogger()

    def filter(self, record):
        return not self.logger.handlers


# This mocks up the historical Pyomo logging system, which uses a
# different formatter based on if the main pyomo logger is enabled for
# debugging.  It has been updated to suppress output if any handlers
# have been defined at the root level.
_pyomoLogger = logging.getLogger('pyomo')
_handler = StdoutHandler()
_handler.setFormatter(LegacyPyomoFormatter(
    base=PYOMO_ROOT_DIR,
    verbosity=lambda: _pyomoLogger.isEnabledFor(logging.DEBUG),
))
_handler.addFilter(_GlobalLogFilter())
_pyomoLogger.addHandler(_handler)


class LogHandler(logging.StreamHandler):
    @deprecated('The pyomo.common.log.LogHandler class has been deprecated '
                'in favor of standard Handlers from the Python logging module '
                'combined with the pyomo.common.log.WrappingFormatter.',
                version='5.7.3')
    def __init__(self, base='', stream=None,
                 level=logging.NOTSET, verbosity=None):
        super(LogHandler, self).__init__(stream)
        self.setLevel(level),
        if verbosity is None:
            verbosity = lambda: True
        self.setFormatter(LegacyPyomoFormatter(
            base=base,
            verbosity=verbosity,
        ))


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
        >>> from pyomo.common.log import LoggingIntercept
        >>> buf = six.StringIO()
        >>> with LoggingIntercept(buf, 'pyomo.core', logging.WARNING):
        ...     logging.getLogger('pyomo.core').warning('a simple message')
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
