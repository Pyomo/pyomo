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

from pyomo.common.fileutils import PYOMO_ROOT_DIR

_indention = re.compile('\s*')
_status_re = re.compile('^\[\s*[A-Za-z0-9\.]+\s*\]')


class LogHandler(logging.Handler):

    def __init__( self, base='', stream=None,
                  level=logging.NOTSET, verbosity=None ):
        logging.Handler.__init__(self, level=level)

        if verbosity is None:
            verbosity = lambda: True
        if stream is None:
            stream = sys.stdout

        self.verbosity = verbosity
        self.stream = stream
        self.basepath = base
        # Public attributes (because embedded constants in functions are evil)
        self.wrap = 78
        self.initial_indent = ''
        self.subsequent_indent = ' '*4

    def emit(self, record):
        level = record.levelname
        msg = record.getMessage()
        # Most of the messages are either unformatted long lines or
        # triple-quote blocks of text.  In the latter case, if the text
        # starts on the same line as the triple-quote, then it is almost
        # certainly NOT indented with the bulk of the text, which will
        # cause dedent to get confused and not strip any leading
        # whitespace.  This attempts to work around that case:
        #
        #if not ( msg.startswith('\n') or _indention.match(msg).group() ):
        #    # copy the indention for the second line to the first:
        #    lines = msg.splitlines()
        #    if len(lines) > 1:
        #        msg = _indention.match(lines[1]).group() + msg
        #
        # The problem with the above logic is that users may want a
        # simple introductory line followed by an intented line (our
        # tests did this!), and cannot specify it without adding an
        # extra blank line to the output.  In contrast, it is possible
        # for the user to fix the scenario above that motivated this
        # code by just indenting their first line correctly.

        # TBD: dedent does not convert \t to ' '*8. Should we do that?
        msg = textwrap.dedent(msg)

        # As textwrap only works on single paragraphs, we need to break
        # up the incoming message into paragraphs before we pass it to
        # textwrap.
        paragraphs = []
        indent = _indention.match(msg).group()
        par_lines = []
        for line in msg.splitlines():
            leading = _indention.match(line).group()
            content = line.strip()
            if not content:
                paragraphs.append((indent, par_lines))
                par_lines = []
                # Blank lines reset the indentation level
                indent = None
            elif indent == leading:
                # Catch things like bulleted lists and '[FAIL]'
                if len(content) > 1 and par_lines and (
                        (content[1] == ' ' and content[0] in '-* ') or
                        (content[0] == '[' and _status_re.match(content))):
                    paragraphs.append((indent, par_lines))
                    par_lines = []
                par_lines.append( content )
            else:
                paragraphs.append((indent, par_lines))
                par_lines = [ content ]
                indent = leading
        # Collect the final paragraph
        if par_lines:
            paragraphs.append((indent, par_lines))

        # Skip any leading/trailing blank lines
        while paragraphs and not paragraphs[-1][1]:
            paragraphs.pop()
        while paragraphs and not paragraphs[0][1]:
            paragraphs.pop(0)

        if self.verbosity():
            #
            # If verbosity is on, the first logged line is the file,
            # line, and function name that called the logger.  The first
            # line of the message is actually the second line of the
            # output (and so is indented/wrapped the same as the rest of
            # the message)
            #
            filename = record.pathname  # file path
            lineno = record.lineno
            try:
                function = record.funcName
            except AttributeError:
                function = '(unknown)'
            if self.basepath and filename.startswith(self.basepath):
                filename = '[base]' + filename[len(self.basepath):]

            self.stream.write(
                '%s: "%s", %d, %s\n' %
                ( level, filename, lineno, function.strip(), ))
        else:
            #
            # If verbosity is off, prepend the log level name to the
            # beginning of the message and format the line without the
            # 'subsequent' indentation of the remainder of the message
            #
            if paragraphs:
                firstPar = ' '.join(paragraphs.pop(0)[1]).strip()
                if level:
                    firstPar = ('%s: %s' % (level, firstPar))
            else:
                firstPar = level
            self.stream.write( '%s\n' % (
                textwrap.fill( firstPar,
                               width=self.wrap,
                               initial_indent=self.initial_indent,
                               subsequent_indent=self.subsequent_indent ), ))
        for indent, par in paragraphs:
            if not indent:
                indent = ''
            # Bulleted lists get indented with a hanging indent
            if par and len(par[0]) > 1 and par[0][0] in '-*':
                hang = ' '*4
            else:
                hang = ''
            self.stream.write( '%s\n' % (
                textwrap.fill(
                    ' '.join(par),
                    width=self.wrap,
                    initial_indent=self.subsequent_indent+indent,
                    subsequent_indent=self.subsequent_indent+indent+hang ), ))

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
