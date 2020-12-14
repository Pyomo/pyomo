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

import sys
import logging
import time
import traceback

_logger = logging.getLogger('pyomo.common.timing')
_logger.propagate = False
_logger.setLevel(logging.WARNING)

class _NotSpecified(object): pass

def report_timing(stream=True):
    if stream:
        _logger.setLevel(logging.INFO)
        if stream is True:
            stream = sys.stdout
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter("      %(message)s"))
        _logger.addHandler(handler)
        return handler
    else:
        _logger.setLevel(logging.WARNING)
        for h in _logger.handlers:
            _logger.removeHandler(h)


_construction_logger = logging.getLogger('pyomo.common.timing.construction')


class ConstructionTimer(object):
    fmt = "%%6.%df seconds to construct %s %s; %d %s total"
    def __init__(self, obj):
        self.obj = obj
        self.timer = TicTocTimer()

    def report(self):
        # Record the elapsed time, as some log handlers may not
        # immediately generate the messge string
        self.timer = self.timer.toc(msg=None)
        _construction_logger.info(self)

    def __str__(self):
        total_time = self.timer
        try:
            idx = len(self.obj.index_set())
        except AttributeError:
            idx = 1
        try:
            name = self.obj.name
        except RuntimeError:
            try:
                name = self.obj.local_name
            except RuntimeError:
                name = '(unknown)'
        except AttributeError:
            name = '(unknown)'
        try:
            _type = self.obj.ctype.__name__
        except AttributeError:
            _type = type(self.obj).__name__
        try:
            return self.fmt % ( 2 if total_time>=0.005 else 0,
                                _type,
                                name,
                                idx,
                                'indices' if idx > 1 else 'index',
                            ) % total_time
        except TypeError:
            return "ConstructionTimer object for %s %s; %s elapsed seconds" % (
                _type,
                name,
                self.timer.toc("") )


_transform_logger = logging.getLogger('pyomo.common.timing.transformation')


class TransformationTimer(object):
    fmt = "%%6.%df seconds to apply Transformation %s%s"
    def __init__(self, obj, mode=None):
        self.obj = obj
        if mode is None:
            self.mode = ''
        else:
            self.mode = " (%s)" % (mode,)
        self.timer = TicTocTimer()

    def report(self):
        # Record the elapsed time, as some log handlers may not
        # immediately generate the message string
        self.timer = self.timer.toc(msg=None)
        _transform_logger.info(self)

    def __str__(self):
        total_time = self.timer
        name = self.obj.__class__.__name__
        try:
            return self.fmt % ( 2 if total_time>=0.005 else 0,
                                name,
                                self.mode,
                            ) % total_time
        except TypeError:
            return "TransformationTimer object for %s; %s elapsed seconds" % (
                name,
                self.timer.toc("") )

#
# Setup the timer
#
# TODO: Remove this bit for Pyomo 6.0 - we won't care about older versions
if sys.version_info >= (3,3):
    # perf_counter is guaranteed to be monotonic and the most accurate timer
    default_timer = time.perf_counter
elif sys.platform.startswith('win'):
    # On old Pythons, clock() is more accurate than time() on Windows
    # (.35us vs 15ms), but time() is more accurate than clock() on Linux
    # (1ns vs 1us).  It is unfortunate that time() is not monotonic, but
    # since the TicTocTimer is used for (potentially very accurate)
    # timers, we will sacrifice monotonicity on Linux for resolution.
    default_timer = time.clock
else:
    default_timer = time.time


class TicTocTimer(object):
    """A class to calculate and report elapsed time.

    Examples:
       >>> from pyomo.common.timing import TicTocTimer
       >>> timer = TicTocTimer()
       >>> timer.tic('starting timer') # starts the elapsed time timer (from 0)
       >>> # ... do task 1
       >>> timer.toc('task 1') # prints the elapsed time for task 1

    If no ostream or logger is provided, then output is printed to sys.stdout

    Args:
        ostream (FILE): an optional output stream to print the timing
            information
        logger (Logger): an optional output stream using the python
           logging package. Note: timing logged using logger.info
    """
    def __init__(self, ostream=_NotSpecified, logger=None):
        self._lastTime = self._loadTime = default_timer()
        self.ostream = ostream
        self.logger = logger
        self._start_count = 0
        self._cumul = 0

    def tic(self, msg=_NotSpecified,
            ostream=_NotSpecified, logger=_NotSpecified):
        """Reset the tic/toc delta timer.

        This resets the reference time from which the next delta time is
        calculated to the current time.

        Args:
            msg (str): The message to print out.  If not specified, then
                prints out "Resetting the tic/toc delta timer"; if msg
                is None, then no message is printed.
            ostream (FILE): an optional output stream (overrides the ostream
                provided when the class was constructed).
            logger (Logger): an optional output stream using the python
                logging package (overrides the ostream provided when the
                class was constructed). Note: timing logged using logger.info

        """
        self._lastTime = default_timer()
        if msg is _NotSpecified:
            msg = "Resetting the tic/toc delta timer"
        if msg is not None:
            self.toc(msg=msg, delta=False, ostream=ostream, logger=logger)


    def toc(self, msg=_NotSpecified, delta=True,
            ostream=_NotSpecified, logger=_NotSpecified):
        """Print out the elapsed time.

        This resets the reference time from which the next delta time is
        calculated to the current time.

        Args:
            msg (str): The message to print out.  If not specified, then
                print out the file name, line number, and function that
                called this method; if msg is None, then no message is
                printed.
            delta (bool): print out the elapsed wall clock time since
                the last call to :meth:`tic` or :meth:`toc`
                (:const:`True` (default)) or since the module was first
                loaded (:const:`False`).
            ostream (FILE): an optional output stream (overrides the ostream
                provided when the class was constructed).
            logger (Logger): an optional output stream using the python
                logging package (overrides the ostream provided when the
                class was constructed). Note: timing logged using logger.info

        """

        if msg is _NotSpecified:
            msg = 'File "%s", line %s in %s' % \
                  traceback.extract_stack(limit=2)[0][:3]

        now = default_timer()
        if self._start_count or self._lastTime is None:
            ans = self._cumul
            if self._lastTime:
                ans += default_timer() - self._lastTime
            if msg is not None:
                msg = "[%8.2f|%4d] %s\n" % (ans, self._start_count, msg)
        elif delta:
            ans = now - self._lastTime
            self._lastTime = now
            if msg is not None:
                msg = "[+%7.2f] %s\n" % (ans, msg)
        else:
            ans = now - self._loadTime
            if msg is not None:
                msg = "[%8.2f] %s\n" % (ans, msg)

        if msg is not None:
            if logger is _NotSpecified:
                logger = self.logger
            if logger is not None:
                logger.info(msg)

            if ostream is _NotSpecified:
                ostream = self.ostream
                if ostream is _NotSpecified and logger is None:
                    ostream = sys.stdout
            if ostream is not None:
                ostream.write(msg)

        return ans

    def stop(self):
        try:
            delta = default_timer() - self._lastTime
        except TypeError:
            if self._lastTime is None:
                raise RuntimeError(
                    "Stopping a TicTocTimer that was already stopped")
            raise
        self._cumul += delta
        self._lastTime = None
        return delta

    def start(self):
        if self._lastTime:
            self.stop()
        self._start_count += 1
        self._lastTime = default_timer()

_globalTimer = TicTocTimer()
tic = _globalTimer.tic
toc = _globalTimer.toc


class _HierarchicalHelper(object):
    def __init__(self):
        self.tic_toc = TicTocTimer()
        self.timers = dict()
        self.total_time = 0
        self.n_calls = 0

    def start(self):
        self.n_calls += 1
        self.tic_toc.start()

    def stop(self):
        self.total_time += self.tic_toc.stop()

    def to_str(self, indent, stage_identifier_lengths):
        s = ''
        if len(self.timers) > 0:
            underline = indent + '-' * (sum(stage_identifier_lengths) + 36) + '\n'
            s += underline
            name_formatter = '{name:<' + str(sum(stage_identifier_lengths)) + '}'
            other_time = self.total_time
            sub_stage_identifier_lengths = stage_identifier_lengths[1:]
            for name, timer in sorted(self.timers.items()):
                if self.total_time > 0:
                    _percent = timer.total_time / self.total_time * 100
                else:
                    _percent = float('nan')
                s += indent
                s += ( name_formatter + '{ncalls:>9d} {cumtime:>9.3f} '
                       '{percall:>9.3f} {percent:>6.1f}\n' ).format(
                           name=name,
                           ncalls=timer.n_calls,
                           cumtime=timer.total_time,
                           percall=timer.total_time/timer.n_calls,
                           percent=_percent )
                s += timer.to_str(
                    indent=indent + ' '*stage_identifier_lengths[0],
                    stage_identifier_lengths=sub_stage_identifier_lengths)
                other_time -= timer.total_time

            if self.total_time > 0:
                _percent = other_time / self.total_time * 100
            else:
                _percent = float('nan')
            s += indent
            s += ( name_formatter + '{ncalls:>9} {cumtime:>9.3f} '
                   '{percall:>9} {percent:>6.1f}\n' ).format(
                       name='other',
                       ncalls='n/a',
                       cumtime=other_time,
                       percall='n/a',
                       percent=_percent )
            s += underline.replace('-', '=')
        return s

    def get_timers(self, res, prefix):
        for name, timer in self.timers.items():
            _name = prefix + '.' + name
            res.append(_name)
            timer.get_timers(res, _name)


class HierarchicalTimer(object):
    """A class for hierarchical timing.

    Examples
    --------
    >>> import time
    >>> from pyomo.common.timing import HierarchicalTimer
    >>> timer = HierarchicalTimer()
    >>> timer.start('all')
    >>> time.sleep(0.2)
    >>> for i in range(10):
    ...     timer.start('a')
    ...     time.sleep(0.1)
    ...     for i in range(5):
    ...         timer.start('aa')
    ...         time.sleep(0.01)
    ...         timer.stop('aa')
    ...     timer.start('ab')
    ...     timer.stop('ab')
    ...     timer.stop('a')
    ...
    >>> for i in range(10):
    ...     timer.start('b')
    ...     time.sleep(0.02)
    ...     timer.stop('b')
    ...
    >>> timer.stop('all')
    >>> print(timer)
    Identifier        ncalls   cumtime   percall      %
    ---------------------------------------------------
    all                    1     2.248     2.248  100.0
         ----------------------------------------------
         a                10     1.787     0.179   79.5
              -----------------------------------------
              aa          50     0.733     0.015   41.0
              ab          10     0.000     0.000    0.0
              other      n/a     1.055       n/a   59.0
              =========================================
         b                10     0.248     0.025   11.0
         other           n/a     0.213       n/a    9.5
         ==============================================
    ===================================================
    <BLANKLINE>

    The columns are:
      ncalls : The number of times the timer was started and stopped
      cumtime: The cumulative time (in seconds) the timer was active
               (started but not stopped)
      percall: cumtime (in seconds) / ncalls
      %      : This is cumtime of the timer divided by cumtime of the
               parent timer times 100


    >>> print('a total time: %f' % timer.get_total_time('all.a'))
    a total time: 1.902037
    >>> print('ab num calls: %d' % timer.get_num_calls('all.a.ab'))
    ab num calls: 10
    >>> print('aa %% time: %f' % timer.get_relative_percent_time('all.a.aa'))
    aa % time: 44.144148
    >>> print('aa %% total: %f' % timer.get_total_percent_time('all.a.aa'))
    aa % total: 35.976058

    Internal Workings
    -----------------
    The HierarchicalTimer use a stack to track which timers are active
    at any point in time. Additionally, each timer has a dictionary of
    timers for its children timers. Consider

    >>> timer = HierarchicalTimer()
    >>> timer.start('all')
    >>> timer.start('a')
    >>> timer.start('aa')

    After the above code is run, self.stack will be ['all', 'a', 'aa']
    and self.timers will have one key, 'all' and one value which will be
    a _HierarchicalHelper. The _HierarchicalHelper has its own timers
    dictionary:

    {'a': _HierarchicalHelper}

    and so on. This way, we can easily access any timer with something
    that looks like the stack. The logic is recursive (although the
    code is not).

    """
    def __init__(self):
        self.stack = list()
        self.timers = dict()

    def _get_timer(self, identifier, should_exist=False):
        """

        This method gets the timer associated with the current state
        of self.stack and the specified identifier.

        Parameters
        ----------
        identifier: str
            The name of the timer
        should_exist: bool
            The should_exist is True, and the timer does not already
            exist, an error will be raised. If should_exist is False, and
            the timer does not already exist, a new timer will be made.

        Returns
        -------
        timer: _HierarchicalHelper

        """
        parent = self._get_timer_from_stack(self.stack)
        if identifier in parent.timers:
            return parent.timers[identifier]
        else:
            if should_exist:
                raise RuntimeError(
                    'Could not find timer {0}'.format(
                        '.'.join(self.stack + [identifier])))
            parent.timers[identifier] = _HierarchicalHelper()
            return parent.timers[identifier]

    def start(self, identifier):
        """
        Start incrementing the timer identified with identifier

        Parameters
        ----------
        identifier: str
            The name of the timer
        """
        timer = self._get_timer(identifier)
        timer.start()
        self.stack.append(identifier)

    def stop(self, identifier):
        """
        Stop incrementing the timer identified with identifier

        Parameters
        ----------
        identifier: str
            The name of the timer
        """
        if identifier != self.stack[-1]:
            raise ValueError(
                str(identifier) + ' is not the currently active timer.  '
                'The only timer that can currently be stopped is '
                + '.'.join(self.stack))
        self.stack.pop()
        timer = self._get_timer(identifier, should_exist=True)
        timer.stop()

    def _get_identifier_len(self):
        stage_timers = list(self.timers.items())
        stage_lengths = list()

        while len(stage_timers) > 0:
            new_stage_timers = list()
            max_len = 0
            for identifier, timer in stage_timers:
                new_stage_timers.extend(timer.timers.items())
                if len(identifier) > max_len:
                    max_len = len(identifier)
            stage_lengths.append(max(max_len, len('other')))
            stage_timers = new_stage_timers

        return stage_lengths

    def __str__(self):
        stage_identifier_lengths = self._get_identifier_len()
        name_formatter = '{name:<' + str(sum(stage_identifier_lengths)) + '}'
        s = ( name_formatter + '{ncalls:>9} {cumtime:>9} '
              '{percall:>9} {percent:>6}\n').format(
                  name='Identifier',
                  ncalls='ncalls',
                  cumtime='cumtime',
                  percall='percall',
                  percent='%')
        underline = '-' * (sum(stage_identifier_lengths) + 36) + '\n'
        s += underline
        sub_stage_identifier_lengths = stage_identifier_lengths[1:]
        for name, timer in sorted(self.timers.items()):
            s += ( name_formatter + '{ncalls:>9d} {cumtime:>9.3f} '
                   '{percall:>9.3f} {percent:>6.1f}\n').format(
                       name=name,
                       ncalls=timer.n_calls,
                       cumtime=timer.total_time,
                       percall=timer.total_time/timer.n_calls,
                       percent=self.get_total_percent_time(name))
            s += timer.to_str(
                indent=' '*stage_identifier_lengths[0],
                stage_identifier_lengths=sub_stage_identifier_lengths)
        s += underline.replace('-', '=')
        return s

    def reset(self):
        """
        Completely reset the timer.
        """
        self.stack = list()
        self.timers = dict()

    def _get_timer_from_stack(self, stack):
        """
        This method gets the timer associated with stack.

        Parameters
        ----------
        stack: list of str
            A list of identifiers.

        Returns
        -------
        timer: _HierarchicalHelper
        """
        tmp = self
        for i in stack:
            tmp = tmp.timers[i]
        return tmp

    def get_total_time(self, identifier):
        """
        Parameters
        ----------
        identifier: str
            The full name of the timer including parent timers separated
            with dots.

        Returns
        -------
        total_time: float
            The total time spent with the specified timer active.
        """
        stack = identifier.split('.')
        timer = self._get_timer_from_stack(stack)
        return timer.total_time

    def get_num_calls(self, identifier):
        """
        Parameters
        ----------
        identifier: str
            The full name of the timer including parent timers separated
            with dots.

        Returns
        -------
        num_calss: int
            The number of times start was called for the specified timer.
        """
        stack = identifier.split('.')
        timer = self._get_timer_from_stack(stack)
        return timer.n_calls

    def get_relative_percent_time(self, identifier):
        """
        Parameters
        ----------
        identifier: str
            The full name of the timer including parent timers separated
            with dots.

        Returns
        -------
        percent_time: float
            The percent of time spent in the specified timer
            relative to the timer's immediate parent.
        """
        stack = identifier.split('.')
        timer = self._get_timer_from_stack(stack)
        parent = self._get_timer_from_stack(stack[:-1])
        if parent is self:
            return self.get_total_percent_time(identifier)
        else:
            if parent.total_time > 0:
                return timer.total_time / parent.total_time * 100
            else:
                return float('nan')

    def get_total_percent_time(self, identifier):
        """
        Parameters
        ----------
        identifier: str
            The full name of the timer including parent timers separated
            with dots.

        Returns
        -------
        percent_time: float
            The percent of time spent in the specified timer
            relative to the total time in all timers.
        """
        stack = identifier.split('.')
        timer = self._get_timer_from_stack(stack)
        total_time = 0
        for _timer in self.timers.values():
            total_time += _timer.total_time
        if total_time > 0:
            return timer.total_time / total_time * 100
        else:
            return float('nan')

    def get_timers(self):
        """
        Returns
        -------
        identifiers: list of str
            Returns a list of all timer identifiers
        """
        res = list()
        for name, timer in self.timers.items():
            res.append(name)
            timer.get_timers(res, name)
        return res
