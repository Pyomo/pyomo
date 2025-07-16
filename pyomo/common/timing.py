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

"""A module of utilities for collecting timing information

.. autosummary::

    report_timing
    TicTocTimer
    tic
    toc
    HierarchicalTimer

"""

import functools
import logging
import sys
import time
import traceback
from pyomo.common.deprecation import deprecation_warning
from pyomo.common.modeling import NOTSET as _NotSpecified

_logger = logging.getLogger('pyomo.common.timing')
_logger.propagate = False
_logger.setLevel(logging.WARNING)

_construction_logger = logging.getLogger('pyomo.common.timing.construction')
_transform_logger = logging.getLogger('pyomo.common.timing.transformation')


class report_timing(object):
    def __init__(self, stream=True, level=logging.INFO):
        """Set reporting of Pyomo timing information.

        For historical reasons, this class may be used as a function
        (the reporting logger is configured as part of the instance
        initializer).  However, the preferred usage is as a context
        manager (thereby ensuring that the timing logger is restored
        upon exit).

        Parameters
        ----------
        stream: bool, TextIOBase

            The destination stream to emit timing information.  If
            ``True``, defaults to ``sys.stdout``.  If ``False`` or
            ``None``, disables reporting of timing information.

        level: int

            The logging level for the timing logger

        """
        self._old_level = _logger.level
        # For historical reasons (because report_timing() used to be a
        # function), we will do what you think should be done in
        # __enter__ here in __init__.
        if stream:
            _logger.setLevel(level)
            if stream is True:
                stream = sys.stdout
            self._handler = logging.StreamHandler(stream)
            self._handler.setFormatter(logging.Formatter("      %(message)s"))
            _logger.addHandler(self._handler)
        else:
            self._handler = list(_logger.handlers)
            _logger.setLevel(logging.WARNING)
            for h in list(_logger.handlers):
                _logger.removeHandler(h)

    def reset(self):
        _logger.setLevel(self._old_level)
        if type(self._handler) is list:
            for h in self._handler:
                _logger.addHandler(h)
        else:
            _logger.removeHandler(self._handler)

    def __enter__(self):
        return self

    def __exit__(self, et, ev, tb):
        self.reset()


class GeneralTimer(object):
    def __init__(self, fmt, data):
        self.fmt = fmt
        self.data = data

    def report(self):
        _logger.info(self)

    @property
    def obj(self):
        return self.data[-1]

    @property
    def timer(self):
        return self.data[:-1]

    def __str__(self):
        return self.fmt % self.data


class ConstructionTimer(object):
    __slots__ = ('obj', 'timer')
    msg = "%6.*f seconds to construct %s %s%s"
    in_progress = "ConstructionTimer object for %s %s; %0.3f elapsed seconds"

    def __init__(self, obj):
        self.obj = obj
        self.timer = -default_timer()

    def report(self):
        # Record the elapsed time, as some log handlers may not
        # immediately generate the message string
        self.timer += default_timer()
        _construction_logger.info(self)

    @property
    def name(self):
        try:
            return self.obj.name
        except RuntimeError:
            try:
                return self.obj.local_name
            except RuntimeError:
                return '(unknown)'
        except AttributeError:
            return '(unknown)'

    def __str__(self):
        try:
            if self.obj.is_indexed():
                # indexed component
                if self.obj.index_set().isfinite():
                    idx = len(self.obj.index_set())
                else:
                    idx = len(self.obj)
                idx_label = f'{idx} indices' if idx != 1 else '1 index'
            elif hasattr(self.obj, 'index_set'):
                # scalar indexed components
                idx = len(self.obj.index_set())
                idx_label = f'{idx} indices' if idx != 1 else '1 index'
            else:
                # other non-indexed component (e.g., Suffix)
                idx_label = ''
        except AttributeError:
            # unknown component
            idx_label = ''
        if idx_label:
            idx_label = f'; {idx_label} total'
        try:
            _type = self.obj.ctype.__name__
        except AttributeError:
            _type = type(self.obj).__name__
        total_time = self.timer
        if total_time < 0:
            total_time += default_timer()
            return self.in_progress % (_type, self.name, total_time)
        return self.msg % (
            2 if total_time >= 0.005 else 0,
            total_time,
            _type,
            self.name,
            idx_label,
        )


class TransformationTimer(object):
    __slots__ = ('obj', 'mode', 'timer')
    msg = "%6.*f seconds to apply Transformation %s%s"
    in_progress = "TransformationTimer object for %s%s; %0.3f elapsed seconds"

    def __init__(self, obj, mode=None):
        self.obj = obj
        if mode is None:
            self.mode = ''
        else:
            self.mode = " (%s)" % (mode,)
        self.timer = -default_timer()

    def report(self):
        # Record the elapsed time, as some log handlers may not
        # immediately generate the message string
        self.timer += default_timer()
        _transform_logger.info(self)

    @property
    def name(self):
        return self.obj.__class__.__name__

    def __str__(self):
        total_time = self.timer
        if total_time < 0:
            total_time += default_timer()
            return self.in_progress % (self.name, self.mode, total_time)
        return self.msg % (
            2 if total_time >= 0.005 else 0,
            total_time,
            self.name,
            self.mode,
        )


#
# Setup the timer
#
# perf_counter is guaranteed to be monotonic and the most accurate
# timer.  It became available in Python 3.3.  Prior to that, clock() was
# more accurate than time() on Windows (.35us vs 15ms), but time() was
# more accurate than clock() on Linux (1ns vs 1us).  It is unfortunate
# that time() is not monotonic, but since the TicTocTimer is used for
# (potentially very accurate) timers, we will sacrifice monotonicity on
# Linux for resolution.
default_timer = time.perf_counter


class TicTocTimer(object):
    """A class to calculate and report elapsed time.

    Examples:
       >>> from pyomo.common.timing import TicTocTimer
       >>> timer = TicTocTimer()
       >>> timer.tic('starting timer') # starts the elapsed time timer (from 0)
       [    0.00] starting timer
       >>> # ... do task 1
       >>> dT = timer.toc('task 1')
       [+   0.00] task 1
       >>> print("elapsed time: %0.1f" % dT)
       elapsed time: 0.0

    If no ostream or logger is provided, then output is printed to sys.stdout

    Args:
        ostream (FILE): an optional output stream to print the timing
            information
        logger (Logger): an optional output stream using the python
           logging package. Note: the timing logged using ``logger.info()``
    """

    def __init__(self, ostream=_NotSpecified, logger=None):
        if ostream is _NotSpecified and logger is not None:
            ostream = None
        self._lastTime = self._loadTime = default_timer()
        self.ostream = ostream
        self.logger = logger
        self.level = logging.INFO
        self._start_count = 0
        self._cumul = 0

    def tic(
        self,
        msg=_NotSpecified,
        *args,
        ostream=_NotSpecified,
        logger=_NotSpecified,
        level=_NotSpecified,
    ):
        """Reset the tic/toc delta timer.

        This resets the reference time from which the next delta time is
        calculated to the current time.

        Args:
            msg (str): The message to print out.  If not specified, then
                prints out "Resetting the tic/toc delta timer"; if msg
                is None, then no message is printed.
            *args (tuple): optional positional arguments used for
                %-formatting the `msg`
            ostream (FILE): an optional output stream (overrides the ostream
                provided when the class was constructed).
            logger (Logger): an optional output stream using the python
                logging package (overrides the ostream provided when the
                class was constructed). Note: timing logged using logger.info
            level (int): an optional logging output level.

        """
        self._lastTime = self._loadTime = default_timer()
        if msg is _NotSpecified:
            msg = "Resetting the tic/toc delta timer"
        if msg is not None:
            if args and '%' not in msg:
                # Note: specify the parent module scope for the logger
                # so this does not hit (and get handled by) the local
                # pyomo.common.timing logger.
                deprecation_warning(
                    "tic(): 'ostream' and 'logger' should be "
                    "specified as keyword arguments",
                    version='6.4.2',
                    logger=__package__,
                )
                ostream, *args = args
                if args:
                    logger, *args = args
            self.toc(
                msg, *args, delta=False, ostream=ostream, logger=logger, level=level
            )

    def toc(
        self,
        msg=_NotSpecified,
        *args,
        delta=True,
        ostream=_NotSpecified,
        logger=_NotSpecified,
        level=_NotSpecified,
    ):
        """Print out the elapsed time.

        This resets the reference time from which the next delta time is
        calculated to the current time.

        Args:
            msg (str): The message to print out.  If not specified, then
                print out the file name, line number, and function that
                called this method; if `msg` is None, then no message is
                printed.
            *args (tuple): optional positional arguments used for
                %-formatting the `msg`
            delta (bool): print out the elapsed wall clock time since
                the last call to :meth:`tic` (``False``) or since the
                most recent call to either :meth:`tic` or :meth:`toc`
                (``True`` (default)).
            ostream (FILE): an optional output stream (overrides the ostream
                provided when the class was constructed).
            logger (Logger): an optional output stream using the python
                logging package (overrides the ostream provided when the
                class was constructed). Note: timing logged using `level`
            level (int): an optional logging output level.
        """

        # Note: important to do this first so that we don't add a random
        # amount of time for I/O operations or extracting the stack.
        # This helps ensure that the timing tests are less fragile.
        now = default_timer()

        if msg is _NotSpecified:
            msg = 'File "%s", line %s in %s' % traceback.extract_stack(limit=2)[0][:3]
        if args and msg is not None and '%' not in msg:
            # Note: specify the parent module scope for the logger
            # so this does not hit (and get handled by) the local
            # pyomo.common.timing logger.
            deprecation_warning(
                "toc(): 'delta', 'ostream', and 'logger' should be "
                "specified as keyword arguments",
                version='6.4.2',
                logger=__package__,
            )
            delta, *args = args
            if args:
                ostream, *args = args
            if args:
                logger, *args = args

        if self._start_count or self._lastTime is None:
            ans = self._cumul
            if self._lastTime:
                ans += now - self._lastTime
            if msg is not None:
                fmt = "[%8.2f|%4d] %s"
                data = (ans, self._start_count, msg)
        elif delta:
            ans = now - self._lastTime
            self._lastTime = now
            if msg is not None:
                fmt = "[+%7.2f] %s"
                data = (ans, msg)
        else:
            ans = now - self._loadTime
            # Even though we are reporting the cumulative time, we will
            # still reset the delta timer.
            self._lastTime = now
            if msg is not None:
                fmt = "[%8.2f] %s"
                data = (ans, msg)

        if msg is not None:
            if logger is _NotSpecified:
                logger = self.logger
            if logger is not None:
                if level is _NotSpecified:
                    level = self.level
                logger.log(level, GeneralTimer(fmt, data), *args)

            if ostream is _NotSpecified:
                ostream = self.ostream
                if ostream is _NotSpecified:
                    if logger is None:
                        ostream = sys.stdout
                    else:
                        ostream = None
            if ostream is not None:
                msg = fmt % data
                if args:
                    msg = msg % args
                ostream.write(msg + '\n')

        return ans

    def stop(self):
        delta, self._lastTime = self._lastTime, None
        if delta is None:
            raise RuntimeError("Stopping a TicTocTimer that was already stopped")
        delta = default_timer() - delta
        self._cumul += delta
        return delta

    def start(self):
        if self._lastTime:
            self.stop()
        self._start_count += 1
        self._lastTime = default_timer()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, et, ev, tb):
        self.stop()


_globalTimer = TicTocTimer()
tic = functools.partial(TicTocTimer.tic, _globalTimer)
tic.__doc__ = """
Reset the global :py:class:`TicTocTimer` instance.

See :py:meth:`TicTocTimer.tic()`.
"""
toc = functools.partial(TicTocTimer.toc, _globalTimer)
toc.__doc__ = """
Print the elapsed time from the global :py:class:`TicTocTimer` instance.

See :py:meth:`TicTocTimer.toc()`.
"""


def _move_grandchildren_to_root(root, child):
    """A helper function to assist with flattening of HierarchicalTimer
    objects

    Parameters
    ----------
    root: HierarchicalTimer or _HierarchicalHelper
        The root node. Children of `child` will become children of
        this node

    child: _HierarchicalHelper
        The child node that will be turned into a leaf by moving
        its children to the root

    """
    for gchild_key, gchild_timer in child.timers.items():
        # For each grandchild, if this key corresponds to a child,
        # combine the information from these timers. Otherwise,
        # add the new timer as a child of the root.
        if gchild_key in root.timers:
            gchild_total_time = gchild_timer.total_time
            gchild_n_calls = gchild_timer.n_calls
            root.timers[gchild_key].total_time += gchild_total_time
            root.timers[gchild_key].n_calls += gchild_n_calls
        else:
            root.timers[gchild_key] = gchild_timer

        # Subtract the grandchild's total time from the child (which
        # will no longer be a parent of the grandchild)
        child.total_time -= gchild_timer.total_time

    # Clear the child timer's dict to make it a leaf node
    child.timers.clear()


def _clear_timers_except(timer, to_retain):
    """A helper function for removing keys, except for those specified,
    from the dictionary of timers

    Parameters
    ----------
    timer: HierarchicalTimer or _HierarchicalHelper
        The timer whose dict of "sub-timers" will be pruned

    to_retain: set
        Set of keys of the "sub-timers" to retain

    """
    keys = list(timer.timers.keys())
    for key in keys:
        if key not in to_retain:
            timer.timers.pop(key)


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
                s += (
                    name_formatter + '{ncalls:>9d} {cumtime:>9.3f} '
                    '{percall:>9.3f} {percent:>6.1f}\n'
                ).format(
                    name=name,
                    ncalls=timer.n_calls,
                    cumtime=timer.total_time,
                    percall=timer.total_time / timer.n_calls,
                    percent=_percent,
                )
                s += timer.to_str(
                    indent=indent + ' ' * stage_identifier_lengths[0],
                    stage_identifier_lengths=sub_stage_identifier_lengths,
                )
                other_time -= timer.total_time

            if self.total_time > 0:
                _percent = other_time / self.total_time * 100
            else:
                _percent = float('nan')
            s += indent
            s += (
                name_formatter + '{ncalls:>9} {cumtime:>9.3f} '
                '{percall:>9} {percent:>6.1f}\n'
            ).format(
                name='other',
                ncalls='n/a',
                cumtime=other_time,
                percall='n/a',
                percent=_percent,
            )
            s += underline.replace('-', '=')
        return s

    def get_timers(self, res, prefix):
        for name, timer in self.timers.items():
            _name = prefix + '.' + name
            res.append(_name)
            timer.get_timers(res, _name)

    def flatten(self):
        # Get keys and values so we don't modify dict while iterating it.
        items = list(self.timers.items())
        for child_key, child_timer in items:
            # Flatten the child timer. Now all grandchildren are leaf nodes
            child_timer.flatten()
            # Flatten by removing grandchildren and adding them as children
            # of the root.
            _move_grandchildren_to_root(self, child_timer)

    def clear_except(self, *args):
        to_retain = set(args)
        _clear_timers_except(self, to_retain)


class HierarchicalTimer(object):
    """A class for collecting and displaying hierarchical timing
    information

    When implementing an iterative algorithm with nested subroutines
    (e.g. an optimization solver), we often want to know the cumulative
    time spent in each subroutine as well as this time as a proportion
    of time spent in the calling routine. This class collects timing
    information, for user-specified keys, that accumulates over the life
    of the timer object and preserves the hierarchical (nested) structure
    of timing categories.

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
    >>> print(timer)       # doctest: +SKIP
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

      ncalls
          The number of times the timer was started and stopped
      cumtime
          The cumulative time (in seconds) the timer was active
          (started but not stopped)
      percall
          cumtime (in seconds) / ncalls
      "%"
          This is cumtime of the timer divided by cumtime of the
          parent timer times 100

    >>> print('a total time: %f' % timer.get_total_time('all.a')) \
        # doctest: +SKIP
    a total time: 1.902037
    >>> print('ab num calls: %d' % timer.get_num_calls('all.a.ab')) \
        # doctest: +SKIP
    ab num calls: 10
    >>> print('aa %% time: %f' % timer.get_relative_percent_time('all.a.aa')) \
        # doctest: +SKIP
    aa % time: 44.144148
    >>> print('aa %% total: %f' % timer.get_total_percent_time('all.a.aa')) \
        # doctest: +SKIP
    aa % total: 35.976058

    When implementing an algorithm, it is often useful to collect detailed
    hierarchical timing information. However, when communicating a timing
    profile, it is often best to retain only the most relevant information
    in a flattened data structure. In the following example, suppose we
    want to compare the time spent in the ``"c"`` and ``"f"`` subroutines.
    We would like to generate a timing profile that displays only the time
    spent in these two subroutines, in a flattened structure so that they
    are easy to compare. To do this, we

    #. Ignore subroutines of ``"c"`` and ``"f"`` that are unnecessary for\
    this comparison

    #. Flatten the hierarchical timing information

    #. Eliminate all the information we don't care about

    >>> import time
    >>> from pyomo.common.timing import HierarchicalTimer
    >>> timer = HierarchicalTimer()
    >>> timer.start("root")
    >>> timer.start("a")
    >>> time.sleep(0.01)
    >>> timer.start("b")
    >>> timer.start("c")
    >>> time.sleep(0.1)
    >>> timer.stop("c")
    >>> timer.stop("b")
    >>> timer.stop("a")
    >>> timer.start("d")
    >>> timer.start("e")
    >>> time.sleep(0.01)
    >>> timer.start("f")
    >>> time.sleep(0.05)
    >>> timer.stop("f")
    >>> timer.start("c")
    >>> timer.start("g")
    >>> timer.start("h")
    >>> time.sleep(0.1)
    >>> timer.stop("h")
    >>> timer.stop("g")
    >>> timer.stop("c")
    >>> timer.stop("e")
    >>> timer.stop("d")
    >>> timer.stop("root")
    >>> print(timer) # doctest: +SKIP
    Identifier                       ncalls   cumtime   percall      %
    ------------------------------------------------------------------
    root                                  1     0.290     0.290  100.0
         -------------------------------------------------------------
         a                                1     0.118     0.118   40.5
              --------------------------------------------------------
              b                           1     0.105     0.105   89.4
                   ---------------------------------------------------
                   c                      1     0.105     0.105  100.0
                   other                n/a     0.000       n/a    0.0
                   ===================================================
              other                     n/a     0.013       n/a   10.6
              ========================================================
         d                                1     0.173     0.173   59.5
              --------------------------------------------------------
              e                           1     0.173     0.173  100.0
                   ---------------------------------------------------
                   c                      1     0.105     0.105   60.9
                        ----------------------------------------------
                        g                 1     0.105     0.105  100.0
                             -----------------------------------------
                             h            1     0.105     0.105  100.0
                             other      n/a     0.000       n/a    0.0
                             =========================================
                        other           n/a     0.000       n/a    0.0
                        ==============================================
                   f                      1     0.055     0.055   31.9
                   other                n/a     0.013       n/a    7.3
                   ===================================================
              other                     n/a     0.000       n/a    0.0
              ========================================================
         other                          n/a     0.000       n/a    0.0
         =============================================================
    ==================================================================
    >>> # Clear subroutines under "c" that we don't care about
    >>> timer.timers["root"].timers["d"].timers["e"].timers["c"].timers.clear()
    >>> # Flatten hierarchy
    >>> timer.timers["root"].flatten()
    >>> # Clear except for the subroutines we care about
    >>> timer.timers["root"].clear_except("c", "f")
    >>> print(timer) # doctest: +SKIP
    Identifier   ncalls   cumtime   percall      %
    ----------------------------------------------
    root              1     0.290     0.290  100.0
         -----------------------------------------
         c            2     0.210     0.105   72.4
         f            1     0.055     0.055   19.0
         other      n/a     0.025       n/a    8.7
         =========================================
    ==============================================

    Notes
    -----

    The :py:class:`HierarchicalTimer` uses a stack to track which timers
    are active at any point in time. Additionally, each timer has a
    dictionary of timers for its children timers. Consider

    >>> timer = HierarchicalTimer()
    >>> timer.start('all')
    >>> timer.start('a')
    >>> timer.start('aa')

    After the above code is run, ``timer.stack`` will be
    ``['all', 'a', 'aa']`` and ``timer.timers`` will have one key,
    ``'all'`` and one value which will be a
    :py:class:`_HierarchicalHelper`. The :py:class:`_HierarchicalHelper`
    has its own timers dictionary:

        ``{'a': _HierarchicalHelper}``

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
                        '.'.join(self.stack + [identifier])
                    )
                )
            parent.timers[identifier] = _HierarchicalHelper()
            return parent.timers[identifier]

    def start(self, identifier):
        """Start incrementing the timer identified with identifier

        Parameters
        ----------
        identifier: str
            The name of the timer
        """
        timer = self._get_timer(identifier)
        timer.start()
        self.stack.append(identifier)

    def stop(self, identifier):
        """Stop incrementing the timer identified with identifier

        Parameters
        ----------
        identifier: str
            The name of the timer
        """
        if identifier != self.stack[-1]:
            raise ValueError(
                str(identifier) + ' is not the currently active timer.  '
                'The only timer that can currently be stopped is '
                + '.'.join(self.stack)
            )
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
        const_indent = 4
        max_name_length = 200 - 36
        stage_identifier_lengths = self._get_identifier_len()
        name_field_width = sum(stage_identifier_lengths)
        if name_field_width > max_name_length:
            # switch to a constant indentation of const_indent spaces
            # (to hopefully shorten the line lengths
            name_field_width = max(
                const_indent * i + l for i, l in enumerate(stage_identifier_lengths)
            )
            for i in range(len(stage_identifier_lengths) - 1):
                stage_identifier_lengths[i] = const_indent
            stage_identifier_lengths[-1] = name_field_width - const_indent * (
                len(stage_identifier_lengths) - 1
            )
        name_formatter = '{name:<' + str(name_field_width) + '}'
        s = (
            name_formatter + '{ncalls:>9} {cumtime:>9} {percall:>9} {percent:>6}\n'
        ).format(
            name='Identifier',
            ncalls='ncalls',
            cumtime='cumtime',
            percall='percall',
            percent='%',
        )
        underline = '-' * (name_field_width + 36) + '\n'
        s += underline
        sub_stage_identifier_lengths = stage_identifier_lengths[1:]
        for name, timer in sorted(self.timers.items()):
            s += (
                name_formatter + '{ncalls:>9d} {cumtime:>9.3f} '
                '{percall:>9.3f} {percent:>6.1f}\n'
            ).format(
                name=name,
                ncalls=timer.n_calls,
                cumtime=timer.total_time,
                percall=timer.total_time / timer.n_calls,
                percent=self.get_total_percent_time(name),
            )
            s += timer.to_str(
                indent=' ' * stage_identifier_lengths[0],
                stage_identifier_lengths=sub_stage_identifier_lengths,
            )
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
        n_calls: int
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

    def flatten(self):
        """Flatten the HierarchicalTimer in-place, moving all the timing
        categories into a single level

        If any timers moved into the same level have the same identifier,
        the ``total_time`` and ``n_calls`` fields are added together.
        The ``total_time`` of a "child timer" that is "moved upwards" is
        subtracted from the ``total_time`` of that timer's original
        parent.

        """
        if self.stack:
            raise RuntimeError(
                "Cannot flatten a HierarchicalTimer while any timers are"
                " active. Current active timer is %s. flatten should only"
                " be called as a post-processing step." % self.stack[-1]
            )
        items = list(self.timers.items())
        for key, timer in items:
            timer.flatten()
            _move_grandchildren_to_root(self, timer)

    def clear_except(self, *args):
        """Prune all "sub-timers" except those specified

        Parameters
        ----------
        args: str
            Keys that will be retained

        """
        if self.stack:
            raise RuntimeError(
                "Cannot clear a HierarchicalTimer while any timers are"
                " active. Current active timer is %s. clear_except should"
                " only be called as a post-processing step." % self.stack[-1]
            )
        to_retain = set(args)
        _clear_timers_except(self, to_retain)
