import sys
import logging
from pyutilib.misc.timing import TicTocTimer

_logger = logging.getLogger('pyomo.common.timing')
_logger.propagate = False
_logger.setLevel(logging.WARNING)

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
        self.timer = self.timer.toc(msg="")
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
            _type = self.obj.type().__name__
        except AttributeError:
            _type = type(self.obj).__name__
        try:
            return self.fmt % ( 2 if total_time>=0.005 else 0,
                                _type,
                                name,
                                idx,
                                'indicies' if idx > 1 else 'index',
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
        self.timer = self.timer.toc(msg="")
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
