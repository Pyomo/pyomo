import sys
import logging
from pyutilib.misc.timing import TicTocTimer

_logger = logging.getLogger('pyomo.util.timing')
_logger.propagate = False
_logger.setLevel(logging.WARNING)

def report_timing(stream=True):
    if stream:
        _logger.setLevel(logging.INFO)
        if stream is True:
            stream = sys.stdout
        handler = logging.StreamHandler(stream)
        _logger.addHandler(handler)
        return handler
    else:
        _logger.setLevel(logging.WARNING)
        for h in _logger.handlers:
            _logger.removeHandler(h)

_construction_logger = logging.getLogger('pyomo.util.timing.construction')
class ConstructionTimer(object):
    def __init__(self, obj):
        self.obj = obj
        self.timer = TicTocTimer()

    def report(self):
        # Record the elapsed time, as some log handlers may not
        # immediately generate the messge string
        self.timer = self.timer.toc(msg="")
        _logger.info(self)

    def __str__(self):
        fmt = "%%6.%df seconds to construct %s %s; %d %s total"
        total_time = self.timer
        idx = len(self.obj.index_set())
        try:
            return fmt % ( 2 if total_time>=0.005 else 0,
                           self.obj.type().__name__,
                           self.obj.name,
                           idx,
                           'indicies' if idx > 1 else 'index' ) \
                % total_time
        except TypeError:
            return "ConstructionTimer object for %s %s; %s elapsed seconds" % (
                self.obj.type().__name__,
                self.obj.name,
                self.timer.toc("") )
