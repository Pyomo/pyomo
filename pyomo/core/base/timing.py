
import logging
from pyutilib.misc.timing import TicTocTimer

_logger = logging.getLogger('pyomo.core.base.timing')
_logger.propagate = False
_logger.setLevel(logging.INFO)

_construction_logger = logging.getLogger('pyomo.core.base.timing.construction')
class ConstructionTimer(object):
    def __init__(self, obj):
        self.obj = obj
        self.timer = TicTocTimer()

    def report(self):
        self.timer = self.timer.toc(msg="")
        _logger.info(self)

    def __str__(self):
        fmt = "%%6.%df seconds to construct %s %s; %d %s total"
        total_time = self.timer
        idx = len(self.obj.index_set())
        return fmt % ( 2 if total_time>=0.005 else 0,
                       self.obj.type().__name__,
                       self.obj.name,
                       idx,
                       'indicies' if idx > 1 else 'index' ) \
            % total_time
