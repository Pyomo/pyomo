import logging
from pyutilib.misc import LogHandler

from os.path import abspath, dirname
coopr_base = dirname(dirname(dirname(dirname(dirname(abspath(__file__))))))

logger = logging.getLogger('coopr.opt')
logger.addHandler( LogHandler(coopr_base) )
logger.setLevel( logging.WARNING )

class NullHandler(logging.Handler):
    def emit(self, record):         #pragma:nocover
        """Do not generate logging record"""

# TODO: move this into a coopr.common project
logger = logging.getLogger('coopr')
logger.addHandler( NullHandler() )
logger.setLevel( logging.WARNING )

logger = logging.getLogger('coopr.solvers')
logger.addHandler( LogHandler(coopr_base) )
logger.setLevel( logging.WARNING )
