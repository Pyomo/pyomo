import logging
from pyutilib.misc import LogHandler

from os.path import abspath, dirname
coopr_base = dirname(dirname(dirname(dirname(abspath(__file__)))))

logger = logging.getLogger('coopr.pysp')
logger.addHandler( LogHandler(coopr_base) )
logger.setLevel( logging.WARNING )
