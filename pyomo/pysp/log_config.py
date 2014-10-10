import logging
from pyutilib.misc import LogHandler

from os.path import abspath, dirname
pyomo_base = dirname(dirname(dirname(dirname(abspath(__file__)))))

logger = logging.getLogger('pyomo.pysp')
logger.addHandler( LogHandler(pyomo_base) )
logger.setLevel( logging.WARNING )
