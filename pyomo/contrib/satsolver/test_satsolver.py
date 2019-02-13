import logging

from six import StringIO
from six.moves import range

import pyutilib.th as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.contrib.satsolver import SMTSatSolver
from pyomo.environ import *
