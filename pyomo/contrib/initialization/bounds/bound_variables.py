from pyomo.core.base.block import BlockData
from pyomo.contrib.fbbt.fbbt import fbbt
from pyomo.contrib.initialization.utils import get_vars
import logging


logger = logging.getLogger(__name__)


def bound_all_nonlinear_variables(m: BlockData, default_bound: float = 1.0e8):
    """
    Attempt to obtain valid bounds on all nonlinear variables based on the 
    constraints in the model, m. If variable bounds cannot be obtained, 
    we use default_bound.
    """
    fbbt(m)
    for v in get_vars(m):
        if v.lb is None or v.lb < -default_bound:
            logger.warning(f'Could not obtain a lower bound for {str(v)} better than {-default_bound}; setting the lower bound to {-default_bound}')
            v.setlb(-default_bound)
        if v.ub is None or v.ub > default_bound:
            logger.warning(f'Could not obtain an upper bound for {str(v)} better than {default_bound}; setting the upper bound to {default_bound}')
            v.setub(default_bound)
    fbbt(m)
