from pyomo.core.base.block import BlockData
from enum import Enum
from pyomo.contrib.initialization.utils import get_vars, shallow_clone
from pyomo.common.collections import ComponentMap
from pyomo.contrib.initialization.pwl_init import _initialize_with_piecewise_linear_approximation


class InitializationMethod(Enum):
    pwl_approximation = "pwl_approximation"


def initialize_nlp(
    nlp: BlockData, 
    method: InitializationMethod = InitializationMethod.pwl_approximation
):
    # get all variable bounds, domains, etc. to restore them later
    orig_vars = get_vars(nlp)
    orig_var_data = ComponentMap(
        (v, (v.lower, v.upper, v.domain, v.fixed, v.value)) for v in orig_vars
    )

    # create a shallow clone of the model so that the initialization method can 
    # can work with the original variables but not make any other 
    # modifications to the model
    nlp = shallow_clone(nlp)

    # run the initialization
    if method == InitializationMethod.pwl_approximation:
        _initialize_with_piecewise_linear_approximation(nlp)
    else:
        raise ValueError(f'unexpected initialization method: {method}')    

    # restore variable bounds, domain, etc.
    for v, (lb, ub, domain, fixed, value) in orig_var_data.items():
        v.setlb(lb)
        v.setub(ub)
        v.domain = domain
        if fixed:
            assert v.value == value
            assert v.fixed
        else:
            v.unfix()
