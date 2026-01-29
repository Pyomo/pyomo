from pyomo.core.base.block import BlockData


def bound_all_nonlinear_variables(m: BlockData, default_bound: float = 1.0e8):
    """
    Attempt to obtain valid bounds on all nonlinear variables based on the 
    constraints in the model, m. If variable bounds cannot be obtained, 
    we use default_bound.
    """
    raise NotImplementedError('not done yet')
