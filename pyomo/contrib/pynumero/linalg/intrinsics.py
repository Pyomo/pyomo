from pyomo.contrib.pynumero.sparse import BlockVector
import numpy as np

__all__ = ['norm']


def norm(x, ord=None):

    f = np.linalg.norm
    if isinstance(x, np.ndarray):
        return f(x, ord=ord)
    elif isinstance(x, BlockVector):
        flat_x = x.flatten()
        return f(flat_x, ord=ord)
    else:
        raise NotImplementedError()

