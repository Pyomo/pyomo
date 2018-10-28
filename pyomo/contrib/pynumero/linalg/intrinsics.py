#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
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

