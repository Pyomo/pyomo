#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

try:
    import numpy as np
    numpy_available = True
except ImportError:
    numpy_available = False

try:
    import scipy 
    scipy_available = True
except ImportError:
    scipy_available = False
    
if numpy_available and scipy_available:
    from .base import SparseBase
    from .coo import *
    from .csc import *
    from .csr import *
    from .block_vector import *
    from .block_matrix import *
    from .extract import tril, triu
else:
    if not numpy_available:
        raise ImportError("Install numpy")
    if not scipy_available:
        raise ImportError("Install scipy")
