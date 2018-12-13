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
    from .coo import empty_matrix, diagonal_matrix
    from .block_vector import BlockVector
    from .block_matrix import BlockMatrix, BlockSymMatrix
else:
    import logging
    _logger = logging.getLogger('pyomo.contrib.pynumero.sparse')
    if not numpy_available:
        #raise ImportError("Install numpy")
        _logger.warn("Install numpy to use pynumero")
    if not scipy_available:
        #raise ImportError("Install scipy")
        _logger.warn("Install scipy to use pynumero")
