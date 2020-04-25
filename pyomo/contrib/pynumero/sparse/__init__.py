#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from ..dependencies import numpy_available, scipy_available

if numpy_available and scipy_available:
    from .block_vector import BlockVector, NotFullyDefinedBlockVectorError
    from .block_matrix import BlockMatrix, NotFullyDefinedBlockMatrixError
