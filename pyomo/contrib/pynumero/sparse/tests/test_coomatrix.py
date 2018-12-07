#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
import sys
import os
import pyutilib.th as unittest

try:
    from scipy.sparse import csr_matrix, csc_matrix, coo_matrix, identity
    import numpy as np
except ImportError:
    raise unittest.SkipTest(
        "Pynumero needs scipy and numpy to run COO matrix tests")

from pyomo.contrib.pynumero.sparse.coo import (diagonal_matrix,
                                               empty_matrix)

@unittest.skipIf(os.name in ['nt', 'dos'], "Do not test on windows")
class TestEmptyMatrix(unittest.TestCase):

    def test_constructor(self):

        m = empty_matrix(3, 3)
        self.assertEqual(m.shape, (3, 3))
        self.assertEqual(m.nnz, 0)





