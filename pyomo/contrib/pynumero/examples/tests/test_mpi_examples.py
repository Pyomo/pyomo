#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyomo.common.unittest as unittest

from pyomo.contrib.pynumero.dependencies import (
    numpy_available,
    scipy_available,
    numpy as np,
)

SKIPTESTS = []
if not numpy_available:
    SKIPTESTS.append("Pynumero needs numpy>=1.13.0 to run BlockMatrix tests")
if not scipy_available:
    SKIPTESTS.append("Pynumero needs scipy to run BlockMatrix tests")

try:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    if comm.Get_size() != 3:
        SKIPTESTS.append(
            f"Pynumero MPI examples require 3 MPI processes (got {comm.Get_size()})"
        )
except ImportError:
    SKIPTESTS.append("Pynumero MPI examples require mpi4py")

if not SKIPTESTS:
    from pyomo.contrib.pynumero.examples import parallel_vector_ops, parallel_matvec


@unittest.pytest.mark.mpi
@unittest.skipIf(SKIPTESTS, "\n".join(SKIPTESTS))
class TestExamples(unittest.TestCase):
    def test_parallel_vector_ops(self):
        z1_local, z2, z3 = parallel_vector_ops.main()
        z1_correct = np.array([6, 6, 6, 2, 2, 2, 4, 4, 4, 2, 4, 6])
        self.assertTrue(np.allclose(z1_local, z1_correct))
        self.assertAlmostEqual(z2, 56)
        self.assertAlmostEqual(z3, 3)

    def test_parallel_matvec(self):
        err = parallel_matvec.main()
        self.assertLessEqual(err, 1e-15)
