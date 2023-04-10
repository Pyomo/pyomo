import pyomo.common.unittest as unittest
from pyomo.common.dependencies import attempt_import

np, numpy_available = attempt_import(
    'numpy', 'Interior point requires numpy', minimum_version='1.13.0'
)
scipy, scipy_available = attempt_import('scipy', 'Interior point requires scipy')
mumps, mumps_available = attempt_import('mumps')
if not (numpy_available and scipy_available):
    raise unittest.SkipTest('Interior point tests require numpy and scipy')
from scipy.sparse import coo_matrix
import pyomo.contrib.interior_point as ip

if mumps_available:
    from pyomo.contrib.interior_point.linalg.mumps_interface import MumpsInterface
from pyomo.contrib.pynumero.linalg.base import LinearSolverStatus


@unittest.skipIf(not mumps_available, 'mumps is not available')
class TestReallocation(unittest.TestCase):
    def test_reallocate_memory_mumps(self):
        # Create a tri-diagonal matrix with small entries on the diagonal
        n = 10000
        small_val = 1e-7
        big_val = 1e2
        irn = []
        jcn = []
        ent = []
        for i in range(n - 1):
            irn.extend([i + 1, i, i])
            jcn.extend([i, i, i + 1])
            ent.extend([big_val, small_val, big_val])
        irn.append(n - 1)
        jcn.append(n - 1)
        ent.append(small_val)
        irn = np.array(irn)
        jcn = np.array(jcn)
        ent = np.array(ent)

        matrix = coo_matrix((ent, (irn, jcn)), shape=(n, n))

        linear_solver = MumpsInterface()
        linear_solver.do_symbolic_factorization(matrix)

        predicted = linear_solver.get_infog(16)

        res = linear_solver.do_numeric_factorization(matrix, raise_on_error=False)
        self.assertEqual(res.status, LinearSolverStatus.not_enough_memory)

        linear_solver.do_symbolic_factorization(matrix)

        factor = 2
        linear_solver.increase_memory_allocation(factor)

        res = linear_solver.do_numeric_factorization(matrix)
        self.assertEqual(res.status, LinearSolverStatus.successful)

        # Expected memory allocation (MB)
        self.assertEqual(linear_solver._prev_allocation, 2 * predicted)

        actual = linear_solver.get_infog(18)

        # Sanity checks:
        # Make sure actual memory usage is greater than initial guess
        self.assertTrue(predicted < actual)
        # Make sure memory allocation is at least as much as was used
        self.assertTrue(actual <= linear_solver._prev_allocation)


if __name__ == '__main__':
    test_realloc = TestReallocation()
    test_realloc.test_reallocate_memory_mumps()
