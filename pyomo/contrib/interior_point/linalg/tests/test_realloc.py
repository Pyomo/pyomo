import pyutilib.th as unittest
from pyomo.common.dependencies import attempt_import

np, numpy_available = attempt_import('numpy', 'Interior point requires numpy',
        minimum_version='1.13.0')
scipy, scipy_available = attempt_import('scipy', 'Interior point requires scipy')
mumps_interface, mumps_available = attempt_import(
        'pyomo.contrib.interior_point.linalg.mumps_interface',
        'Interior point requires mumps')
if not (numpy_available and scipy_available):
    raise unittest.SkipTest('Interior point tests require numpy and scipy')

from pyomo.contrib.pynumero.extensions.asl import AmplInterface
asl_available = AmplInterface.available()

from pyomo.contrib.interior_point.interior_point import InteriorPointSolver
from pyomo.contrib.interior_point.interface import InteriorPointInterface

from scipy.sparse import coo_matrix


class TestReallocation(unittest.TestCase):
    @unittest.skipIf(not asl_available, 'asl is not available')
    @unittest.skipIf(not mumps_available, 'mumps is not available')
    def test_reallocate_memory(self):
        interface = InteriorPointInterface('realloc.nl')
        '''This NLP is the steady state optimization of a moving bed
        chemical looping reduction reactor.'''

        linear_solver = mumps_interface.MumpsInterface()
        linear_solver.allow_reallocation = True
        ip_solver = InteriorPointSolver(linear_solver)

        x, duals_eq, duals_ineq = ip_solver.solve(interface, max_iter=5)

        # Predicted memory requirement after symbolic factorization
        init_alloc = linear_solver.get_infog(16)

        # Maximum memory allocation (presumably after reallocation)
        # Stored in icntl(23), here accessed with C indexing:
        realloc = linear_solver._mumps.mumps.id.icntl[22]

        # Actual memory used:
        i_actually_used = linear_solver.get_infog(18) # Integer
        r_actually_used = linear_solver.get_rinfog(18) # Real

        # Sanity check:
        self.assertEqual(round(r_actually_used), i_actually_used)
        self.assertTrue(init_alloc <= r_actually_used and
                        r_actually_used <= realloc)

        # Expected memory allocation in MB:
        self.assertEqual(init_alloc, 2)
        self.assertEqual(realloc, 4)

        # Repeat, this time without reallocation
        interface = InteriorPointInterface('realloc.nl')

        # Reduce maximum memory allocation
        linear_solver.set_icntl(23, 2)
        linear_solver.allow_reallocation = False

        with self.assertRaises(RuntimeError):
            # Should be Mumps error: -9
            x, duals_eq, duals_ineq = ip_solver.solve(interface, max_iter=5)
            

    @unittest.skipIf(not mumps_available, 'mumps is not available')
    def test_reallocate_matrix_only(self):
        irn = np.array([0,1,2,3,4,5,6,7,8,9,0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9])
        jcn = np.array([0,1,2,3,4,5,6,7,8,9,1,9,2,8,3,7,4,6,5,4,6,4,7,3,8,2,9,1,0,1])
        ent = np.array([0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,
                        1.,3.,5.,7.,9.,2.,4.,6.,8.,1.,
                        1.1,2.2,3.3,4.4,5.5,6.6,7.7,8.8,9.9,0.1])

        matrix = coo_matrix((ent, (irn, jcn)), shape=(10,10))

        linear_solver = mumps_interface.MumpsInterface()
        linear_solver.do_symbolic_factorization(matrix)
        linear_solver.do_numeric_factorization(matrix)

        import pdb; pdb.set_trace()


if __name__ == '__main__':
    test_realloc = TestReallocation()
    test_realloc.test_reallocate_memory()
    test_realloc.test_reallocate_matrix_only()
