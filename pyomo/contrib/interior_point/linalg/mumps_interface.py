from .base_linear_solver_interface import LinearSolverInterface
from pyomo.contrib.pynumero.linalg.mumps_solver import MumpsCentralizedAssembledLinearSolver
from scipy.sparse import isspmatrix_coo, tril
from collections import OrderedDict
import logging


class MumpsInterface(LinearSolverInterface):
    def __init__(self, par=1, comm=None, cntl_options=None, icntl_options=None,
                 log_filename=None, allow_reallocation=False):
        self._mumps = MumpsCentralizedAssembledLinearSolver(sym=2,
                                                            par=par,
                                                            comm=comm)

        if cntl_options is None:
            cntl_options = dict()
        if icntl_options is None:
            icntl_options = dict()

        # These options are set in order to get the correct inertia.
        if 13 not in icntl_options:
            icntl_options[13] = 1
        if 24 not in icntl_options:
            icntl_options[24] = 0
            
        for k, v in cntl_options.items():
            self.set_cntl(k, v)
        for k, v in icntl_options.items():
            self.set_icntl(k, v)

        self._dim = None

        self.logger = logging.getLogger('mumps')
        self.logger.propagate = False
        if log_filename:
            self.log_switch = True 
            open(log_filename, 'w').close()
            self.logger.setLevel(logging.DEBUG)

            fh = logging.FileHandler(log_filename)
            fh.setLevel(logging.DEBUG)
            self.logger.addHandler(fh)
            # Now logger will not propagate to the root logger.
            # This is probably bad because I might want to 
            # propagate ERROR to the console, but I can't figure
            # out how to disable console logging otherwise

        self.allow_reallocation = allow_reallocation
        self._prev_allocation = None
        # Max number of reallocations per iteration:
        self.max_num_realloc = 5
        # TODO: Should probably set more reallocation options here,
        #       and allow the user to specify them.
        #       (e.g. max memory usage)

    def do_symbolic_factorization(self, matrix):
        if not isspmatrix_coo(matrix):
            matrix = matrix.tocoo()
        matrix = tril(matrix)
        nrows, ncols = matrix.shape
        self._dim = nrows

        self._mumps.do_symbolic_factorization(matrix)
        self._prev_allocation = self.get_infog(16)

    def do_numeric_factorization(self, matrix):
        if not isspmatrix_coo(matrix):
            matrix = matrix.tocoo()
        matrix = tril(matrix)

        if not self.allow_reallocation:
            self._mumps.do_numeric_factorization(matrix)
        else:
            success = False
            for count in range(self.max_num_realloc):
                try:
                    self._mumps.do_numeric_factorization(matrix)
                    success = True
                    break
                except RuntimeError as err:
                    # What is the proper error to indicate that numeric
                    # factorization needs reallocation?
                    msg = str(err)
                    if ('MUMPS error: -9' not in msg and 
                        'MUMPS error: -8' not in msg):
                        raise

                    status = self.get_infog(1)
                    if status != -8 and status != -9:
                        raise

                    # Increase the amount of memory allocated to this
                    # factorization.
                    new_allocation = self.increase_memory_allocation()

                    # Should probably handle propagation with a context manager
                    self.logger.propagate = True
                    self.logger.info(
                            'Reallocating memory for MUMPS Linear Solver. '
                            'New memory allocation is ' + str(new_allocation)
                            + ' MB.')
                    self.logger.propagate = False
                    
            if not success:
                raise RuntimeError(
                        'Maximum number of reallocations exceeded in the '
                        'numeric factorization.')

    def increase_memory_allocation(self):
        new_allocation = 2*self._prev_allocation
        self._prev_allocation = new_allocation

        # Here I set the memory allocation directly instead of increasing
        # the "percent-increase-from-predicted" parameter ICNTL(14)
        self.set_icntl(23, new_allocation)
        return new_allocation

    def try_factorization(self, kkt):
        try:
            self.do_symbolic_factorization(kkt)
            self.do_numeric_factorization(kkt)
        except RuntimeError as err:
            return err
        return None

    def is_numerically_singular(self, err=None, raise_if_not=True):
        num_sing_err = True
        if err:
            # -6: Structural singularity in symbolic factorization
            # -10: Singularity in numeric factorization
            if ('MUMPS error: -10' not in str(err) and
                'MUMPS error: -6' not in str(err)):
                num_sing_err = False
                if raise_if_not:
                    raise err
        status = self.get_info(1)
        if status == -10 or status == -6:
            # Only return True if status and error both imply singularity
            return True and num_sing_err
        else:
            return False

    def do_back_solve(self, rhs):
        return self._mumps.do_back_solve(rhs)

    def get_inertia(self):
        num_negative_eigenvalues = self.get_infog(12)
        num_positive_eigenvalues = self._dim - num_negative_eigenvalues
        return (num_positive_eigenvalues, num_negative_eigenvalues, 0)

    def get_error_info(self):
        # Access error level contained in ICNTL(11) (Fortran indexing).
        # Assuming this value has not changed since the solve was performed.
        error_level = self._mumps.mumps.id.icntl[10]
        info = OrderedDict()
        if error_level == 0:
            return info
        elif error_level == 1:
            info['||A||'] = self.get_rinfog(4)
            info['||x||'] = self.get_rinfog(5)
            info['Max resid'] = self.get_rinfog(6)
            info['Max error'] = self.get_rinfog(9)
            return info
        elif error_level == 2:
            info['||A||'] = self.get_rinfog(4)
            info['||x||'] = self.get_rinfog(5)
            info['Max resid'] = self.get_rinfog(6)
            return info

    def set_icntl(self, key, value):
        if key == 13:
            if value <= 0:
                raise ValueError(
                    'ICNTL(13) must be positive for the MumpsInterface.')
        elif key == 24:
            if value != 0:
                raise ValueError(
                    'ICNTL(24) must be 0 for the MumpsInterface.')
        self._mumps.set_icntl(key, value)

    def set_cntl(self, key, value):
        self._mumps.set_cntl(key, value)

    def get_info(self, key):
        return self._mumps.get_info(key)

    def get_infog(self, key):
        return self._mumps.get_infog(key)

    def get_rinfo(self, key):
        return self._mumps.get_rinfo(key)

    def get_rinfog(self, key):
        return self._mumps.get_rinfog(key)

    def log_header(self, include_error=True, extra_fields=[]):
        header_fields = []
        header_fields.append('Iter')
        header_fields.append('Status')
        header_fields.append('n_null')
        header_fields.append('n_neg')

        if include_error:
            header_fields.extend(self.get_error_info().keys())

        header_fields.extend(extra_fields)

        # Allocate 10 spaces for integer values
        header_string = '{0:<10}'
        header_string += '{1:<10}'
        header_string += '{2:<10}'
        header_string += '{3:<10}'

        # Allocate 15 spsaces for the rest, which I assume are floats
        for i in range(4, len(header_fields)):
            header_string += '{' + str(i) + ':<15}'

        self.logger.debug(header_string.format(*header_fields))

    def log_info(self, iter_no='', include_error=True, extra_fields=[]):
        fields = [iter_no]
        fields.append(self.get_infog(1))   # Status, 0 for success
        fields.append(self.get_infog(28))  # Number of null pivots
        fields.append(self.get_infog(12))  # Number of negative pivots

        if include_error:
            fields.extend(self.get_error_info().values())

        fields.extend(extra_fields)

        # Allocate 10 spaces for integer values
        log_string = '{0:<10}'
        log_string += '{1:<10}'
        log_string += '{2:<10}'
        log_string += '{3:<10}'

        # Allocate 15 spsaces for the rest, which I assume are floats
        for i in range(4, len(fields)):
            log_string += '{' + str(i) + ':<15.3e}'

        self.logger.debug(log_string.format(*fields))

