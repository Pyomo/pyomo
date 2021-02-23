#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from .base_linear_solver_interface import LinearSolverInterface
from .results import LinearSolverStatus, LinearSolverResults
from pyomo.common.dependencies import attempt_import
from scipy.sparse import isspmatrix_coo, tril
from collections import OrderedDict

mumps, mumps_available = attempt_import(name='pyomo.contrib.pynumero.linalg.mumps_interface',
                                        error_message='pymumps is required to use the MumpsInterface')


class MumpsInterface(LinearSolverInterface):

    @classmethod
    def getLoggerName(cls):
        return 'mumps'

    def __init__(self, par=1, comm=None, cntl_options=None, icntl_options=None):
        self._mumps = mumps.MumpsCentralizedAssembledLinearSolver(sym=2,
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
        
        self.error_level = self.get_icntl(11)
        self.log_error = bool(self.error_level)
        self._dim = None
        self.logger = self.getLogger()
        self.log_header(include_error=self.log_error)
        self._prev_allocation = None

    def do_symbolic_factorization(self, matrix, raise_on_error=True):
        if not isspmatrix_coo(matrix):
            matrix = matrix.tocoo()
        matrix = tril(matrix)
        nrows, ncols = matrix.shape
        self._dim = nrows

        try:
            self._mumps.do_symbolic_factorization(matrix)
            self._prev_allocation = max(self.get_infog(16),
                                        self.get_icntl(23))
            # INFOG(16) is the Mumps estimate for memory usage; ICNTL(23)
            # is the override used in increase_memory_allocation. Both are
            # already rounded to MB, so neither should every be negative.
        except RuntimeError as err:
            if raise_on_error:
                raise err

        stat = self.get_infog(1)
        res = LinearSolverResults()
        if stat == 0:
            res.status = LinearSolverStatus.successful
        elif stat in {-6, -10}:
            res.status = LinearSolverStatus.singular
        elif stat < 0:
            res.status = LinearSolverStatus.error
        else:
            res.status = LinearSolverStatus.warning
        return res

    def do_numeric_factorization(self, matrix, raise_on_error=True):
        if not isspmatrix_coo(matrix):
            matrix = matrix.tocoo()
        matrix = tril(matrix)
        try:
            self._mumps.do_numeric_factorization(matrix)
        except RuntimeError as err:
            if raise_on_error:
                raise err

        stat = self.get_infog(1)
        res = LinearSolverResults()
        if stat == 0:
            res.status = LinearSolverStatus.successful
        elif stat in {-6, -10}:
            res.status = LinearSolverStatus.singular
        elif stat in {-8, -9}:
            res.status = LinearSolverStatus.not_enough_memory
        elif stat < 0:
            res.status = LinearSolverStatus.error
        else:
            res.status = LinearSolverStatus.warning
        return res

    def increase_memory_allocation(self, factor):
        # info(16) is rounded to the nearest MB, so it could be zero
        if self._prev_allocation == 0:
            new_allocation = 1
        else:
            new_allocation = int(factor*self._prev_allocation)
        # Here I set the memory allocation directly instead of increasing
        # the "percent-increase-from-predicted" parameter ICNTL(14)
        self.set_icntl(23, new_allocation)
        self._prev_allocation = new_allocation
        return new_allocation

    def do_back_solve(self, rhs):
        res = self._mumps.do_back_solve(rhs)
        self.log_info()
        return res

    def get_inertia(self):
        num_negative_eigenvalues = self.get_infog(12)
        num_zero_eigenvalues = self.get_infog(28)
        num_positive_eigenvalues = self._dim - num_negative_eigenvalues - num_zero_eigenvalues
        return num_positive_eigenvalues, num_negative_eigenvalues, num_zero_eigenvalues

    def get_error_info(self):
        # Access error level contained in ICNTL(11) (Fortran indexing).
        # Assuming this value has not changed since the solve was performed.
        error_level = self.get_icntl(11)
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
        value = int(value)
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

    def get_icntl(self, key):
        return self._mumps.get_icntl(key)

    def get_cntl(self, key):
        return self._mumps.get_cntl(key)

    def get_info(self, key):
        return self._mumps.get_info(key)

    def get_infog(self, key):
        return self._mumps.get_infog(key)

    def get_rinfo(self, key):
        return self._mumps.get_rinfo(key)

    def get_rinfog(self, key):
        return self._mumps.get_rinfog(key)

    def log_header(self, include_error=True, extra_fields=None):
        if extra_fields is None:
            extra_fields = list()
        header_fields = []
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

        # Allocate 15 spaces for the rest, which I assume are floats
        for i in range(4, len(header_fields)):
            header_string += '{' + str(i) + ':<15}'

        self.logger.info(header_string.format(*header_fields))

    def log_info(self):
        # Which fields to log should be specified at the instance level
        # Any logging that should be done on an iteration-specific case
        # should be handled by the IP solver
        fields=[]
        fields.append(self.get_infog(1))   # Status, 0 for success
        fields.append(self.get_infog(28))  # Number of null pivots
        fields.append(self.get_infog(12))  # Number of negative pivots

        include_error = self.log_error
        if include_error:
            fields.extend(self.get_error_info().values())

        extra_fields = []
        fields.extend(extra_fields)

        # Allocate 10 spaces for integer values
        log_string = '{0:<10}'
        log_string += '{1:<10}'
        log_string += '{2:<10}'

        # Allocate 15 spsaces for the rest, which I assume are floats
        for i in range(4, len(fields)):
            log_string += '{' + str(i) + ':<15.3e}'

        self.logger.info(log_string.format(*fields))

