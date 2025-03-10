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

from .base_linear_solver_interface import IPLinearSolverInterface
from pyomo.contrib.pynumero.linalg.base import LinearSolverStatus, LinearSolverResults
from pyomo.common.dependencies import attempt_import
from collections import OrderedDict
from typing import Union, Optional, Tuple
from pyomo.contrib.pynumero.sparse import BlockVector
import numpy as np

mumps, mumps_available = attempt_import(
    name='pyomo.contrib.pynumero.linalg.mumps_interface',
    error_message='pymumps is required to use the MumpsInterface',
)

from pyomo.contrib.pynumero.linalg.mumps_interface import (
    MumpsCentralizedAssembledLinearSolver,
)


class MumpsInterface(MumpsCentralizedAssembledLinearSolver, IPLinearSolverInterface):
    @classmethod
    def getLoggerName(cls):
        return 'mumps'

    def __init__(self, par=1, comm=None, cntl_options=None, icntl_options=None):
        if icntl_options is None:
            icntl_options = dict()

        # These options are set in order to get the correct inertia.
        if 13 not in icntl_options:
            icntl_options[13] = 1
        if 24 not in icntl_options:
            icntl_options[24] = 0

        super(MumpsInterface, self).__init__(
            sym=2,
            par=par,
            comm=comm,
            cntl_options=cntl_options,
            icntl_options=icntl_options,
        )

        self.error_level = self.get_icntl(11)
        self.log_error = bool(self.error_level)
        self.logger = self.getLogger()
        self.log_header(include_error=self.log_error)

    def do_back_solve(
        self, rhs: Union[np.ndarray, BlockVector], raise_on_error: bool = True
    ) -> Tuple[Optional[Union[np.ndarray, BlockVector]], LinearSolverResults]:
        res, status = super(MumpsInterface, self).do_back_solve(
            rhs, raise_on_error=raise_on_error
        )
        self.log_info()
        return res, status

    def get_inertia(self):
        num_negative_eigenvalues = self.get_infog(12)
        num_zero_eigenvalues = self.get_infog(28)
        num_positive_eigenvalues = (
            self._dim - num_negative_eigenvalues - num_zero_eigenvalues
        )
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
        fields = []
        fields.append(self.get_infog(1))  # Status, 0 for success
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

        # Allocate 15 spaces for the rest, which I assume are floats
        for i in range(4, len(fields)):
            log_string += '{' + str(i) + ':<15.3e}'

        self.logger.info(log_string.format(*fields))
