#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2024
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from abc import ABCMeta, abstractmethod
import enum
from typing import Optional, Union, Tuple
from scipy.sparse import spmatrix
import numpy as np

from pyomo.contrib.pynumero.sparse.base_block import BaseBlockVector, BaseBlockMatrix

try:
    from pyomo.contrib.pynumero.sparse import BlockVector, BlockMatrix
except ImportError as e:
    print("IMPORT ERROR: ", e)
    print("Current environment information...")
    import sys
    import platform
    import pkg_resources

    print(f"Python version: {platform.python_version()}")
    print(f"Python executable: {sys.executable}")
    print(f"Platform: {platform.system()} {platform.release()} ({platform.platform()})")

    print("\nInstalled packages:")
    installed_packages = pkg_resources.working_set
    installed_packages_list = sorted(
        [f"{pkg.key}=={pkg.version}" for pkg in installed_packages]
    )
    print("\n".join(installed_packages_list))

    print("\nImported packages:")
    imported_packages = sorted(sys.modules.keys())
    print("\n".join(imported_packages))
    raise e


class LinearSolverStatus(enum.Enum):
    successful = 0
    not_enough_memory = 1
    singular = 2
    error = 3
    warning = 4
    max_iter = 5


class LinearSolverResults(object):
    def __init__(self, status: Optional[LinearSolverStatus] = None):
        self.status = status


class LinearSolverInterface(object, metaclass=ABCMeta):
    @abstractmethod
    def solve(
        self,
        matrix: Union[spmatrix, BlockMatrix],
        rhs: Union[np.ndarray, BlockVector],
        raise_on_error: bool = True,
    ) -> Tuple[Optional[Union[np.ndarray, BlockVector]], LinearSolverResults]:
        pass


class DirectLinearSolverInterface(LinearSolverInterface, metaclass=ABCMeta):
    @abstractmethod
    def do_symbolic_factorization(
        self, matrix: Union[spmatrix, BlockMatrix], raise_on_error: bool = True
    ) -> LinearSolverResults:
        pass

    @abstractmethod
    def do_numeric_factorization(
        self, matrix: Union[spmatrix, BlockMatrix], raise_on_error: bool = True
    ) -> LinearSolverResults:
        pass

    @abstractmethod
    def do_back_solve(
        self, rhs: Union[np.ndarray, BlockVector], raise_on_error: bool = True
    ) -> Tuple[Optional[Union[np.ndarray, BlockVector]], LinearSolverResults]:
        pass

    def solve(
        self,
        matrix: Union[spmatrix, BlockMatrix],
        rhs: Union[np.ndarray, BlockVector],
        raise_on_error: bool = True,
    ) -> Tuple[Optional[Union[np.ndarray, BlockVector]], LinearSolverResults]:
        symbolic_res = self.do_symbolic_factorization(
            matrix, raise_on_error=raise_on_error
        )
        if symbolic_res.status != LinearSolverStatus.successful:
            return None, symbolic_res
        numeric_res = self.do_numeric_factorization(
            matrix, raise_on_error=raise_on_error
        )
        if numeric_res.status != LinearSolverStatus.successful:
            return None, numeric_res
        return self.do_back_solve(rhs, raise_on_error=raise_on_error)
