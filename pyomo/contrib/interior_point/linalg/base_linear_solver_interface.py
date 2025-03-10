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

from pyomo.contrib.pynumero.linalg.base import DirectLinearSolverInterface
from abc import ABCMeta, abstractmethod
import logging


class IPLinearSolverInterface(DirectLinearSolverInterface, metaclass=ABCMeta):
    @classmethod
    def getLoggerName(cls):
        return 'linear_solver'

    @classmethod
    def getLogger(cls):
        name = 'interior_point.' + cls.getLoggerName()
        return logging.getLogger(name)

    def increase_memory_allocation(self, factor):
        raise NotImplementedError('Should be implemented by base class.')

    @abstractmethod
    def get_inertia(self):
        pass
