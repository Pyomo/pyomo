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
