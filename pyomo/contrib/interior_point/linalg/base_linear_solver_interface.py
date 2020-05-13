from abc import ABCMeta, abstractmethod
import six
import logging


class LinearSolverInterface(six.with_metaclass(ABCMeta, object)):
    @classmethod
    def getLoggerName(cls):
        return 'linear_solver'

    @classmethod
    def getLogger(cls):
        name = 'interior_point.' + cls.getLoggerName()
        return logging.getLogger(name)

    @abstractmethod
    def do_symbolic_factorization(self, matrix, raise_on_error=True):
        pass

    @abstractmethod
    def do_numeric_factorization(self, matrix, raise_on_error=True):
        pass

    @abstractmethod
    def do_back_solve(self, rhs):
        pass

    @abstractmethod
    def get_inertia(self):
        pass
