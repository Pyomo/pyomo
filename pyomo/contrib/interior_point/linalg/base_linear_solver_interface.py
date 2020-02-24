from collections.abc import ABCMeta, abstractmethod
import six


class LinearSolverInterface(six.with_metaclass(ABCMeta, object)):
    @abstractmethod
    def do_symbolic_factorization(matrix):
        pass

    @abstractmethod
    def do_numeric_factorization(matrix):
        pass

    @abstractmethod
    def do_back_solve(rhs):
        pass

    @abc.abstractmethod
    def get_inertia():
        pass
    
