from abc import ABCMeta, abstractmethod
import six


class LinearSolverInterface(six.with_metaclass(ABCMeta, object)):
    @abstractmethod
    def do_symbolic_factorization(self, matrix):
        pass

    @abstractmethod
    def do_numeric_factorization(self, matrix):
        pass

    @abstractmethod
    def do_back_solve(self, rhs):
        pass

    @abstractmethod
    def get_inertia(self):
        pass

    def log_header(self):
        pass

    def log_info(self):
        pass
