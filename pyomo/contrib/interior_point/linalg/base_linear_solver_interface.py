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
    def is_numerically_singular(self, err=None, raise_if_not=True):
        pass

    @abstractmethod
    def get_inertia(self):
        pass

    def set_outer_iteration_number(self, num):
        pass

    def set_regularization_switch(self, reg_switch):
        pass

    def set_reg_coef(self, reg_coef):
        pass
