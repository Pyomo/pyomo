from abc import ABCMeta, abstractmethod
import six
from pyomo.contrib.pynumero.interfaces import pyomo_nlp, ampl_nlp
from pyomo.contrib.pynumero.interfaces.utils import build_bounds_mask, build_compression_matrix
from pyomo.contrib.pynumero.sparse import BlockMatrix, BlockVector
import numpy as np
import scipy.sparse


class BaseInteriorPointInterface(six.with_metaclass(ABCMeta, object)):
    @abstractmethod
    def init_primals(self):
        pass

    @abstractmethod
    def init_slacks(self):
        pass

    @abstractmethod
    def init_duals_eq(self):
        pass

    @abstractmethod
    def init_duals_ineq(self):
        pass

    @abstractmethod
    def init_duals_primals_lb(self):
        pass
    
    @abstractmethod
    def init_duals_primals_ub(self):
        pass

    @abstractmethod
    def init_duals_slacks_lb(self):
        pass
    
    @abstractmethod
    def init_duals_slacks_ub(self):
        pass
    
    @abstractmethod
    def set_primals(self, primals):
        pass

    @abstractmethod
    def set_slacks(self, slacks):
        pass

    @abstractmethod
    def set_duals_eq(self, duals):
        pass

    @abstractmethod
    def set_duals_ineq(self, duals):
        pass

    @abstractmethod
    def set_duals_primals_lb(self, duals):
        pass

    @abstractmethod
    def set_duals_primals_ub(self, duals):
        pass

    @abstractmethod
    def set_duals_slacks_lb(self, duals):
        pass

    @abstractmethod
    def set_duals_slacks_ub(self, duals):
        pass
    
    @abstractmethod
    def get_primals(self):
        pass

    @abstractmethod
    def get_slacks(self):
        pass

    @abstractmethod
    def get_duals_eq(self):
        pass

    @abstractmethod
    def get_duals_ineq(self):
        pass

    @abstractmethod
    def get_duals_primals_lb(self):
        pass

    @abstractmethod
    def get_duals_primals_ub(self):
        pass

    @abstractmethod
    def get_duals_slacks_lb(self):
        pass

    @abstractmethod
    def get_duals_slacks_ub(self):
        pass

    @abstractmethod
    def get_primals_lb(self):
        pass

    @abstractmethod
    def get_primals_ub(self):
        pass

    @abstractmethod
    def get_ineq_lb(self):
        pass

    @abstractmethod
    def get_ineq_ub(self):
        pass

    @abstractmethod
    def set_barrier_parameter(self, barrier):
        pass

    @abstractmethod
    def evaluate_primal_dual_kkt_matrix(self):
        pass

    @abstractmethod
    def evaluate_primal_dual_kkt_rhs(self):
        pass

    @abstractmethod
    def set_primal_dual_kkt_solution(self, sol):
        pass

    @abstractmethod
    def get_delta_primals(self):
        pass

    @abstractmethod
    def get_delta_slacks(self):
        pass

    @abstractmethod
    def get_delta_duals_eq(self):
        pass

    @abstractmethod
    def get_delta_duals_ineq(self):
        pass

    @abstractmethod
    def get_delta_duals_primals_lb(self):
        pass

    @abstractmethod
    def get_delta_duals_primals_ub(self):
        pass

    @abstractmethod
    def get_delta_duals_slacks_lb(self):
        pass

    @abstractmethod
    def get_delta_duals_slacks_ub(self):
        pass

    @abstractmethod
    def evaluate_objective(self):
        pass

    @abstractmethod
    def evaluate_eq_constraints(self):
        pass

    @abstractmethod
    def evaluate_ineq_constraints(self):
        pass

    @abstractmethod
    def evaluate_grad_objective(self):
        pass

    @abstractmethod
    def evaluate_jacobian_eq(self):
        pass

    @abstractmethod
    def evaluate_jacobian_ineq(self):
        pass

    @abstractmethod
    def get_primals_lb_compression_matrix(self):
        pass

    @abstractmethod
    def get_primals_ub_compression_matrix(self):
        pass

    @abstractmethod
    def get_ineq_lb_compression_matrix(self):
        pass

    @abstractmethod
    def get_ineq_ub_compression_matrix(self):
        pass

    @abstractmethod
    def get_primals_lb_compressed(self):
        pass

    @abstractmethod
    def get_primals_ub_compressed(self):
        pass

    @abstractmethod
    def get_ineq_lb_compressed(self):
        pass

    @abstractmethod
    def get_ineq_ub_compressed(self):
        pass

    # These should probably be methods of some InteriorPointSolver class
    def regularize_equality_gradient(self):
        raise RuntimeError(
            'Equality gradient regularization is necessary but no '
            'function has been implemented for doing so.')

    def regularize_hessian(self):
        raise RuntimeError(
            'Hessian of Lagrangian regularization is necessary but no '
            'function has been implemented for doing so.')


class InteriorPointInterface(BaseInteriorPointInterface):
    def __init__(self, pyomo_model):
        if type(pyomo_model) is str:
            # Assume argument is the name of an nl file
            self._nlp = ampl_nlp.AmplNLP(pyomo_model)
        else:
            self._nlp = pyomo_nlp.PyomoNLP(pyomo_model)
        lb = self._nlp.primals_lb()
        ub = self._nlp.primals_ub()
        self._primals_lb_compression_matrix = \
                build_compression_matrix(build_bounds_mask(lb)).tocsr()
        self._primals_ub_compression_matrix = \
                build_compression_matrix(build_bounds_mask(ub)).tocsr()
        ineq_lb = self._nlp.ineq_lb()
        ineq_ub = self._nlp.ineq_ub()
        self._ineq_lb_compression_matrix = \
                build_compression_matrix(build_bounds_mask(ineq_lb)).tocsr()
        self._ineq_ub_compression_matrix = \
                build_compression_matrix(build_bounds_mask(ineq_ub)).tocsr()
        self._primals_lb_compressed = self._primals_lb_compression_matrix * lb
        self._primals_ub_compressed = self._primals_ub_compression_matrix * ub
        self._ineq_lb_compressed = self._ineq_lb_compression_matrix * ineq_lb
        self._ineq_ub_compressed = self._ineq_ub_compression_matrix * ineq_ub
        self._slacks = self.init_slacks()
        self._duals_primals_lb = np.ones(self._primals_lb_compression_matrix.shape[0])
        self._duals_primals_ub = np.ones(self._primals_ub_compression_matrix.shape[0])
        self._duals_slacks_lb = np.ones(self._ineq_lb_compression_matrix.shape[0])
        self._duals_slacks_ub = np.ones(self._ineq_ub_compression_matrix.shape[0])
        self._delta_primals = None
        self._delta_slacks = None
        self._delta_duals_eq = None
        self._delta_duals_ineq = None
        self._barrier = None

    def init_primals(self):
        primals = self._nlp.init_primals()
        return primals

    def init_slacks(self):
        slacks = self._nlp.evaluate_ineq_constraints()
        return slacks

    def init_duals_eq(self):
        return self._nlp.init_duals_eq()

    def init_duals_ineq(self):
        return self._nlp.init_duals_ineq()

    def init_duals_primals_lb(self):
        return np.ones(self._primals_lb_compressed.size)

    def init_duals_primals_ub(self):
        return np.ones(self._primals_ub_compressed.size)

    def init_duals_slacks_lb(self):
        return np.ones(self._ineq_lb_compressed.size)

    def init_duals_slacks_ub(self):
        return np.ones(self._ineq_ub_compressed.size)
    
    def set_primals(self, primals):
        self._nlp.set_primals(primals)

    def set_slacks(self, slacks):
        self._slacks = slacks

    def set_duals_eq(self, duals):
        self._nlp.set_duals_eq(duals)

    def set_duals_ineq(self, duals):
        self._nlp.set_duals_ineq(duals)

    def set_duals_primals_lb(self, duals):
        self._duals_primals_lb = duals

    def set_duals_primals_ub(self, duals):
        self._duals_primals_ub = duals

    def set_duals_slacks_lb(self, duals):
        self._duals_slacks_lb = duals

    def set_duals_slacks_ub(self, duals):
        self._duals_slacks_ub = duals
    
    def get_primals(self):
        return self._nlp.get_primals()

    def get_slacks(self):
        return self._slacks

    def get_duals_eq(self):
        return self._nlp.get_duals_eq()

    def get_duals_ineq(self):
        return self._nlp.get_duals_ineq()

    def get_duals_primals_lb(self):
        return self._duals_primals_lb

    def get_duals_primals_ub(self):
        return self._duals_primals_ub

    def get_duals_slacks_lb(self):
        return self._duals_slacks_lb

    def get_duals_slacks_ub(self):
        return self._duals_slacks_ub

    def get_primals_lb(self):
        return self._nlp.primals_lb()

    def get_primals_ub(self):
        return self._nlp.primals_ub()

    def get_ineq_lb(self):
        return self._nlp.ineq_lb()

    def get_ineq_ub(self):
        return self._nlp.ineq_ub()

    def set_barrier_parameter(self, barrier):
        self._barrier = barrier

    def evaluate_primal_dual_kkt_matrix(self):
        hessian = self._nlp.evaluate_hessian_lag()
        jac_eq = self._nlp.evaluate_jacobian_eq()
        jac_ineq = self._nlp.evaluate_jacobian_ineq()

        primals_lb_diff_inv = self._get_primals_lb_diff_inv()
        primals_ub_diff_inv = self._get_primals_ub_diff_inv()
        slacks_lb_diff_inv = self._get_slacks_lb_diff_inv()
        slacks_ub_diff_inv = self._get_slacks_ub_diff_inv()

        duals_primals_lb = self._duals_primals_lb
        duals_primals_ub = self._duals_primals_ub
        duals_slacks_lb = self._duals_slacks_lb
        duals_slacks_ub = self._duals_slacks_ub

        duals_primals_lb = scipy.sparse.coo_matrix(
                            (duals_primals_lb, 
                             (np.arange(duals_primals_lb.size), 
                              np.arange(duals_primals_lb.size))), 
                            shape=(duals_primals_lb.size, 
                                   duals_primals_lb.size))
        duals_primals_ub = scipy.sparse.coo_matrix(
                            (duals_primals_ub, 
                             (np.arange(duals_primals_ub.size), 
                              np.arange(duals_primals_ub.size))), 
                            shape=(duals_primals_ub.size, 
                                   duals_primals_ub.size))
        duals_slacks_lb = scipy.sparse.coo_matrix(
                            (duals_slacks_lb, 
                             (np.arange(duals_slacks_lb.size), 
                              np.arange(duals_slacks_lb.size))), 
                            shape=(duals_slacks_lb.size, 
                                   duals_slacks_lb.size))
        duals_slacks_ub = scipy.sparse.coo_matrix(
                            (duals_slacks_ub, 
                             (np.arange(duals_slacks_ub.size), 
                              np.arange(duals_slacks_ub.size))), 
                            shape=(duals_slacks_ub.size, 
                                   duals_slacks_ub.size))

        kkt = BlockMatrix(4, 4)
        kkt.set_block(0, 0, (hessian +
                             self._primals_lb_compression_matrix.transpose() * 
                             primals_lb_diff_inv * 
                             duals_primals_lb * 
                             self._primals_lb_compression_matrix +
                             self._primals_ub_compression_matrix.transpose() * 
                             primals_ub_diff_inv * 
                             duals_primals_ub * 
                             self._primals_ub_compression_matrix))

        kkt.set_block(1, 1, (self._ineq_lb_compression_matrix.transpose() * 
                             slacks_lb_diff_inv * 
                             duals_slacks_lb * 
                             self._ineq_lb_compression_matrix +
                             self._ineq_ub_compression_matrix.transpose() * 
                             slacks_ub_diff_inv * 
                             duals_slacks_ub * 
                             self._ineq_ub_compression_matrix))

        kkt.set_block(2, 0, jac_eq)
        kkt.set_block(0, 2, jac_eq.transpose())
        kkt.set_block(3, 0, jac_ineq)
        kkt.set_block(0, 3, jac_ineq.transpose())
        kkt.set_block(3, 1, -scipy.sparse.identity(
                                            self._nlp.n_ineq_constraints(),
                                            format='coo'))
        kkt.set_block(1, 3, -scipy.sparse.identity(
                                            self._nlp.n_ineq_constraints(),
                                            format='coo'))
        return kkt

    def evaluate_primal_dual_kkt_rhs(self):
        grad_obj = self.evaluate_grad_objective()
        jac_eq = self._nlp.evaluate_jacobian_eq()
        jac_ineq = self._nlp.evaluate_jacobian_ineq()

        primals_lb_diff_inv = self._get_primals_lb_diff_inv()
        primals_ub_diff_inv = self._get_primals_ub_diff_inv()
        slacks_lb_diff_inv = self._get_slacks_lb_diff_inv()
        slacks_ub_diff_inv = self._get_slacks_ub_diff_inv()

        rhs = BlockVector(4)
        rhs.set_block(0, (grad_obj +
                          jac_eq.transpose() * self._nlp.get_duals_eq() +
                          jac_ineq.transpose() * self._nlp.get_duals_ineq() -
                          self._barrier * 
                          self._primals_lb_compression_matrix.transpose() * 
                          primals_lb_diff_inv * 
                          np.ones(primals_lb_diff_inv.size) +
                          self._barrier * 
                          self._primals_ub_compression_matrix.transpose() * 
                          primals_ub_diff_inv * 
                          np.ones(primals_ub_diff_inv.size)))

        rhs.set_block(1, (-self._nlp.get_duals_ineq() -
                          self._barrier * 
                          self._ineq_lb_compression_matrix.transpose() * 
                          slacks_lb_diff_inv * 
                          np.ones(slacks_lb_diff_inv.size) +
                          self._barrier * 
                          self._ineq_ub_compression_matrix.transpose() * 
                          slacks_ub_diff_inv * 
                          np.ones(slacks_ub_diff_inv.size)))

        rhs.set_block(2, self._nlp.evaluate_eq_constraints())
        rhs.set_block(3, self._nlp.evaluate_ineq_constraints() - self._slacks)
        rhs = -rhs
        return rhs

    def set_primal_dual_kkt_solution(self, sol):
        self._delta_primals = sol.get_block(0)
        self._delta_slacks = sol.get_block(1)
        self._delta_duals_eq = sol.get_block(2)
        self._delta_duals_ineq = sol.get_block(3)

    def get_delta_primals(self):
        return self._delta_primals

    def get_delta_slacks(self):
        return self._delta_slacks

    def get_delta_duals_eq(self):
        return self._delta_duals_eq

    def get_delta_duals_ineq(self):
        return self._delta_duals_ineq

    def get_delta_duals_primals_lb(self):
        primals_lb_diff_inv = self._get_primals_lb_diff_inv()
        duals_primals_lb_matrix = scipy.sparse.coo_matrix(
                                    (self._duals_primals_lb, 
                                     (np.arange(self._duals_primals_lb.size), 
                                      np.arange(self._duals_primals_lb.size))), 
                                    shape=(self._duals_primals_lb.size, 
                                           self._duals_primals_lb.size))
        res = -self._duals_primals_lb + primals_lb_diff_inv * (self._barrier - 
              duals_primals_lb_matrix * self._primals_lb_compression_matrix * 
              self.get_delta_primals())
        return res

    def get_delta_duals_primals_ub(self):
        primals_ub_diff_inv = self._get_primals_ub_diff_inv()
        duals_primals_ub_matrix = scipy.sparse.coo_matrix(
                                    (self._duals_primals_ub, 
                                     (np.arange(self._duals_primals_ub.size), 
                                      np.arange(self._duals_primals_ub.size))), 
                                    shape=(self._duals_primals_ub.size, 
                                           self._duals_primals_ub.size))
        res = -self._duals_primals_ub + primals_ub_diff_inv * (self._barrier + 
                duals_primals_ub_matrix * self._primals_ub_compression_matrix * 
                self.get_delta_primals())
        return res

    def get_delta_duals_slacks_lb(self):
        slacks_lb_diff_inv = self._get_slacks_lb_diff_inv()
        duals_slacks_lb_matrix = scipy.sparse.coo_matrix(
                                    (self._duals_slacks_lb, 
                                     (np.arange(self._duals_slacks_lb.size), 
                                      np.arange(self._duals_slacks_lb.size))), 
                                    shape=(self._duals_slacks_lb.size, 
                                           self._duals_slacks_lb.size))
        res = -self._duals_slacks_lb + slacks_lb_diff_inv * (self._barrier - 
                duals_slacks_lb_matrix * self._ineq_lb_compression_matrix * 
                self.get_delta_slacks())
        return res

    def get_delta_duals_slacks_ub(self):
        slacks_ub_diff_inv = self._get_slacks_ub_diff_inv()
        duals_slacks_ub_matrix = scipy.sparse.coo_matrix(
                                    (self._duals_slacks_ub, 
                                     (np.arange(self._duals_slacks_ub.size), 
                                      np.arange(self._duals_slacks_ub.size))), 
                                    shape=(self._duals_slacks_ub.size, 
                                           self._duals_slacks_ub.size))
        res = -self._duals_slacks_ub + slacks_ub_diff_inv * (self._barrier + 
                duals_slacks_ub_matrix * self._ineq_ub_compression_matrix * 
                self.get_delta_slacks())
        return res

    def evaluate_objective(self):
        return self._nlp.evaluate_objective()

    def evaluate_eq_constraints(self):
        return self._nlp.evaluate_eq_constraints()

    def evaluate_ineq_constraints(self):
        return self._nlp.evaluate_ineq_constraints()

    def evaluate_grad_objective(self):
        return self._nlp.get_obj_factor() * self._nlp.evaluate_grad_objective()

    def evaluate_jacobian_eq(self):
        return self._nlp.evaluate_jacobian_eq()

    def evaluate_jacobian_ineq(self):
        return self._nlp.evaluate_jacobian_ineq()

    def _get_primals_lb_diff_inv(self):
        res = (self._primals_lb_compression_matrix * self._nlp.get_primals() - 
               self._primals_lb_compressed)
        res = scipy.sparse.coo_matrix(
            (1 / res, (np.arange(res.size), np.arange(res.size))),
            shape=(res.size, res.size))
        return res

    def _get_primals_ub_diff_inv(self):
        res = (self._primals_ub_compressed - 
               self._primals_ub_compression_matrix * self._nlp.get_primals())
        res = scipy.sparse.coo_matrix(
            (1 / res, (np.arange(res.size), np.arange(res.size))),
            shape=(res.size, res.size))
        return res

    def _get_slacks_lb_diff_inv(self):
        res = (self._ineq_lb_compression_matrix * self._slacks - 
               self._ineq_lb_compressed)
        res = scipy.sparse.coo_matrix(
            (1 / res, (np.arange(res.size), np.arange(res.size))),
            shape=(res.size, res.size))
        return res

    def _get_slacks_ub_diff_inv(self):
        res = (self._ineq_ub_compressed - 
               self._ineq_ub_compression_matrix * self._slacks)
        res = scipy.sparse.coo_matrix(
            (1 / res, (np.arange(res.size), np.arange(res.size))),
            shape=(res.size, res.size))
        return res

    def get_primals_lb_compression_matrix(self):
        return self._primals_lb_compression_matrix

    def get_primals_ub_compression_matrix(self):
        return self._primals_ub_compression_matrix

    def get_ineq_lb_compression_matrix(self):
        return self._ineq_lb_compression_matrix

    def get_ineq_ub_compression_matrix(self):
        return self._ineq_ub_compression_matrix

    def get_primals_lb_compressed(self):
        return self._primals_lb_compressed

    def get_primals_ub_compressed(self):
        return self._primals_ub_compressed

    def get_ineq_lb_compressed(self):
        return self._ineq_lb_compressed

    def get_ineq_ub_compressed(self):
        return self._ineq_ub_compressed

    def regularize_equality_gradient(self, kkt):
        # Not technically regularizing the equality gradient ...
        # Replace this with a regularize_diagonal_block function?
        # Then call with kkt matrix and the value of the perturbation?

        # Use a constant perturbation to regularize the equality constraint
        # gradient
        kkt = kkt.copy()
        reg_coef = -1e-8*self._barrier**(1/4)
        ptb = (reg_coef *
               scipy.sparse.identity(self._nlp.n_eq_constraints(), 
                                     format='coo'))

        kkt.set_block(2, 2, ptb)
        return kkt

    def regularize_hessian(self, kkt, coef):
        hess = kkt.get_block(0, 0).copy()
        kkt = kkt.copy()

        ptb = coef * scipy.sparse.identity(self._nlp.n_primals(), format='coo')
        hess = hess + ptb
        kkt.set_block(0, 0, hess)
        return kkt

