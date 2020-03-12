from collections.abc import ABCMeta, abstractmethod
import six
from pyomo.contrib.pynumero.interfaces import pyomo_nlp
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
    def evaluate_primal_dual_kkt_matrix(self, barrier_parameter):
        pass

    @abstractmethod
    def evaluate_primal_dual_kkt_rhs(self, barrier_parameter):
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
    def evaluate_objective(self):
        pass

    @abstractmethod
    def evaluate_eq_constraints(self):
        pass

    @abstractmethod
    def evalute_ineq_constraints(self):
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


class InteriorPointInterface(BaseInteriorPointInterface):
    def __init__(self, pyomo_model):
        self._nlp = pyomo_nlp.PyomoNLP(pyomo_model)
        lb = self._nlp.primals_lb()
        ub = self._nlp.primals_ub()
        self._primals_lb_compression_matrix = build_compression_matrix(build_bounds_mask(lb)).tocsr()
        self._primals_ub_compression_matrix = build_compression_matrix(build_bounds_mask(ub)).tocsr()
        ineq_lb = self._nlp.ineq_lb()
        ineq_ub = self._nlp.ineq_ub()
        self._ineq_lb_compression_matrix = build_compression_matrix(build_bounds_mask(ineq_lb)).tocsr()
        self._ineq_ub_compression_matrix = build_compression_matrix(build_bounds_mask(ineq_ub)).tocsr()
        self._primals_lb_compressed = self._primals_lb_compression_matrix * lb
        self._primals_ub_compressed = self._primals_ub_compression_matrix * ub
        self._ineq_lb_compressed = self._ineq_lb_compression_matrix * ineq_lb
        self._ineq_ub_compressed = self._ineq_ub_compression_matrix * ineq_ub
        self._init_slacks = self._nlp.evaluate_ineq_constraints()
        self._slacks = self._init_slacks
        self._duals_primals_lb = np.zeros(self._primals_lb_compression_matrix.shape[0])
        self._duals_primals_ub = np.zeros(self._primals_ub_compression_matrix.shape[0])
        self._duals_slacks_lb = np.zeros(self._ineq_lb_compression_matrix.shape[0])
        self._duals_slacks_ub = np.zeros(self._ineq_ub_compression_matrix.shape[0])
        self._delta_primals = None
        self._delta_slacks = None
        self._delta_duals_eq = None
        self._delta_duals_ineq = None

    def init_primals(self):
        primals = self._nlp.init_primals().copy()
        lb = self._nlp.primals_lb().copy()
        ub = self._nlp.primals_ub().copy()
        out_of_bounds = ((primals > ub) + (primals < lb)).nonzero()[0]
        neg_inf_indices = np.isneginf(lb).nonzero()[0]
        np.put(lb, neg_inf_indices, ub[neg_inf_indices])
        pos_inf_indices = np.isposinf(lb).nonzero()[0]
        np.put(lb, pos_inf_indices, 0)
        pos_inf_indices = np.isposinf(ub).nonzero()[0]
        np.put(ub, pos_inf_indices, lb[pos_inf_indices])
        tmp = 0.5 * (lb + ub)
        np.put(primals, out_of_bounds, tmp[out_of_bounds])
        return primals

    def init_slacks(self):
        return self._init_slacks

    def init_duals_eq(self):
        return self._nlp.init_duals_eq()

    def init_duals_ineq(self):
        return self._nlp.init_duals_ineq()

    def init_duals_primals_lb(self):
        return np.zeros(self._primals_lb_compression_matrix.shape[0])

    def init_duals_primals_ub(self):
        return np.zeros(self._primals_ub_compression_matrix.shape[0])

    def init_duals_slacks_lb(self):
        return np.zeros(self._ineq_lb_compression_matrix.shape[0])

    def init_duals_slacks_ub(self):
        return np.zeros(self._ineq_ub_compression_matrix.shape[0])
    
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

    def evaluate_primal_dual_kkt_matrix(self, barrier_parameter):
        hessian = self._nlp.evaluate_hessian_lag()
        primals = self._nlp.get_primals()
        jac_eq = self._nlp.evaluate_jacobian_eq()
        jac_ineq = self._nlp.evaluate_jacobian_ineq()
        
        primals_lb_diff = self._primals_lb_compression_matrix * primals - self._primals_lb_compressed
        primals_ub_diff = self._primals_ub_compressed - self._primals_ub_compression_matrix * primals
        slacks_lb_diff = self._ineq_lb_compression_matrix * self._slacks - self._ineq_lb_compressed
        slacks_ub_diff = self._ineq_ub_compressed - self._ineq_ub_compression_matrix * self._slacks

        primals_lb_diff = scipy.sparse.coo_matrix((1/primals_lb_diff, (np.arange(primals_lb_diff.size), np.arange(primals_lb_diff.size))), shape=(primals_lb_diff.size, primals_lb_diff.size))
        primals_ub_diff = scipy.sparse.coo_matrix((1/primals_ub_diff, (np.arange(primals_ub_diff.size), np.arange(primals_ub_diff.size))), shape=(primals_ub_diff.size, primals_ub_diff.size))
        slacks_lb_diff = scipy.sparse.coo_matrix((1/slacks_lb_diff, (np.arange(slacks_lb_diff.size), np.arange(slacks_lb_diff.size))), shape=(slacks_lb_diff.size, slacks_lb_diff.size))
        slacks_ub_diff = scipy.sparse.coo_matrix((1/slacks_ub_diff, (np.arange(slacks_ub_diff.size), np.arange(slacks_ub_diff.size))), shape=(slacks_ub_diff.size, slacks_ub_diff.size))

        duals_primals_lb = self._duals_primals_lb
        duals_primals_ub = self._duals_primals_ub
        duals_slacks_lb = self._duals_slacks_lb
        duals_slacks_ub = self._duals_slacks_ub

        duals_primals_lb = scipy.sparse.coo_matrix((duals_primals_lb, (np.arange(duals_primals_lb.size), np.arange(duals_primals_lb.size))), shape=(duals_primals_lb.size, duals_primals_lb.size))
        duals_primals_ub = scipy.sparse.coo_matrix((duals_primals_ub, (np.arange(duals_primals_ub.size), np.arange(duals_primals_ub.size))), shape=(duals_primals_ub.size, duals_primals_ub.size))
        duals_slacks_lb = scipy.sparse.coo_matrix((duals_slacks_lb, (np.arange(duals_slacks_lb.size), np.arange(duals_slacks_lb.size))), shape=(duals_slacks_lb.size, duals_slacks_lb.size))
        duals_slacks_ub = scipy.sparse.coo_matrix((duals_slacks_ub, (np.arange(duals_slacks_ub.size), np.arange(duals_slacks_ub.size))), shape=(duals_slacks_ub.size, duals_slacks_ub.size))

        kkt = BlockMatrix(4, 4)
        kkt.set_block(0, 0, (hessian +
                             self._primals_lb_compression_matrix.transpose() * primals_lb_diff * duals_primals_lb * self._primals_lb_compression_matrix +
                             self._primals_ub_compression_matrix.transpose() * primals_ub_diff * duals_primals_ub * self._primals_ub_compression_matrix))
        kkt.set_block(1, 1, (self._ineq_lb_compression_matrix.transpose() * slacks_lb_diff * duals_slacks_lb * self._ineq_lb_compression_matrix +
                             self._ineq_ub_compression_matrix.transpose() * slacks_ub_diff * duals_slacks_ub * self._ineq_ub_compression_matrix))
        kkt.set_block(2, 0, jac_eq)
        kkt.set_block(0, 2, jac_eq.transpose())
        kkt.set_block(3, 0, jac_ineq)
        kkt.set_block(0, 3, jac_ineq.transpose())
        kkt.set_block(3, 1, scipy.sparse.identity(self._nlp.n_ineq_constraints, format='coo'))
        kkt.set_block(1, 3, scipy.sparse.identity(self._nlp.n_ineq_constraints, format='coo'))
        return kkt

    def evaluate_primal_dual_kkt_rhs(self, barrier_parameter):
        grad_obj = self._nlp.evaluate_grad_objective()
        jac_eq = self._nlp.evaluate_jacobian_eq()
        jac_ineq = self._nlp.evaluate_jacobian_ineq()

        primals_lb_diff = self._primals_lb_compression_matrix * primals - self._primals_lb_compressed
        primals_ub_diff = self._primals_ub_compressed - self._primals_ub_compression_matrix * primals
        slacks_lb_diff = self._ineq_lb_compression_matrix * self._slacks - self._ineq_lb_compressed
        slacks_ub_diff = self._ineq_ub_compressed - self._ineq_ub_compression_matrix * self._slacks

        primals_lb_diff = scipy.sparse.coo_matrix((1/primals_lb_diff, (np.arange(primals_lb_diff.size), np.arange(primals_lb_diff.size))), shape=(primals_lb_diff.size, primals_lb_diff.size))
        primals_ub_diff = scipy.sparse.coo_matrix((1/primals_ub_diff, (np.arange(primals_ub_diff.size), np.arange(primals_ub_diff.size))), shape=(primals_ub_diff.size, primals_ub_diff.size))
        slacks_lb_diff = scipy.sparse.coo_matrix((1/slacks_lb_diff, (np.arange(slacks_lb_diff.size), np.arange(slacks_lb_diff.size))), shape=(slacks_lb_diff.size, slacks_lb_diff.size))
        slacks_ub_diff = scipy.sparse.coo_matrix((1/slacks_ub_diff, (np.arange(slacks_ub_diff.size), np.arange(slacks_ub_diff.size))), shape=(slacks_ub_diff.size, slacks_ub_diff.size))

        rhs = BlockVector(4)
        rhs.set_block(0, (grad_obj +
                          jac_eq.transpose() * self._nlp.get_duals_eq() +
                          jac_ineq.transpose() * self._nlp.get_duals_ineq() -
                          barrier_parameter * self._primals_lb_compression_matrix.transpose() * primals_lb_diff * np.ones(primals_lb_diff.size) +
                          barrier_parameter * self._primals_ub_compression_matrix.transpose() * primals_ub_diff * np.ones(primals_ub_diff.size)))
        rhs.set_block(1, (-self._nlp.get_duals_ineq() -
                          barrier_parameter * self._ineq_lb_compression_matrix.transpose() * slacks_lb_diff * np.ones(slacks_lb_diff.size) +
                          barrier_parameter * self._ineq_ub_compression_matrix.transpose() * slacks_ub_diff * np.ones(slacks_ub_diff.size)))
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

    def evaluate_objective(self):
        return self._nlp.evaluate_objective()

    def evaluate_eq_constraints(self):
        return self._nlp.evaluate_eq_constraints()

    def evalute_ineq_constraints(self):
        return self._nlp.evaluate_ineq_constraints()

    def evaluate_grad_objective(self):
        return self._nlp.evaluate_grad_objective()

    def evaluate_jacobian_eq(self):
        return self._nlp.evaluate_jacobian_eq()

    def evaluate_jacobian_ineq(self):
        return self._nlp.evaluate_jacobian_ineq()
