from pyomo.contrib.pynumero.linalg.solvers.regularization import InertiaCorrectionParams
try:
    from pyomo.contrib.pynumero.linalg.solvers import ma27_solver
    found_hsl = True
except ImportError as e:
    found_hsl = False

try:
    from pyomo.contrib.pynumero.linalg.solvers.mumps_solver import MUMPSSymLinearSolver
    found_mumps = True
except ImportError as e:
    found_mumps = False

import numpy as np
from pyomo.contrib.pynumero.sparse import (BlockSymMatrix,
                                           BlockVector,
                                           BlockMatrix,
                                           empty_matrix)

from scipy.sparse.linalg import spsolve, inv, splu
from scipy.sparse import coo_matrix
import six
import abc


@six.add_metaclass(abc.ABCMeta)
class KKTSolver(object):

    def __init__(self, linear_solver, **kwargs):
        self._lsolver = linear_solver

    @abc.abstractmethod
    def solve(self, kkt_matrix, rhs, *args, **kwargs):
        return


class FullKKTSolver(KKTSolver):

    def __init__(self, linear_solver, with_regularization=True):

        # create linear solver
        if linear_solver == 'mumps':
            if found_mumps:
                lsolver = MUMPSSymLinearSolver()
            else:
                if found_hsl:
                    print('WARNING: Running with ma27 linear solver. Mumps not available')
                    lsolver = ma27_solver.MA27LinearSolver()
                raise RuntimeError('Did not found MUMPS linear solver')
        elif linear_solver == 'ma27':
            if found_hsl:
                lsolver = ma27_solver.MA27LinearSolver()
            else:
                if found_mumps:
                    print('WARNING: Running with mumps linear solver. Ma27 not available')
                    lsolver = MUMPSSymLinearSolver()
                raise RuntimeError('Did not found MA27 linear solver')
        else:
            raise RuntimeError('{} Linear solver not recognized'.format(linear_solver))

        # call parent class to set model
        super(FullKKTSolver, self).__init__(lsolver)
        self._inertia_params = InertiaCorrectionParams()
        self._with_regularization = with_regularization
        self._diagonal = None

    def do_symbolic_factorization(self, kkt, **kwargs):
        nlp = kwargs.pop('nlp', None)

        lsolver = self._lsolver
        wr = self._with_regularization
        lsolver.do_symbolic_factorization(kkt, include_diagonal=wr)
        if wr:
            self._diagonal = np.zeros(kkt.shape[0])

    def do_numeric_factorization(self, kkt, **kwargs):

        nlp = kwargs.pop('nlp', None)
        max_iter_reg = kwargs.pop('max_iter_reg', 40)
        desired_num_neg_eval = kwargs.pop('desired_num_neg_eval', -1)

        if nlp is not None:
            desired_num_neg_eval = nlp.nc + nlp.nd

        lsolver = self._lsolver
        wr = self._with_regularization
        nneval = desired_num_neg_eval
        diagonal = self._diagonal
        nvars = kkt.shape[0]
        val_reg = 0.0
        count_iter = 0
        status = lsolver.do_numeric_factorization(kkt,
                                                  diagonal=diagonal,
                                                  desired_num_neg_eval=nneval)
        done = self._inertia_params.ibr1(status)

        if wr:
            assert nneval >= 0, 'regularization needs the number of negative eigenvalues'

            nx = nvars - nneval

            while not done and count_iter < max_iter_reg:

                diagonal[0: nx] = self._inertia_params.delta_w
                diagonal[nx: nvars] = self._inertia_params.delta_a
                status = self._lsolver.do_numeric_factorization(kkt,
                                                                diagonal=diagonal,
                                                                desired_num_neg_eval=nneval)

                if self._inertia_params.delta_w > 0.0:
                    val_reg = self._inertia_params.delta_w
                done = self._inertia_params.ibr4(status)
                count_iter += 1

            self._diagonal.fill(0.0)
            if count_iter >= max_iter_reg:
                print('WARNING: REACHED MAXIMUM ITERATIONS IN REGULARIZATION')

        return {'status': status, 'delta_reg': val_reg, 'reg_iter': count_iter}

    def do_back_solve(self, rhs, **kwargs):
        return self._lsolver.do_back_solve(rhs)

    def solve(self, kkt, rhs, *args, **kwargs):

        do_symbolic = kwargs.pop('do_symbolic', True)
        desired_num_neg_eval = kwargs.pop('desired_num_neg_eval',-1)
        max_iter_reg = kwargs.pop('max_iter_reg', 40)
        nlp = kwargs.pop('nlp', None)

        if do_symbolic:
            self.do_symbolic_factorization(kkt)

        info = self.do_numeric_factorization(kkt,
                                             desired_num_neg_eval=desired_num_neg_eval,
                                             max_iter_reg=max_iter_reg,
                                             nlp=nlp)
        return self.do_back_solve(rhs), info


class SchurComplementKKTSolver(KKTSolver):

    def __init__(self, linear_solver, with_regularization=True):

        if linear_solver != 'ma27':
            raise RuntimeError('Only support ma27')

        # call parent class to set model
        super(SchurComplementKKTSolver, self).__init__(linear_solver)
        self._lsolver = linear_solver
        self._inertia_params = None
        self._with_regularization = with_regularization
        self._diagonal = None

        self._sc_solver = ma27_solver.MA27LinearSolver()

    def do_symbolic_factorization(self, kkt, **kwargs):

        nlp = kwargs.pop('nlp', None)
        assert nlp is not None, 'SchurComplementSolver requires nlp to be passed'

        nblocks = nlp.nblocks
        wr = self._with_regularization
        self._lsolver = [ma27_solver.MA27LinearSolver() for i in range(nblocks)]
        self._diagonal = [None for i in range(nblocks)]
        if wr:
            self._inertia_params = []
            diagonals = []
            for name, scenario in nlp.nlps():
                self._inertia_params.append(InertiaCorrectionParams())
                nx = scenario.nx
                nd = scenario.nd
                nc = scenario.nc
                diagonals.append(np.zeros(nx + 2*nd + nc + nlp.nz))
            self._diagonal = diagonals

        permuted_kkt = self.build_permuted_kkt(kkt, nlp)

        # perform symbolic factorization in each block kkt
        for bid in range(nblocks):
            block_kkt = permuted_kkt[bid, bid]
            self._lsolver[bid].do_symbolic_factorization(block_kkt, include_diagonal=wr)

        return permuted_kkt

    @staticmethod
    def build_permuted_kkt(kkt, nlp):

        # Note: this ignores Jd_coupling since they are empty blocks
        nblocks = nlp.nblocks

        permuted_kkt = BlockSymMatrix(nblocks+1)
        for bid in range(nblocks):

            hess = kkt[0, 0][bid, bid]
            Ds = kkt[1, 1][bid, bid]
            Jc = kkt[2, 0][bid, bid]
            Jd = kkt[3, 0][bid, bid]
            A = kkt[2, 0][bid + nblocks, bid]
            B = kkt[2, 0][bid + nblocks, nblocks]

            block_kkt = BlockSymMatrix(5)
            block_kkt[0, 0] = hess
            block_kkt[1, 1] = Ds
            block_kkt[2, 0] = Jc
            block_kkt[3, 0] = Jd
            block_kkt[4, 0] = A

            nzeros = hess.shape[0] + Ds.shape[0] + Jc.shape[0] + Jd.shape[0]
            nz = B.shape[0]
            block_B = BlockMatrix(1, 2)
            block_B[0, 0] = empty_matrix(nz, nzeros)
            block_B[0, 1] = B

            permuted_kkt[bid, bid] = block_kkt
            permuted_kkt[nblocks, bid] = block_B

        hess_z = kkt[0, 0][nblocks, nblocks]
        permuted_kkt[nblocks, nblocks] = hess_z
        return permuted_kkt

    @staticmethod
    def build_permuted_rhs(rhs, nlp):

        # Note: this ignores Jd_coupling since they are empty blocks
        nblocks = nlp.nblocks
        permuted_rhs = BlockVector(nblocks + 1)
        for bid in range(nblocks):

            grad_x_lag = rhs[0][bid]
            grad_s_lag = rhs[1][bid]
            grad_yc_lag = rhs[2][bid]
            grad_yc_coupling_lag = rhs[2][bid+nblocks]
            grad_yd_lag = rhs[3][bid]

            block_rhs = BlockVector(5)
            block_rhs[0] = grad_x_lag
            block_rhs[1] = grad_s_lag
            block_rhs[2] = grad_yc_lag
            block_rhs[3] = grad_yd_lag
            block_rhs[4] = grad_yc_coupling_lag

            permuted_rhs[bid] = block_rhs

        grad_z_lag = rhs[0][nblocks]
        permuted_rhs[nblocks] = grad_z_lag

        return permuted_rhs

    def build_schur_complement(self, kkt, nlp):
        # Note: just for testing
        nblocks = nlp.nblocks
        permuted_kkt = self.build_permuted_kkt(kkt, nlp)
        # build schur-complement
        S = empty_matrix(nlp.nz, nlp.nz)
        for bid in range(nblocks):
            Ki = permuted_kkt[bid, bid].tocoo()
            BiT = permuted_kkt[bid, nblocks].tocoo()
            Bi = permuted_kkt[nblocks, bid].tocoo()
            Ri = spsolve(Ki, BiT)
            Si = Bi.dot(Ri)
            S -= Si
        return S

    def solve(self, kkt, rhs, **kwargs):

        do_symbolic = kwargs.pop('do_symbolic', True)
        desired_num_neg_eval = kwargs.pop('desired_num_neg_eval', -1)
        max_iter_reg = kwargs.pop('max_iter_reg', 40)
        nlp = kwargs.pop('nlp', None)
        assert nlp is not None, 'SchurComplementSolver requires nlp to be passed'

        if do_symbolic:
            permuted_kkt = self.do_symbolic_factorization(kkt, nlp=nlp)
        else:
            # ToDo: add safeguard to see if symbolic needed
            permuted_kkt = self.build_permuted_kkt(kkt, nlp)

        # numeric factorization
        nblocks = nlp.nblocks
        wr = self._with_regularization

        nlps = []
        for name, nlp_block in nlp.nlps():
            nlps.append(nlp_block)

        # perform numeric factorization in each block kkt
        status = [None for i in range(nblocks)]
        for bid in range(nblocks):
            block_kkt = permuted_kkt[bid, bid]
            if wr:
                nlp_block = nlps[bid]
                nneval = nlp_block.nc + nlp_block.nd + nlp.nz
                status[bid] = self._lsolver[bid].do_numeric_factorization(block_kkt,
                                                                          diagonal=self._diagonal[bid],
                                                                          desired_num_neg_eval=nneval)
            else:
                nneval = desired_num_neg_eval
                status[bid] = self._lsolver[bid].do_numeric_factorization(block_kkt,
                                                                          diagonal=self._diagonal[bid],
                                                                          desired_num_neg_eval=nneval)

        val_reg_w = 0.0
        val_reg_a = 0.0
        max_count_iter = 0
        overall_status = 0
        if wr:

            # first pass to see which block needs the largest regularization
            for bid in range(nblocks):
                block_kkt = permuted_kkt[bid, bid]
                done = self._inertia_params[bid].ibr1(status[bid])
                count_iter = 0
                block_nlp = nlps[bid]
                nneval = block_nlp.nc + block_nlp.nd + nlp.nz
                nxd = block_nlp.nx + block_nlp.nd
                nvars = block_nlp.nx + 2 * block_nlp.nd + block_nlp.nc + nlp.nz

                while not done and count_iter < max_iter_reg:

                    self._diagonal[bid][0: nxd] = self._inertia_params[bid].delta_w
                    self._diagonal[bid][nxd: nvars] = self._inertia_params[bid].delta_a
                    status = self._lsolver[bid].do_numeric_factorization(block_kkt,
                                                                         diagonal=self._diagonal[bid],
                                                                         desired_num_neg_eval=nneval)

                    if val_reg_w > self._inertia_params[bid].delta_w:
                        val_reg_w = self._inertia_params[bid].delta_w

                    if val_reg_a > self._inertia_params[bid].delta_a:
                        val_reg_a = self._inertia_params[bid].delta_a

                    if count_iter > max_count_iter:
                        max_count_iter = count_iter

                    done = self._inertia_params[bid].ibr4(status)
                    count_iter += 1

            if val_reg_a > 0.0 or val_reg_w > 0.0:

                for bid in range(nblocks):
                    block_kkt = permuted_kkt[bid, bid]
                    block_nlp = nlps[bid]
                    nxd = block_nlp.nx + block_nlp.nd
                    nvars = block_nlp.nx + 2 * block_nlp.nd + block_nlp.nc + nlp.nz

                    self._diagonal[bid][0: nxd] = val_reg_w
                    self._diagonal[bid][nxd: nvars] = val_reg_a
                    status = self._lsolver[bid].do_numeric_factorization(block_kkt,
                                                                         diagonal=self._diagonal[bid],
                                                                         desired_num_neg_eval=nneval)

                    if status > overall_status:
                        overall_status = status

        # permute rhs
        permuted_rhs = self.build_permuted_rhs(rhs, nlp)
        # build schur-complement and rhs-schur-complement
        sc_row = []
        sc_col = []
        sc_values = []
        rhs_sc = permuted_rhs[nblocks].copy()
        for bid in range(nblocks):
            lsolver = self._lsolver[bid]
            # compute schur-complement contribution of bid block
            BiT = permuted_kkt[bid, nblocks].tocoo()
            Bi = permuted_kkt[nblocks, bid].tocsr()
            nnz_cols = np.unique(BiT.col)
            BiT_csc = BiT.tocsc()
            for j in nnz_cols:
                si = BiT_csc.getcol(j).toarray()
                sij = lsolver.do_back_solve(si)
                sc_values.append(-Bi.dot(sij))
                sc_row.append(np.arange(nlp.nz))
                tmp = np.ones(nlp.nz, dtype='i')
                tmp.fill(j)
                sc_col.append(tmp)
            # compute rhs_sc contribution of bid block
            ri = permuted_rhs[bid].flatten()
            pi = lsolver.do_back_solve(ri)
            rhs_sc -= Bi.dot(pi)

        row = np.concatenate(sc_row)
        col = np.concatenate(sc_col)
        data = np.concatenate(sc_values)
        S = coo_matrix((data, (row, col)), shape=(nlp.nz, nlp.nz))
        S.sum_duplicates()

        # solve schur-complement
        # delta_z = spsolve(S, rhs_sc)
        delta_z = self._sc_solver.solve(S, rhs_sc)

        # perform backsolves each block kkt
        permuted_sol = BlockVector(nblocks)
        for bid in range(nblocks):
            lsolver = self._lsolver[bid]
            ri = permuted_rhs[bid]
            rhs_i = ri - permuted_kkt[bid, nblocks].tocsr().dot(delta_z)
            xi = lsolver.do_back_solve(rhs_i)
            permuted_sol[bid] = xi

        # permute solution back
        sol = BlockVector(4)
        x_vals = []
        s_vals = []
        s_linking_vals = []
        yc_vals = []
        yc_linking_vals = []
        yd_vals = []
        yd_linking_vals = []
        for bid in range(nblocks):
            x_vals.append(permuted_sol[bid][0])
            s_vals.append(permuted_sol[bid][1])
            yc_vals.append(permuted_sol[bid][2])
            yd_vals.append(permuted_sol[bid][3])
            yc_linking_vals.append(permuted_sol[bid][4])
            yd_linking_vals.append(np.zeros(0))
            s_linking_vals.append(np.zeros(0))

        x_vals.append(delta_z)
        sol[0] = BlockVector(x_vals)
        sol[1] = BlockVector(s_vals + s_linking_vals)
        sol[2] = BlockVector(yc_vals + yc_linking_vals)
        sol[3] = BlockVector(yd_vals + yd_linking_vals)

        info = {'status': overall_status, 'delta_reg': val_reg_w, 'reg_iter': max_count_iter}
        return sol, info
