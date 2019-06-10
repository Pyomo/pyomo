#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
from pyomo.contrib.pynumero.interfaces.nlp import NLP
from pyomo.contrib.pynumero.sparse import (BlockMatrix,
                                           BlockSymMatrix,
                                           BlockVector,
                                           empty_matrix)
from scipy.sparse import coo_matrix, csr_matrix, identity
from collections import OrderedDict
import pyomo.contrib.pynumero as pn
import pyomo.environ as aml
import pyomo.dae as dae
import numpy as np
import six
import abc

# ToDo: improve the design of the abstract class
@six.add_metaclass(abc.ABCMeta)
class CompositeNLP(object):

    def __init__(self):
        pass

    def nblocks(self):
        return 0

__all__ = ['TwoStageStochasticNLP']


class TwoStageStochasticNLP(NLP, CompositeNLP):
    """
    Nonlinear program interface for composite NLP that result from
    two-stage stochastic programming problems
    """

    def __init__(self,
                 nlps,
                 complicating_vars):

        """

        Parameters
        ----------
        nlps: dictionary with scenarios (scenario names to NLPs)
        complicating_vars: dictionary with complicated variables
        (scenario names to list of variable indices)

        """
        if not isinstance(nlps, dict):
            raise RuntimeError("Model must be a dictionary")
        if not isinstance(complicating_vars, dict):
            raise RuntimeError("complicating_vars must be a dictionary")
        if len(complicating_vars) != len(nlps):
            raise RuntimeError("Each scenario must have a list of complicated variables")

        # call parent class to set model
        super(TwoStageStochasticNLP, self).__init__(None)

        # initialize components
        self._initialize_nlp_components(nlps, complicating_vars)

    def _initialize_nlp_components(self, *args, **kwargs):

        nlps = args[0]
        complicating_vars = args[1]

        aux_counter = 0
        n_z = 0

        # check inputs
        for k, l in complicating_vars.items():
            if k not in nlps:
                raise RuntimeError("{} not a scenario name".format(k))
            if aux_counter == 0:
                n_z = len(l)
            else:
                if len(l) != n_z:
                    err_msg = "All scenarios must have the same number of complicating variables"
                    raise RuntimeError(err_msg)

            if len(set(l)) != len(l):
                err_msg = 'Scenario {} has duplicates in list of complicating variables'
                raise RuntimeError(err_msg.format(k))

            for val in l:
                nlp = nlps[k]
                if val > nlp.nx:
                    raise RuntimeError("Variable index cannot be greater than number of vars in NLP")
            aux_counter += 1

        # map of scenario name to indices
        self._sname_to_sid = dict()
        # map of scenario id to scenario name
        self._sid_to_sname = list()

        # populate containers
        ordered_keys = sorted(nlps.keys())
        new_dict = OrderedDict()
        self._nlps = list()
        for k in ordered_keys:
            nlp = nlps[k]
            if not isinstance(nlp, NLP):
                raise RuntimeError("Scenarios must be NLP objects")
            self._sname_to_sid[k] = len(self._sid_to_sname)
            self._sid_to_sname.append(k)
            self._nlps.append(nlp)
            # make model a dictionary of original models (PyomoModels or nl-files)
            new_dict[k] = nlp.model
        self._model = new_dict

        # set number of complicated variables
        self._nz = n_z

        # define map of complicated variables
        # this goes from [scnario_id][zid] - > vid
        self._zid_to_vid = list()
        for sid, sname in enumerate(self._sid_to_sname):
            self._zid_to_vid.append(complicating_vars[sname])

        # defines vectors
        self._create_vectors()

        # define sizes
        self._nx = self._init_x.size  # this includes nz
        self._ng = self._init_y.size
        self._nc = sum(nlp.nc for nlp in self._nlps) + self.nz * self.nblocks
        self._nd = sum(nlp.nd for nlp in self._nlps)

        # define structure of jacobians
        self._create_jacobian_structures()

        # define structure Hessian
        self._create_hessian_structure()

        # cache coupling matrices
        self._AB_csr = self.coupling_matrix()
        self._AB_coo = BlockMatrix(self.nblocks+1, self.nblocks+1)
        nb = self.nblocks
        for i in range(nb):
            self._AB_coo[i, i] = self._AB_csr[i, i].tocoo()
        self._AB_coo[nb, nb] = self._AB_csr[nb, nb]

    def _make_unmutable_caches(self):
        # no need for caches here
        pass

    def _create_vectors(self):

        # Note: This method requires the complicated vars nz to be defined beforehand

        # init values
        self._init_x = BlockVector([nlp.x_init() for nlp in self._nlps] +
                                   [np.zeros(self.nz, dtype=np.double)])

        self._init_y = BlockVector([nlp.y_init() for nlp in self._nlps] +
                       [np.zeros(self.nz, dtype=np.double) for i in range(self.nblocks)])

        # lower and upper bounds

        self._lower_x = BlockVector([nlp.xl() for nlp in self._nlps] +
                                    [np.full(self.nz, -np.inf, dtype=np.double)])
        self._upper_x = BlockVector([nlp.xu() for nlp in self._nlps] +
                                    [np.full(self.nz, np.inf, dtype=np.double)])

        self._lower_g = BlockVector([nlp.gl() for nlp in self._nlps] +
                        [np.zeros(self.nz, dtype=np.double) for i in range(self.nblocks)])
        self._upper_g = BlockVector([nlp.gu() for nlp in self._nlps] +
                        [np.zeros(self.nz, dtype=np.double) for i in range(self.nblocks)])

        # define x maps and masks
        self._lower_x_mask = np.isfinite(self._lower_x)
        self._lower_x_map = self._lower_x_mask.nonzero()[0]
        self._upper_x_mask = np.isfinite(self._upper_x)
        self._upper_x_map = self._upper_x_mask.nonzero()[0]

        # define gcd maps and masks
        bounds_difference = self._upper_g - self._lower_g
        abs_bounds_difference = np.absolute(bounds_difference)
        tolerance_equalities = 1e-8
        self._c_mask = abs_bounds_difference < tolerance_equalities
        self._c_map = self._c_mask.nonzero()[0]
        self._d_mask = abs_bounds_difference >= tolerance_equalities
        self._d_map = self._d_mask.nonzero()[0]

        self._lower_g_mask = np.isfinite(self._lower_g) * self._d_mask + self._c_mask
        self._lower_g_map = self._lower_g_mask.nonzero()[0]
        self._upper_g_mask = np.isfinite(self._upper_g) * self._d_mask + self._c_mask
        self._upper_g_map = self._upper_g_mask.nonzero()[0]

        self._lower_d_mask = pn.isin(self._d_map, self._lower_g_map)
        self._upper_d_mask = pn.isin(self._d_map, self._upper_g_map)

        # define lower and upper d maps
        self._lower_d_map = pn.where(self._lower_d_mask)[0]
        self._upper_d_map = pn.where(self._upper_d_mask)[0]

        # get lower and upper d values
        self._lower_d = np.compress(self._d_mask, self._lower_g)
        self._upper_d = np.compress(self._d_mask, self._upper_g)

    def _create_jacobian_structures(self):

        # Note: This method requires the complicated vars map to be
        # created beforehand

        # build general jacobian
        jac_g = BlockMatrix(2 * self.nblocks, self.nblocks + 1)
        for sid, nlp in enumerate(self._nlps):
            xi = nlp.x_init()
            jac_g[sid, sid] = nlp.jacobian_g(xi)

            # coupling matrices Ai
            scenario_vids = self._zid_to_vid[sid]
            col = np.array([vid for vid in scenario_vids])
            row = np.arange(0, self.nz)
            data = np.ones(self.nz, dtype=np.double)
            jac_g[sid + self.nblocks, sid] = coo_matrix((data, (row, col)),
                                                       shape=(self.nz, nlp.nx))

            # coupling matrices Bi
            jac_g[sid + self.nblocks, self.nblocks] = -identity(self.nz)

        self._internal_jacobian_g = jac_g
        flat_jac_g = jac_g.tocoo()
        self._irows_jac_g = flat_jac_g.row
        self._jcols_jac_g = flat_jac_g.col
        self._nnz_jac_g = flat_jac_g.nnz

        # build jacobian equality constraints
        jac_c = BlockMatrix(2 * self.nblocks, self.nblocks + 1)
        for sid, nlp in enumerate(self._nlps):
            xi = nlp.x_init()
            jac_c[sid, sid] = nlp.jacobian_c(xi)

            # coupling matrices Ai
            scenario_vids = self._zid_to_vid[sid]
            col = np.array([vid for vid in scenario_vids])
            row = np.arange(0, self.nz)
            data = np.ones(self.nz, dtype=np.double)
            jac_c[sid + self.nblocks, sid] = coo_matrix((data, (row, col)),
                                                       shape=(self.nz, nlp.nx))

            # coupling matrices Bi
            jac_c[sid + self.nblocks, self.nblocks] = -identity(self.nz)

        self._internal_jacobian_c = jac_c
        flat_jac_c = jac_c.tocoo()
        self._irows_jac_c = flat_jac_c.row
        self._jcols_jac_c = flat_jac_c.col
        self._nnz_jac_c = flat_jac_c.nnz

        # build jacobian inequality constraints
        jac_d = BlockMatrix(2 * self.nblocks, self.nblocks + 1)
        for sid, nlp in enumerate(self._nlps):
            xi = nlp.x_init()
            jac_d[sid, sid] = nlp.jacobian_d(xi)
            jac_d[sid + self.nblocks, sid] = empty_matrix(0, xi.size)
            jac_d[sid + self.nblocks, self.nblocks] = empty_matrix(0, self.nz)
        self._internal_jacobian_d = jac_d
        flat_jac_d = jac_d.tocoo()
        self._irows_jac_d = flat_jac_d.row
        self._jcols_jac_d = flat_jac_d.col
        self._nnz_jac_d = flat_jac_d.nnz

    def _create_hessian_structure(self):

        # Note: This method requires the complicated vars map to be
        # created beforehand

        hess_lag = BlockSymMatrix(self.nblocks + 1)
        for sid, nlp in enumerate(self._nlps):
            xi = nlp.x_init()
            yi = nlp.y_init()
            hess_lag[sid, sid] = nlp.hessian_lag(xi, yi)

        hess_lag[self.nblocks, self.nblocks] = empty_matrix(self.nz, self.nz)

        flat_hess = hess_lag.tocoo()
        self._irows_hess = flat_hess.row
        self._jcols_hess = flat_hess.col
        self._nnz_hess_lag = flat_hess.nnz

        # ToDo: decide if we cache _irows and _jcols pointers for composite nlp

    @property
    def model(self):
        """
        Return optimization model
        """
        return self._model

    @property
    def nblocks(self):
        """
        Returns number of blocks (nlps)
        """
        return len(self._nlps)

    @property
    def nz(self):
        """
        Return number of complicated variables
        """
        return self._nz

    def nlps(self):
        """Creates generator scenario name to nlp """
        for sid, name in enumerate(self._sid_to_sname):
            yield name, self._nlps[sid]

    def create_vector(self, vector_type):

        if vector_type == 'x':
            subvectors = [np.zeros(nlp.nx, dtype=np.double) for nlp in self._nlps] + \
                         [np.zeros(self.nz, dtype=np.double)]
            return BlockVector(subvectors)
        elif vector_type == 'xl':
            vectors = list()
            for nlp in self._nlps:
                nx_l = len(nlp._lower_x_map)
                xl = np.zeros(nx_l, dtype=np.double)
                vectors.append(xl)
            # complicated variables have no lower bounds
            vectors.append(np.zeros(0, dtype=np.double))
            return BlockVector(vectors)
        elif vector_type == 'xu':
            vectors = list()
            for nlp in self._nlps:
                nx_u = len(nlp._upper_x_map)
                xu = np.zeros(nx_u, dtype=np.double)
                vectors.append(xu)
            # complicated variables have no upper bounds
            vectors.append(np.zeros(0, dtype=np.double))
            return BlockVector(vectors)
        elif vector_type == 'g' or vector_type == 'y':
            return BlockVector([np.zeros(nlp.ng, dtype=np.double) for nlp in self._nlps] +
                               [np.zeros(self.nz, dtype=np.double) for i in range(self.nblocks)])
        elif vector_type == 'c' or vector_type == 'yc':
            return BlockVector([np.zeros(nlp.nc, dtype=np.double) for nlp in self._nlps] +
                               [np.zeros(self.nz, dtype=np.double) for i in range(self.nblocks)])
        elif vector_type == 'd' or vector_type == 'yd' or vector_type == 's':
            return BlockVector([np.zeros(nlp.nd, dtype=np.double) for nlp in self._nlps] +
                               [np.zeros(0, dtype=np.double) for i in range(self.nblocks)])
        elif vector_type == 'dl' or vector_type == 'du':
            return BlockVector([nlp.create_vector(vector_type) for nlp in self._nlps] +
                               [np.zeros(0, dtype=np.double) for i in range(self.nblocks)])
        else:
            raise RuntimeError('Vector_type not recognized')

    def objective(self, x, **kwargs):
        """Returns value of objective function evaluated at x

        Parameters
        ----------
        x : array_like
            Array with values of primal variables.

        Returns
        -------
        float

        """
        if isinstance(x, BlockVector):
            return sum(self._nlps[i].objective(x[i]) for i in range(self.nblocks))
        elif isinstance(x, np.ndarray):
            block_x = self.create_vector('x')
            block_x.copyfrom(x)
            x_ = block_x
            return sum(self._nlps[i].objective(x_[i]) for i in range(self.nblocks))
        else:
            raise NotImplementedError("x must be a numpy array or a BlockVector")

    def grad_objective(self, x, out=None, **kwargs):
        """Returns gradient of the objective function evaluated at x

        Parameters
        ----------
        x : array_like
            Array with values of primal variables.
        out : array_like
            Output array. Its type is preserved and it
            must be of the right shape to hold the output.

        Returns
        -------
        array_like

        """
        if out is None:
            df = self.create_vector('x')
        else:
            assert isinstance(out, BlockVector), 'Composite NLP takes block vector to evaluate g'
            assert out.nblocks == self.nblocks + 1
            assert out.size == self.nx
            df = out

        if isinstance(x, BlockVector):
            assert x.size == self.nx
            assert x.nblocks == self.nblocks + 1
            for i in range(self.nblocks):
                self._nlps[i].grad_objective(x[i], out=df[i])
            return df
        elif isinstance(x, np.ndarray):
            assert x.size == self.nx
            block_x = self.create_vector('x')
            block_x.copyfrom(x)
            x_ = block_x
            for i in range(self.nblocks):
                self._nlps[i].grad_objective(x_[i], out=df[i])
            return df
        else:
            raise NotImplementedError("x must be a numpy array or a BlockVector")

    def evaluate_g(self, x, out=None, **kwargs):
        """Returns general inequality constraints evaluated at x

        Parameters
        ----------
        x : array_like
            Array with values of primal variables.
        out : array_like
            Output array. Its type is preserved and it
            must be of the right shape to hold the output.

        Returns
        -------
        array_like

        """
        if out is None:
            res = self.create_vector('y')
        else:
            assert isinstance(out, BlockVector), 'Composite NLP takes block vector to evaluate g'
            assert out.nblocks == 2 * self.nblocks
            assert out.size == self.ng
            res = out

        if isinstance(x, BlockVector):
            assert x.size == self.nx
            assert x.nblocks == self.nblocks + 1
            for sid in range(self.nblocks):
                # evaluate gi
                self._nlps[sid].evaluate_g(x[sid], out=res[sid])

                # evaluate coupling Ax-z
                A = self._AB_csr[sid, sid]
                res[sid + self.nblocks] = A * x[sid] - x[self.nblocks]
            return res
        elif isinstance(x, np.ndarray):
            assert x.size == self.nx
            block_x = self.create_vector('x')
            block_x.copyfrom(x)  # this is expensive
            x_ = block_x
            for sid in range(self.nblocks):
                self._nlps[sid].evaluate_g(x_[sid], out=res[sid])
                # evaluate coupling Ax-z
                A = self._AB_csr[sid, sid]
                res[sid + self.nblocks] = A * x_[sid] - x_[self.nblocks]
            return res
        else:
            raise NotImplementedError("x must be a numpy array or a BlockVector")

    def evaluate_c(self, x, out=None, **kwargs):
        """Returns the equality constraints evaluated at x

        Parameters
        ----------
        x : array_like
            Array with values of primal variables.
        out : array_like
            Output array. Its type is preserved and it
            must be of the right shape to hold the output.

        Returns
        -------
        array_like

        """

        evaluated_g = kwargs.pop('evaluated_g', None)

        if out is None:
            res = self.create_vector('c')
        else:
            assert isinstance(out, BlockVector), 'Composite NLP takes block vector to evaluate g'
            assert out.nblocks == 2 * self.nblocks
            assert out.size == self.nc
            res = out

        if evaluated_g is not None:
            assert isinstance(evaluated_g, BlockVector), 'evaluated_g must be a BlockVector'
            assert evaluated_g.nblocks == 2 * self.nblocks
            assert evaluated_g.size == self.ng
            g = evaluated_g.compress(self._c_mask)
            if out is None:
                return g
            for bid, blk in enumerate(g):
                out[bid] = blk
            return out

        if isinstance(x, BlockVector):
            assert x.size == self.nx
            assert x.nblocks == self.nblocks + 1
            for sid in range(self.nblocks):
                self._nlps[sid].evaluate_c(x[sid], out=res[sid])
                A = self._AB_csr[sid, sid]
                res[sid + self.nblocks] = A * x[sid] - x[self.nblocks]
            return res
        elif isinstance(x, np.ndarray):
            assert x.size == self.nx
            block_x = self.create_vector('x')
            block_x.copyfrom(x)
            x_ = block_x
            for sid in range(self.nblocks):
                self._nlps[sid].evaluate_c(x_[sid], out=res[sid])
                A = self._AB_csr[sid, sid]
                res[sid + self.nblocks] = A * x_[sid] - x_[self.nblocks]
            return res
        else:
            raise NotImplementedError('x must be a numpy array or a BlockVector')

    def evaluate_d(self, x, out=None, **kwargs):
        """Returns the inequality constraints evaluated at x

        Parameters
        ----------
        x : array_like
            Array with values of primal variables.
        out : array_like
            Output array. Its type is preserved and it
            must be of the right shape to hold the output.

        Returns
        -------
        array_like

        """
        evaluated_g = kwargs.pop('evaluated_g', None)

        if out is None:
            res = self.create_vector('d')
        else:
            assert isinstance(out, BlockVector), 'Composite NLP takes block vector to evaluate g'
            assert out.nblocks == self.nblocks * 2
            assert out.size == self.nd
            res = out

        if evaluated_g is not None:
            assert isinstance(evaluated_g, BlockVector), 'evaluated_g must be a BlockVector'
            assert evaluated_g.nblocks == 2 * self.nblocks
            assert evaluated_g.size == self.ng
            d = evaluated_g.compress(self._d_mask)
            if out is None:
                return d
            for bid in range(self.nblocks):
                out[bid] = d[bid]
            return out

        if isinstance(x, BlockVector):
            assert x.size == self.nx
            assert x.nblocks == self.nblocks + 1
            for sid in range(self.nblocks):
                self._nlps[sid].evaluate_d(x[sid], out=res[sid])
                res[sid + self.nblocks] = np.zeros(0)  # empty z inequalities
            return res
        elif isinstance(x, np.ndarray):
            assert x.size == self.nx
            block_x = self.create_vector('x')
            block_x.copyfrom(x)
            x_ = block_x
            for sid in range(self.nblocks):
                self._nlps[sid].evaluate_d(x_[sid], out=res[sid])
                res[sid + self.nblocks] = np.zeros(0)  # empty z inequalities
            return res
        else:
            raise NotImplementedError("x must be a numpy array or a BlockVector")

    def jacobian_g(self, x, out=None, **kwargs):
        """Returns the Jacobian of the general inequalities evaluated at x

        Parameters
        ----------
        x : array_like
            Array with values of primal variables.
        out : BlockMatrix, optional
            Output matrix with the structure of the jacobian already defined.

        Returns
        -------
        BlockMatrix

        """
        assert x.size == self.nx, "Dimension mismatch"

        if isinstance(x, BlockVector):
            assert x.nblocks == self.nblocks + 1
            x_ = x
        elif isinstance(x, np.ndarray):
            block_x = self.create_vector('x')
            block_x.copyfrom(x)
            x_ = block_x
        else:
            raise RuntimeError("Input vector format not recognized")

        if out is None:
            jac_g = BlockMatrix(2 * self.nblocks, self.nblocks + 1)
            for sid, nlp in enumerate(self._nlps):
                xi = x_[sid]
                jac_g[sid, sid] = nlp.jacobian_g(xi)
                # coupling matrices Ai
                jac_g[sid + self.nblocks, sid] = self._AB_coo[sid, sid]
                # coupling matrices Bi
                jac_g[sid + self.nblocks, self.nblocks] = -identity(self.nz)
            return jac_g
        else:
            assert isinstance(out, BlockMatrix), 'out must be a BlockMatrix'
            assert out.bshape == (2 * self.nblocks, self.nblocks + 1), "Block shape mismatch"
            jac_g = out
            for sid, nlp in enumerate(self._nlps):
                xi = x_[sid]
                nlp.jacobian_g(xi, out=jac_g[sid, sid])
                # coupling matrices Ai
                jac_g[sid + self.nblocks, sid] = self._AB_coo[sid, sid]
                # coupling matrices Bi
                jac_g[sid + self.nblocks, self.nblocks] = -identity(self.nz)
            return jac_g

    def jacobian_c(self, x, out=None, **kwargs):
        """Returns the Jacobian of the equalities evaluated at x

        Parameters
        ----------
        x : array_like
            Array with values of primal variables.
        out : BlockMatrix, optional
            Output matrix with the structure of the jacobian already defined.

        Returns
        -------
        BlockMatrix

        """
        assert x.size == self.nx, 'Dimension mismatch'

        if isinstance(x, BlockVector):
            assert x.nblocks == self.nblocks + 1
            x_ = x
        elif isinstance(x, np.ndarray):
            block_x = self.create_vector('x')
            block_x.copyfrom(x)
            x_ = block_x
        else:
            raise RuntimeError('Input vector format not recognized')

        if out is None:
            jac_c = BlockMatrix(2 * self.nblocks, self.nblocks + 1)
            for sid, nlp in enumerate(self._nlps):
                xi = x_[sid]
                jac_c[sid, sid] = nlp.jacobian_c(xi)
                # coupling matrices Ai
                jac_c[sid + self.nblocks, sid] = self._AB_coo[sid, sid]
                # coupling matrices Bi
                jac_c[sid + self.nblocks, self.nblocks] = -identity(self.nz)
            return jac_c
        else:
            assert isinstance(out, BlockMatrix), 'out must be a BlockMatrix'
            assert out.bshape == (2 * self.nblocks, self.nblocks + 1), "Block shape mismatch"
            jac_c = out
            for sid, nlp in enumerate(self._nlps):
                xi = x_[sid]
                nlp.jacobian_c(xi, out=jac_c[sid, sid])
                # coupling matrices Ai
                jac_c[sid + self.nblocks, sid] = self._AB_coo[sid, sid]
                # coupling matrices Bi
                jac_c[sid + self.nblocks, self.nblocks] = -identity(self.nz)
            return jac_c

    def jacobian_d(self, x, out=None, **kwargs):
        """Returns the Jacobian of the inequalities evaluated at x

        Parameters
        ----------
        x : array_like
            Array with values of primal variables.
        out : coo_matrix, optional
            Output matrix with the structure of the jacobian already defined.

        Returns
        -------
        BlockMatrix

        """
        assert x.size == self.nx, "Dimension mismatch"

        if isinstance(x, BlockVector):
            assert x.nblocks == self.nblocks + 1
            x_ = x
        elif isinstance(x, np.ndarray):
            block_x = self.create_vector('x')
            block_x.copyfrom(x)
            x_ = block_x
        else:
            raise RuntimeError('Input vector format not recognized')

        if out is None:
            jac_d = BlockMatrix(2 * self.nblocks, self.nblocks + 1)
            for sid, nlp in enumerate(self._nlps):
                xi = x_[sid]
                jac_d[sid, sid] = nlp.jacobian_d(xi)
                jac_d[sid + self.nblocks, sid] = empty_matrix(0, xi.size)
                jac_d[sid + self.nblocks, self.nblocks] = empty_matrix(0, self.nz)
            return jac_d
        else:
            assert isinstance(out, BlockMatrix), 'out must be a BlockMatrix'
            assert out.bshape == (2 * self.nblocks, self.nblocks + 1), 'Block shape mismatch'
            jac_d = out
            for sid, nlp in enumerate(self._nlps):
                xi = x_[sid]
                nlp.jacobian_d(xi, out=jac_d[sid, sid])
                jac_d[sid + self.nblocks, sid] = empty_matrix(0, xi.size)
                jac_d[sid + self.nblocks, self.nblocks] = empty_matrix(0, self.nz)
            return jac_d

    def hessian_lag(self, x, y, out=None, **kwargs):
        """Return the Hessian of the Lagrangian function evaluated at x and y

        Parameters
        ----------
        x : array_like
            Array with values of primal variables.
        y : array_like
            Array with values of dual variables.
        out : BlockMatrix
            Output matrix with the structure of the hessian already defined. Optional

        Returns
        -------
        BlockMatrix

        """
        assert x.size == self.nx, 'Dimension mismatch'
        assert y.size == self.ng, 'Dimension mismatch'

        eval_f_c = kwargs.pop('eval_f_c', True)

        if isinstance(x, BlockVector) and isinstance(y, BlockVector):
            assert x.nblocks == self.nblocks + 1
            assert y.nblocks == 2 * self.nblocks
            x_ = x
            y_ = y
        elif isinstance(x, np.ndarray) and isinstance(y, BlockVector):
            assert y.nblocks == 2 * self.nblocks
            block_x = self.create_vector('x')
            block_x.copyfrom(x)
            x_ = block_x
            y_ = y
        elif isinstance(x, BlockVector) and isinstance(y, np.ndarray):
            assert x.nblocks == self.nblocks + 1
            x_ = x
            block_y = self.create_vector('y')
            block_y.copyfrom(y)
            y_ = block_y
        elif isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            block_x = self.create_vector('x')
            block_x.copyfrom(x)
            x_ = block_x
            block_y = self.create_vector('y')
            block_y.copyfrom(y)
            y_ = block_y
        else:
            raise NotImplementedError('Input vector format not recognized')

        if out is None:
            hess_lag = BlockSymMatrix(self.nblocks + 1)
            for sid, nlp in enumerate(self._nlps):
                xi = x_[sid]
                yi = y_[sid]
                hess_lag[sid, sid] = nlp.hessian_lag(xi, yi, eval_f_c=eval_f_c)

            hess_lag[self.nblocks, self.nblocks] = empty_matrix(self.nz, self.nz)
            return hess_lag
        else:
            assert isinstance(out, BlockSymMatrix), \
                'out must be a BlockSymMatrix'
            assert out.bshape == (self.nblocks + 1, self.nblocks + 1), \
                'Block shape mismatch'
            hess_lag = out
            for sid, nlp in enumerate(self._nlps):
                xi = x_[sid]
                yi = y_[sid]
                nlp.hessian_lag(xi,
                                yi,
                                out=hess_lag[sid, sid],
                                eval_f_c=eval_f_c)

            Hz = hess_lag[self.nblocks, self.nblocks]
            nb = self.nblocks
            assert Hz.shape == (self.nz, self.nz), \
                'out must have an {}x{} empty matrix in block {}'.format(nb,
                                                                         nb,
                                                                         (nb, nb))
            assert Hz.nnz == 0, \
                'out must have an empty matrix in block {}'.format((nb, nb))
            return hess_lag

    def block_id(self, scneario_name):
        """
        Returns idx of corresponding nlp for scenario_name

        Parameters
        ----------
        scenario_name : str
            name of scenario

        Returns
        -------
        int

        """
        return self._sname_to_sid[scneario_name]

    def block_name(self, bid):
        """
        Returns scenario name for given bid index

        Parameters
        ----------
        bid : int
            index of a given scenario

        Returns
        -------
        int

        """
        return self._sid_to_sname[bid]

    def get_block(self, scneario_name):
        """
        Returns nlp corresponding to scenario_name

        Parameters
        ----------
        scenario_name : str
            name of scenario

        Returns
        -------
        NLP

        """
        bid = self._sname_to_sid[scneario_name]
        return self._nlps[bid]

    def complicated_vars_ids(self, scenario_name):
        return self._zid_to_vid[scenario_name]

    def variable_order(self):

        var_order = list()
        for sid, nlp in enumerate(self._nlps):
            var_order.append(nlp.variable_order())
        var_order.append(['z{}'.format(i) for i in range(self.nz)])
        return var_order

    def constraint_order(self):

        constraint_order = list()
        for sid, nlp in enumerate(self._nlps):
            constraint_order.append(nlp.constraint_order())
        for sid, nlp in enumerate(self._nlps):
            constraint_order.append(['linking_b{}_{}'.format(sid, i) for i in range(self.nz)])
        return constraint_order

    def projection_matrix_xl(self):

        Pxl = BlockMatrix(self.nblocks + 1, self.nblocks + 1)
        for sid, nlp in enumerate(self._nlps):
            Pxl[sid, sid] = nlp.projection_matrix_xl()
        Pxl[self.nblocks, self.nblocks] = empty_matrix(self.nz, 0).tocsr()
        return Pxl

    def projection_matrix_xu(self):

        Pxu = BlockMatrix(self.nblocks + 1, self.nblocks + 1)
        for sid, nlp in enumerate(self._nlps):
            Pxu[sid, sid] = nlp.projection_matrix_xu()
        Pxu[self.nblocks, self.nblocks] = empty_matrix(self.nz, 0).tocsr()
        return Pxu

    def projection_matrix_dl(self):

        Pdl = BlockMatrix(2 * self.nblocks, 2 * self.nblocks)
        for sid, nlp in enumerate(self._nlps):
            Pdl[sid, sid] = nlp.projection_matrix_dl()
            Pdl[sid + self.nblocks, sid + self.nblocks] = empty_matrix(0, 0).tocsr()
        return Pdl

    def projection_matrix_du(self):

        Pdu = BlockMatrix(2 * self.nblocks, 2 * self.nblocks)
        for sid, nlp in enumerate(self._nlps):
            Pdu[sid, sid] = nlp.projection_matrix_du()
            Pdu[sid + self.nblocks, sid + self.nblocks] = empty_matrix(0, 0).tocsr()
        return Pdu

    def projection_matrix_c(self):

        Pc = BlockMatrix(2 * self.nblocks, 2 * self.nblocks)
        for sid, nlp in enumerate(self._nlps):
            Pc[sid, sid] = nlp.projection_matrix_c()
            Pc[sid + self.nblocks, sid + self.nblocks] = identity(self.nz).tocsr()
        return Pc

    def projection_matrix_d(self):

        Pd = BlockMatrix(2 * self.nblocks, 2 * self.nblocks)
        for sid, nlp in enumerate(self._nlps):
            Pd[sid, sid] = nlp.projection_matrix_d()
            Pd[sid + self.nblocks, sid + self.nblocks] = empty_matrix(self.nz, 0).tocsr()
        return Pd

    def coupling_matrix(self):

        AB = BlockMatrix(self.nblocks + 1, self.nblocks + 1)
        for sid, nlp in enumerate(self._nlps):
            col = self._zid_to_vid[sid]
            row = np.arange(self.nz, dtype=np.int)
            data = np.ones(self.nz)
            AB[sid, sid] = csr_matrix((data, (row, col)), shape=(self.nz, nlp.nx))
        AB[self.nblocks, self.nblocks] = -identity(self.nz)
        return AB

    def scenarios_order(self):
        return [self._sid_to_sname[i] for i in range(self.nblocks)]

    def report_solver_status(self, status_num, status_msg, x, y):
        raise NotImplementedError('TwoStageStochasticNLP does not support report_solver_status')


class DynoptNLP(NLP, CompositeNLP):
    """
    Nonlinear program interface for composite NLP that result from
    two-stage stochastic programming problems
    """

    def __init__(self,
                 nlps,
                 init_state_vars,
                 end_state_vars,
                 initial_conditions):

        """

        Parameters
        ----------
        model: dictionary with scenarios (scenario names to NLPs)

        """

        # call parent class to set model
        super(DynoptNLP, self).__init__(None)

        # initialize components
        self._initialize_nlp_components(nlps,
                                        init_state_vars,
                                        end_state_vars,
                                        initial_conditions)

    def _initialize_nlp_components(self, *args, **kwargs):

        nlps = args[0]
        init_state_vars = args[1]
        end_state_vars = args[2]
        initial_conditions = args[3]

        _n_states = -1
        # check initial state variables
        for k, l in enumerate(init_state_vars):
            if k == 0:
                _n_states = len(l)
            else:
                if len(l) != _n_states:
                    err_msg = "All blocks must have the same number of state variables"
                    raise RuntimeError(err_msg)
            for val in l:
                nlp = nlps[k]
                if val > nlp.nx:
                    raise RuntimeError("Variable index cannot be greater than number of vars in NLP")

        # check initial state variables
        for k, l in enumerate(end_state_vars):
            if k == 0:
                _n_states = len(l)
            else:
                if len(l) != _n_states:
                    err_msg = "All blocks must have the same number of state variables"
                    raise RuntimeError(err_msg)
            for val in l:
                nlp = nlps[k]
                if val > nlp.nx:
                    raise RuntimeError("Variable index cannot be greater than number of vars in NLP")

        # add blocks
        self._nlps = list()
        for blk in nlps:
            assert isinstance(blk, NLP), "Blocks must be instances of NLP class"
            self._nlps.append(blk)

        self._nstates = _n_states
        self._nz = (self.nblocks-1) * self.nstates

        # store initial conditions
        assert len(initial_conditions) == _n_states
        self._init_conditions = np.array([v for v in initial_conditions])

        # define map of block to initial state var
        # this goes from [block_id][zid] - > vid
        self._init_state_vars = list()
        for bid in range(self.nblocks):
            self._init_state_vars.append(init_state_vars[bid])

        # define map of block to end state var
        # this goes from [block_id][zid] - > vid
        self._end_state_vars = list()
        for bid in range(self.nblocks):
            self._end_state_vars.append(end_state_vars[bid])

        self._create_vectors()

        # define sizes
        self._nx = self._init_x.size  # this includes nz
        self._ng = self._init_y.size
        self._nc = sum(nlp.nc for nlp in self._nlps) + 2 * self.nstates * (self.nblocks - 1) + self.nstates
        self._nd = sum(nlp.nd for nlp in self._nlps)

        # cache coupling matrices
        self._A_start_csr = self.start_extraction_matrix()
        self._A_end_csr = self.end_extraction_matrix()
        self._A_start_coo = BlockMatrix(self.nblocks, self.nblocks)
        self._A_end_coo = BlockMatrix(self.nblocks, self.nblocks)

        for i in range(self.nblocks):
            self._A_start_coo[i, i] = self._A_start_csr[i, i].tocoo()
            self._A_end_coo[i, i] = self._A_end_csr[i, i]. tocoo()

        # define structure of jacobians
        self._create_jacobian_structures()

        # define structure Hessian
        self._create_hessian_structure()

    def _make_unmutable_caches(self):
        # no need for caches here
        pass

    def _create_vectors(self):

        # init values
        x_vectors = [nlp.x_init() for nlp in self._nlps]
        q_vectors = [np.zeros(self.nstates, dtype=np.double) for i in range(self.nblocks - 1)]
        self._init_x = BlockVector(x_vectors + q_vectors)

        y_vectors = [nlp.y_init() for nlp in self._nlps]
        q_vectors = [np.zeros(self.nstates, dtype=np.double)]  # for the initial conditions
        q_vectors += [np.zeros(self.nstates, dtype=np.double) for i in range(2*(self.nblocks-1))]
        self._init_y = BlockVector(y_vectors + q_vectors)

        # lower and upper bounds
        x_vectors = [nlp.xl() for nlp in self._nlps]
        q_vectors = [np.full(self.nstates, -np.inf, dtype=np.double) for i in range(self.nblocks - 1)]
        self._lower_x = BlockVector(x_vectors + q_vectors)

        x_vectors = [nlp.xu() for nlp in self._nlps]
        q_vectors = [np.full(self.nstates, np.inf, dtype=np.double) for i in range(self.nblocks - 1)]
        self._upper_x = BlockVector(x_vectors + q_vectors)

        y_vectors = [nlp.gl() for nlp in self._nlps]
        q_vectors = [np.zeros(self.nstates, dtype=np.double)]  # for the initial conditions
        q_vectors += [np.zeros(self.nstates, dtype=np.double) for i in range(2 * (self.nblocks - 1))]
        self._lower_g = BlockVector(y_vectors + q_vectors)

        y_vectors = [nlp.gu() for nlp in self._nlps]
        q_vectors = [np.zeros(self.nstates, dtype=np.double)]  # for the initial conditions
        q_vectors += [np.zeros(self.nstates, dtype=np.double) for i in range(2 * (self.nblocks - 1))]
        self._upper_g = BlockVector(y_vectors + q_vectors)

        # define x maps and masks
        self._lower_x_mask = np.isfinite(self._lower_x)
        self._lower_x_map = self._lower_x_mask.nonzero()[0]
        self._upper_x_mask = np.isfinite(self._upper_x)
        self._upper_x_map = self._upper_x_mask.nonzero()[0]

        # define gcd maps and masks
        bounds_difference = self._upper_g - self._lower_g
        abs_bounds_difference = np.absolute(bounds_difference)
        tolerance_equalities = 1e-8
        self._c_mask = abs_bounds_difference < tolerance_equalities
        self._c_map = self._c_mask.nonzero()[0]
        self._d_mask = abs_bounds_difference >= tolerance_equalities
        self._d_map = self._d_mask.nonzero()[0]

        self._lower_g_mask = np.isfinite(self._lower_g) * self._d_mask + self._c_mask
        self._lower_g_map = self._lower_g_mask.nonzero()[0]
        self._upper_g_mask = np.isfinite(self._upper_g) * self._d_mask + self._c_mask
        self._upper_g_map = self._upper_g_mask.nonzero()[0]

        self._lower_d_mask = pn.isin(self._d_map, self._lower_g_map)
        self._upper_d_mask = pn.isin(self._d_map, self._upper_g_map)

        # define lower and upper d maps
        self._lower_d_map = pn.where(self._lower_d_mask)[0]
        self._upper_d_map = pn.where(self._upper_d_mask)[0]

        # get lower and upper d values
        self._lower_d = np.compress(self._d_mask, self._lower_g)
        self._upper_d = np.compress(self._d_mask, self._upper_g)

    def _create_jacobian_structures(self):

        # Note: This method requires the init_state_var and end_state_var map to be
        # created beforehand
        jac_g = BlockMatrix(3 * self.nblocks - 1, 2 * self.nblocks - 1)
        for bid, nlp in enumerate(self._nlps):
            xi = nlp.x_init()
            jac_g[bid, bid] = nlp.jacobian_g(xi)
            jac_g[bid + self.nblocks, bid] = self._A_start_coo[bid, bid]
            if bid > 0:
                jac_g[bid + self.nblocks, self.nblocks + bid-1] = -identity(self.nstates)
            if bid < self.nblocks-1:
                jac_g[bid + 2 * self.nblocks, bid] = -self._A_end_coo[bid, bid]
                jac_g[bid + 2 * self.nblocks, self.nblocks + bid] = identity(self.nstates)

        self._internal_jacobian_g = jac_g
        flat_jac_g = jac_g.tocoo()
        self._irows_jac_g = flat_jac_g.row
        self._jcols_jac_g = flat_jac_g.col
        self._nnz_jac_g = flat_jac_g.nnz

        jac_c = BlockMatrix(3 * self.nblocks - 1, 2 * self.nblocks - 1)
        for bid, nlp in enumerate(self._nlps):
            xi = nlp.x_init()
            jac_c[bid, bid] = nlp.jacobian_c(xi)
            jac_c[bid + self.nblocks, bid] = self._A_start_coo[bid, bid]
            if bid > 0:
                jac_c[bid + self.nblocks, self.nblocks + bid - 1] = -identity(self.nstates)
            if bid < self.nblocks - 1:
                jac_c[bid + 2 * self.nblocks, bid] = -self._A_end_coo[bid, bid]
                jac_c[bid + 2 * self.nblocks, self.nblocks + bid] = identity(self.nstates)

        self._internal_jacobian_c = jac_c
        flat_jac_c = jac_c.tocoo()
        self._irows_jac_c = flat_jac_c.row
        self._jcols_jac_c = flat_jac_c.col
        self._nnz_jac_c = flat_jac_c.nnz

        jac_d = BlockMatrix(3 * self.nblocks - 1, 2 * self.nblocks - 1)
        for bid, nlp in enumerate(self._nlps):
            xi = nlp.x_init()
            jac_d[bid, bid] = nlp.jacobian_d(xi)
            jac_d[bid + self.nblocks, bid] = empty_matrix(0, xi.size)
            if bid > 0:
                jac_d[bid + self.nblocks, self.nblocks + bid - 1] = empty_matrix(0, self.nstates)
            if bid < self.nblocks - 1:
                jac_d[bid + 2 * self.nblocks, bid] = empty_matrix(0, xi.size)
                jac_d[bid + 2 * self.nblocks, self.nblocks + bid] = empty_matrix(0, self.nstates)

        self._internal_jacobian_d = jac_d
        flat_jac_d = jac_d.tocoo()
        self._irows_jac_d = flat_jac_d.row
        self._jcols_jac_d = flat_jac_d.col
        self._nnz_jac_d = flat_jac_d.nnz

    def _create_hessian_structure(self):

        # Note: This method requires the init_state_var and end_state_var map to be
        # created beforehand
        hess_lag = BlockSymMatrix(2 * self.nblocks - 1)
        for bid, nlp in enumerate(self._nlps):
            xi = nlp.x_init()
            yi = nlp.y_init()
            hess_lag[bid, bid] = nlp.hessian_lag(xi, yi)
            if bid < self.nblocks - 1:
                hess_lag[self.nblocks + bid, self.nblocks + bid] = empty_matrix(self.nstates,
                                                                                self.nstates)

        self._internal_hess_lag = hess_lag
        flat_hess = hess_lag.tocoo()
        self._irows_hess = flat_hess.row
        self._jcols_hess = flat_hess.col
        self._nnz_hess_lag = flat_hess.nnz

    @property
    def model(self):
        """
        Return optimization model
        """
        return self._model

    @property
    def nblocks(self):
        """
        Returns number of blocks (nlps)
        """
        return len(self._nlps)

    @property
    def nstates(self):
        """
        Returns number of blocks (nlps)
        """
        return self._nstates

    @property
    def nz(self):
        """
        Return number of complicated variables
        """
        return self._nz

    def nlps(self):
        """Creates generator scenario name to nlp """
        for sid, block_nlp in enumerate(self._nlps):
            yield "block {}".format(sid), block_nlp

    def create_vector(self, vector_type):
        if vector_type == 'x':
            x_vectors = [np.zeros(nlp.nx, dtype=np.double) for nlp in self._nlps]
            q_vectors = [np.zeros(self.nstates, dtype=np.double) for i in range(self.nblocks - 1)]
            return BlockVector(x_vectors + q_vectors)
        elif vector_type == 'xl':
            vectors = list()
            for nlp in self._nlps:
                nx_l = len(nlp._lower_x_map)
                xl = np.zeros(nx_l, dtype=np.double)
                vectors.append(xl)
            # complicated variables have no lower bounds
            q_vectors = [np.zeros(0, dtype=np.double) for i in range(self.nblocks - 1)]
            vectors += q_vectors
            return BlockVector(vectors)
        elif vector_type == 'xu':
            vectors = list()
            for nlp in self._nlps:
                nx_u = len(nlp._upper_x_map)
                xu = np.zeros(nx_u, dtype=np.double)
                vectors.append(xu)
            q_vectors = [np.zeros(0, dtype=np.double) for i in range(self.nblocks - 1)]
            vectors += q_vectors
            return BlockVector(vectors)
        if vector_type == 'g' or vector_type == 'y':
            y_vectors = [np.zeros(nlp.ng, dtype=np.double) for nlp in self._nlps]
            q_vectors = [np.zeros(self.nstates, dtype=np.double)]  # for the initial conditions
            q_vectors += [np.zeros(self.nstates, dtype=np.double) for i in range(2 * (self.nblocks - 1))]
            return BlockVector(y_vectors + q_vectors)
        elif vector_type == 'c' or vector_type == 'yc':
            y_vectors = [np.zeros(nlp.nc, dtype=np.double) for nlp in self._nlps]
            q_vectors = [np.zeros(self.nstates, dtype=np.double)]  # for the initial conditions
            q_vectors += [np.zeros(self.nstates, dtype=np.double) for i in range(2 * (self.nblocks - 1))]
            return BlockVector(y_vectors + q_vectors)
        elif vector_type == 'd' or vector_type == 'd' or vector_type == 's':
            y_vectors = [np.zeros(nlp.nd, dtype=np.double) for nlp in self._nlps]
            q_vectors = [np.zeros(0, dtype=np.double)]  # for the initial conditions
            q_vectors += [np.zeros(0, dtype=np.double) for i in range(2 * (self.nblocks - 1))]
            return BlockVector(y_vectors + q_vectors)
        elif vector_type == 'dl' or vector_type == 'du':
            y_vectors = [nlp.create_vector(vector_type) for nlp in self._nlps]
            q_vectors = [np.zeros(0, dtype=np.double)]  # for the initial conditions
            q_vectors += [np.zeros(0, dtype=np.double) for i in range(2 * (self.nblocks - 1))]
            return BlockVector(y_vectors + q_vectors)
        else:
            raise RuntimeError('Vector_type not recognized')

    def objective(self, x, **kwargs):
        """Returns value of objective function evaluated at x

        Parameters
        ----------
        x : array_like
            Array with values of primal variables.

        Returns
        -------
        float

        """
        if isinstance(x, BlockVector):
            return sum(self._nlps[i].objective(x[i]) for i in range(self.nblocks))
        elif isinstance(x, np.ndarray):
            block_x = self.create_vector('x')
            block_x.copyfrom(x)
            x_ = block_x
            return sum(self._nlps[i].objective(x_[i]) for i in range(self.nblocks))
        else:
            raise NotImplementedError("x must be a numpy array or a BlockVector")

    def grad_objective(self, x, out=None, **kwargs):
        """Returns gradient of the objective function evaluated at x

        Parameters
        ----------
        x : array_like
            Array with values of primal variables.
        out : array_like
            Output array. Its type is preserved and it
            must be of the right shape to hold the output.

        Returns
        -------
        array_like

        """
        if out is None:
            df = self.create_vector('x')
        else:
            assert isinstance(out, BlockVector), 'Composite NLP takes block vector to evaluate g'
            assert out.nblocks == 2 * self.nblocks - 1
            assert out.size == self.nx
            df = out

        if isinstance(x, BlockVector):
            assert x.size == self.nx
            assert x.nblocks == 2 * self.nblocks - 1
            for i in range(self.nblocks):
                self._nlps[i].grad_objective(x[i], out=df[i])
            return df
        elif isinstance(x, np.ndarray):
            assert x.size == self.nx
            block_x = self.create_vector('x')
            block_x.copyfrom(x)
            x_ = block_x
            for i in range(self.nblocks):
                self._nlps[i].grad_objective(x_[i], out=df[i])
            return df
        else:
            raise NotImplementedError("x must be a numpy array or a BlockVector")

    def evaluate_g(self, x, out=None, **kwargs):
        """Returns general inequality constraints evaluated at x

        Parameters
        ----------
        x : array_like
            Array with values of primal variables.
        out : array_like
            Output array. Its type is preserved and it
            must be of the right shape to hold the output.

        Returns
        -------
        array_like

        """
        if out is None:
            res = self.create_vector('y')
        else:
            assert isinstance(out, BlockVector), 'Composite NLP takes block vector to evaluate g'
            assert out.nblocks == 3 * self.nblocks - 1
            assert out.size == self.ng
            res = out

        if isinstance(x, BlockVector):
            assert x.size == self.nx
            assert x.nblocks == 2 * self.nblocks - 1
            for bid in range(self.nblocks):
                # evaluate gi
                self._nlps[bid].evaluate_g(x[bid], out=res[bid])

                # evaluate coupling start
                A = self._A_start_csr[bid, bid]
                if bid == 0:
                    res[bid + self.nblocks] = A * x[bid] - self._init_conditions
                else:
                    res[bid + self.nblocks] = A * x[bid] - x[self.nblocks + bid-1]

                # evaluate coupling end
                A = self._A_end_csr[bid, bid]
                if bid < self.nblocks - 1:
                    res[bid + 2 * self.nblocks] = x[self.nblocks + bid] - A * x[bid]

            return res
        elif isinstance(x, np.ndarray):
            assert x.size == self.nx
            block_x = self.create_vector('x')
            block_x.copyfrom(x)  # this is expensive
            x_ = block_x
            for bid in range(self.nblocks):
                self._nlps[bid].evaluate_g(x_[bid], out=res[bid])

                # evaluate coupling start
                A = self._A_start_csr[bid, bid]
                if bid == 0:
                    res[bid + self.nblocks] = A * x_[bid] - self._init_conditions
                else:
                    res[bid + self.nblocks] = A * x_[bid] - x_[self.nblocks + bid - 1]

                # evaluate coupling end
                A = self._A_end_csr[bid, bid]
                if bid < self.nblocks - 1:
                    res[bid + 2 * self.nblocks] = x_[self.nblocks + bid] - A * x_[bid]
            return res
        else:
            raise NotImplementedError("x must be a numpy array or a BlockVector")

    def evaluate_c(self, x, out=None, **kwargs):
        """Returns the equality constraints evaluated at x

        Parameters
        ----------
        x : array_like
            Array with values of primal variables.
        out : array_like
            Output array. Its type is preserved and it
            must be of the right shape to hold the output.

        Returns
        -------
        array_like

        """

        evaluated_g = kwargs.pop('evaluated_g', None)

        if out is None:
            res = self.create_vector('c')
        else:
            assert isinstance(out, BlockVector), 'Composite NLP takes block vector to evaluate g'
            assert out.nblocks == 3 * self.nblocks - 1
            assert out.size == self.nc
            res = out

        if evaluated_g is not None:
            assert isinstance(evaluated_g, BlockVector), 'evaluated_g must be a BlockVector'
            assert evaluated_g.nblocks == 3 * self.nblocks - 1
            assert evaluated_g.size == self.ng
            g = evaluated_g.compress(self._c_mask)
            if out is None:
                return g
            for bid, blk in enumerate(g):
                out[bid] = blk
            return out

        if isinstance(x, BlockVector):
            assert x.size == self.nx
            assert x.nblocks == 2 * self.nblocks - 1
            for bid in range(self.nblocks):
                self._nlps[bid].evaluate_c(x[bid], out=res[bid])
                # evaluate coupling start
                A = self._A_start_csr[bid, bid]
                if bid == 0:
                    res[bid + self.nblocks] = A * x[bid] - self._init_conditions
                else:
                    res[bid + self.nblocks] = A * x[bid] - x[self.nblocks + bid - 1]

                # evaluate coupling end
                A = self._A_end_csr[bid, bid]
                if bid < self.nblocks - 1:
                    res[bid + 2 * self.nblocks] = x[self.nblocks + bid] - A * x[bid]
            return res
        elif isinstance(x, np.ndarray):
            assert x.size == self.nx
            block_x = self.create_vector('x')
            block_x.copyfrom(x)
            x_ = block_x
            for bid in range(self.nblocks):
                self._nlps[bid].evaluate_c(x_[bid], out=res[bid])
                # evaluate coupling start
                A = self._A_start_csr[bid, bid]
                if bid == 0:
                    res[bid + self.nblocks] = A * x_[bid] - self._init_conditions
                else:
                    res[bid + self.nblocks] = A * x_[bid] - x_[self.nblocks + bid - 1]

                # evaluate coupling end
                A = self._A_end_csr[bid, bid]
                if bid < self.nblocks - 1:
                    res[bid + 2 * self.nblocks] = x_[self.nblocks + bid] - A * x_[bid]
            return res
        else:
            raise NotImplementedError('x must be a numpy array or a BlockVector')

    def evaluate_d(self, x, out=None, **kwargs):
        """Returns the inequality constraints evaluated at x

        Parameters
        ----------
        x : array_like
            Array with values of primal variables.
        out : array_like
            Output array. Its type is preserved and it
            must be of the right shape to hold the output.

        Returns
        -------
        array_like

        """
        evaluated_g = kwargs.pop('evaluated_g', None)

        if out is None:
            res = self.create_vector('d')
        else:
            assert isinstance(out, BlockVector), 'Composite NLP takes block vector to evaluate g'
            assert out.nblocks == 3 * self.nblocks - 1
            assert out.size == self.nd
            res = out

        if evaluated_g is not None:
            assert isinstance(evaluated_g, BlockVector), 'evaluated_g must be a BlockVector'
            assert evaluated_g.nblocks == 3 * self.nblocks - 1
            assert evaluated_g.size == self.ng
            d = evaluated_g.compress(self._d_mask)
            if out is None:
                return d
            for bid in range(self.nblocks):
                out[bid] = d[bid]
            return out

        if isinstance(x, BlockVector):
            assert x.size == self.nx
            assert x.nblocks == 2 * self.nblocks - 1
            for bid in range(self.nblocks):
                self._nlps[bid].evaluate_d(x[bid], out=res[bid])
                if bid == 0:
                    res[bid + self.nblocks] = np.zeros(0)  # empty z inequalities
                else:
                    res[bid + self.nblocks] = np.zeros(0)  # empty z inequalities

                if bid < self.nblocks - 1:
                    res[bid + 2 * self.nblocks] = np.zeros(0)  # empty z inequalities
            return res
        elif isinstance(x, np.ndarray):
            assert x.size == self.nx
            block_x = self.create_vector('x')
            block_x.copyfrom(x)
            x_ = block_x
            for bid in range(self.nblocks):
                self._nlps[bid].evaluate_d(x_[bid], out=res[bid])
                if bid == 0:
                    res[bid + self.nblocks] = np.zeros(0)  # empty z inequalities
                else:
                    res[bid + self.nblocks] = np.zeros(0)  # empty z inequalities

                if bid < self.nblocks - 1:
                    res[bid + 2 * self.nblocks] = np.zeros(0)  # empty z inequalities
            return res
        else:
            raise NotImplementedError("x must be a numpy array or a BlockVector")

    def jacobian_g(self, x, out=None, **kwargs):
        """Returns the Jacobian of the general inequalities evaluated at x

        Parameters
        ----------
        x : array_like
            Array with values of primal variables.
        out : BlockMatrix, optional
            Output matrix with the structure of the jacobian already defined.

        Returns
        -------
        BlockMatrix

        """
        assert x.size == self.nx, "Dimension mismatch"

        if isinstance(x, BlockVector):
            assert x.nblocks == 2 * self.nblocks - 1
            x_ = x
        elif isinstance(x, np.ndarray):
            block_x = self.create_vector('x')
            block_x.copyfrom(x)
            x_ = block_x
        else:
            raise RuntimeError("Input vector format not recognized")

        if out is None:
            jac_g = BlockMatrix(3 * self.nblocks - 1, 2 * self.nblocks - 1)
            for bid, nlp in enumerate(self._nlps):
                xi = x_[bid]
                jac_g[bid, bid] = nlp.jacobian_g(xi)
                # coupling matrices start block
                jac_g[bid + self.nblocks, bid] = self._A_start_coo[bid, bid]
                if bid > 0:
                    jac_g[bid + self.nblocks, self.nblocks + bid - 1] = -identity(self.nstates)
                if bid < self.nblocks - 1:
                    # coupling matrices end block
                    jac_g[bid + 2 * self.nblocks, bid] = -self._A_end_coo[bid, bid]
                    jac_g[bid + 2 * self.nblocks, self.nblocks + bid] = identity(self.nstates)
            return jac_g
        else:
            assert isinstance(out, BlockMatrix), 'out must be a BlockMatrix'
            assert out.bshape == (3 * self.nblocks - 1, 2 * self.nblocks - 1), "Block shape mismatch"
            jac_g = out
            for bid, nlp in enumerate(self._nlps):
                xi = x_[bid]
                nlp.jacobian_g(xi, out=jac_g[bid, bid])

                # coupling matrices start block
                jac_g[bid + self.nblocks, bid] = self._A_start_coo[bid, bid]
                if bid > 0:
                    jac_g[bid + self.nblocks, self.nblocks + bid - 1] = -identity(self.nstates)
                if bid < self.nblocks - 1:
                    # coupling matrices end block
                    jac_g[bid + 2 * self.nblocks, bid] = -self._A_end_coo[bid, bid]
                    jac_g[bid + 2 * self.nblocks, self.nblocks + bid] = identity(self.nstates)

            return jac_g

    def jacobian_c(self, x, out=None, **kwargs):
        """Returns the Jacobian of the equalities evaluated at x

        Parameters
        ----------
        x : array_like
            Array with values of primal variables.
        out : BlockMatrix, optional
            Output matrix with the structure of the jacobian already defined.

        Returns
        -------
        BlockMatrix

        """
        assert x.size == self.nx, "Dimension mismatch"

        if isinstance(x, BlockVector):
            assert x.nblocks == 2 * self.nblocks - 1
            x_ = x
        elif isinstance(x, np.ndarray):
            block_x = self.create_vector('x')
            block_x.copyfrom(x)
            x_ = block_x
        else:
            raise RuntimeError("Input vector format not recognized")

        if out is None:
            jac_c = BlockMatrix(3 * self.nblocks - 1, 2 * self.nblocks - 1)
            for bid, nlp in enumerate(self._nlps):
                xi = x_[bid]
                jac_c[bid, bid] = nlp.jacobian_c(xi)
                # coupling matrices start block
                jac_c[bid + self.nblocks, bid] = self._A_start_coo[bid, bid]
                if bid > 0:
                    jac_c[bid + self.nblocks, self.nblocks + bid - 1] = -identity(self.nstates)
                if bid < self.nblocks - 1:
                    # coupling matrices end block
                    jac_c[bid + 2 * self.nblocks, bid] = -self._A_end_coo[bid, bid]
                    jac_c[bid + 2 * self.nblocks, self.nblocks + bid] = identity(self.nstates)
            return jac_c
        else:
            assert isinstance(out, BlockMatrix), 'out must be a BlockMatrix'
            assert out.bshape == (3 * self.nblocks - 1, 2 * self.nblocks - 1), "Block shape mismatch"
            jac_c = out
            for bid, nlp in enumerate(self._nlps):
                xi = x_[bid]
                nlp.jacobian_c(xi, out=jac_c[bid, bid])

                # coupling matrices start block
                jac_c[bid + self.nblocks, bid] = self._A_start_coo[bid, bid]
                if bid > 0:
                    jac_c[bid + self.nblocks, self.nblocks + bid - 1] = -identity(self.nstates)
                if bid < self.nblocks - 1:
                    # coupling matrices end block
                    jac_c[bid + 2 * self.nblocks, bid] = -self._A_end_coo[bid, bid]
                    jac_c[bid + 2 * self.nblocks, self.nblocks + bid] = identity(self.nstates)
            return jac_c

    def jacobian_d(self, x, out=None, **kwargs):
        """Returns the Jacobian of the inequalities evaluated at x

        Parameters
        ----------
        x : array_like
            Array with values of primal variables.
        out : coo_matrix, optional
            Output matrix with the structure of the jacobian already defined.

        Returns
        -------
        BlockMatrix

        """
        assert x.size == self.nx, "Dimension mismatch"

        if isinstance(x, BlockVector):
            assert x.nblocks == 2 * self.nblocks - 1
            x_ = x
        elif isinstance(x, np.ndarray):
            block_x = self.create_vector('x')
            block_x.copyfrom(x)
            x_ = block_x
        else:
            raise RuntimeError('Input vector format not recognized')

        if out is None:
            jac_d = BlockMatrix(3 * self.nblocks - 1, 2 * self.nblocks - 1)
            for bid, nlp in enumerate(self._nlps):
                xi = x_[bid]
                jac_d[bid, bid] = nlp.jacobian_d(xi)
                jac_d[bid + self.nblocks, bid] = empty_matrix(0, xi.size)
                if bid > 0:
                    jac_d[bid + self.nblocks, self.nblocks + bid - 1] = empty_matrix(0, self.nstates)
                if bid < self.nblocks - 1:
                    jac_d[bid + 2 * self.nblocks, bid] = empty_matrix(0, xi.size)
                    jac_d[bid + 2 * self.nblocks, self.nblocks + bid] = empty_matrix(0, self.nstates)
            return jac_d
        else:
            assert isinstance(out, BlockMatrix), 'out must be a BlockMatrix'
            assert out.bshape == (3 * self.nblocks - 1, 2 * self.nblocks - 1), 'Block shape mismatch'
            jac_d = out
            for bid, nlp in enumerate(self._nlps):
                xi = x_[bid]
                nlp.jacobian_d(xi, out=jac_d[bid, bid])
                jac_d[bid + self.nblocks, bid] = empty_matrix(0, xi.size)
                if bid > 0:
                    jac_d[bid + self.nblocks, self.nblocks + bid - 1] = empty_matrix(0, self.nstates)
                if bid < self.nblocks - 1:
                    jac_d[bid + 2 * self.nblocks, bid] = empty_matrix(0, xi.size)
                    jac_d[bid + 2 * self.nblocks, self.nblocks + bid] = empty_matrix(0, self.nstates)
            return jac_d

    def hessian_lag(self, x, y, out=None, **kwargs):
        """Return the Hessian of the Lagrangian function evaluated at x and y

        Parameters
        ----------
        x : array_like
            Array with values of primal variables.
        y : array_like
            Array with values of dual variables.
        out : BlockMatrix
            Output matrix with the structure of the hessian already defined. Optional

        Returns
        -------
        BlockMatrix

        """
        assert x.size == self.nx, 'Dimension mismatch'
        assert y.size == self.ng, 'Dimension mismatch'

        eval_f_c = kwargs.pop('eval_f_c', True)

        if isinstance(x, BlockVector) and isinstance(y, BlockVector):
            assert x.nblocks == 2 * self.nblocks - 1
            assert y.nblocks == 3 * self.nblocks - 1
            x_ = x
            y_ = y
        elif isinstance(x, np.ndarray) and isinstance(y, BlockVector):
            assert y.nblocks == 3 * self.nblocks - 1
            block_x = self.create_vector('x')
            block_x.copyfrom(x)
            x_ = block_x
            y_ = y
        elif isinstance(x, BlockVector) and isinstance(y, np.ndarray):
            assert x.nblocks == 2 * self.nblocks - 1
            x_ = x
            block_y = self.create_vector('y')
            block_y.copyfrom(y)
            y_ = block_y
        elif isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            block_x = self.create_vector('x')
            block_x.copyfrom(x)
            x_ = block_x
            block_y = self.create_vector('y')
            block_y.copyfrom(y)
            y_ = block_y
        else:
            raise NotImplementedError('Input vector format not recognized')

        if out is None:

            hess_lag = BlockSymMatrix(2 * self.nblocks - 1)
            for bid, nlp in enumerate(self._nlps):
                xi = x_[bid]
                yi = y_[bid]
                hess_lag[bid, bid] = nlp.hessian_lag(xi, yi, eval_f_c=eval_f_c)
                if bid < self.nblocks - 1:
                    hess_lag[self.nblocks + bid, self.nblocks + bid] = empty_matrix(self.nstates,
                                                                                    self.nstates)
            return hess_lag
        else:
            assert isinstance(out, BlockSymMatrix), \
                'out must be a BlockSymMatrix'
            assert out.bshape == (2 * self.nblocks - 1, 2 * self.nblocks - 1), \
                'Block shape mismatch'
            hess_lag = out
            for bid, nlp in enumerate(self._nlps):
                xi = x_[bid]
                yi = y_[bid]
                nlp.hessian_lag(xi,
                                yi,
                                out=hess_lag[bid, bid],
                                eval_f_c=eval_f_c)

                if bid < self.nblocks - 1:
                    hess_lag[self.nblocks + bid, self.nblocks + bid] = empty_matrix(self.nstates,
                                                                                    self.nstates)
            return hess_lag

    def start_extraction_matrix(self):

        A = BlockMatrix(self.nblocks, self.nblocks)

        for bid in range(self.nblocks):
            nlp = self._nlps[bid]
            col = np.array(self._init_state_vars[bid])
            row = np.arange(self.nstates)
            data = np.ones(self.nstates)
            A[bid, bid] = csr_matrix((data, (row, col)), shape=(self.nstates, nlp.nx))

        return A

    def end_extraction_matrix(self):

        A = BlockMatrix(self.nblocks, self.nblocks)

        for bid in range(self.nblocks):
            nlp = self._nlps[bid]
            col = np.array(self._end_state_vars[bid])
            row = np.arange(self.nstates)
            data = np.ones(self.nstates)
            A[bid, bid] = csr_matrix((data, (row, col)), shape=(self.nstates, nlp.nx))

        return A

    def projection_matrix_xl(self):

        Pxl = BlockMatrix(2 * self.nblocks - 1, 2 * self.nblocks - 1)
        for bid, nlp in enumerate(self._nlps):
            Pxl[bid, bid] = nlp.projection_matrix_xl()
            if bid < self.nblocks - 1:
                Pxl[self.nblocks + bid, self.nblocks + bid] = empty_matrix(self.nstates, 0)
        return Pxl

    def projection_matrix_xu(self):

        Pxu = BlockMatrix(2 * self.nblocks - 1, 2 * self.nblocks - 1)
        for bid, nlp in enumerate(self._nlps):
            Pxu[bid, bid] = nlp.projection_matrix_xu()
            if bid < self.nblocks - 1:
                Pxu[self.nblocks + bid, self.nblocks + bid] = empty_matrix(self.nstates, 0)
        return Pxu

    def projection_matrix_dl(self):

        Pdl = BlockMatrix(3 * self.nblocks - 1, 3 * self.nblocks - 1)
        for bid, nlp in enumerate(self._nlps):
            Pdl[bid, bid] = nlp.projection_matrix_dl()
            Pdl[bid + self.nblocks, bid + self.nblocks] = empty_matrix(0, 0).tocsr()
            if bid < self.nblocks - 1:
                Pdl[bid + 2 * self.nblocks, bid + 2 * self.nblocks] = empty_matrix(0, 0).tocsr()
                Pdl[bid + 2 * self.nblocks, bid + 2 * self.nblocks] = empty_matrix(0, 0).tocsr()
        return Pdl

    def projection_matrix_du(self):

        Pdu = BlockMatrix(3 * self.nblocks - 1, 3 * self.nblocks - 1)
        for bid, nlp in enumerate(self._nlps):
            Pdu[bid, bid] = nlp.projection_matrix_du()
            Pdu[bid + self.nblocks, bid + self.nblocks] = empty_matrix(0, 0).tocsr()
            if bid < self.nblocks - 1:
                Pdu[bid + 2 * self.nblocks, bid + 2 * self.nblocks] = empty_matrix(0, 0).tocsr()
                Pdu[bid + 2 * self.nblocks, bid + 2 * self.nblocks] = empty_matrix(0, 0).tocsr()
        return Pdu

    def projection_matrix_c(self):

        Pc = BlockMatrix(3 * self.nblocks - 1, 3 * self.nblocks - 1)
        for bid, nlp in enumerate(self._nlps):
            Pc[bid, bid] = nlp.projection_matrix_c()
            Pc[bid + self.nblocks, bid + self.nblocks] = identity(self.nstates).tocsr()
            if bid < self.nblocks - 1:
                Pc[bid + 2 * self.nblocks, bid + 2 * self.nblocks] = identity(self.nstates).tocsr()
                Pc[bid + 2 * self.nblocks, bid + 2 * self.nblocks] = identity(self.nstates).tocsr()
        return Pc

    def projection_matrix_d(self):

        Pd = BlockMatrix(3 * self.nblocks - 1, 3 * self.nblocks - 1)
        for bid, nlp in enumerate(self._nlps):
            Pd[bid, bid] = nlp.projection_matrix_d()
            Pd[bid + self.nblocks, bid + self.nblocks] = empty_matrix(self.nstates, 0).tocsr()
            if bid < self.nblocks - 1:
                Pd[bid + 2 * self.nblocks, bid + 2 * self.nblocks] = empty_matrix(self.nstates, 0).tocsr()
                Pd[bid + 2 * self.nblocks, bid + 2 * self.nblocks] = empty_matrix(self.nstates, 0).tocsr()
        return Pd

    def variable_order(self):

        var_order = list()
        for bid, nlp in enumerate(self._nlps):
            var_order.append(nlp.variable_order())
        for bid in range(self.nblocks-1):
            var_order.append(['q{}_{}'.format(bid, i) for i in range(self.nstates)])
        return var_order

    def constraint_order(self):

        constraint_order = list()
        for bid, nlp in enumerate(self._nlps):
            constraint_order.append(nlp.constraint_order())
        for bid in range(self.nblocks):
            if bid == 0:
                constraint_order.append(['initial_condition_{}'.format(i) for i in range(self.nstates)])
            else:
                constraint_order.append(['linking_start_b{}_{}'.format(bid, i) for i in range(self.nstates)])
        for bid in range(self.nblocks-1):
            constraint_order.append(['linking_end_b{}_{}'.format(bid, i) for i in range(self.nstates)])
        return constraint_order

    def initial_conditions(self):
        return np.array(self._init_conditions, dtype=np.double)

    def report_solver_status(self, status_num, status_msg, x, y):
        raise NotImplementedError('DynoptNLP does not support report_solver_status')
