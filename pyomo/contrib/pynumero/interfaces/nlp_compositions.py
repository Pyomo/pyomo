from pyomo.contrib.pynumero.interfaces.nlp import NLP
from pyomo.contrib.pynumero.sparse import (BlockMatrix,
                             BlockSymMatrix,
                             BlockVector,
                             EmptyMatrix,
                             IdentityMatrix,
                             COOMatrix)
from collections import OrderedDict
import pyomo.environ as aml
import numpy as np

# ToDo: Create another one on top for general composite nlps?

__all__ = ['TwoStageStochasticNLP']


class TwoStageStochasticNLP(NLP):
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
        model: dictionary with scenarios (scenario names to NLPs)
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
                    err_msg = "All scenarios must have the same number of complicated variables"
                    raise RuntimeError(err_msg)
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

        # ToDo: workout masks and maps as block vectors
        """
            _lower_x_mask
            _upper_x_mask
            _lower_x_map
            _upper_x_map
            _lower_g_mask
            _upper_g_mask
            _lower_g_map
            _upper_g_map

            _d_mask
            _d_map
            _lower_d
            _upper_d
            _lower_d_mask
            _upper_d_mask
            _lower_d_map
            _upper_d_map

            _c_mask
            _c_map
        """

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
            jac_g[sid + self.nblocks, sid] = COOMatrix((data, (row, col)),
                                                       shape=(self.nz, nlp.nx))

            # coupling matrices Bi
            jac_g[sid + self.nblocks, self.nblocks] = -IdentityMatrix(self.nz)

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
            jac_c[sid + self.nblocks, sid] = COOMatrix((data, (row, col)),
                                                       shape=(self.nz, nlp.nx))

            # coupling matrices Bi
            jac_c[sid + self.nblocks, self.nblocks] = -IdentityMatrix(self.nz)

        self._internal_jacobian_c = jac_c
        flat_jac_c = jac_c.tocoo()
        self._irows_jac_c = flat_jac_c.row
        self._jcols_jac_c = flat_jac_c.col
        self._nnz_jac_c = flat_jac_c.nnz

        # build jacobian inequality constraints
        jac_d = BlockMatrix(self.nblocks, self.nblocks)
        for sid, nlp in enumerate(self._nlps):
            xi = nlp.x_init()
            jac_d[sid, sid] = nlp.jacobian_d(xi)
        self._internal_jacobian_d = jac_d
        flat_jac_d = jac_d.tocoo()
        self._irows_jac_d = flat_jac_d.row
        self._jcols_jac_d = flat_jac_d.col
        self._nnz_jac_d = flat_jac_d.nnz

        # ToDo: decide if we cache _irows and _jcols pointers for composite nlp

    def _create_hessian_structure(self):

        # Note: This method requires the complicated vars map to be
        # created beforehand

        hess_lag = BlockSymMatrix(self.nblocks + 1)
        for sid, nlp in enumerate(self._nlps):
            xi = nlp.x_init()
            yi = nlp.y_init()
            hess_lag[sid, sid] = nlp.hessian_lag(xi, yi)

        hess_lag[self.nblocks, self.nblocks] = EmptyMatrix(self.nz, self.nz)

        flat_hess = hess_lag.tocoo()
        self._irows_hess = flat_hess.row
        self._jcols_hess = flat_hess.col
        self._nnz_hess_lag = flat_hess.nnz

        # ToDo: decide if we cache _irows and _jcols pointers for composite nlp

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

    def xl(self, condensed=False):
        """
        Returns array of primal variable lower bounds

        Parameters
        ----------
        condensed :  bool, optional
            Boolean flag to indicate if array excludes -numpy.inf values (default False)

        Returns
        -------
        BlockVector

        """
        if condensed:
            raise NotImplementedError()
        return self._lower_x

    def xu(self, condensed=False):
        """
        Returns array of primal variable upper bounds

        Parameters
        ----------
        condensed :  bool, optional
            Boolean flag to indicate if array excludes -numpy.inf values (default False)

        Returns
        -------
        BlockVector

        """
        if condensed:
            raise NotImplementedError()
        return self._upper_x

    def gl(self, condensed=False):
        """
        Returns array of general inequalities lower bounds

        Parameters
        ----------
        condensed :  bool, optional
            Boolean flag to indicate if array excludes -numpy.inf values (default False)

        Returns
        -------
        BlockVector

        """
        if condensed:
            raise NotImplementedError()
        return self._lower_g

    def gu(self, condensed=False):
        """
        Returns array of general inequalities upper bounds

        Parameters
        ----------
        condensed :  bool, optional
            Boolean flag to indicate if array excludes numpy.inf values (default False)

        Returns
        -------
        BlockVector

        """
        if condensed:
            raise NotImplementedError()
        return self._upper_g

    def dl(self, condensed=False):
        """
        Returns array of inequalities lower bounds

        Parameters
        ----------
        condensed :  bool, optional
            Boolean flag to indicate if array excludes -numpy.inf values (default False)

        Returns
        -------
        BlockVector

        """
        raise NotImplementedError()

    def du(self, condensed=False):
        """
        Returns array of inequalities upper bounds

        Parameters
        ----------
        condensed :  bool, optional
            Boolean flag to indicate if array excludes numpy.inf values (default False)

        Returns
        -------
        BlockVector

        """
        raise NotImplementedError()

    def nlps(self):
        """Creates generator scenario name to nlp """
        for sid, name in enumerate(self._sid_to_sname):
            yield name, self._nlps[sid]

    def create_vector_x(self, subset=None):
        """Returns ndarray of primal variables

        Parameters
        ----------
        subset : str, optional
            determines size of vector.
            `l`: only primal variables with lower bounds
            `u`: only primal variables with upper bounds

        Returns
        -------
        BlockVector

        """
        if subset is None:
            subvectors = [np.zeros(nlp.nx, dtype=np.double) for nlp in self._nlps] + \
                         [np.zeros(self.nz, dtype=np.double)]
            return BlockVector(subvectors)
        elif subset == 'l':
            vectors = list()
            for nlp in self._nlps:
                nx_l = len(nlp._lower_x_map)
                xl = np.zeros(nx_l, dtype=np.double)
                vectors.append(xl)
            # complicated variables have no lower bounds
            vectors.append(np.zeros(0, dtype=np.double))
            return BlockVector(vectors)
        elif subset == 'u':
            vectors = list()
            for nlp in self._nlps:
                nx_u = len(nlp._upper_x_map)
                xu = np.zeros(nx_u, dtype=np.double)
                vectors.append(xu)
            # complicated variables have no upper bounds
            vectors.append(np.zeros(0, dtype=np.double))
            return BlockVector(vectors)
        else:
            raise RuntimeError('Subset not recognized')

    def create_vector_y(self, subset=None):
        """Return ndarray of vector of constraints

        Parameters
        ----------
        subset : str, optional
            determines size of vector.
            `c`: only equality constraints
            `d`: only inequality constraints
            `dl`: only inequality constraints with lower bound
            `du`: only inequality constraints with upper bound

        Returns
        -------
        BlockVector

        """
        if subset is None:
            return BlockVector([np.zeros(nlp.ng, dtype=np.double) for nlp in self._nlps] +
                               [np.zeros(self.nz, dtype=np.double) for i in range(self.nblocks)])
        elif subset == 'c':
            return BlockVector([np.zeros(nlp.nc, dtype=np.double) for nlp in self._nlps] +
                               [np.zeros(self.nz, dtype=np.double) for i in range(self.nblocks)])
        elif subset == 'd':
            return BlockVector([np.zeros(nlp.nd, dtype=np.double) for nlp in self._nlps])
        else:
            raise RuntimeError('Subset not recognized')

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
            block_x = self.create_vector_x()
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
            df = self.create_vector_x()
        else:
            raise NotImplementedError("ToDo")

        if isinstance(x, BlockVector):
            assert x.size == self.nx
            assert x.nblocks == self.nblocks + 1
            for i in range(self.nblocks):
                self._nlps[i].grad_objective(x[i], out=df[i])
            return df
        elif isinstance(x, np.ndarray):
            assert x.size == self.nx
            block_x = self.create_vector_x()
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
            res = self.create_vector_y()
        else:
            raise NotImplementedError("ToDo")

        if isinstance(x, BlockVector):
            assert x.size == self.nx
            assert x.nblocks == self.nblocks + 1
            for sid in range(self.nblocks):
                self._nlps[sid].evaluate_g(x[sid], out=res[sid])
                scenario_vids = self._zid_to_vid[sid]
                diff = []
                for zid in range(self.nz):
                    vid = scenario_vids[zid]
                    diff.append(x[sid][vid] - x[self.nblocks][zid])
                res[sid + self.nblocks] = np.array(diff, dtype=np.double)
            return res
        elif isinstance(x, np.ndarray):
            assert x.size == self.nx
            block_x = self.create_vector_x()
            block_x.copyfrom(x)
            x_ = block_x
            for sid in range(self.nblocks):
                self._nlps[sid].evaluate_g(x_[sid], out=res[sid])
                scenario_vids = self._zid_to_vid[sid]
                diff = []
                # this could be done more efficient with a mask of z
                for zid in range(self.nz):
                    vid = scenario_vids[zid]
                    diff.append(x_[sid][vid] - x_[self.nblocks][zid])
                res[sid + self.nblocks] = np.array(diff, dtype=np.double)
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
        if out is None:
            res = self.create_vector_y(subset='c')
        else:
            raise NotImplementedError("ToDo")

        if isinstance(x, BlockVector):
            assert x.size == self.nx
            assert x.nblocks == self.nblocks + 1
            for sid in range(self.nblocks):
                self._nlps[sid].evaluate_c(x[sid], out=res[sid])
                scenario_vids = self._zid_to_vid[sid]
                diff = []
                for zid in range(self.nz):
                    vid = scenario_vids[zid]
                    diff.append(x[sid][vid] - x[self.nblocks][zid])
                res[sid + self.nblocks] = np.array(diff, dtype=np.double)
            return res
        elif isinstance(x, np.ndarray):
            assert x.size == self.nx
            block_x = self.create_vector_x()
            block_x.copyfrom(x)
            x_ = block_x
            for sid in range(self.nblocks):
                self._nlps[sid].evaluate_c(x_[sid], out=res[sid])
                scenario_vids = self._zid_to_vid[sid]
                diff = []
                for zid in range(self.nz):
                    vid = scenario_vids[zid]
                    diff.append(x_[sid][vid] - x_[self.nblocks][zid])
                res[sid + self.nblocks] = np.array(diff, dtype=np.double)
            return res
        else:
            raise NotImplementedError("x must be a numpy array or a BlockVector")

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

        if out is None:
            res = self.create_vector_y(subset='d')
        else:
            raise NotImplementedError("ToDo")

        if isinstance(x, BlockVector):
            assert x.size == self.nx
            assert x.nblocks == self.nblocks + 1
            for sid in range(self.nblocks):
                self._nlps[sid].evaluate_d(x[sid], out=res[sid])
            return res
        elif isinstance(x, np.ndarray):
            raise NotImplementedError("ToDo")
        else:
            raise NotImplementedError("x must be a numpy array or a BlockVector")

    def jacobian_g(self, x, out=None, **kwargs):
        """Returns the Jacobian of the general inequalities evaluated at x

        Parameters
        ----------
        x : array_like
            Array with values of primal variables.
        out : COOMatrix, optional
            Output matrix with the structure of the jacobian already defined.

        Returns
        -------
        BlockMatrix

        """
        assert x.size == self.nx, "Dimension missmatch"

        if out is None:
            if isinstance(x, BlockVector):
                assert x.nblocks == self.nblocks + 1
                jac_g = BlockMatrix(2 * self.nblocks, self.nblocks + 1)
                for sid, nlp in enumerate(self._nlps):
                    xi = x[sid]
                    jac_g[sid, sid] = nlp.jacobian_g(xi)

                    # coupling matrices Ai
                    scenario_vids = self._zid_to_vid[sid]
                    col = np.array([vid for vid in scenario_vids])
                    row = np.arange(0, self.nz)
                    data = np.ones(self.nz, dtype=np.double)
                    jac_g[sid + self.nblocks, sid] = COOMatrix((data, (row, col)),
                                                               shape=(self.nz, nlp.nx))

                    # coupling matrices Bi
                    jac_g[sid + self.nblocks, self.nblocks] = -IdentityMatrix(self.nz)
                return jac_g
            elif isinstance(x, np.ndarray):
                assert x.size == self.nx
                block_x = self.create_vector_x()
                block_x.copyfrom(x)
                x_ = block_x
                jac_g = BlockMatrix(2 * self.nblocks, self.nblocks + 1)
                for sid, nlp in enumerate(self._nlps):
                    xi = x_[sid]
                    jac_g[sid, sid] = nlp.jacobian_g(xi)

                    # coupling matrices Ai
                    scenario_vids = self._zid_to_vid[sid]
                    col = np.array([vid for vid in scenario_vids])
                    row = np.arange(0, self.nz)
                    data = np.ones(self.nz, dtype=np.double)
                    jac_g[sid + self.nblocks, sid] = COOMatrix((data, (row, col)),
                                                               shape=(self.nz, nlp.nx))

                    # coupling matrices Bi
                    jac_g[sid + self.nblocks, self.nblocks] = -IdentityMatrix(self.nz)
                return jac_g
            else:
                raise RuntimeError("Input vector format not recognized")
        else:
            raise NotImplementedError("ToDo")

    def jacobian_c(self, x, out=None, **kwargs):
        """Returns the Jacobian of the equalities evaluated at x

        Parameters
        ----------
        x : array_like
            Array with values of primal variables.
        out : COOMatrix, optional
            Output matrix with the structure of the jacobian already defined.

        Returns
        -------
        BlockMatrix

        """
        assert x.size == self.nx, "Dimension missmatch"

        if out is None:
            if isinstance(x, BlockVector):
                assert x.nblocks == self.nblocks + 1
                jac_c = BlockMatrix(2 * self.nblocks, self.nblocks + 1)
                for sid, nlp in enumerate(self._nlps):
                    xi = x[sid]
                    jac_c[sid, sid] = nlp.jacobian_c(xi)

                    # coupling matrices Ai
                    scenario_vids = self._zid_to_vid[sid]
                    col = np.array([vid for vid in scenario_vids])
                    row = np.arange(0, self.nz)
                    data = np.ones(self.nz, dtype=np.double)
                    jac_c[sid + self.nblocks, sid] = COOMatrix((data, (row, col)),
                                                               shape=(self.nz, nlp.nx))

                    # coupling matrices Bi
                    jac_c[sid + self.nblocks, self.nblocks] = -IdentityMatrix(self.nz)
                return jac_c
            elif isinstance(x, np.ndarray):
                assert x.size == self.nx
                block_x = self.create_vector_x()
                block_x.copyfrom(x)
                x_ = block_x
                jac_c = BlockMatrix(2 * self.nblocks, self.nblocks + 1)
                for sid, nlp in enumerate(self._nlps):
                    xi = x_[sid]
                    jac_c[sid, sid] = nlp.jacobian_c(xi)

                    # coupling matrices Ai
                    scenario_vids = self._zid_to_vid[sid]
                    col = np.array([vid for vid in scenario_vids])
                    row = np.arange(0, self.nz)
                    data = np.ones(self.nz, dtype=np.double)
                    jac_c[sid + self.nblocks, sid] = COOMatrix((data, (row, col)),
                                                               shape=(self.nz, nlp.nx))

                    # coupling matrices Bi
                    jac_c[sid + self.nblocks, self.nblocks] = -IdentityMatrix(self.nz)
                return jac_c
            else:
                raise RuntimeError("Input vector format not recognized")
        else:
            raise NotImplementedError("ToDo")

    def jacobian_d(self, x, out=None, **kwargs):
        """Returns the Jacobian of the inequalities evaluated at x

        Parameters
        ----------
        x : array_like
            Array with values of primal variables.
        out : COOMatrix, optional
            Output matrix with the structure of the jacobian already defined.

        Returns
        -------
        BlockMatrix

        """
        assert x.size == self.nx, "Dimension missmatch"

        if out is None:
            if isinstance(x, BlockVector):
                assert x.nblocks == self.nblocks + 1
                jac_d = BlockMatrix(self.nblocks, self.nblocks)
                for sid, nlp in enumerate(self._nlps):
                    xi = x[sid]
                    jac_d[sid, sid] = nlp.jacobian_d(xi)
                return jac_d
            elif isinstance(x, np.ndarray):
                raise NotImplementedError("ToDo")
        else:
            raise NotImplementedError("ToDo")

    def hessian_lag(self, x, y, out=None, **kwargs):
        """Return the Hessian of the Lagrangian function evaluated at x and y

        Parameters
        ----------
        x : array_like
            Array with values of primal variables.
        y : array_like
            Array with values of dual variables.
        out : SymCOOMatrix
            Output matrix with the structure of the hessian already defined. Optional

        Returns
        -------
        BlockSymMatrix

        """
        assert x.size == self.nx, "Dimension missmatch"
        assert y.size == self.ng, "Dimension missmatch"

        if out is None:
            if isinstance(x, BlockVector) and isinstance(y, BlockVector):
                assert x.nblocks == self.nblocks + 1
                assert y.nblocks == 2 * self.nblocks

                hess_lag = BlockSymMatrix(self.nblocks + 1)
                for sid, nlp in enumerate(self._nlps):
                    xi = x[sid]
                    yi = y[sid]
                    hess_lag[sid, sid] = nlp.hessian_lag(xi, yi)

                hess_lag[self.nblocks, self.nblocks] = EmptyMatrix(self.nz, self.nz)
                return hess_lag
            else:
                raise NotImplementedError("ToDo")
        else:
            raise NotImplementedError("ToDo")

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
        return self._sname_to_sidp[scneario_name]

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

    # ToDo: order of variables?
    # ToDo: order of constraints?

    def expansion_matrix_xl(self):
        raise NotImplementedError()

    def expansion_matrix_xu(self):
        raise NotImplementedError()

    def expansion_matrix_dl(self):
        raise NotImplementedError()

    def expansion_matrix_du(self):
        raise NotImplementedError()
