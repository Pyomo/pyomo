from pyomo.contrib.pynumero.interfaces.nlp import NLP, PyomoNLP
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
                 model,
                 complicated_vars):

        """

        Parameters
        ----------
        model: dict
            dictionary with scenarios. The keys are scenario names, and the values
            are PyomoModels
        probabilities: dict
            dictionary with scenario probabilities
        """

        # call parent class to set model
        super(TwoStageStochasticNLP, self).__init__(model)

        # make model an ordered dictionary
        ordered_keys = sorted(self._model.keys())
        new_dict = OrderedDict()
        for k in ordered_keys:
            new_dict[k] = self._model[k]
        self._model = new_dict

        # map of scenario name to indices
        self._sname_to_sid = dict()
        # map of scenario id to scenario name
        self._sid_to_sname = list()
        # scenario nlps
        self._nlps = list()

        for sname, instance in self._model.items():
            self._sname_to_sid[sname] = len(self._sid_to_sname)
            self._sid_to_sname.append(sname)
            self._nlps.append(PyomoNLP(instance))

        # build map of complicated vars
        # this defines _nz, _zid_to_name, and _zid_to_vid
        self._build_complicated_vars_map(complicated_vars)

        # defines vectors
        self._create_vectors()

        # define sizes
        self._nx = self._init_x.size
        self._ng = self._init_y.size
        self._nc = sum(nlp.nc for nlp in self._nlps) + self.nz * self.nblocks
        self._nd = sum(nlp.nd for nlp in self._nlps)

        # define structure of jacobians
        self._create_jacobian_structures()

        # define structure Hessian
        self._create_hessian_structure()

    @property
    def nblocks(self):
        """
        Return number of blocks (nlps)
        """
        return len(self._nlps)

    @nblocks.setter
    def nblocks(self, other):
        """
        Prevent changes in number of blocks
        """
        raise RuntimeError("Change in number of blocks not supported")

    @property
    def nz(self):
        """
        Return number of complicated variables
        """
        return self._nz

    @nz.setter
    def nz(self, other):
        """
        Prevent changes in number of complicated variables
        """
        raise RuntimeError("Change in number of blocks not supported")

    @property
    def xl(self):
        """
        Return lower bounds of primal variables in a BlockVector
        """
        return self._lower_x

    @property
    def xu(self):
        """
        Return upper bounds of primal variables in a BlockVector
        """
        return self._upper_x

    @property
    def gl(self):
        """
        Return lower bounds of general inequality constraints.
        in a BlockVector
        """
        return self._lower_g

    @property
    def gu(self):
        """
        Return upper bounds of general inequality constraints.
        in a BlockVector
        """
        return self._upper_g

    @property
    def x_init(self):
        """
        Return initial guess of primal variables in a BlockVector
        """
        return self._init_x

    @property
    def y_init(self):
        """
        Return initial guess of dual variables in a BlockVector
        """
        return self._init_y

    def _build_complicated_vars_map(self, complicated_vars):
        """
        Return dictionary of complicated var_id to subproblem var_id
        """

        # get "first scenario"
        sname0 = self._sid_to_sname[0]
        s0 = self._model[sname0]
        self._nz = 0
        self._zid_to_name = []
        # count number of complicating variables
        for zname in complicated_vars:
            assert hasattr(s0, zname), "scenario {} does not have variable {}.".format(sname0, zname)
            z_var = getattr(s0, zname)
            assert isinstance(z_var, aml.Var), "{} is not a variable of scenario".format(zname, sname0)
            self._nz += len(z_var)
            if z_var.is_indexed():
                indexed_set = sorted([k for k in z_var.keys()])
                for k in indexed_set:
                    local_v = z_var[k].local_name
                    self._zid_to_name.append(local_v)
            else:
                local_v = z_var.local_name
                self._zid_to_name.append(local_v)

        # this goes from [scnario_id][zid] - > vid
        self._zid_to_vid = list()
        for sid, sname in enumerate(self._sid_to_sname):
            local_list = [None] * self._nz
            self._zid_to_vid.append(local_list)
            counter = 0
            for zname in complicated_vars:
                scenario = self._model[sname]
                assert hasattr(scenario, zname), "scenario {} does not have variable {}.".format(sname, zname)
                z_var = getattr(scenario, zname)
                assert isinstance(z_var, aml.Var), "{} is not a variable of scenario".format(zname, sname)
                if z_var.is_indexed():
                    indexed_set = sorted([k for k in z_var.keys()])
                    for k in indexed_set:
                        local_v = z_var[k].local_name
                        self._zid_to_vid[-1][counter] = self._nlps[sid]._name_to_vid[local_v]
                        counter += 1
                else:
                    local_v = z_var.local_name
                    self._zid_to_vid[-1][counter] = self._nlps[sid]._name_to_vid[local_v]
                    counter += 1
                assert counter == self._nz

        #for sid in range(self.nblocks):
        #    for zid in range(self._nz):
        #        print(self._sid_to_sname[sid], self._zid_to_name[zid], self._zid_to_vid[sid][zid])

    def _create_vectors(self):

        # Note: This method requires the complicated vars map to be
        # created beforehand

        # init values
        self._init_x = BlockVector([nlp.x_init for nlp in self._nlps] +
                                   [np.zeros(self.nz, dtype=np.double)])

        self._init_y = BlockVector([nlp.y_init for nlp in self._nlps] +
                                   [np.zeros(self.nz, dtype=np.double) for i in range(self.nblocks)])

        # lower and upper bounds
        self._lower_x = BlockVector([nlp.xl for nlp in self._nlps] +
                                    [np.full(self.nz, -np.inf, dtype=np.double)])
        self._upper_x = BlockVector([nlp.xu for nlp in self._nlps] +
                                    [np.full(self.nz, -np.inf, dtype=np.double)])

        self._lower_g = BlockVector([nlp.gl for nlp in self._nlps] +
                                    [np.zeros(self.nz, dtype=np.double) for i in range(self.nblocks)])
        self._upper_g = BlockVector([nlp.gu for nlp in self._nlps] +
                                    [np.zeros(self.nz, dtype=np.double) for i in range(self.nblocks)])

    def _create_jacobian_structures(self):

        # Note: This method requires the complicated vars map to be
        # created beforehand

        # build general jacobian
        jac_g = BlockMatrix(2 * self.nblocks, self.nblocks + 1)
        for sid, nlp in enumerate(self._nlps):
            xi = nlp.x_init
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
            xi = nlp.x_init
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
            xi = nlp.x_init
            jac_d[sid, sid] = nlp.jacobian_d(xi)
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
            xi = nlp.x_init
            yi = nlp.y_init
            hess_lag[sid, sid] = nlp.hessian_lag(xi, yi)

        hess_lag[self.nblocks, self.nblocks] = EmptyMatrix(self.nz, self.nz)

        flat_hess = hess_lag.tocoo()
        self._irows_hess = flat_hess.row
        self._jcols_hess = flat_hess.col
        self._nnz_hess_lag = flat_hess.nnz


    def nlps(self):
        for sid, name in enumerate(self._sid_to_sname):
            yield name, self._nlps[sid]

    def create_vector_x(self, subset=None):
        """Return BlockVector of primal variables

        Parameters
        ----------
        subset : str
            type of vector. xl returns a vector of
            variables with lower bounds. xu returns a
            vector of variables with upper bounds (optional)

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
        """Return BlockVector of constraints

        Parameters
        ----------
        subset : str
            type of vector. yd returns a vector of
            inequality constriants. yc returns a
            vector of equality constraints (optional)

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

    def objective(self, x):

        if isinstance(x, BlockVector):
            return sum(self._nlps[i].objective(x[i]) for i in range(self.nblocks))
        elif isinstance(x, np.ndarray):
            raise NotImplementedError("ToDo")
        else:
            raise NotImplementedError("x must be a numpy array or a BlockVector")

    def grad_objective(self, x, out=None):

        if out is None:
            df = self.create_vector_x()
        else:
            raise NotImplementedError("ToDo")

        if isinstance(x, BlockVector):
            assert x.size == self.nx
            assert x.nblocks == self.nblocks # this will change as z are included in x
            for i in range(self.nblocks):
                self._nlps[i].grad_objective(x[i], other=df[i])
            return df
        elif isinstance(x, np.ndarray):
            raise NotImplementedError("ToDo")
        else:
            raise NotImplementedError("x must be a numpy array or a BlockVector")

    def evaluate_g(self, x, out=None):

        if out is None:
            res = self.create_vector_y()
        else:
            raise NotImplementedError("ToDo")

        if isinstance(x, BlockVector):
            assert x.size == self.nx
            assert x.nblocks == 2 * self.nblocks
            for sid in range(self.nblocks):
                self._nlps[sid].evaluate_g(x[sid], other=res[sid])
                scenario_vids = self._zid_to_vid[sid]
                diff = []
                for zid in range(self.nz):
                    vid = scenario_vids[zid]
                    diff.append(x[sid][vid] - x[sid + self.nblocks][zid])
                res[sid + self.nblocks] = np.array(diff, dtype=np.double)
            return res
        elif isinstance(x, np.ndarray):
            raise NotImplementedError("ToDo")
        else:
            raise NotImplementedError("x must be a numpy array or a BlockVector")

    def evaluate_c(self, x, out=None, **kwargs):

        if out is None:
            res = self.create_vector_y(subset='c')
        else:
            raise NotImplementedError("ToDo")

        if isinstance(x, BlockVector):
            assert x.size == self.nx
            assert x.nblocks == 2 * self.nblocks
            for sid in range(self.nblocks):
                self._nlps[sid].evaluate_c(x[sid], other=res[sid])
                scenario_vids = self._zid_to_vid[sid]
                diff = []
                for zid in range(self.nz):
                    vid = scenario_vids[zid]
                    diff.append(x[sid][vid] - x[sid + self.nblocks][zid])
                res[sid + self.nblocks] = np.array(diff, dtype=np.double)
            return res
        elif isinstance(x, np.ndarray):
            raise NotImplementedError("ToDo")
        else:
            raise NotImplementedError("x must be a numpy array or a BlockVector")

    def evaluate_d(self, x, out=None, **kwargs):
        if out is None:
            res = self.create_vector_y(subset='d')
        else:
            raise NotImplementedError("ToDo")

        if isinstance(x, BlockVector):
            assert x.size == self.nx
            assert x.nblocks == self.nblocks
            for sid in range(self.nblocks):
                self._nlps[sid].evaluate_d(x[sid], other=res[sid])
            return res
        elif isinstance(x, np.ndarray):
            raise NotImplementedError("ToDo")
        else:
            raise NotImplementedError("x must be a numpy array or a BlockVector")

    def jacobian_g(self, x, out=None):

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
                raise NotImplementedError("ToDo")
        else:
            raise NotImplementedError("ToDo")

    def jacobian_c(self, x, out=None, **kwargs):

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
                raise NotImplementedError("ToDo")
        else:
            raise NotImplementedError("ToDo")

    def jacobian_d(self, x, out=None, **kwargs):

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





