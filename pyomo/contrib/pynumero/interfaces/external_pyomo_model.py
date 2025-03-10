#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import itertools
from pyomo.core.base.var import Var
from pyomo.core.base.constraint import Constraint
from pyomo.core.base.objective import Objective
from pyomo.core.expr.visitor import identify_variables
from pyomo.common.timing import HierarchicalTimer
from pyomo.util.subsystems import create_subsystem_block
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxModel
from pyomo.contrib.pynumero.algorithms.solvers.implicit_functions import (
    SccImplicitFunctionSolver,
)
import numpy as np
import scipy.sparse as sps


def _dense_to_full_sparse(matrix):
    """
    Used to convert a dense matrix (2d NumPy array) to SciPy sparse matrix
    with explicit coordinates for every entry, including zeros. This is
    used because _ExternalGreyBoxAsNLP methods rely on receiving sparse
    matrices where sparsity structure does not change between calls.
    This is difficult to achieve for matrices obtained via the implicit
    function theorem unless an entry is returned for every coordinate
    of the matrix.

    Note that this does not mean that the Hessian of the entire NLP will
    be dense, only that the block corresponding to this external model
    will be dense.
    """
    # TODO: Allow methods to hard-code Jacobian/Hessian sparsity structure
    # in the case it is known a priori.
    # TODO: Decompose matrices to infer maximum fill-in sparsity structure.
    nrow, ncol = matrix.shape
    row = []
    col = []
    data = []
    for i, j in itertools.product(range(nrow), range(ncol)):
        row.append(i)
        col.append(j)
        data.append(matrix[i, j])
    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    return sps.coo_matrix((data, (row, col)), shape=(nrow, ncol))


def get_hessian_of_constraint(constraint, wrt1=None, wrt2=None, nlp=None):
    constraints = [constraint]
    if wrt1 is None and wrt2 is None:
        variables = list(identify_variables(constraint.expr, include_fixed=False))
        wrt1 = variables
        wrt2 = variables
    elif wrt1 is not None and wrt2 is not None:
        variables = wrt1 + wrt2
    elif wrt1 is not None:  # but wrt2 is None
        wrt2 = wrt1
        variables = wrt1
    else:
        # wrt2 is not None and wrt1 is None
        wrt1 = wrt2
        variables = wrt1

    if nlp is None:
        block = create_subsystem_block(constraints, variables=variables)
        # Could fix input_vars so I don't evaluate the Hessian with respect
        # to variables I don't care about...

        # HUGE HACK: Variables not included in a constraint are not written
        # to the nl file, so we cannot take the derivative with respect to
        # them, even though we know this derivative is zero. To work around,
        # we make sure all variables appear on the block in the form of a
        # dummy constraint. Then we can take derivatives of any constraint
        # with respect to them. Conveniently, the extract_submatrix_
        # call deals with extracting the variables and constraint we care
        # about, in the proper order.
        block._dummy_var = Var()
        block._dummy_con = Constraint(expr=sum(variables) == block._dummy_var)
        block._obj = Objective(expr=0.0)
        nlp = PyomoNLP(block)

    saved_duals = nlp.get_duals()
    saved_obj_factor = nlp.get_obj_factor()
    temp_duals = np.zeros(len(saved_duals))

    # NOTE: This makes some assumption about how the Lagrangian is constructed.
    # TODO: Define the convention we assume and convert if necessary.
    idx = nlp.get_constraint_indices(constraints)[0]
    temp_duals[idx] = 1.0
    nlp.set_duals(temp_duals)
    nlp.set_obj_factor(0.0)

    # NOTE: The returned matrix preserves explicit zeros. I.e. it contains
    # coordinates for every entry that could possibly be nonzero.
    submatrix = nlp.extract_submatrix_hessian_lag(wrt1, wrt2)

    nlp.set_obj_factor(saved_obj_factor)
    nlp.set_duals(saved_duals)
    return submatrix


class ExternalPyomoModel(ExternalGreyBoxModel):
    """
    This is an ExternalGreyBoxModel used to create an external model
    from existing Pyomo components. Given a system of variables and
    equations partitioned into "input" and "external" variables and
    "residual" and "external" equations, this class computes the
    residual of the "residual equations," as well as their Jacobian
    and Hessian, as a function of only the inputs.

    Pyomo components:
        f(x, y) == 0 # "Residual equations"
        g(x, y) == 0 # "External equations", dim(g) == dim(y)

    Effective constraint seen by this "external model":
        F(x) == f(x, y(x)) == 0
        where y(x) solves g(x, y) == 0

    """

    def __init__(
        self,
        input_vars,
        external_vars,
        residual_cons,
        external_cons,
        solver_class=None,
        solver_options=None,
        timer=None,
    ):
        """
        Arguments:
        ----------
        input_vars: list
            List of variables sent to this system by the outer solver
        external_vars: list
            List of variables that are solved for internally by this system
        residual_cons: list
            List of equality constraints whose residuals are exposed to
            the outer solver
        external_cons: list
            List of equality constraints used to solve for the external
            variables
        solver_class: Subclass of ImplicitFunctionSolver
            The solver object that is used to converge the system of
            equations defining the implicit function.
        solver_options: dict
            Options dict for the ImplicitFunctionSolver
        timer: HierarchicalTimer
            HierarchicalTimer object to which new timing categories introduced
            will be attached. If None, a new timer will be created.

        """
        if timer is None:
            timer = HierarchicalTimer()
        self._timer = timer
        if solver_class is None:
            solver_class = SccImplicitFunctionSolver
        self._solver_class = solver_class
        if solver_options is None:
            solver_options = {}

        self._timer.start("__init__")

        # We only need this block to construct the NLP, which wouldn't
        # be necessary if we could compute Hessians of Pyomo constraints.
        self._block = create_subsystem_block(
            residual_cons + external_cons, input_vars + external_vars
        )
        self._block._obj = Objective(expr=0.0)
        self._timer.start("PyomoNLP")
        self._nlp = PyomoNLP(self._block)
        self._timer.stop("PyomoNLP")

        # Instantiate a solver with the ImplicitFunctionSolver API:
        self._solver = self._solver_class(
            external_vars,
            external_cons,
            input_vars,
            timer=self._timer,
            **solver_options,
        )

        assert len(external_vars) == len(external_cons)

        self.input_vars = input_vars
        self.external_vars = external_vars
        self.residual_cons = residual_cons
        self.external_cons = external_cons

        self.residual_con_multipliers = [None for _ in residual_cons]
        self.residual_scaling_factors = None

        self._input_output_coords = self._nlp.get_primal_indices(
            input_vars + external_vars
        )

        self._timer.stop("__init__")

    def n_inputs(self):
        return len(self.input_vars)

    def n_equality_constraints(self):
        return len(self.residual_cons)

    # I would like to try to get by without using the following "name" methods.
    def input_names(self):
        return ["input_%i" % i for i in range(self.n_inputs())]

    def equality_constraint_names(self):
        return ["residual_%i" % i for i in range(self.n_equality_constraints())]

    def set_input_values(self, input_values):
        self._timer.start("set_inputs")

        solver = self._solver
        external_cons = self.external_cons
        external_vars = self.external_vars
        input_vars = self.input_vars

        solver.set_parameters(input_values)
        outputs = solver.evaluate_outputs()
        solver.update_pyomo_model()

        #
        # Send updated variable values to NLP for dervative evaluation
        #
        primals = self._nlp.get_primals()
        values = np.concatenate((input_values, outputs))
        primals[self._input_output_coords] = values
        self._nlp.set_primals(primals)

        self._timer.stop("set_inputs")

    def set_equality_constraint_multipliers(self, eq_con_multipliers):
        """
        Sets multipliers for residual equality constraints seen by the
        outer solver.

        """
        for i, val in enumerate(eq_con_multipliers):
            self.residual_con_multipliers[i] = val

    def set_external_constraint_multipliers(self, eq_con_multipliers):
        eq_con_multipliers = np.array(eq_con_multipliers)
        external_multipliers = self.calculate_external_constraint_multipliers(
            eq_con_multipliers
        )
        multipliers = np.concatenate((eq_con_multipliers, external_multipliers))
        cons = self.residual_cons + self.external_cons
        n_con = len(cons)
        assert n_con == self._nlp.n_constraints()
        duals = np.zeros(n_con)
        indices = self._nlp.get_constraint_indices(cons)
        duals[indices] = multipliers
        self._nlp.set_duals(duals)

    def calculate_external_constraint_multipliers(self, resid_multipliers):
        """
        Calculates the multipliers of the external constraints from the
        multipliers of the residual constraints (which are provided by
        the "outer" solver).

        """
        # NOTE: This method implicitly relies on the value of inputs stored
        # in the nlp. Should we also rely on the multiplier that are in
        # the nlp?
        # We would then need to call nlp.set_duals twice. Once with the
        # residual multipliers and once with the full multipliers.
        # I like the current approach better for now.
        nlp = self._nlp
        y = self.external_vars
        f = self.residual_cons
        g = self.external_cons
        jfy = nlp.extract_submatrix_jacobian(y, f)
        jgy = nlp.extract_submatrix_jacobian(y, g)

        jgy_t = jgy.transpose()
        jfy_t = jfy.transpose()
        dfdg = -sps.linalg.splu(jgy_t.tocsc()).solve(jfy_t.toarray())
        resid_multipliers = np.array(resid_multipliers)
        external_multipliers = dfdg.dot(resid_multipliers)
        return external_multipliers

    def get_full_space_lagrangian_hessians(self):
        """
        Calculates terms of Hessian of full-space Lagrangian due to
        external and residual constraints. Note that multipliers are
        set by set_equality_constraint_multipliers. These matrices
        are used to calculate the Hessian of the reduced-space
        Lagrangian.

        """
        nlp = self._nlp
        x = self.input_vars
        y = self.external_vars
        hlxx = nlp.extract_submatrix_hessian_lag(x, x)
        hlxy = nlp.extract_submatrix_hessian_lag(x, y)
        hlyy = nlp.extract_submatrix_hessian_lag(y, y)
        return hlxx, hlxy, hlyy

    def calculate_reduced_hessian_lagrangian(self, hlxx, hlxy, hlyy):
        """
        Performs the matrix multiplications necessary to get the
        reduced space Hessian-of-Lagrangian term from the full-space
        terms.

        """
        # Converting to dense is faster for the distillation
        # example. Does this make sense?
        hlxx = hlxx.toarray()
        hlxy = hlxy.toarray()
        hlyy = hlyy.toarray()
        dydx = self.evaluate_jacobian_external_variables()
        term1 = hlxx
        prod = hlxy.dot(dydx)
        term2 = prod + prod.transpose()
        term3 = hlyy.dot(dydx).transpose().dot(dydx)
        hess_lag = term1 + term2 + term3
        return hess_lag

    def evaluate_equality_constraints(self):
        return self._nlp.extract_subvector_constraints(self.residual_cons)

    def evaluate_jacobian_equality_constraints(self):
        self._timer.start("jacobian")

        nlp = self._nlp
        x = self.input_vars
        y = self.external_vars
        f = self.residual_cons
        g = self.external_cons
        jfx = nlp.extract_submatrix_jacobian(x, f)
        jfy = nlp.extract_submatrix_jacobian(y, f)
        jgx = nlp.extract_submatrix_jacobian(x, g)
        jgy = nlp.extract_submatrix_jacobian(y, g)

        nf = len(f)
        nx = len(x)
        n_entries = nf * nx

        # TODO: Does it make sense to cast dydx to a sparse matrix?
        # My intuition is that it does only if jgy is "decomposable"
        # in the strongly connected component sense, which is probably
        # not usually the case.
        dydx = -1 * sps.linalg.splu(jgy.tocsc()).solve(jgx.toarray())
        # NOTE: PyNumero block matrices require this to be a sparse matrix
        # that contains coordinates for every entry that could possibly
        # be nonzero. Here, this is all of the entries.
        dfdx = jfx + jfy.dot(dydx)

        full_sparse = _dense_to_full_sparse(dfdx)

        self._timer.stop("jacobian")
        return full_sparse

    def evaluate_jacobian_external_variables(self):
        nlp = self._nlp
        x = self.input_vars
        y = self.external_vars
        g = self.external_cons
        jgx = nlp.extract_submatrix_jacobian(x, g)
        jgy = nlp.extract_submatrix_jacobian(y, g)
        jgy_csc = jgy.tocsc()
        dydx = -1 * sps.linalg.splu(jgy_csc).solve(jgx.toarray())
        return dydx

    def evaluate_hessian_external_variables(self):
        nlp = self._nlp
        x = self.input_vars
        y = self.external_vars
        g = self.external_cons
        jgx = nlp.extract_submatrix_jacobian(x, g)
        jgy = nlp.extract_submatrix_jacobian(y, g)
        jgy_csc = jgy.tocsc()
        jgy_fact = sps.linalg.splu(jgy_csc)
        dydx = -1 * jgy_fact.solve(jgx.toarray())

        ny = len(y)
        nx = len(x)

        hgxx = np.array(
            [get_hessian_of_constraint(con, x, nlp=nlp).toarray() for con in g]
        )
        hgxy = np.array(
            [get_hessian_of_constraint(con, x, y, nlp=nlp).toarray() for con in g]
        )
        hgyy = np.array(
            [get_hessian_of_constraint(con, y, nlp=nlp).toarray() for con in g]
        )

        # This term is sparse, but we do not exploit it.
        term1 = hgxx

        # This is what we want.
        # prod[i,j,k] = sum(hgxy[i,:,j] * dydx[:,k])
        prod = hgxy.dot(dydx)
        # Swap the second and third axes of the tensor
        term2 = prod + prod.transpose((0, 2, 1))
        # The term2 tensor could have some sparsity worth exploiting.

        # matrix.dot(tensor) is not what we want, so we reverse the order of the
        # product. Exploit symmetry of hgyy to only perform one transpose.
        term3 = hgyy.dot(dydx).transpose((0, 2, 1)).dot(dydx)

        rhs = term1 + term2 + term3

        rhs.shape = (ny, nx * nx)
        sol = jgy_fact.solve(rhs)
        sol.shape = (ny, nx, nx)
        d2ydx2 = -sol

        return d2ydx2

    def evaluate_hessians_of_residuals(self):
        """
        This method computes the Hessian matrix of each equality
        constraint individually, rather than the sum of Hessians
        times multipliers.
        """
        nlp = self._nlp
        x = self.input_vars
        y = self.external_vars
        f = self.residual_cons
        g = self.external_cons
        jfx = nlp.extract_submatrix_jacobian(x, f)
        jfy = nlp.extract_submatrix_jacobian(y, f)

        dydx = self.evaluate_jacobian_external_variables()

        ny = len(y)
        nf = len(f)
        nx = len(x)

        hfxx = np.array(
            [get_hessian_of_constraint(con, x, nlp=nlp).toarray() for con in f]
        )
        hfxy = np.array(
            [get_hessian_of_constraint(con, x, y, nlp=nlp).toarray() for con in f]
        )
        hfyy = np.array(
            [get_hessian_of_constraint(con, y, nlp=nlp).toarray() for con in f]
        )

        d2ydx2 = self.evaluate_hessian_external_variables()

        term1 = hfxx
        prod = hfxy.dot(dydx)
        term2 = prod + prod.transpose((0, 2, 1))
        term3 = hfyy.dot(dydx).transpose((0, 2, 1)).dot(dydx)

        d2ydx2.shape = (ny, nx * nx)
        term4 = jfy.dot(d2ydx2)
        term4.shape = (nf, nx, nx)

        d2fdx2 = term1 + term2 + term3 + term4
        return d2fdx2

    def evaluate_hessian_equality_constraints(self):
        """
        This method actually evaluates the sum of Hessians times
        multipliers, i.e. the term in the Hessian of the Lagrangian
        due to these equality constraints.

        """
        self._timer.start("hessian")

        # External multipliers must be calculated after both primals and duals
        # are set, and are only necessary for this Hessian calculation.
        # We know this Hessian calculation wants to use the most recently
        # set primals and duals, so we can safely calculate external
        # multipliers here.
        eq_con_multipliers = self.residual_con_multipliers
        self.set_external_constraint_multipliers(eq_con_multipliers)

        # These are full-space Hessian-of-Lagrangian terms
        hlxx, hlxy, hlyy = self.get_full_space_lagrangian_hessians()

        # These terms can be used to calculate the corresponding
        # Hessian-of-Lagrangian term in the full space.
        hess_lag = self.calculate_reduced_hessian_lagrangian(hlxx, hlxy, hlyy)
        sparse = _dense_to_full_sparse(hess_lag)
        lower_triangle = sps.tril(sparse)
        self._timer.stop("hessian")
        return lower_triangle

    def set_equality_constraint_scaling_factors(self, scaling_factors):
        """
        Set scaling factors for the equality constraints that are exposed
        to a solver. These are the "residual equations" in this class.
        """
        self.residual_scaling_factors = np.array(scaling_factors)

    def get_equality_constraint_scaling_factors(self):
        """
        Get scaling factors for the equality constraints that are exposed
        to a solver. These are the "residual equations" in this class.
        """
        return self.residual_scaling_factors
