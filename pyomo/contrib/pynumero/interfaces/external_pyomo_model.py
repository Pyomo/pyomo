#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.environ import SolverFactory
from pyomo.core.base.objective import Objective
from pyomo.util.subsystems import (
        create_subsystem_block,
        TemporarySubsystemManager,
        )
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.contrib.pynumero.interfaces.external_grey_box import (
        ExternalGreyBoxModel,
        )
import scipy.sparse as sps


class ExternalPyomoModel(ExternalGreyBoxModel):
    """
    This is an ExternalGreyBoxModel used to create an exteral model
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

    def __init__(self,
            input_vars,
            external_vars,
            residual_cons,
            external_cons,
            ):
        # We only need this block to construct the NLP, which wouldn't
        # be necessary if we could compute Hessians of Pyomo constraints.
        self._block = create_subsystem_block(
                residual_cons+external_cons,
                input_vars+external_vars,
                )
        self._block._obj = Objective(expr=0.0)
        self._nlp = PyomoNLP(self._block)

        assert len(external_vars) == len(external_cons)

        self.input_vars = input_vars
        self.external_vars = external_vars
        self.residual_cons = residual_cons
        self.external_cons = external_cons

    def n_inputs(self):
        return len(self.input_vars)

    def n_equality_constraints(self):
        return len(self.residual_equations)

    # I would like to try to get by without using the following "name" methods.
    def input_names(self):
        return ["input_%i" % i for i in range(self.n_inputs())]
    def equality_constraint_names(self):
        return ["residual_%i" % i for i in range(self.n_equality_constraints())]

    def set_input_values(self, input_values):
        external_cons = self.external_cons
        external_vars = self.external_vars
        input_vars = self.input_vars

        for var, val in zip(input_vars, input_values):
            var.set_value(val)

        _temp = create_subsystem_block(external_cons, variables=external_vars)
        # Make sure that no additional variables appear in the
        # "external constraints." Not sure if this is necessary.
        assert len(_temp.input_vars) == len(input_vars)

        # TODO: Make this solver a configurable option
        solver = SolverFactory("ipopt")
        with TemporarySubsystemManager(to_fix=input_vars):
            solver.solve(_temp)

        # Should we create the NLP from the original block or the temp block?
        # Need to create it from the original block because temp block won't
        # have residual constraints, whose derivatives are necessary.
        self._nlp = PyomoNLP(self._block)

    def set_equality_constraint_multipliers(self, eq_con_multipliers):
        raise NotImplementedError()

    def evaluate_equality_constraints(self):
        return self._nlp.extract_subvector_constraints(self.residual_cons)

    def evaluate_jacobian_equality_constraints(self):
        nlp = self._nlp
        x = self.input_vars
        y = self.external_vars
        f = self.residual_cons
        g = self.external_cons
        jfx = nlp.extract_submatrix_jacobian(x, f)
        jfy = nlp.extract_submatrix_jacobian(y, f)
        jgx = nlp.extract_submatrix_jacobian(x, g)
        jgy = nlp.extract_submatrix_jacobian(y, g)

        # TODO: Does it make sense to cast dydx to a sparse matrix?
        # My intuition is that it does only if jgy is "decomposable"
        # in the strongly connected component sense, which is probably
        # not usually the case.
        dydx = -1 * sps.linalg.splu(jgy.tocsc()).solve(jgx.toarray())
        return (jfx + jfy.dot(dydx))

    def evaluate_hessian_equality_constraints(self):
        raise NotImplementedError()
