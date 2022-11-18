#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from collections import namedtuple
from pyomo.common.timing import HierarchicalTimer
from pyomo.common.config import ConfigBlock
from pyomo.util.subsystems import create_subsystem_block


# This is like a DirectSolver or PersistentSolver, but with a more limited API.
# It may make sense to inherit from one of these classes at some point.
# It may also make sense to merge this with the sensitivity toolbox class,
# so that sensitivities wrt this model may be calculated. However, for this
# solve, we limit ourselves to square problems, and don't immediately require
# derivatives.
#
# The purpose of this class is to define an API that we can use for implicit
# function solves. The reason this API is useful is that it allows resolves
# with different parameters without rewriting the NL file.
#
# Question on the API:
# - should this accept *values* as inputs, or just automatically update
#   from the Pyomo variables?
#
# A parameterized solver on a Pyomo model needs:
# - A list of variables/parameters (need only be "set value"-compatible)
# - A solve method
#
# Does a "parameterized solver" make sense for a Pyomo model?
# Not really, but it's not necessarily invalid.
#
# I think this model-based API actually makes sense.
# We accept a Pyomo model, but then have an API that allows for separation
# between model an NLP during the solve.
# What output data does this class allow us to access?
# Currently, only the model, with its values after the solve. Which we
# send to another NLP for derivative evaluation.
# The caller just needs to get values into an NLP object. Should this
# class have more of an "input-output" interface? I can take the outputs, i.e.
# values of all variables, and load them into model, NLP, or whatever else
# I want. This makes the most sense. However, for now, is ParamSquareSolver
# an okay abstraction?
#
# I need something that operates on a model, so that this can still work
# if CyIpopt is not available. Well in that case we just use a SciPy solver.
# This is all a silly distinction. I don't need to support a model-only case.
#
# If I don't need to support a model-only API, what is the simplest API
# I can implement?
# Accept: NLP, list of parameters
# Methods: Update parameters, solve.
# ... but this is not consistent with the decomposed solver ...
# So far everything I do relies on a "global" model to store values.
# I need either:
# - global model/NLP
#   - Receive model, write NLP and NLPs for decomposition blocks
#   - Solve NLPs, transfer values into "global" NLP
#   - Transfer values from global NLP into global NLP used for
#     derivative evaluations
# - A vector of outputs in pre-specified variables
#   - This seems much simpler
#   - Just need to know the coordinates in the "derivative NLP"
#   - How do I get the output variables?
#   - Are they specified a priori?
# Currently, this class is instantiated with _external_block in
# EPM. I explicitly specify input variables. Should I explicitly
# specify output variables?
class PyomoImplicitFunctionBase(object):

    def __init__(self, variables, constraints, parameters):
        # TODO: If I'm going to add these attributes in the base class,
        # I should probably add an interface to access them. Otherwise,
        # derived classes have to access these private attributes.
        self._variables = variables
        self._constraints = constraints
        self._parameters = parameters
        self._block_variables = variables + parameters
        self._block = create_subsystem_block(
            constraints,
            self._block_variables,
        )

    def get_variables(self):
        return self._variables

    def get_constraints(self):
        return self._constraints

    def get_parameters(self):
        return self._parameters

    def get_block(self):
        return self._block

    def set_parameters(self, values):
        # A derived class may implement whatever storage scheme for the
        # parameter values they wish
        raise NotImplementedError()

    def evaluate_outputs(self):
        # Should return an array in the size of the output variables
        raise NotImplementedError()

    def update_pyomo_model(self):
        raise NotImplementedError()


class ParameterizedSquareSolver(object):
    """
    Given a square Pyomo model representing a system:
    g(x, y) = 0
    where x are variables and y are parameters.
    This class allows updating parameters and solving for variables.

    """

    def __init__(self, model, param_vars):
        # Is param_vars a confusing name? These must be variables, but they
        # will be treated as paramters
        # In a proper Pyomo interface, these could be actual parameters
        # or variables. Maybe this should even go through the existing
        # sensitivity interface.
        self._model = model
        self._param_vars = param_vars

    def update_parameters(self, values):
        # Does this make sense? The purpose of this class is to cache some
        # data structure
        raise NotImplementedError()

    def solve(self):
        raise NotImplementedError()


# This is the class that I actually want to use in the implicit function.
# But I want to be able to fall back on a writer-based Pyomo implementation
# if for some reason this is not available.
class ParameterizedSquareNlpSolver(object):

    def __init__(self, nlp, parameters):
        # 1. Identify parameters in the NLP
        # 2. "Project out" these parameters
        # 3. Make sure the result is square
        raise NotImplementedError()

    def set_parameters(self, values):
        # We need a method to update parameters, as we do not want to
        # rely on the Pyomo model during a solve.
        raise NotImplementedError()

    def solve(self):
        raise NotImplementedError()


class SquareNlpSolverBase(object):
    """A base class for NLP solvers that act on a square system
    of equality constraints.

    """
    OPTIONS = ConfigBlock()

    def __init__(self, nlp, timer=None, options=None):
        """
        Arguments
        ---------
        nlp: ExtendedNLP
            An instance of ExtendedNLP that will be solved.
            ExtendedNLP is required to ensure that the NLP has equal
            numbers of primal variables and equality constraints.

        """
        if timer is None:
            timer = HierarchicalTimer()
        if options is None:
            options = {}
        self.options = self.OPTIONS(options)

        self._timer = timer
        self._nlp = nlp
        self._function_values = None
        self._jacobian = None

        if self._nlp.n_eq_constraints() != self._nlp.n_primals():
            raise RuntimeError(
                "Cannot construct a square solver for an NLP that"
                " does not have the same numbers of variables as"
                " equality constraints. Got %s variables and %s"
                " equalities."
                % (self._nlp.n_primals(), self._nlp.n_eq_constraints())
            )
        # Checking for a square system of equalities is easy, but checking
        # bounds is a little difficult. We don't know how an NLP will
        # implement bounds (no bound could be None, np.nan, or np.inf),
        # so it is annoying to check that bounds are not present.
        # Instead, we just ignore bounds, and the user must know that the
        # result of this solver is not guaranteed to respect bounds.
        # While it is easier to check that inequalities are absent,
        # for consistency, we take the same approach and simply ignore
        # them.

    def solve(self, x0=None):
        # the NLP has a natural initial guess - the cached primal
        # values. x0 may be provided if a different initial guess
        # is desired.
        raise NotImplementedError(
            "%s has not implemented the solve method" % self.__class__
        )

    def evaluate_function(self, x0):
        # NOTE: NLP object should handle any caching
        self._timer.start("eval_f")
        self._nlp.set_primals(x0)
        values = self._nlp.evaluate_eq_constraints()
        self._timer.stop("eval_f")
        return values

    def evaluate_jacobian(self, x0):
        # NOTE: NLP object should handle any caching
        self._timer.start("eval_j")
        self._nlp.set_primals(x0)
        self._jacobian = self._nlp.evaluate_jacobian_eq(out=self._jacobian)
        self._timer.stop("eval_j")
        return self._jacobian


class DenseSquareNlpSolver(SquareNlpSolverBase):
    """A square NLP solver that uses a dense Jacobian
    """

    def evaluate_jacobian(self, x0):
        sparse_jac = super().evaluate_jacobian(x0)
        dense_jac = sparse_jac.toarray()
        return dense_jac
