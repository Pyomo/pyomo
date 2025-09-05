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

from pyomo.common.autoslots import AutoSlots
from pyomo.common.collections import ComponentMap
from pyomo.common.config import ConfigDict, ConfigValue
from pyomo.common.dependencies import scipy
from pyomo.core import (
    ConcreteModel,
    Block,
    Var,
    Constraint,
    Objective,
    TransformationFactory,
    NonNegativeReals,
    NonPositiveReals,
    maximize,
    minimize,
    RangeSet,
    Reals,
)
from pyomo.core.expr.numvalue import native_numeric_types
from pyomo.opt import WriterFactory
from pyomo.repn.standard_repn import isclose_const
from pyomo.util.config_domains import ComponentDataSet


class _LPDualData(AutoSlots.Mixin):
    __slots__ = ('primal_var', 'dual_var', 'primal_constraint', 'dual_constraint')

    def __init__(self):
        self.primal_var = {}
        self.dual_var = {}
        self.primal_constraint = ComponentMap()
        self.dual_constraint = ComponentMap()


Block.register_private_data_initializer(_LPDualData)


@TransformationFactory.register(
    'core.lp_dual', 'Generate the linear programming dual of the given model'
)
class LinearProgrammingDual(object):
    CONFIG = ConfigDict("core.lp_dual")
    CONFIG.declare(
        'parameterize_wrt',
        ConfigValue(
            default=None,
            domain=ComponentDataSet(Var),
            description="Vars to treat as data for the purposes of taking the dual",
            doc="""
            Optional list of Vars to be treated as data while taking the LP dual.

            For example, if this is the dual of the inner problem in a multilevel
            optimization problem, then the outer problem's Vars would be specified
            in this list since they are not variables from the perspective of the 
            inner problem.
            """,
        ),
    )

    def apply_to(self, model, **options):
        raise NotImplementedError(
            "The 'core.lp_dual' transformation does not implement "
            "apply_to since it is ambiguous what it means to take a dual "
            "in place. Please use 'create_using' and do what you wish with the "
            "returned model."
        )

    def create_using(self, model, ostream=None, **kwds):
        """Take linear programming dual of a model

        Returns
        -------
        ConcreteModel containing linear programming dual

        Parameters
        ----------
        model: ConcreteModel
            The concrete Pyomo model to take the dual of

        ostream: None
            This is provided for API compatibility with other writers
            and is ignored here.

        """
        config = self.CONFIG(kwds.pop('options', {}))
        config.set_value(kwds)

        if config.parameterize_wrt is None:
            std_form = WriterFactory('compile_standard_form').write(
                model, mixed_form=True, set_sense=None
            )
        else:
            std_form = WriterFactory('compile_parameterized_standard_form').write(
                model, wrt=config.parameterize_wrt, mixed_form=True, set_sense=None
            )
        return self._take_dual(model, std_form)

    def _take_dual(self, model, std_form):
        if len(std_form.objectives) != 1:
            raise ValueError(
                "Model '%s' has no objective or multiple active objectives. Can "
                "only take dual with exactly one active objective!" % model.name
            )
        if len(std_form.columns) == 0 and std_form.c.shape[1] == 0:
            raise ValueError(
                f"Model '{model.name}' has no variables in the active Constraints "
                f"or Objective."
            )
        primal_sense = std_form.objectives[0].sense

        dual = ConcreteModel(name="%s dual" % model.name)
        # This is a csc matrix, so we'll skip transposing and just work off
        # of the columns
        A = std_form.A
        c = std_form.c.todense().ravel()
        dual_rows = range(A.shape[1])
        dual_cols = range(A.shape[0])
        dual.x = Var(dual_cols, domain=NonNegativeReals)
        trans_info = dual.private_data()
        A_csr = A.tocsr()
        for j, (primal_cons, ineq) in enumerate(std_form.rows):
            # We need to check this constraint isn't trivial due to the
            # parameterization, which we can detect if the row is all 0's.
            if A_csr.indptr[j] == A_csr.indptr[j + 1]:
                # All 0's in the coefficient matrix: check what's on the RHS
                b = std_form.rhs[j]
                if type(b) not in native_numeric_types:
                    # The parameterization made this trivial. I'm not sure what's
                    # really expected here, so maybe we just scream? Or we leave
                    # the constraint in the model as it is written...
                    raise ValueError(
                        f"The primal model contains a constraint that the "
                        f"parameterization makes trivial: '{primal_cons.name}'"
                        f"\nPlease deactivate it or declare it on another Block "
                        f"before taking the dual."
                    )
                else:
                    # The whole constraint is trivial--it will already have been
                    # checked compiling the standard form, so we can safely ignore
                    # it.
                    pass

            # maximize is -1 and minimize is +1 and ineq is +1 for <= and -1 for
            # >=, so we need to change domain to NonPositiveReals if the product
            # of these is +1.
            if primal_sense * ineq == 1:
                dual.x[j].domain = NonPositiveReals
            elif ineq == 0:
                # equality
                dual.x[j].domain = Reals
            trans_info.primal_constraint[dual.x[j]] = primal_cons
            trans_info.dual_var[primal_cons] = dual.x[j]

        dual.constraints = Constraint(dual_rows)
        for i, primal in enumerate(std_form.columns):
            lhs = 0
            for j in range(A.indptr[i], A.indptr[i + 1]):
                coef = A.data[j]
                primal_row = A.indices[j]
                lhs += coef * dual.x[primal_row]

            domain = primal.domain
            lb, ub = domain.bounds()
            # Note: the following checks the domain for continuity and compactness:
            if not domain == RangeSet(*domain.bounds(), 0):
                raise ValueError(
                    f"The domain of the primal variable '{primal.name}' "
                    f"is not continuous."
                )
            unrestricted = (lb is None or lb < 0) and (ub is None or ub > 0)
            nonneg = (lb is not None) and lb >= 0

            if unrestricted:
                dual.constraints[i] = lhs == c[i]
            elif primal_sense is minimize:
                if nonneg:
                    dual.constraints[i] = lhs <= c[i]
                else:  # primal domain is nonpositive
                    dual.constraints[i] = lhs >= c[i]
            else:
                if nonneg:
                    dual.constraints[i] = lhs >= c[i]
                else:  # primal domain is nonpositive
                    dual.constraints[i] = lhs <= c[i]
            trans_info.dual_constraint[primal] = dual.constraints[i]
            trans_info.primal_var[dual.constraints[i]] = primal

        dual.obj = Objective(
            expr=sum(std_form.rhs[j] * dual.x[j] for j in dual_cols),
            sense=-primal_sense,
        )

        return dual

    def get_primal_constraint(self, model, dual_var):
        """Return the primal constraint corresponding to 'dual_var'

        Returns
        -------
        Constraint

        Parameters
        ----------
        model: ConcreteModel
            A dual model returned from the 'core.lp_dual' transformation
        dual_var: Var
            A dual variable on 'model'

        """
        primal_constraint = model.private_data().primal_constraint
        if dual_var in primal_constraint:
            return primal_constraint[dual_var]
        else:
            raise ValueError(
                "It does not appear that Var '%s' is a dual variable on model '%s'"
                % (dual_var.name, model.name)
            )

    def get_dual_constraint(self, model, primal_var):
        """Return the dual constraint corresponding to 'primal_var'

        Returns
        -------
        Constraint

        Parameters
        ----------
        model: ConcreteModel
            A primal model passed as an argument to the 'core.lp_dual' transformation
        primal_var: Var
            A primal variable on 'model'

        """
        dual_constraint = model.private_data().dual_constraint
        if primal_var in dual_constraint:
            return dual_constraint[primal_var]
        else:
            raise ValueError(
                "It does not appear that Var '%s' is a primal variable on model '%s'"
                % (primal_var.name, model.name)
            )

    def get_primal_var(self, model, dual_constraint):
        """Return the primal variable corresponding to 'dual_constraint'

        Returns
        -------
        Var

        Parameters
        ----------
        model: ConcreteModel
            A dual model returned from the 'core.lp_dual' transformation
        dual_constraint: Constraint
            A constraint on 'model'

        """
        primal_var = model.private_data().primal_var
        if dual_constraint in primal_var:
            return primal_var[dual_constraint]
        else:
            raise ValueError(
                "It does not appear that Constraint '%s' is a dual constraint on "
                "model '%s'" % (dual_constraint.name, model.name)
            )

    def get_dual_var(self, model, primal_constraint):
        """Return the dual variable corresponding to 'primal_constraint'

        Returns
        -------
        Var

        Parameters
        ----------
        model: ConcreteModel
            A primal model passed as an argument to the 'core.lp_dual' transformation
        primal_constraint: Constraint
            A constraint on 'model'

        """
        dual_var = model.private_data().dual_var
        if primal_constraint in dual_var:
            return dual_var[primal_constraint]
        else:
            raise ValueError(
                "It does not appear that Constraint '%s' is a primal constraint on "
                "model '%s'" % (primal_constraint.name, model.name)
            )
