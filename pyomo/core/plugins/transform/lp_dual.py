#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2024
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.common.autoslots import AutoSlots
from pyomo.common.collections import ComponentMap
from pyomo.common.config import ConfigDict, ConfigValue
from pyomo.common.errors import MouseTrap
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
    Reals,
)
from pyomo.opt import WriterFactory
from pyomo.repn.standard_repn import isclose_const
from pyomo.util.var_list_domain import var_component_set


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
            domain=var_component_set,
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
        raise MouseTrap(
            "The 'core.lp_dual' transformation does not currently implement "
            "apply_to since it is a bit ambiguous what it means to take a dual "
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
            std_form = WriterFactory('parameterized_standard_form_compiler').write(
                model, wrt=config.parameterize_wrt, mixed_form=True, set_sense=None
            )
        return self._take_dual(model, std_form)

    def _take_dual(self, model, std_form):
        if len(std_form.objectives) != 1:
            raise ValueError(
                "Model '%s' has no objective or multiple active objectives. Cannot "
                "take dual with more than one objective!" % model.name
            )
        primal_sense = std_form.objectives[0].sense

        dual = ConcreteModel(name="%s dual" % model.name)
        # This is a csc matrix, so we'll skip transposing and just work off
        # of the columns
        A = std_form.A
        c = std_form.c.todense()
        dual_rows = range(A.shape[1])
        dual_cols = range(A.shape[0])
        dual.x = Var(dual_cols, domain=NonNegativeReals)
        trans_info = dual.private_data()
        for j, (primal_cons, ineq) in enumerate(std_form.rows):
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

            if primal.domain is Reals:
                dual.constraints[i] = lhs == c[0, i]
            elif primal_sense is minimize:
                if primal.domain is NonNegativeReals:
                    dual.constraints[i] = lhs <= c[0, i]
                else:  # primal.domain is NonPositiveReals
                    dual.constraints[i] = lhs >= c[0, i]
            else:
                if primal.domain is NonNegativeReals:
                    dual.constraints[i] = lhs >= c[0, i]
                else:  # primal.domain is NonPositiveReals
                    dual.constraints[i] = lhs <= c[0, i]
            trans_info.dual_constraint[primal] = dual.constraints[i]
            trans_info.primal_var[dual.constraints[i]] = primal

        dual.obj = Objective(
            expr=sum(std_form.rhs[j] * dual.x[j] for j in dual_cols),
            sense=-primal_sense,
        )

        return dual

    def _get_corresponding_component(
        self, model, model_type, component, component_type, mapping
    ):
        """Return the corresponding component based on the provided mapping.

        Parameters
        ----------
        model: ConcreteModel
            The model containing the mappings.
        component: Var or Constraint
            The primal/dual component for which we want to find the corresponding
            dual/primal component.
        component_type: str
            A string indicating whether the component is 'dual' or 'primal'.
        mapping: dict
            The mapping to look up the corresponding component.

        Returns
        -------
        Var or Constraint
            The corresponding component.

        Raises
        ------
        ValueError
            If the component is not found in the mapping.
        """
        if component in mapping:
            return mapping[component]
        else:
            raise ValueError(
                "It does not appear that %s '%s' is a %s %s on model '%s'"
                % (
                    component_type,
                    component.name,
                    model_type,
                    'variable' if component_type == 'Var' else 'constraint',
                    model.name,
                )
            )

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
        return self._get_corresponding_component(
            model, 'dual', dual_var, 'Var', primal_constraint
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
        return self._get_corresponding_component(
            model, 'primal', primal_var, 'Var', dual_constraint
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
        return self._get_corresponding_component(
            model, 'dual', dual_constraint, 'Constraint', primal_var
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
        return self._get_corresponding_component(
            model, 'primal', primal_constraint, 'Constraint', dual_var
        )
