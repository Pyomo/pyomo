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
                "Model '%s' has n o objective or multiple active objectives. Cannot "
                "take dual with more than one objective!" % model.name
            )
        primal_sense = std_form.objectives[0].sense

        dual = ConcreteModel(name="%s dual" % model.name)
        # This is a csc matrix, so we'll skipping transposing and just work off
        # of the folumns
        A = std_form.A
        rows = range(A.shape[1])
        cols = range(A.shape[0])
        dual.x = Var(cols, domain=NonNegativeReals)
        trans_info = dual.private_data()
        for j, (primal_cons, ineq) in enumerate(std_form.rows):
            if primal_sense is minimize and ineq == 1:
                dual.x[j].domain = NonPositiveReals
            elif primal_sense is maximize and ineq == -1:
                dual.x[j].domain = NonPositiveReals
            if ineq == 0:
                # equality
                dual.x[j].domain = Reals
            trans_info.primal_constraint[dual.x[j]] = primal_cons
            trans_info.dual_var[primal_cons] = dual.x[j]

        dual.constraints = Constraint(rows)
        for i, primal in enumerate(std_form.columns):
            lhs = 0
            for j in cols:
                coef = A[j, i]
                if not coef:
                    continue
                elif isclose_const(coef, 1.0):
                    lhs += dual.x[j]
                elif isclose_const(coef, -1.0):
                    lhs -= dual.x[j]
                else:
                    lhs += float(A[j, i]) * dual.x[j]

            if primal_sense is minimize:
                if primal.domain is NonNegativeReals:
                    dual.constraints[i] = lhs <= float(std_form.c[0, i])
                elif primal.domain is NonPositiveReals:
                    dual.constraints[i] = lhs >= float(std_form.c[0, i])
            else:
                if primal.domain is NonNegativeReals:
                    dual.constraints[i] = lhs >= float(std_form.c[0, i])
                elif primal.domain is NonPositiveReals:
                    dual.constraints[i] = lhs <= float(std_form.c[0, i])
            if primal.domain is Reals:
                dual.constraints[i] = lhs == float(std_form.c[0, i])
            trans_info.dual_constraint[primal] = dual.constraints[i]
            trans_info.primal_var[dual.constraints[i]] = primal

        dual.obj = Objective(
            expr=sum(std_form.rhs[j] * dual.x[j] for j in cols), sense=-primal_sense
        )

        return dual

    def get_primal_constraint(self, model, dual_var):
        primal_constraint = model.private_data().primal_constraint
        if dual_var in primal_constraint:
            return primal_constraint[dual_var]
        else:
            raise ValueError(
                "It does not appear that Var '%s' is a dual variable on model '%s'"
                % (dual_var.name, model.name)
            )

    def get_dual_constraint(self, model, primal_var):
        dual_constraint = model.private_data().dual_constraint
        if primal_var in dual_constraint:
            return dual_constraint[primal_var]
        else:
            raise ValueError(
                "It does not appear that Var '%s' is a primal variable from model '%s'"
                % (primal_var.name, model.name)
            )

    def get_primal_var(self, model, dual_constraint):
        primal_var = model.private_data().primal_var
        if dual_constraint in primal_var:
            return primal_var[dual_constraint]
        else:
            raise ValueError(
                "It does not appear that Constraint '%s' is a dual constraint on "
                "model '%s'" % (dual_constraint.name, model.name)
            )

    def get_dual_var(self, model, primal_constraint):
        dual_var = model.private_data().dual_var
        if primal_constraint in dual_var:
            return dual_var[primal_constraint]
        else:
            raise ValueError(
                "It does not appear that Constraint '%s' is a primal constraint from "
                "model '%s'" % (primal_constraint.name, model.name)
            )
