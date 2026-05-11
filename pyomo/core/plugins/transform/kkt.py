# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________


from pyomo.common.autoslots import AutoSlots
from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.common.config import ConfigDict, ConfigValue
from pyomo.core import (
    Block,
    Constraint,
    ConstraintList,
    Expression,
    NonNegativeReals,
    Objective,
    TransformationFactory,
    Var,
    VarList,
)
from pyomo.core.expr.calculus.diff_with_pyomo import reverse_sd
from pyomo.core.expr.visitor import identify_variables
from pyomo.mpec import ComplementarityList, complements
from pyomo.util.config_domains import ComponentDataSet


class _KKTReformulationData(AutoSlots.Mixin):
    __slots__ = ("obj_dual_map", "dual_obj_map")

    def __init__(self):
        self.obj_dual_map = ComponentMap()
        self.dual_obj_map = ComponentMap()


Block.register_private_data_initializer(_KKTReformulationData)


@TransformationFactory.register(
    'core.kkt', 'Generate KKT reformulation of the given model'
)
class NonLinearProgrammingKKT:
    CONFIG = ConfigDict("core.kkt")
    CONFIG.declare(
        'kkt_block_name',
        ConfigValue(
            default='kkt',
            doc="""
           Name of the block on which the kkt variables and constraints will be stored. 
           """,
        ),
    )
    CONFIG.declare(
        'parameterize_wrt',
        ConfigValue(
            default=[],
            domain=ComponentDataSet(Var),
            description='Vars to treat as data for the purposes of generating KKT reformulation',
            doc="""
            Optional list of Vars to be treated as data while generating the KKT reformulation.
            """,
        ),
    )

    def apply_to(self, model, **kwds):
        """
        Reformulate model with KKT conditions.
        """
        config = self.CONFIG(kwds.pop('options', {}))
        config.set_value(kwds)

        if hasattr(model, config.kkt_block_name):
            raise ValueError(
                "model already has an attribute with the "
                f"specified kkt_block_name: '{config.kkt_block_name}'"
            )

        # We will check below that all vars the user fixed are included in
        # parameterize_wrt
        params = config.parameterize_wrt

        kkt_block = Block(concrete=True)
        kkt_block.parameterize_wrt = params
        self._reformulate(model, kkt_block, params)
        model.add_component(config.kkt_block_name, kkt_block)
        return model

    def _reformulate(self, model, kkt_block, params):
        # initialize
        info = model.private_data()
        lagrangean = 0
        all_vars_set = ComponentSet()

        # collect the active Objectives
        active_objs = list(
            model.component_data_objects(Objective, active=True, descend_into=True)
        )
        if len(active_objs) != 1:
            raise ValueError(
                f"model must have exactly one active objective; found {len(active_objs)}"
            )
        # collect vars from active objective
        obj = active_objs[0]
        all_vars_set.update(identify_variables(obj.expr, include_fixed=True))
        lagrangean += obj.sense * obj.expr

        # list of equality multipliers
        kkt_block.gamma = VarList()
        # list of inequality multipliers
        kkt_block.alpha = VarList(domain=NonNegativeReals)
        # define inequality complements
        kkt_block.complements = ComplementarityList()

        for con in model.component_data_objects(
            Constraint, descend_into=True, active=True
        ):
            lower, body, upper = con.to_bounded_expression()

            # collect variables in constraint
            for expr in (lower, body, upper):
                if expr is None:
                    continue
                all_vars_set.update(identify_variables(expr=expr, include_fixed=True))

            if con.equality:
                gamma_i = kkt_block.gamma.add()
                lagrangean += (upper - body) * gamma_i
                info.obj_dual_map[con] = gamma_i
                info.dual_obj_map[gamma_i] = con

            else:
                alpha_l = None
                if lower is not None:
                    alpha_l = kkt_block.alpha.add()
                    con_expr = lower - body
                    lagrangean += con_expr * alpha_l
                    kkt_block.complements.add(complements(alpha_l >= 0, con_expr <= 0))
                    info.dual_obj_map[alpha_l] = con

                alpha_u = None
                if upper is not None:
                    alpha_u = kkt_block.alpha.add()
                    con_expr = body - upper
                    lagrangean += con_expr * alpha_u
                    kkt_block.complements.add(complements(alpha_u >= 0, con_expr <= 0))
                    info.dual_obj_map[alpha_u] = con

                info.obj_dual_map[con] = (alpha_l, alpha_u)

        fixed_vars = ComponentSet(v for v in all_vars_set if v.is_fixed())
        var_set = ComponentSet(all_vars_set)
        var_set -= fixed_vars

        # do error checking on parameterize_wrt
        missing = fixed_vars - params
        if missing:
            raise ValueError(
                "All fixed variables must be included in parameterize_wrt. "
                "Missing variables:\n\t" + "\n\t".join(v.name for v in missing)
            )

        extra = params - all_vars_set
        if extra:
            raise ValueError(
                "A variable passed in parameterize_wrt does not exist in an "
                "active constraint or objective within the model. "
                "Invalid variables:\n\t" + "\n\t".join(v.name for v in extra)
            )

        var_set = var_set - params
        for var in var_set:
            alpha_l = None
            if var.has_lb():
                alpha_l = kkt_block.alpha.add()
                con_expr = var.lb - var
                lagrangean += con_expr * alpha_l
                kkt_block.complements.add(complements(alpha_l >= 0, con_expr <= 0))
                info.dual_obj_map[alpha_l] = var

            alpha_u = None
            if var.has_ub():
                alpha_u = kkt_block.alpha.add()
                con_expr = var - var.ub
                lagrangean += con_expr * alpha_u
                kkt_block.complements.add(complements(alpha_u >= 0, con_expr <= 0))
                info.dual_obj_map[alpha_u] = var

            info.obj_dual_map[var] = (alpha_l, alpha_u)

        kkt_block.lagrangean = Expression(expr=lagrangean)

        # enforce stationarity conditions
        deriv_lagrangean = reverse_sd(kkt_block.lagrangean.expr)
        kkt_block.stationarity_conditions = ConstraintList()
        for var in var_set:
            kkt_block.stationarity_conditions.add(deriv_lagrangean[var] == 0)

        active_objs[0].deactivate()

    def get_object_from_multiplier(self, model, multiplier_var):
        """
        Return the constraint corresponding to a KKT multiplier variable. If the
        multiplier corresponds to an inequality formed by a variable bound, the variable
        is returned.

        Parameters
        ----------
        model: ConcreteModel
            The model on which the kkt transformation was applied
        multiplier_var: Var
            A KKT multiplier created by the transformation.

        Returns
        -------
        Object
            - Constraint object
            - Variable
        """

        info = model.private_data()
        if multiplier_var in info.dual_obj_map:
            return info.dual_obj_map[multiplier_var]
        raise ValueError(
            f"The KKT multiplier: {multiplier_var.name}, does not exist on {model.name}."
        )

    def get_multiplier_from_object(self, model, component):
        """
        Return the multiplier for the object. If the object is a normal constraint, a single
        multiplier is returned. If the object is a ranged constraint or a variable, a tuple
        containing the lower and upper bound multipliers is returned.

        Parameters
        ----------
        model: ConcreteModel
            The model to which the kkt transformation was applied to
        component: Constraint or Variable

        Returns
        -------
        VarData | tuple[VarData | None, VarData | None]
            The KKT multiplier(s) corresponding to the component.
            For ranged constraints/variables, returns (lb_mult, ub_mult),
            where an entry is 'None' if that bound doesn't exist.
        """

        info = model.private_data()
        if component in info.obj_dual_map:
            return info.obj_dual_map[component]
        raise ValueError(
            f"The component '{component.name}' either does not exist on "
            f"'{model.name}', or is not associated with a multiplier."
        )
