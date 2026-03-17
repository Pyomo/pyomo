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
    RangeSet,
    Reals,
    Set,
    TransformationFactory,
    Var,
    maximize,
    minimize,
)
from pyomo.core.expr.calculus.diff_with_pyomo import reverse_sd
from pyomo.mpec import ComplementarityList, complements
from pyomo.util.vars_from_expressions import get_vars_from_components
from pyomo.util.config_domains import ComponentDataSet


class _KKTReformulationData(AutoSlots.Mixin):
    __slots__ = (
        "equality_multiplier_from_con",
        "equality_con_from_multiplier",
        "inequality_multiplier_from_con",
        "inequality_con_from_multiplier",
        "var_bound_multiplier_index_to_con",
        "equality_multiplier_index_to_con",
        "inequality_multiplier_index_to_con",
        "equality_con_to_expr",
        "inequality_con_to_expr",
        "ranged_constraints",
    )

    def __init__(self):
        self.equality_multiplier_from_con = ComponentMap()
        self.equality_con_from_multiplier = ComponentMap()
        self.inequality_multiplier_from_con = ComponentMap()
        self.inequality_con_from_multiplier = ComponentMap()

        self.var_bound_multiplier_index_to_con = {}
        self.equality_multiplier_index_to_con = {}
        self.inequality_multiplier_index_to_con = {}
        self.equality_con_to_expr = ComponentMap()
        self.inequality_con_to_expr = ComponentMap()

        self.ranged_constraints = ComponentSet()


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
        'parametrize_wrt',
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
                f"""model already has an attribute with the 
                specified kkt_block_name: '{config.kkt_block_name}'"""
            )

        # we should check that all vars the user fixed are included
        # in parametrize_wrt
        params = config.parametrize_wrt
        vars_in_cons = ComponentSet(
            get_vars_from_components(model, Constraint, active=True, descend_into=True)
        )
        vars_in_obj = ComponentSet(
            get_vars_from_components(model, Objective, active=True, descend_into=True)
        )
        vars_in_model = vars_in_cons | vars_in_obj
        fixed_vars_in_model = ComponentSet(v for v in vars_in_model if v.is_fixed())
        missing = fixed_vars_in_model - params
        if missing:
            raise ValueError("All fixed variables must be included in parametrize_wrt.")

        # we should also check that all vars the user passes in parametrize_wrt
        # exist on an active constraint or objective within the model
        unknown = params - vars_in_model
        if unknown:
            raise ValueError(
                "A variable passed in parametrize_wrt does not exist on an "
                "active constraint or objective within the model."
            )

        kkt_block = Block(concrete=True)
        kkt_block.parametrize_wrt = params
        self._reformulate(model, kkt_block)
        model.add_component(config.kkt_block_name, kkt_block)
        return model

    def _reformulate(self, model, kkt_block):
        active_objs = list(
            model.component_data_objects(Objective, active=True, descend_into=True)
        )
        if len(active_objs) != 1:
            raise ValueError(
                f"model must have only one active objective; found {len(active_objs)}"
            )

        self._construct_lagrangean(model, kkt_block)
        self._enforce_stationarity_conditions(kkt_block)
        self._enforce_complementarity_conditions(model, kkt_block)

        active_objs[0].deactivate()
        kkt_block.dummy_obj = Objective(expr=1)

        return model

    def _construct_lagrangean(self, model, kkt_block):
        # we need to loop through the model and store the
        # equality and inequality constraints
        info = model.private_data()
        equality_cons = []
        inequality_cons = []
        for con in model.component_data_objects(
            Constraint, descend_into=True, active=True
        ):
            lower, body, upper = con.to_bounded_expression()
            if con.equality:
                equality_cons.append(con)
                info.equality_con_to_expr[con] = upper - body
            else:
                if lower is not None:
                    inequality_cons.append((con, "lb"))
                    info.inequality_con_to_expr.setdefault(con, {})["lb"] = lower - body
                if upper is not None:
                    inequality_cons.append((con, "ub"))
                    info.inequality_con_to_expr.setdefault(con, {})["ub"] = body - upper

                    # lower is not None and upper is not None -> ranged constraint
                    if lower is not None:
                        # we want to keep track of the ranged constraints because the mapping between
                        # multipliers and ranged constraints will be a tuple (to indicate bound as well)
                        # instead of simply the model object
                        info.ranged_constraints.add(con)

        kkt_block.equality_cons_list = list(equality_cons)
        kkt_block.gamma_set = RangeSet(0, len(equality_cons) - 1)
        kkt_block.gamma = Var(kkt_block.gamma_set, domain=Reals)
        info.equality_multiplier_index_to_con = dict(
            enumerate(kkt_block.equality_cons_list)
        )
        kkt_block.inequality_cons_list = list(inequality_cons)
        kkt_block.alpha_con_set = RangeSet(0, len(inequality_cons) - 1)
        kkt_block.alpha_con = Var(kkt_block.alpha_con_set, domain=NonNegativeReals)
        info.inequality_multiplier_index_to_con = dict(
            enumerate(kkt_block.inequality_cons_list)
        )

        # we also need to consider inequality constraints
        # formed by the user specifying variable bounds
        var_bound_sides = []
        vars_in_cons = ComponentSet(
            get_vars_from_components(model, Constraint, active=True, descend_into=True)
        )
        vars_in_obj = ComponentSet(
            get_vars_from_components(model, Objective, active=True, descend_into=True)
        )
        kkt_block.var_set = vars_in_cons | vars_in_obj
        kkt_block.var_set = kkt_block.var_set - kkt_block.parametrize_wrt
        for var in kkt_block.var_set:
            if var.has_lb():
                var_bound_sides.append((var, "lb"))
            if var.has_ub():
                var_bound_sides.append((var, "ub"))
        kkt_block.var_bound_set = RangeSet(0, len(var_bound_sides) - 1)
        kkt_block.alpha_var_bound = Var(
            kkt_block.var_bound_set, domain=NonNegativeReals
        )

        info.var_bound_multiplier_index_to_con = dict(enumerate(var_bound_sides))

        # indexing the inequality constraint expressions will help
        # with constructing the lagrangean later
        def _var_bound_expr_rule(kkt, i):
            var, side = info.var_bound_multiplier_index_to_con[i]
            return (var.lb - var) if side == "lb" else (var - var.ub)

        kkt_block.var_bound_expr = Expression(
            kkt_block.var_bound_set, rule=_var_bound_expr_rule
        )

        # we will construct the lagrangean by first adding the objective,
        # and then looping through and adding the product of each constraint and
        # the corresponding multiplier
        obj = list(
            model.component_data_objects(Objective, active=True, descend_into=True)
        )
        # Note: maximize is -1 and minimize is +1
        lagrangean = obj[0].sense * obj[0].expr

        for index, con in enumerate(kkt_block.equality_cons_list):
            lagrangean += info.equality_con_to_expr[con] * kkt_block.gamma[index]
            info.equality_con_from_multiplier[kkt_block.gamma[index]] = con
            info.equality_multiplier_from_con[con] = kkt_block.gamma[index]

        for index, (con, bound) in enumerate(kkt_block.inequality_cons_list):
            lagrangean += (
                info.inequality_con_to_expr[con][bound] * kkt_block.alpha_con[index]
            )
            # mappings for ranged constraints will consider bounds as well
            if con in info.ranged_constraints:
                info.inequality_con_from_multiplier[kkt_block.alpha_con[index]] = (
                    con,
                    bound,
                )
                info.inequality_multiplier_from_con[(con, bound)] = kkt_block.alpha_con[
                    index
                ]
            else:
                info.inequality_con_from_multiplier[kkt_block.alpha_con[index]] = con
                info.inequality_multiplier_from_con[con] = kkt_block.alpha_con[index]

        for i in kkt_block.var_bound_set:
            lagrangean += kkt_block.var_bound_expr[i] * kkt_block.alpha_var_bound[i]
            # mappings for ranged constraints built from variable bounds
            var, bound = info.var_bound_multiplier_index_to_con[i]
            info.inequality_con_from_multiplier[kkt_block.alpha_var_bound[i]] = (
                var,
                bound,
            )
            info.inequality_multiplier_from_con[(var, bound)] = (
                kkt_block.alpha_var_bound[i]
            )

        kkt_block.lagrangean = Expression(expr=lagrangean)

    def _enforce_stationarity_conditions(self, kkt_block):
        deriv_lagrangean = reverse_sd(kkt_block.lagrangean.expr)
        kkt_block.stationarity_conditions = ConstraintList()
        for var in kkt_block.var_set:
            kkt_block.stationarity_conditions.add(deriv_lagrangean[var] == 0)

    def _enforce_complementarity_conditions(self, model, kkt_block):
        info = model.private_data()
        kkt_block.complements = ComplementarityList()
        for index, (con, bound) in enumerate(kkt_block.inequality_cons_list):
            kkt_block.complements.add(
                complements(
                    kkt_block.alpha_con[index] >= 0,
                    info.inequality_con_to_expr[con][bound] <= 0,
                )
            )
        # we also need to consider the inequality constraints
        # formed from the user specifying the variable bounds
        for i in kkt_block.var_bound_set:
            kkt_block.complements.add(
                complements(
                    kkt_block.alpha_var_bound[i] >= 0, kkt_block.var_bound_expr[i] <= 0
                )
            )

    def get_constraint_from_multiplier(self, model, multiplier_var):
        """
        Return the constraint or variable bound corresponding to a KKT multiplier variable.

        Parameters
        ----------
        model: ConcreteModel
            The model on which the kkt transformation was applied
        multiplier_var: Var
            A KKT multiplier created by the transformation.

        Returns
        -------
        Constraint or Tuple
            - Constraint object for simple constraints
            - (Constraint, bound) tuple for ranged constraints
            - (Var, bound) tuple for variable bounds
        """

        info = model.private_data()
        if multiplier_var in info.equality_con_from_multiplier:
            return info.equality_con_from_multiplier[multiplier_var]
        if multiplier_var in info.inequality_con_from_multiplier:
            # if this multiplier var maps to a ranged constraint, we will return a tuple
            # so that we can indicate which bound the multiplier var maps to
            return info.inequality_con_from_multiplier[multiplier_var]
        raise ValueError(
            f"The KKT multiplier: {multiplier_var.name}, does not exist on {model.name}."
        )

    def get_multiplier_from_constraint(self, model, constraint=None, variable=None):
        """
        Return the multiplier variable corresponding to a constraint or variable bound.

        Parameters
        ----------
        model: ConcreteModel
            The model on which the kkt transformation was applied to
        constraint: Constraint or Tuple, optional
            - A primal Constraint object for simple constraints, OR
            - A tuple (constraint, 'lb'|'ub') for ranged constraints
            Mutually exclusive with variables.
        Variable: Tuple, optional
            A tuple (Var, 'lb'|'ub') for variable bounds.
            Mutually exclusive with constraint.

        Returns
        -------
        Var
            The KKT multiplier variable corresponding to the constraint or variable bound.
        """

        if (constraint is None) and (variable is None):
            raise ValueError("Must provide 'constraint' or 'variable'.")
        if (constraint is not None) and (variable is not None):
            raise ValueError(
                "Cannot provide both 'constraint' and 'variable'. " "Provide only one."
            )

        info = model.private_data()
        if constraint is not None:
            if isinstance(constraint, tuple):
                if len(constraint) != 2:
                    raise ValueError(
                        f"constraint tuple must be (Constraint, bound), "
                        f"got tuple of length {len(constraint)}"
                    )
                con_obj, bound = constraint
                if bound not in ['lb', 'ub']:
                    raise ValueError(f"Bound must be 'lb' or 'ub', got: '{bound}'")
                key = (con_obj, bound)
                if key in info.inequality_multiplier_from_con:
                    return info.inequality_multiplier_from_con[key]
                raise ValueError(
                    f"Ranged constraint '{con_obj.name}' with bound='{bound}' "
                    f"does not exist on {model.name}."
                )

            # simple constraints are much easier to deal with
            else:
                if constraint in info.equality_multiplier_from_con:
                    return info.equality_multiplier_from_con[constraint]
                if constraint in info.inequality_multiplier_from_con:
                    return info.inequality_multiplier_from_con[constraint]
                # may be a ranged constraint
                is_ranged = constraint in info.ranged_constraints
                if is_ranged:
                    raise ValueError(
                        f"Constraint '{constraint.name}' is a ranged constraint. "
                        "Provide as tuple: constraint=(constraint_obj, 'lb'|'ub')."
                    )
                raise ValueError(
                    f"Constraint '{constraint.name}' does not exist on {model.name}."
                )

        # we need to deal with the case that the user wants multipliers associated with variable bounds
        if variable is not None:
            # variable bounds must be provided as tuple
            if not isinstance(variable, tuple):
                raise ValueError(
                    "variable must be a tuple (Var, 'lb'|'ub'), "
                    f"got: {type(variable).__name__}"
                )
            if len(variable) != 2:
                raise ValueError(
                    f"variable tuple must be (Var, bound), "
                    f"got tuple of length {len(variable)}"
                )
            var_obj, bound = variable
            if bound not in ['lb', 'ub']:
                raise ValueError(f"Bound must be 'lb' or 'ub', got: '{bound}'")
            key = (var_obj, bound)
            if key in info.inequality_multiplier_from_con:
                return info.inequality_multiplier_from_con[key]
            raise ValueError(
                f"Variable bound {var_obj.name} (bound='{bound}') "
                f"does not exist on {model.name}. "
                f"The variable may not have a {bound} bound defined."
            )
