#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.core import (Var, Block, Constraint, Param, Set, Suffix, Expression,
                        Objective, SortComponents, value, ConstraintList)
from pyomo.core.base import (TransformationFactory, _VarData)
from pyomo.core.base.block import _BlockData
from pyomo.core.base.param import _ParamData
from pyomo.core.base.constraint import _ConstraintData
from pyomo.core.plugins.transform.hierarchy import Transformation
from pyomo.common.config import ConfigBlock, ConfigValue
from pyomo.common.modeling import unique_component_name
from pyomo.repn.standard_repn import generate_standard_repn
from pyomo.core.kernel.component_map import ComponentMap
from pyomo.core.kernel.component_set import ComponentSet

def vars_to_eliminate_list(x):
    if isinstance(x, (Var, _VarData)):
        if not x.is_indexed():
            return ComponentSet([x])
        ans = ComponentSet()
        for j in x.index_set():
            ans.add(x[j])
        return ans
    elif hasattr(x, '__iter__'):
        ans = ComponentSet()
        for i in x:
            ans.update(vars_to_eliminate_list(i))
        return ans
    else:
        raise ValueError(
            "Expected Var or list of Vars."
            "\n\tRecieved %s" % type(x))

@TransformationFactory.register('contrib.fourier_motzkin_elimination',
                                doc="Project out specified (continuous) "
                                "variables from a linear model.")
class Fourier_Motzkin_Elimination_Transformation(Transformation):
    """Project out specified variables from a linear model.

    This transformation requires the following keyword argument:
        vars_to_eliminate: A user-specified list of continuous variables to 
                           project out of the model

    The transformation will deactivate the original constraints of the model
    and create a new block named "_pyomo_contrib_fme_transformation" with the 
    projected constraints. Note that this transformation will flatten the 
    structure of the original model since there is no obvious mapping between 
    the original model and the transformed one.

    """

    CONFIG = ConfigBlock("contrib.fourier_motzkin_elimination")
    CONFIG.declare('vars_to_eliminate', ConfigValue(
        default=None,
        domain=vars_to_eliminate_list,
        description="Continuous variable or list of continuous variables to "
        "project out of the model", 
        doc="""
        This specifies the list of variables to project out of the model.
        Note that these variables must all be continuous and the model must be 
        linear."""
    ))

    def __init__(self):
        """Initialize transformation object"""
        super(Fourier_Motzkin_Elimination_Transformation, self).__init__()
    
    def _apply_to(self, instance, **kwds):
        config = self.CONFIG(kwds.pop('options', {}))
        config.set_value(kwds)
        vars_to_eliminate = config.vars_to_eliminate
        if vars_to_eliminate is None:
            raise RuntimeError("The Fourier-Motzkin Elimination transformation "
                               "requires the argument vars_to_eliminate, a "
                               "list of Vars to be projected out of the model.")
        
        # make transformation block
        transBlockName = unique_component_name(
            instance,
            '_pyomo_contrib_fme_transformation')
        transBlock = Block()
        instance.add_component(transBlockName, transBlock)
        projected_constraints = transBlock.projected_constraints = \
                                ConstraintList()

        # collect all of the constraints
        # NOTE that we are ignoring deactivated constraints
        constraints = []
        ctypes_not_to_transform = set((Block, Param, Objective, Set, Expression,
                                       Suffix))
        for obj in instance.component_data_objects(
                descend_into=Block,
                sort=SortComponents.deterministic,
                active=True):
            if obj.type() in ctypes_not_to_transform:
                continue
            elif obj.type() is Constraint:
                cons_list = self._process_constraint(obj)
                constraints.extend(cons_list)
                obj.deactivate() # the truth will be on our transformation block
            elif obj.type() is Var:
                # variable bounds are constraints, but we only need them if this
                # is a variable we are projecting out
                if obj not in vars_to_eliminate:
                    continue
                if obj.lb is not None:
                    constraints.append({'body': generate_standard_repn(obj),
                                        'lower': value(obj.lb),
                                        'map': ComponentMap([(obj, 1)])})
                if obj.ub is not None:
                    constraints.append({'body': generate_standard_repn(-obj),
                                        'lower': -value(obj.ub),
                                        'map': ComponentMap([(obj, -1)])})
            else:
                raise RuntimeError(
                    "Found active component %s of type %s. The "
                    "Fourier-Motzkin Elimination transformation can only "
                    "handle purely algebraic models. That is, only "
                    "Sets, Params, Vars, Constraints, Expressions, Blocks, "
                    "and Objectives may be active on the model." % (obj.name, 
                                                                    obj.type()))

        new_constraints = self._fourier_motzkin_elimination(constraints,
                                                            vars_to_eliminate)

        # put the new constraints on the transformation block
        for cons in new_constraints:
            body = cons['body']
            lhs = sum(coef*var for (coef, var) in zip(body.linear_coefs,
                                                      body.linear_vars)) + \
                sum(coef*v1*v2 for (coef, (v1, v2)) in zip(body.quadratic_coefs,
                                                           body.quadratic_vars))
            if body.nonlinear_expr is not None:
                lhs += body.nonlinear_expr
            lower = cons['lower']
            if type(lhs >= lower) is bool:
                if lhs >= lower:
                    continue
                else:
                    # This would actually make a lot of sense in this case...
                    #projected_constraints.add(Constraint.Infeasible)
                    raise RuntimeError("Fourier-Motzkin found the model is "
                                       "infeasible!")
            else:
                projected_constraints.add(lhs >= lower)

    def _process_constraint(self, constraint):
        """Transforms a pyomo Constraint objective into a list of dictionaries
        representing only >= constraints. That is, if the constraint has both an
        ub and a lb, it is transformed into two constraints. Otherwise it is
        flipped if it is <=. Each dictionary contains the keys 'lower',
        and 'body' where, after the process, 'lower' will be a constant, and 
        'body' will be the standard repn of the body. (The constant will be 
        moved to the RHS and we know that the upper bound is None after this).
        """
        body = constraint.body
        std_repn = generate_standard_repn(body)
        cons_dict = {'lower': constraint.lower,
                     'body': std_repn
                     }
        upper = constraint.upper
        constraints_to_add = [cons_dict]
        if upper is not None:
            # if it has both bounds
            if cons_dict['lower'] is not None:
                # copy the constraint and flip
                leq_side = {'lower': -upper,
                            'body': generate_standard_repn(-1.0*body)}
                self._move_constant_and_add_map(leq_side)
                constraints_to_add.append(leq_side)

            # If it has only an upper bound, we just need to flip it
            else:
                # just flip the constraint
                cons_dict['lower'] = -upper
                cons_dict['body'] = generate_standard_repn(-1.0*body)
        self._move_constant_and_add_map(cons_dict)

        return constraints_to_add

    def _move_constant_and_add_map(self, cons_dict):
        """Takes constraint in dicionary form already in >= form, 
        and moves the constant to the RHS
        """
        body = cons_dict['body']
        constant = body.constant
        cons_dict['lower'] -= constant
        body.constant = 0

        # store a map of vars to coefficients. We can't use this in place of
        # standard repn because determinism, but this will save a lot of linear
        # time searches later.
        cons_dict['map'] = ComponentMap(zip(body.linear_vars, body.linear_coefs))

    def _fourier_motzkin_elimination(self, constraints, vars_to_eliminate):
        """Performs FME on the constraint list in the argument 
        (which is assumed to be all >= constraints and stored in the 
        dictionary representation), projecting out each of the variables in 
        vars_to_eliminate"""

        # We only need to eliminate variables that actually appear in
        # this set of constraints... Revise our list.
        vars_that_appear = []
        for cons in constraints:
            std_repn = cons['body']
            if not std_repn.is_linear():
                # as long as none of vars_that_appear are in the nonlinear part,
                # we are actually okay.
                nonlinear_vars = ComponentSet(v for two_tuple in 
                                        std_repn.quadratic_vars for
                                        v in two_tuple)
                nonlinear_vars.update(v for v in std_repn.nonlinear_vars)
                for var in nonlinear_vars:
                    if var in vars_to_eliminate:
                        raise RuntimeError("Variable %s appears in a nonlinear "
                                           "constraint. The Fourier-Motzkin "
                                           "Elimination transformation can only "
                                           "be used to eliminate variables "
                                           "which only appear linearly." % 
                                           var.name)
            for var in std_repn.linear_vars:
                if var in vars_to_eliminate:
                    vars_that_appear.append(var)

        # we actually begin the recursion here
        while vars_that_appear:
            # first var we will project out
            the_var = vars_that_appear.pop()

            # we are 'reorganizing' the constraints, we sort based on the sign
            # of the coefficient of the_var: This tells us whether we have
            # the_var <= other stuff or vice versa.
            leq_list = []
            geq_list = []
            waiting_list = []

            for cons in constraints:
                leaving_var_coef = cons['map'].get(the_var)
                if leaving_var_coef is None or leaving_var_coef == 0:
                    waiting_list.append(cons)
                    continue

                # we know the constraint is a >= constraint, using that
                # assumption below.
                # NOTE: neither of the scalar multiplications below flip the
                # constraint. So we are sure to have only geq constraints
                # forever, which is exactly what we want.
                if leaving_var_coef < 0:
                    leq_list.append(
                        self._nonneg_scalar_multiply_linear_constraint(
                            cons, -1.0/leaving_var_coef))
                else:
                    geq_list.append(
                        self._nonneg_scalar_multiply_linear_constraint(
                            cons, 1.0/leaving_var_coef))

            constraints = waiting_list
            for leq in leq_list:
                for geq in geq_list:
                    constraints.append(self._add_linear_constraints(leq, geq))

        return constraints

    def _nonneg_scalar_multiply_linear_constraint(self, cons, scalar):
        """Multiplies all coefficients and the RHS of a >= constraint by scalar.
        There is no logic for flipping the equality, so this is just the 
        special case with a nonnegative scalar, which is all we need.
        """
        body = cons['body']
        body.linear_coefs = [scalar*coef for coef in body.linear_coefs]
        body.quadratic_coefs = [scalar*coef for coef in body.quadratic_coefs]
        body.nonlinear_expr = scalar*body.nonlinear_expr if \
                              body.nonlinear_expr is not None else None
        # and update the map... (It isn't lovely that I am storing this in two
        # places...)
        for var, coef in cons['map'].items():
            cons['map'][var] = coef*scalar

        # assume scalar >= 0 and constraint only has lower bound
        if cons['lower'] is not None:
            cons['lower'] *= scalar
        
        return cons

    def _add_linear_constraints(self, cons1, cons2):
        """Adds two >= constraints"""
        ans = {'lower': None, 'body': None, 'map': ComponentMap()}
        cons1_body = cons1['body']
        cons2_body = cons2['body']

        # Need this to be both deterministic and to account for the fact that
        # Vars aren't hashable.
        all_vars = list(cons1_body.linear_vars)
        seen = ComponentSet(all_vars)
        for v in cons2_body.linear_vars:
            if v not in seen:
                all_vars.append(v)
        
        expr = 0
        for var in all_vars:
            coef = cons1['map'].get(var, 0) + cons2['map'].get(var, 0)
            ans['map'][var] = coef
            expr += coef*var
        # deal with nonlinear stuff if there is any
        for cons in [cons1_body, cons2_body]:
            if cons.nonlinear_expr is not None:
                expr += cons.nonlinear_expr
            expr += sum(coef*v1*v2 for (coef, (v1, v2)) in
                        zip(cons.quadratic_coefs, cons.quadratic_vars)) 
        
        ans['body'] = generate_standard_repn(expr)

        # upper is None and lower exists, so this gets the constant
        ans['lower'] = cons1['lower'] + cons2['lower']

        return ans
