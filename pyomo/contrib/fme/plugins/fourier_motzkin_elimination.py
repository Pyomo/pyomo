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
from pyomo.core.plugins.transform.hierarchy import LinearTransformation
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
            if isinstance(i, (Var, _VarData)):
                # flatten indexed things
                if i.is_indexed():
                    for j in i.index_set():
                        ans.add(i[j])
                else:
                    ans.add(i)
            else:
                raise ValueError(
                    "Expected Var or list of Vars."
                    "\n\tRecieved %s" % type(x))
        return ans
    else:
        raise ValueError(
            "Expected Var or list of Vars."
            "\n\tRecieved %s" % type(x))

@TransformationFactory.register('contrib.fourier_motzkin_elimination',
                                doc="Project out specified (continuous) "
                                "variables from a linear model.")
class Fourier_Motzkin_Elimination_Transformation(LinearTransformation):
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
        for obj in instance.component_data_objects(
                descend_into=Block,
                sort=SortComponents.deterministic,
                active=True):
            if obj.type() in (Block, _BlockData, Param, _ParamData, Objective,
                              Set, Expression, Suffix):
                continue
            elif obj.type() in (Constraint, _ConstraintData):
                cons_list = self._process_constraint(obj)
                constraints.extend(cons_list)
                obj.deactivate() # the truth will be on our transformation block
            elif obj.type() in (Var, _VarData):
                # variable bounds are constraints, but we only need them if this
                # is a variable we are projecting out
                if obj not in vars_to_eliminate:
                    continue
                if obj.lb is not None:
                    constraints.append({'body': generate_standard_repn(obj),
                                        'upper': None,
                                        'lower': value(obj.lb),
                                        'map': ComponentMap([(obj, 1)])})
                if obj.ub is not None:
                    constraints.append({'body': generate_standard_repn(-obj),
                                        'upper': None,
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
                                                      body.linear_vars))
            lower = cons['lower']
            if type(lhs >= lower) is bool:
                if lhs >= lower:
                    continue
                else:
                    # This would actually make a lot of sense in this case...
                    #projected_constraints.add(Constraint.Infeasible)
                    raise RuntimeError("Fourier-Motzkin found that model is "
                                       "infeasible!")
            else:
                projected_constraints.add(lhs >= lower)

    def _process_constraint(self, constraint):
        """Transforms a pyomo Constraint objective into a list of dictionaries
        representing only >= constraints. That is, if the constraint has both an
        ub and a lb, it is transformed into two constraints. Otherwise it is
        flipped if it is <=. Each dictionary contains the keys 'lower', 'upper'
        and 'body' where, after the process, 'upper' will be None, 'lower' will
        be a constant, and 'body' will be the standard repn of the body.
        (The constant will be moved to the RHS).
        """
        body = constraint.body
        std_repn = generate_standard_repn(body)
        # linear only!!
        if not std_repn.is_linear():
            raise RuntimeError("Found nonlinear constraint %s. The "
                               "Fourier-Motzkin Elimination transformation "
                               "can only be applied to linear models!"
                               % constraint.name)
        cons_dict = {'lower': constraint.lower,
                     'upper': constraint.upper,
                     'body': std_repn
                     }
        constraints_to_add = [cons_dict]
        if cons_dict['upper'] is not None:
            # if it has both bounds
            if cons_dict['lower'] is not None:
                # copy the constraint and flip
                leq_side = {'lower': -cons_dict['upper'],
                            'upper': None,
                            'body': generate_standard_repn(-1.0*body)}
                self._move_constant_and_add_map(leq_side)
                constraints_to_add.append(leq_side)
                cons_dict['upper'] = None

            # If it has only an upper bound, we just need to flip it
            else:
                # just flip the constraint
                cons_dict['lower'] = -cons_dict['upper']
                cons_dict['upper'] = None
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
            for var in cons['body'].linear_vars:
                if var in vars_to_eliminate:
                    vars_that_appear.append(var)

        # we actually begin the recursion here
        while vars_that_appear:
            # first var we will project out
            the_var = vars_that_appear.pop()

            # we are 'reorganizing' the constraints, we will map the coefficient
            # of the_var from that constraint and the rest of the expression and
            # sorting based on whether we have the_var <= other stuff or vice
            # versa.
            leq_list = []
            geq_list = []
            waiting_list = []

            while(constraints):
                cons = constraints.pop()
                leaving_var_coef = cons['map'].get(the_var)
                if leaving_var_coef is None or leaving_var_coef == 0:
                    waiting_list.append(cons)
                    continue

                # we know the constraints is a >= constraint, using that
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

            for leq in leq_list:
                for geq in geq_list:
                    constraints.append(self._add_linear_constraints(leq, geq))

            # add back in the constraints that didn't have the variable we were
            # projecting out
            constraints.extend(waiting_list)

        return constraints

    def _nonneg_scalar_multiply_linear_constraint(self, cons, scalar):
        """Multiplies all coefficients and the RHS of a >= constraint by scalar.
        There is no logic for flipping the equality, so this is just the 
        special case with a nonnegative scalar, which is all we need.
        """
        cons['body'].linear_coefs = [scalar*coef for coef in
                                     cons['body'].linear_coefs]
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
        ans = {'lower': None, 'upper': None, 'body': None, 'map': ComponentMap()}

        # This is not beautiful, but it needs to be both deterministic and
        # account for the fact that Vars aren't hashable.
        seen = ComponentSet()
        all_vars = []
        for v in cons1['body'].linear_vars:
            all_vars.append(v)
            seen.add(v)
        for v in cons2['body'].linear_vars:
            if v not in seen:
                all_vars.append(v)
        
        expr = 0
        for var in all_vars:
            cons1_coef = cons1['map'].get(var)
            cons2_coef = cons2['map'].get(var)
            if cons2_coef is not None and cons1_coef is not None:
                ans['map'][var] = new_coef = cons1_coef + cons2_coef
            elif cons1_coef is not None:
                ans['map'][var] = new_coef = cons1_coef
            elif cons2_coef is not None:
                ans['map'][var] = new_coef = cons2_coef
            expr += new_coef*var
        ans['body'] = generate_standard_repn(expr)

        # upper is None, so we just deal with the constants here.
        cons1_lower = cons1['lower']
        cons2_lower = cons2['lower']
        if cons1_lower is not None and cons2_lower is not None:
            ans['lower'] = cons1_lower + cons2_lower

        return ans
