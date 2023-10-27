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

from pyomo.gdp import GDP_Error
from pyomo.common.collections import ComponentSet
from pyomo.contrib.fbbt.expression_bounds_walker import ExpressionBoundsVisitor
import pyomo.contrib.fbbt.interval as interval
from pyomo.core import Suffix


def _convert_M_to_tuple(M, constraint, disjunct=None):
    if not isinstance(M, (tuple, list)):
        if M is None:
            M = (None, None)
        else:
            try:
                M = (-M, M)
            except:
                logger.error(
                    "Error converting scalar M-value %s "
                    "to (-M,M).  Is %s not a numeric type?" % (M, type(M))
                )
                raise
    if len(M) != 2:
        constraint_name = constraint.name
        if disjunct is not None:
            constraint_name += " relative to Disjunct %s" % disjunct.name
        raise GDP_Error(
            "Big-M %s for constraint %s is not of "
            "length two. "
            "Expected either a single value or "
            "tuple or list of length two specifying M values for "
            "the lower and upper sides of the constraint "
            "respectively." % (str(M), constraint.name)
        )

    return M


def _get_bigM_suffix_list(block, stopping_block=None):
    # Note that you can only specify suffixes on BlockData objects or
    # ScalarBlocks. Though it is possible at this point to stick them
    # on whatever components you want, we won't pick them up.
    suffix_list = []

    # go searching above block in the tree, stop when we hit stopping_block
    # (This is so that we can search on each Disjunct once, but get any
    # information between a constraint and its Disjunct while transforming
    # the constraint).
    while block is not None:
        bigm = block.component('BigM')
        if type(bigm) is Suffix:
            suffix_list.append(bigm)
        if block is stopping_block:
            break
        block = block.parent_block()

    return suffix_list


def _warn_for_unused_bigM_args(bigM, used_args, logger):
    # issue warnings about anything that was in the bigM args dict that we
    # didn't use
    if bigM is not None:
        unused_args = ComponentSet(bigM.keys()) - ComponentSet(used_args.keys())
        if len(unused_args) > 0:
            warning_msg = (
                "Unused arguments in the bigM map! "
                "These arguments were not used by the "
                "transformation:\n"
            )
            for component in unused_args:
                if isinstance(component, (tuple, list)) and len(component) == 2:
                    warning_msg += "\t(%s, %s)\n" % (
                        component[0].name,
                        component[1].name,
                    )
                elif hasattr(component, 'name'):
                    warning_msg += "\t%s\n" % component.name
                else:
                    warning_msg += "\t%s\n" % component
            logger.warning(warning_msg)


class _BigM_MixIn(object):
    def _get_bigM_arg_list(self, bigm_args, block):
        # Gather what we know about blocks from args exactly once. We'll still
        # check for constraints in the moment, but if that fails, we've
        # preprocessed the time-consuming part of traversing up the tree.
        arg_list = []
        if bigm_args is None:
            return arg_list
        while block is not None:
            if block in bigm_args:
                arg_list.append({block: bigm_args[block]})
            block = block.parent_block()
        return arg_list

    def _set_up_expr_bound_visitor(self):
        # we assume the default config arg for 'assume_fixed_vars_permanent,`
        # and we will change it during apply_to if we need to
        self._expr_bound_visitor = ExpressionBoundsVisitor(
            use_fixed_var_values_as_bounds=False
        )

    def _process_M_value(
        self,
        m,
        lower,
        upper,
        need_lower,
        need_upper,
        src,
        key,
        constraint,
        from_args=False,
    ):
        m = _convert_M_to_tuple(m, constraint)
        if need_lower and m[0] is not None:
            if from_args:
                self.used_args[key] = m
            lower = (m[0], src, key)
            need_lower = False
        if need_upper and m[1] is not None:
            if from_args:
                self.used_args[key] = m
            upper = (m[1], src, key)
            need_upper = False
        return lower, upper, need_lower, need_upper

    def _get_M_from_args(self, constraint, bigMargs, arg_list, lower, upper):
        # check args: we first look in the keys for constraint and
        # constraintdata. In the absence of those, we traverse up the blocks,
        # and as a last resort check for a value for None
        if bigMargs is None:
            return (lower, upper)

        # since we check for args first, we know lower[0] and upper[0] are both
        # None
        need_lower = constraint.lower is not None
        need_upper = constraint.upper is not None

        # check for the constraint itself and its container
        parent = constraint.parent_component()
        if constraint in bigMargs:
            m = bigMargs[constraint]
            (lower, upper, need_lower, need_upper) = self._process_M_value(
                m,
                lower,
                upper,
                need_lower,
                need_upper,
                bigMargs,
                constraint,
                constraint,
                from_args=True,
            )
            if not need_lower and not need_upper:
                return lower, upper
        elif parent in bigMargs:
            m = bigMargs[parent]
            (lower, upper, need_lower, need_upper) = self._process_M_value(
                m,
                lower,
                upper,
                need_lower,
                need_upper,
                bigMargs,
                parent,
                constraint,
                from_args=True,
            )
            if not need_lower and not need_upper:
                return lower, upper

        # use the precomputed traversal up the blocks
        for arg in arg_list:
            for block, val in arg.items():
                (lower, upper, need_lower, need_upper) = self._process_M_value(
                    val,
                    lower,
                    upper,
                    need_lower,
                    need_upper,
                    bigMargs,
                    block,
                    constraint,
                    from_args=True,
                )
                if not need_lower and not need_upper:
                    return lower, upper

        # last check for value for None!
        if None in bigMargs:
            m = bigMargs[None]
            (lower, upper, need_lower, need_upper) = self._process_M_value(
                m,
                lower,
                upper,
                need_lower,
                need_upper,
                bigMargs,
                None,
                constraint,
                from_args=True,
            )
            if not need_lower and not need_upper:
                return lower, upper

        return lower, upper

    def _estimate_M(self, expr, constraint):
        expr_lb, expr_ub = self._expr_bound_visitor.walk_expression(expr)
        if expr_lb == -interval.inf or expr_ub == interval.inf:
            raise GDP_Error(
                "Cannot estimate M for unbounded "
                "expressions.\n\t(found while processing "
                "constraint '%s'). Please specify a value of M "
                "or ensure all variables that appear in the "
                "constraint are bounded." % constraint.name
            )
        else:
            M = (expr_lb, expr_ub)
        return tuple(M)

    def _add_constraint_expressions(
        self, c, i, M, indicator_var, newConstraint, constraintMap
    ):
        # Since we are both combining components from multiple blocks and using
        # local names, we need to make sure that the first index for
        # transformedConstraints is guaranteed to be unique. We just grab the
        # current length of the list here since that will be monotonically
        # increasing and hence unique. We'll append it to the
        # slightly-more-human-readable constraint name for something familiar
        # but unique. (Note that we really could do this outside of the loop
        # over the constraint indices, but I don't think it matters a lot.)
        unique = len(newConstraint)
        name = c.local_name + "_%s" % unique

        if c.lower is not None:
            if M[0] is None:
                raise GDP_Error(
                    "Cannot relax disjunctive constraint '%s' "
                    "because M is not defined." % name
                )
            M_expr = M[0] * (1 - indicator_var)
            newConstraint.add((name, i, 'lb'), c.lower <= c.body - M_expr)
            constraintMap['transformedConstraints'][c] = [newConstraint[name, i, 'lb']]
            constraintMap['srcConstraints'][newConstraint[name, i, 'lb']] = c
        if c.upper is not None:
            if M[1] is None:
                raise GDP_Error(
                    "Cannot relax disjunctive constraint '%s' "
                    "because M is not defined." % name
                )
            M_expr = M[1] * (1 - indicator_var)
            newConstraint.add((name, i, 'ub'), c.body - M_expr <= c.upper)
            transformed = constraintMap['transformedConstraints'].get(c)
            if transformed is not None:
                constraintMap['transformedConstraints'][c].append(
                    newConstraint[name, i, 'ub']
                )
            else:
                constraintMap['transformedConstraints'][c] = [
                    newConstraint[name, i, 'ub']
                ]
            constraintMap['srcConstraints'][newConstraint[name, i, 'ub']] = c
