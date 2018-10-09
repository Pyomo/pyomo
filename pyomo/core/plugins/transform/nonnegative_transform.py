#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import copy

from pyomo.core.expr import current as EXPR

from pyomo.core import *
from pyomo.core.base.expression import _ExpressionData
from pyomo.core.base.var import SimpleVar, _VarData
from pyomo.core.base.misc import create_name
from pyomo.core.plugins.transform.util import partial
from pyomo.core.plugins.transform.hierarchy import IsomorphicTransformation
from pyomo.core.plugins.transform.util import collectAbstractComponents


class VarmapVisitor(EXPR.ExpressionReplacementVisitor):

    def __init__(self, varmap):
        super(VarmapVisitor, self).__init__()
        self.varmap = varmap

    def visiting_potential_leaf(self, node):
        if node.__class__ in nonpyomo_leaf_types:
            return True, node
        #
        # Clone leaf nodes in the expression tree
        #
        if node.is_variable_type():
            if node.local_name in self.varmap:
                return True, self.varmap[node.local_name]
            else: 
                return True, node

        if isinstance(node, EXPR.LinearExpression):
            with EXPR.nonlinear_expression() as expr:
                for c, v in zip(node.linear_coefs, node.linear_vars):
                    if hasattr(v, 'local_name'):
                        expr += c * self.varmap.get(v.local_name)
                    else:
                        expr += c * v
            return True, expr

        return False, None


def _walk_expr(expr, varMap):
    """
    Walks an expression tree, making the replacements defined in varMap
    """
    visitor = VarmapVisitor(varMap)
    return visitor.dfs_postorder_stack(expr)


@TransformationFactory.register("core.nonnegative_vars", doc="Create an equivalent model in which all variables lie in the nonnegative orthant.")
class NonNegativeTransformation(IsomorphicTransformation):
    """
    Creates a new, equivalent model by forcing all variables to lie in
    the nonnegative orthant by introducing auxiliary variables.
    """

    def __init__(self, **kwds):
        kwds["name"] = kwds.pop("name", "vars")
        super(NonNegativeTransformation, self).__init__(**kwds)

        self.realSets = (Reals, PositiveReals, NonNegativeReals, NegativeReals,
                         NonPositiveReals, PercentFraction, RealSet)
        # Intentionally leave out Binary, Boolean, BinarySet, and BooleanSet;
        # we check for those explicitly
        self.discreteSets = (IntegerSet, Integers, PositiveIntegers,
                             NonPositiveIntegers, NegativeIntegers,
                             NonNegativeIntegers)


    def _create_using(self, model, **kwds):
        """
        Force all variables to lie in the nonnegative orthant.

        Required arguments:
            model       The model to transform.

        Optional keyword arguments:
          pos_suffix    The suffix applied to the 'positive' component of
                            converted variables. Default is '_plus'.
          neg_suffix    The suffix applied to the 'positive' component of
                            converted variables. Default is '_minus'.
        """
        #
        # Optional naming schemes
        #
        pos_suffix = kwds.pop("pos_suffix", "_plus")
        neg_suffix = kwds.pop("neg_suffix", "_minus")
        #
        # We first perform an abstract problem transformation. Then, if model
        # data is available, we instantiate the new model. If not, we construct
        # a mapping that can later be used to populate the new model.
        #
        nonneg = model.clone()
        components = collectAbstractComponents(nonneg)

        # Map from variable base names to a {index, rule} map
        constraint_rules = {}

        # Map from variable base names to a rule defining the domains for that
        # variable
        domain_rules = {}

        # Map from variable base names to its set of indices
        var_indices = {}

        # Map from fully qualified variable names to replacement expressions.
        # For now, it is actually a map from a variable name to a closure that
        # must later be evaulated with a model containing the replacement
        # variables.
        var_map = {}

        #
        # Get the constraints that enforce the bounds and domains of each
        # variable
        #
        for var_name in components["Var"]:
            var = nonneg.__getattribute__(var_name)

            # Individual bounds and domains
            orig_bounds = {}
            orig_domain = {}

            # New indices
            indices = set()

            # Map from constraint names to a constraint rule.
            constraints = {}

            # Map from variable indices to a domain
            domains = {}

            for ndx in var:
                # Fully qualified variable name
                vname = create_name(str(var_name), ndx)

                # We convert each index to a string to avoid difficult issues
                # regarding appending a suffix to tuples.
                #
                # If the index is None, this casts the index to a string,
                # which doesn't match up with how Pyomo treats None indices
                # internally. Replace with "" to be consistent.
                if ndx is None:
                    v_ndx = ""
                else:
                    v_ndx = str(ndx)

                # Get the variable bounds
                lb = var[ndx].lb
                ub = var[ndx].ub
                if lb is not None:
                    lb = value(lb)
                if ub is not None:
                    ub = value(ub)
                orig_bounds[ndx] = (lb, ub)

                # Get the variable domain
                if var[ndx].domain is not None:
                    orig_domain[ndx] = var[ndx].domain
                else:
                    orig_domain[ndx] = var.domain

                # Determine the replacement expression. Either a new single
                # variable with the same attributes, or a sum of two new
                # variables.
                #
                # If both the bounds and domain allow for negative values,
                # replace the variable with the sum of nonnegative ones.

                bounds_neg = (orig_bounds[ndx] == (None, None) or
                              orig_bounds[ndx][0] is None or
                              orig_bounds[ndx][0] < 0)
                domain_neg = (orig_domain[ndx] is None or
                              orig_domain[ndx].bounds()[0] is None or
                              orig_domain[ndx].bounds()[0] < 0)
                if bounds_neg and domain_neg:
                    # Make two new variables.
                    posVarSuffix = "%s%s" % (v_ndx, pos_suffix)
                    negVarSuffix = "%s%s" % (v_ndx, neg_suffix)

                    new_indices = (posVarSuffix, negVarSuffix)

                    # Replace the original variable with a sum expression
                    expr_dict = {posVarSuffix: 1, negVarSuffix: -1}
                else:
                    # Add the new index. Lie if is 'None', since Pyomo treats
                    # 'None' specially as a key.
                    #
                    # More lies: don't let a blank index exist. Replace it with
                    # '_'. I don't actually have a justification for this other
                    # than that allowing "" as a key will eventually almost
                    # certainly lead to a strange bug.
                    if v_ndx is None:
                        t_ndx = "None"
                    elif v_ndx == "":
                        t_ndx = "_"
                    else:
                        t_ndx = v_ndx

                    new_indices = (t_ndx,)

                    # Replace the original variable with a sum expression
                    expr_dict = {t_ndx: 1}

                # Add the new indices
                for x in new_indices:
                    indices.add(x)

                # Replace the original variable with an expression
                var_map[vname] = partial(self.sumRule,
                                         var_name,
                                         expr_dict)

                # Enforce bounds as constraints
                if orig_bounds[ndx] != (None, None):
                    cname = "%s_%s" % (vname, "bounds")
                    tmp = orig_bounds[ndx]
                    constraints[cname] = partial(
                        self.boundsConstraintRule,
                        tmp[0],
                        tmp[1],
                        var_name,
                        expr_dict)

                # Enforce the bounds of the domain as constraints
                if orig_domain[ndx] != None:
                    cname = "%s_%s" % (vname, "domain_bounds")
                    tmp = orig_domain[ndx].bounds()
                    constraints[cname] = partial(
                        self.boundsConstraintRule,
                        tmp[0],
                        tmp[1],
                        var_name,
                        expr_dict)

                # Domain will either be NonNegativeReals, NonNegativeIntegers,
                # or Binary. We consider Binary because some solvers may
                # optimize over binary variables.
                if isinstance(orig_domain[ndx], RealSet):
                    for x in new_indices:
                        domains[x] = NonNegativeReals
                elif isinstance(orig_domain[ndx], IntegerSet):
                    for x in new_indices:
                        domains[x] = NonNegativeIntegers
                elif isinstance(orig_domain[ndx], BooleanSet):
                    for x in new_indices:
                        domains[x] = Binary
                else:
                    print ("Warning: domain '%s' not recognized, " + \
                           "defaulting to 'Reals'") % (str(var.domain))
                    for x in new_indices:
                        domains[x] = Reals

            constraint_rules[var_name] = constraints
            domain_rules[var_name] = partial(self.exprMapRule, domains)
            var_indices[var_name] = indices

        # Remove all existing variables.
        toRemove = []
        for (attr_name, attr) in nonneg.__dict__.items():
            if isinstance(attr, Var):
                toRemove.append(attr_name)
        for attr_name in toRemove:
            nonneg.__delattr__(attr_name)

        # Add the sets defining the variables, then the variables
        for (k, v) in var_indices.items():
            sname = "%s_indices" % k
            nonneg.__setattr__(sname, Set(initialize=v))
            nonneg.__setattr__(k, Var(nonneg.__getattribute__(sname),
                                      domain = domain_rules[k],
                                      bounds = (0, None)))

        # Construct the model to get the variables and their indices
        # recognized in the model
        ##nonneg = nonneg.create()

        # Safe to evaluate the modifiedVars mapping
        for var in var_map:
            var_map[var] = var_map[var](nonneg)

        # Map from constraint base names to maps from indices to expressions
        constraintExprs = {}

        #
        # Convert all modified variables in all constraints in the original
        # problem
        #
        for conName in components["Constraint"]:
            con = nonneg.__getattribute__(conName)

            # Map from constraint indices to a corrected expression
            exprMap = {}

            for (ndx, cdata) in con._data.items():
                lower = _walk_expr(cdata.lower, var_map)
                body  = _walk_expr(cdata.body,  var_map)
                upper = _walk_expr(cdata.upper, var_map)

                # Lie if ndx is None. Pyomo treats 'None' indices specially.
                if ndx is None:
                    ndx = "None"

                # Cast indices to strings, otherwise tuples ruin everything
                exprMap[str(ndx)] = (lower, body, upper)

            # Add to list of expression maps
            constraintExprs[conName] = exprMap

        # Map from constraint base names to maps from indices to expressions
        objectiveExprs = {}

        #
        # Convert all modified variables in all objectives in the original
        # problem
        #
        for objName in components["Objective"]:
            obj = nonneg.__getattribute__(objName)

            # Map from objective indices to a corrected expression
            exprMap = {}

            for (ndx, odata) in obj._data.items():
                exprMap[ndx] = _walk_expr(odata.expr, var_map)

            # Add to list of expression maps
            objectiveExprs[objName] = exprMap


        # Make the modified original constraints
        for (conName, ruleMap) in constraintExprs.items():
            # Make the set of indices
            sname = conName + "_indices"
            _set = Set(initialize=ruleMap.keys())
            nonneg.__setattr__(sname, _set)
            _set.construct()

            # Define the constraint
            _con = Constraint( nonneg.__getattribute__(sname),
                               rule=partial(self.exprMapRule, ruleMap) )
            nonneg.__setattr__(conName, _con)
            _con.construct()

        # Make the bounds constraints
        for (varName, ruleMap) in constraint_rules.items():
            conName = varName + "_constraints"
            # Make the set of indices
            sname = conName + "_indices"
            _set = Set(initialize=ruleMap.keys())
            nonneg.__setattr__(sname, _set)
            _set.construct()

            # Define the constraint
            _con = Constraint(nonneg.__getattribute__(sname),
                              rule=partial(self.delayedExprMapRule, ruleMap))
            nonneg.__setattr__(conName, _con)
            _con.construct()

        # Make the objectives
        for (objName, ruleMap) in objectiveExprs.items():
            # Make the set of indices
            sname = objName + "_indices"
            _set = Set(initialize=ruleMap.keys())
            nonneg.__setattr__(sname, _set)
            _set.construct()

            # Define the constraint
            _obj = Objective(nonneg.__getattribute__(sname),
                             rule=partial(self.exprMapRule, ruleMap))
            nonneg.__setattr__(objName, _obj)
            _obj.construct()

        return nonneg

    @staticmethod
    def boundsConstraintRule(lb, ub, attr, vars, model):
        """
        Produces 'lb < x^+ - x^- < ub' style constraints. Designed to
        be made a closer through functools.partial, across lb, ub, attr,
        and vars. vars is a {varname: coefficient} dictionary. attr is the
        base variable name; that is, X[1] would be referenced by

          model.__getattribute__('X')[1]

        and so attr='X', and 1 is a key of vars.

        """
        return (lb,
                sum(c * model.__getattribute__(attr)[v] \
                    for (v,c) in vars.items()),
                ub)

    @staticmethod
    def noConstraint(*args):
        return None

    @staticmethod
    def sumRule(attr, vars, model):
        """
        Returns a sum expression.
        """
        return sum(c*model.__getattribute__(attr)[v] for (v, c) in vars.items())

    @staticmethod
    def exprMapRule(ruleMap, model, ndx=None):
        """ Rule intended to return expressions from a lookup table """
        return ruleMap[ndx]

    @staticmethod
    def delayedExprMapRule(ruleMap, model, ndx=None):
        """
        Rule intended to return expressions from a lookup table. Each entry
        in the lookup table is a functor that needs to be evaluated before
        returning.
        """
        return ruleMap[ndx](model)

