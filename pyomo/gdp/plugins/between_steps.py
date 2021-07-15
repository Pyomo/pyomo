#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""
Between Steps (P-Split) reformulation for GDPs from:

J. Kronqvist, R. Misener, and C. Tsay, "Between Steps: Intermediate 
Relaxations between big-M and Convex Hull Reformulations," 2021.
"""
from __future__ import division

from pyomo.common.config import (ConfigBlock, ConfigValue)
from pyomo.common.modeling import unique_component_name
from pyomo.core import ( Block, Constraint, Var, SortComponents, Transformation,
                         TransformationFactory, TraversalStrategy,
                         NonNegativeIntegers, value, ConcreteModel,
                         NonNegativeIntegers, Objective, ComponentMap)
from pyomo.common.collections import ComponentSet
from pyomo.repn import generate_standard_repn
from pyomo.core.expr import current as EXPR
from pyomo.opt import SolverFactory

from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.gdp.util import (preprocess_targets, is_child_of, target_list,
                            _to_dict, verify_successful_solve, NORMAL,
                            clone_without_expression_components )

import logging
logger = logging.getLogger('pyomo.gdp.between_steps')

from nose.tools import set_trace

NAME_BUFFER = {}

def _generate_additively_separable_repn(nonlinear_part):
    if nonlinear_part.__class__ is not EXPR.SumExpression:
        # This isn't separable, so we just have the one expression
        return {'nonlinear_vars': [(v for v in
                                    EXPR.identify_variables(
                                        nonlinear_part))],
                'nonlinear_exprs': [nonlinear_part]}

    # else, it was a SumExpression, and we will break it into the summands,
    # recording which variables are there.
    nonlinear_decomp = {'nonlinear_vars': [],
                        'nonlinear_exprs': []}
    for summand in nonlinear_part.args:
        nonlinear_decomp['nonlinear_exprs'].append(summand)
        nonlinear_decomp['nonlinear_vars'].append(
            (v for v in EXPR.identify_variables(summand)))

    return nonlinear_decomp

def _arbitrary_partition(disjunction):
    pass



@TransformationFactory.register('gdp.between_steps',
                                doc="Reformulates a convex disjunctive model "
                                "by splitting additively separable constraints"
                                "on P sets of variables, then taking hull "
                                "reformulation.")
class BetweenSteps_Transformation(Transformation):
    """
    TODO: This transformation does stuff!

    References
    ----------
        [1] J. Kronqvist, R. Misener, and C. Tsay, "Between Steps: Intermediate 
            Relaxations between big-M and Convex Hull Reformulations," 2021.
    """
    CONFIG = ConfigBlock("gdp.between_steps")
    CONFIG.declare('targets', ConfigValue(
        default=None,
        domain=target_list,
        description="target or list of targets that will be relaxed",
        doc="""

        This specifies the target or list of targets to relax as either a
        component or a list of components. If None (default), the entire model
        is transformed. Note that if the transformation is done out of place,
        the list of targets should be attached to the model before it is cloned,
        and the list will specify the targets on the cloned instance."""
    ))
    CONFIG.declare('variable_partitions', ConfigValue(
        default=None,
        domain=_to_dict,
        description="""Set of sets of variables which define valid partitions
        (i.e., the constraints are additively separable across these
        partitions). These can be specified globally (for all active 
        Disjunctions), or by Disjunction.""",
        doc="""
        Specified variable partitions, either globally or per Disjunction.

        Expects either a set of disjoint ComponentSets whose union is all the
        variables that appear in all Disjunctions or a mapping from each active
        Disjunction to a set of disjoint ComponentSets whose union is the set
        of variables that appear in that Disjunction. In either case, if any 
        constraints in the Disjunction are only partially additively separable,
        these sets must be a valid partition so that these constraints are
        additively separable with respect to this partition. To specify a
        default partition for Disjunctions that do not appear as keys in the
        map, map the partition to 'None.'

        Last, note that in the case of constraints containing partially 
        additively separable functions, it is required that the user specify 
        the variable parition(s).
        """
    ))
    CONFIG.declare('P', ConfigValue(
        default=None,
        domain=_to_dict,
        description="""Number of partitions of variables, if variable_paritions
        is not specifed. Can be specified separately for specific Disjunctions 
        if desired.""",
        doc="""
        Either a single value so that all Disjunctions will have variables
        partitioned into P sets, or a map of Disjunctions to a value of P
        for each active Disjunction. Mapping None to a value of P will specify
        the default value of P to use if the value for a given Disjunction
        is not explicitly specified.

        Note that if any constraints contain partially additively separable
        functions, the partitions for the Disjunctions with these Constraints
        must be specified in the variable_partitions argument.
        """
    ))
    CONFIG.declare('variable_partitioning_method', ConfigValue(
        default=_arbitrary_partition,
        domain=_to_dict,
        description="""Method to partition the variables. By default, the 
        partitioning will be done arbitrarily. Other options include: TODO""",
        doc="""
        A function which takes some stuff and return variable partitions. 

        Note that if any constraints contain partially additively separable
        functions, the partitions for the Disjunctions cannot be calculated
        automatically. Please specify the paritions for the Disjunctions with 
        these Constraints in the variable_partitions argument.
        """
    ))
    CONFIG.declare('assume_fixed_vars_permanent', ConfigValue(
        default=True,# TODO: John, I'm kind of tempted to make this the
                     # default... We have to do more work if it's not True. It's
                     # inconsistent with bigm and hull, but matches cutting
                     # planes kind of (because cutting planes gives up... Wait,
                     # cutting planes could also do what I'm doing here...)
                     # Anyway, thoughts?
        domain=bool,
        description="Boolean indicating whether or not to transform so that the "
        "the transformed model will still be valid when fixed Vars are unfixed.",
        doc="""
        If True, the transformation will create a correct model even if fixed
        variable are later unfixed. That is, bounds will be calculated based
        on fixed variables' bounds, not their values. However, if fixed 
        variables will never be unfixed, a possibly tigher model will result, 
        and fixed variables need not have bounds.
        """
    ))
    CONFIG.declare('verbose', ConfigValue(
        default=False,
        domain=bool,
        description="""Enable verbose output.""",
        doc="""
        Set to True for verbose output, False otherwise.
        """
    ))
    # ESJ: TODO: I'm not quite sure how best to do this... I kinda want a
    # callback, but I kinda want to not mess with solver stuff... I guess it
    # could optionally take a SolverFactory as an argument, would be the
    # cleanest way to do it?
    CONFIG.declare('compute_bounds_method', ConfigValue(
        default=None,
        description="""Function which takes an expression and returns both a 
        lower and upper bound for it.""",
        doc="""Callback for computing bounds on expressions, in order to bound
        the auxilary variables created by the transformation. Some 
        pre-implemented options include
            * compute_optimal_bounds (the default), and
            * pyomo.contrib.fbbt.fbbt.compute_bounds_on_expr (a faster but 
              weaker alternative),
        or you can write your own callback which accepts an Expression object
        and returns a tuple (LB, UB) where either element can be None if no
        valid bound could be found.
        """
    ))
    CONFIG.declare('subproblem_solver', ConfigValue(
        default=SolverFactory('ipopt'),
        description="""SolverFactory object to use for computing expression 
        bounds, if doing so optimally.""",
        doc="""
        If compute_bounds_method is 'compute_optimal_bounds,' this SolverFactory
        will be used to solve the subproblems.
        """
    ))
    # TODO: Maybe a way to specify your own bounds??

    def __init__(self):
        super(BetweenSteps_Transformation, self).__init__()

    def _apply_to(self, instance, **kwds):
        if not instance.ctype in (Block, Disjunct):
            raise GDP_Error("Transformation called on %s of type %s. 'instance' "
                            "must be a ConcreteModel, Block, or Disjunct (in "
                            "the case of nested disjunctions)." %
                            (instance.name, instance.ctype))

        original_log_level = logger.level
        log_level = logger.getEffectiveLevel()
        try:
            assert not NAME_BUFFER
            self._config = self.CONFIG(kwds.pop('options', {}))
            self._config.set_value(kwds)

            if self._config.verbose and log_level > logging.INFO:
                logger.setLevel(logging.INFO)
                self.verbose = True
            elif log_level <= logging.INFO:
                self.verbose = True
            else:
                self.verbose = False

            if self._config.compute_bounds_method is None:
                self._config.compute_bounds_method = self.compute_optimal_bounds

            if not self._config.assume_fixed_vars_permanent:
                # TODO: This actually a place where I want everything that
                # appears in an accessible expression, so component_data_objects
                # is wrong...
                fixed_vars = ComponentMap()
                for v in instance.component_data_objects(
                        Var, active=True, descend_into=(Block, Disjunct)):
                    if v.fixed:
                        fixed_vars[v] = v.value
                        v.fixed = False

            self._apply_to_impl(instance)

        finally:
            # restore fixed variables
            if not self._config.assume_fixed_vars_permanent:
                for v, val in fixed_vars.items():
                    v.fix(value)

            del self._config
            del self.verbose
            # clear the global name buffer
            NAME_BUFFER.clear()
            # restore logging level
            logger.setLevel(original_log_level)

    def _apply_to_impl(self, instance):
        # create a transformation block on which we will create the reformulated
        # GDP...
        transformation_block = Block()
        instance.add_component(unique_component_name( 
            instance, '_pyomo_gdp_between_steps_reformulation'),
                               transformation_block)
        self.variable_partitions = self._config.variable_partitions
        self.partitioning_method = self._config.variable_partitioning_method

        # we can support targets as usual.
        targets = self._config.targets
        if targets is None:
            targets = ( instance, )
        else:
            # we need to preprocess targets to make sure that if there are any
            # disjunctions in targets that their disjuncts appear before them in
            # the list.
            targets = preprocess_targets(targets)
        knownBlocks = {}
        for t in targets:
            # check that t is in fact a child of instance
            if not is_child_of(parent=instance, child=t,
                               knownBlocks=knownBlocks):
                raise GDP_Error(
                    "Target '%s' is not a component on instance '%s'!"
                    % (t.name, instance.name))
            elif t.ctype is Disjunction:
                if t.is_indexed():
                    self._transform_disjunction(t, transformation_block)
                else:
                    self._transform_disjunctionData(t, t.index(),
                                                    transformation_block)
            elif t.ctype in (Block, Disjunct):
                if t.is_indexed():
                    self._transform_block(t, transformation_block)
                else:
                    self._transform_blockData(t, transformation_block)
            else:
                raise GDP_Error(
                    "Target '%s' was not a Block, Disjunct, or Disjunction. "
                    "It was of type %s and can't be transformed."
                    % (t.name, type(t)) )

        # call hull on the whole thing and return.
        TransformationFactory('gdp.hull').apply_to(instance,
                                                   targets=transformation_block)

        # TODO: What do we want to do about mappings? We could go through now
        # and follow the chain, so to speak. I think that might be the right
        # answer. I've never really found the perfect solution. But we will need
        # some of them when we try using this inside of cuttingplanes.

    def _transform_block(self, obj, transBlock):
        for i in sorted(obj.keys()):
            self._transform_blockData(obj[i], transBlock)

    def _transform_blockData(self, obj, transBlock):
        # Transform every (active) disjunction in the block
        for disjunction in obj.component_data_objects(
                Disjunction,
                active=True,
                sort=SortComponents.deterministic,
                descend_into=(Block, Disjunct),
                descent_order=TraversalStrategy.PostfixDFS):
            self._transform_disjunction(disjunction, transBlock)
    
    def _transform_disjunction(self, obj, transBlock):
        if not obj.active:
            return
            
        # relax each of the disjunctionDatas
        for i in sorted(obj.keys()):
            self._transform_disjunctionData(obj[i], i, transBlock)

        obj.deactivate()

    def _transform_disjunctionData(self, obj, idx, transBlock):
        if not obj.active:
            return

        variable_partitions = self.variable_partitions
        partition_method = self.partitioning_method

        # was it specified for the disjunct?
        partition = variable_partitions.get(obj)
        if partition is None:
            # was there a default
            partition = variable_partitions.get(None)
            if partition is None:
                # If not, see what method to use to calculate one
                method = partition_method.get(obj)
                # was there a default method?
                if method is None:
                    method = partition_method.get(None)
                # if all else fails, set it to our default
                method = method if method is not None else _arbitrary_partition
                # it's this method's job to scream if it can't handle what's
                # here, we can only assume it worked for now, since it's a
                # callback.
                partition = method(obj)
        # these have to be ComponentSets
        partition = [ComponentSet(var_list) for var_list in partition]
                
        transformed_disjuncts = []
        for disjunct in obj.disjuncts:
            transformed_disjuncts.append(self._transform_disjunct(disjunct,
                                                                  partition,
                                                                  transBlock))

        # make a new disjunction with the transformed guys
        transBlock.add_component(unique_component_name(
            transBlock,
            obj.getname(fully_qualified=True, name_buffer=NAME_BUFFER)),
                                 Disjunction(expr=[disj for disj in
                                                   transformed_disjuncts]))

        obj.deactivate()

    def _get_leq_constraints(self, cons):
        constraints = []
        if cons.lower is not None:
            constraints.append((-cons.body, -cons.lower))
        if cons.upper is not None:
            constraints.append((cons.body, cons.upper))
        return constraints

    def _transform_disjunct(self, obj, partition, transBlock):
        # deactivated -> either we've already transformed or user deactivated
        if not obj.active:
            if obj.indicator_var.is_fixed():
                if not value(obj.indicator_var):
                    # The user cleanly deactivated the disjunct: there
                    # is nothing for us to do here.
                    return
                else:
                    raise GDP_Error(
                        "The disjunct '%s' is deactivated, but the "
                        "indicator_var is fixed to %s. This makes no sense."
                        % ( obj.name, value(obj.indicator_var) ))

        transformed_disjunct = Disjunct()
        transBlock.add_component(unique_component_name(
            transBlock,
            obj.getname(fully_qualified=True, name_buffer=NAME_BUFFER)),
                                 transformed_disjunct)
        for cons in obj.component_data_objects(Constraint, active=True,
                                               sort=SortComponents.deterministic,
                                               descend_into=Block):
            cons_name = cons.getname(fully_qualified=True,
                                     name_buffer=NAME_BUFFER)

            # create place on transformed Disjunct for the new constraint and
            # for the auxilary variables
            transformed_constraint = Constraint(NonNegativeIntegers)
            transformed_disjunct.add_component(unique_component_name(
                transformed_disjunct, cons_name), transformed_constraint)
            aux_vars = Var(NonNegativeIntegers, dense=False)
            transformed_disjunct.add_component(unique_component_name(
                transformed_disjunct, cons_name + "_aux_vars"), aux_vars)

            # create a place on the transBlock for the split constraints
            split_constraints = Constraint(NonNegativeIntegers)
            transBlock.add_component(unique_component_name(
                transBlock, cons_name + "_split_constraints"), split_constraints)

            # this is a list which might have two constraints in it if we had
            # both a lower and upper value.
            leq_constraints = self._get_leq_constraints(cons)
            for (body, rhs) in leq_constraints:
                repn = generate_standard_repn(body)
                nonlinear_repn = None
                if repn.nonlinear_expr is not None:
                    nonlinear_repn = _generate_additively_separable_repn(
                        repn.nonlinear_expr)
                split_exprs = []
                split_aux_vars = []
                for var_list in partition:
                    # we are going to recreate the piece of the expression
                    # involving the vars in var_list
                    split_exprs.append(0)
                    expr = split_exprs[-1]
                    for i, v in enumerate(repn.linear_vars):
                        if v in var_list:
                            expr += repn.linear_coefs[i]*v
                    for i, (v1, v2) in enumerate(repn.quadratic_vars):
                        if v1 in var_list:
                            if v2 not in var_list:
                                raise GDP_Error("Variables %s and %s are "
                                                "multiplied in Constraint %s,"
                                                "but they are in different "
                                                "partitions! Please ensure that "
                                                "all the constraints in the "
                                                "disjunction are "
                                                "additively separable with "
                                                "respect to the specified "
                                                "partition." % (v1.name, v2.name,
                                                                cons.name))
                            expr += repn.quadratic_coefs[i]*v1*v2
                    if nonlinear_repn is not None:
                        for i, v_list in enumerate(
                                nonlinear_repn['nonlinear_vars']):
                            # check if v_list is a subset of var_list. If it is
                            # not and there is no intersection, we move on. If
                            # it is not and there is an intersection, we raise
                            # an error: It's not a valid partition. If it is,
                            # then we add this piece of the expression.
                            expr_var_set = ComponentSet(v_list)
                            # subset?
                            if all(v in var_list for v in expr_var_set):
                                expr += nonlinear_repn['nonlinear_exprs'][i]
                            # intersection?
                            elif len(expr_var_set & var_list) != 0:
                                raise GDP_Error("Variables which appear in the "
                                                "expression %s are in different "
                                                "partitions, but this "
                                                "expression doesn't appear "
                                                "additively separable. Please "
                                                "expand it if it is additively "
                                                "separable or, more likely, "
                                                "ensure that all the "
                                                "constraints in the disjunction "
                                                "are additively separable with "
                                                "respect to the specified "
                                                "partition." % 
                                                nonlinear_repn[
                                                    'nonlinear_exprs'][i])
                                                
                    # now we have the piece of the expression, we need bounds on
                    # it. We first check if they were specified via args. If not
                    # we use specified method to calculate them. (Probably fbbt
                    # or actually optimizing the function over the box defined
                    # by the variable bounds)

                    # TODO: need to implement a bounds arg and try to retrieve
                    # them here. Only do the below if that fails.

                    expr_lb, expr_ub = self._config.compute_bounds_method(expr)
                    if expr_lb is None or expr_ub is None:
                        raise GDP_Error("Expression %s from constraint %s "
                                        "is unbounded! Please specify a bound "
                                        "in the TODO-some-bounds-arg arg "
                                        "or ensure all variables that appear "
                                        "in the constraint are bounded." % 
                                        (expr, cons.name))
                    aux_var = aux_vars[len(aux_vars)]
                    aux_var.setlb(expr_lb)
                    aux_var.setub(expr_ub)                    
                    split_aux_vars.append(aux_var)
                    split_constraints[
                        len(split_constraints)] = expr <= aux_var
                transformed_constraint[
                    len(transformed_constraint)] = sum(v for v in
                                                       split_aux_vars) <= \
                    rhs - repn.constant
                                         
            obj.deactivate()
            return transformed_disjunct

    # ESJ: TODO: This is a terrible name.. Like, wrt WHAT, Emma?
    def compute_optimal_bounds(self, expr):
        m = ConcreteModel()
        m.x = Var(NonNegativeIntegers, dense=False)
        substitute_map = {}
        for i,v in enumerate(EXPR.identify_variables(expr)):
            m.x[i].setlb(v.lb)
            m.x[i].setub(v.ub)
            substitute_map[id(v)] = m.x[i]
        m.obj = Objective(expr=clone_without_expression_components(
            expr,
            substitute=substitute_map))
        opt = self._config.subproblem_solver
        results = opt.solve(m)
        if verify_successful_solve(results) is not NORMAL:
            logger.warning("Problem to find lower bound for expression %s"
                           "did not solve normally.\n\n%s" % (expr, results))
            LB = None
        else:
            # TODO: we probably need tolerance or rounding or something
            # here... Ideally we'd get the LB from the solver, but ipopt for
            # example doesn't even gives us one so...
            LB = value(m.obj.expr)
        m.obj.sense = -1
        results = opt.solve(m)
        if verify_successful_solve(results) is not NORMAL:
            logger.warning("Problem to find upper bound for expression %s"
                           "did not solve normally.\n\n%s" % (expr, results))
            UB = None
        else:
            UB = value(m.obj.expr)
        
        return (LB, UB)
