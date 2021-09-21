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
                         NonNegativeIntegers, value, ConcreteModel, Objective,
                         ComponentMap, BooleanVar, LogicalConstraint, Connector,
                         Expression, Suffix, Param, Set, SetOf, RangeSet,
                         Reference)
from pyomo.core.base.external import ExternalFunction
from pyomo.network import Port
from pyomo.common.collections import ComponentSet
from pyomo.repn import generate_standard_repn
from pyomo.core.expr import current as EXPR
from pyomo.opt import SolverFactory
from pyomo.util.vars_from_expressions import get_vars_from_constraints

from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.gdp.util import (preprocess_targets, is_child_of, target_list,
                            _to_dict, verify_successful_solve, NORMAL,
                            clone_without_expression_components,
                            _warn_for_active_disjunct,
                            _warn_for_active_logical_constraint)
from pyomo.contrib.fbbt.fbbt import compute_bounds_on_expr
from weakref import ref as weakref_ref

from math import floor

import logging
logger = logging.getLogger('pyomo.gdp.partition_disjuncts')

NAME_BUFFER = {}

def _generate_additively_separable_repn(nonlinear_part):
    if nonlinear_part.__class__ is not EXPR.SumExpression:
        # This isn't separable, so we just have the one expression
        return {'nonlinear_vars': [tuple(v for v in EXPR.identify_variables(
            nonlinear_part))], 'nonlinear_exprs': [nonlinear_part]}

    # else, it was a SumExpression, and we will break it into the summands,
    # recording which variables are there.
    nonlinear_decomp = {'nonlinear_vars': [],
                        'nonlinear_exprs': []}
    for summand in nonlinear_part.args:
        nonlinear_decomp['nonlinear_exprs'].append(summand)
        nonlinear_decomp['nonlinear_vars'].append(
            tuple(v for v in EXPR.identify_variables(summand)))

    return nonlinear_decomp

def arbitrary_partition(disjunction, P):
    # collect variables
    v_set = ComponentSet()
    for disj in disjunction.disjuncts:
        v_set.update(get_vars_from_constraints(disj, descend_into=Block,
                                               active=True))
    # assign them to partitions
    partitions = []
    V = len(v_set)
    whole = floor(V/P)
    for partition in range(V % P):
        partitions.append(ComponentSet())
        # add whole + 1 vars
        for i in range(whole + 1):
            partitions[partition].add(v_set.pop())
    # for the rest, add whole vars
    for partition in range(V % P, P):
        partitions.append(ComponentSet())
        for i in range(whole):
            partitions[partition].add(v_set.pop())

    return partitions

def compute_optimal_bounds(expr, instance, global_constraints, opt):
    # computes bounds on expr by minimizing and maximizing expr over the
    # variable bounds and the global constraints. Note that even if expr is
    # convex and the global constraints are a convex set, the max problem is
    # nonconvex!
    if opt is None:
        raise GDP_Error("No solver was specified to optimize the "
                        "subproblems for computing expression bounds!"
                        "Please specify a configured solver in the "
                        "'compute_bounds_solver' argument if using "
                        "'compute_optimal_bounds.'")

    # add temporary objective and calculate bounds
    obj = Objective(expr=expr)
    global_constraints.add_component(unique_component_name(instance, "tmp_obj"),
                                     obj)
    results = opt.solve(global_constraints)
    if verify_successful_solve(results) is not NORMAL:
        logger.warning("Problem to find lower bound for expression %s"
                       "did not solve normally.\n\n%s" % (expr, results))
        LB = None
    else:
        # This has some risks, if you're using a solver the gives a lower bound,
        # getting that would be better. But this is why this is a callback.
        LB = value(obj.expr)
    obj.sense = -1
    results = opt.solve(global_constraints)
    if verify_successful_solve(results) is not NORMAL:
        logger.warning("Problem to find upper bound for expression %s"
                       "did not solve normally.\n\n%s" % (expr, results))
        UB = None
    else:
        UB = value(obj.expr)

    # clean up
    global_constraints.del_component(obj)
    del obj

    return (LB, UB)

def compute_fbbt_bounds(expr, instance, transBlock, opt):
    return compute_bounds_on_expr(expr)

@TransformationFactory.register('gdp.partition_disjuncts',
                                doc="Reformulates a convex disjunctive model "
                                "into a new GDP by splitting additively "
                                "separable constraintson P sets of variables")
class PartitionDisjuncts_Transformation(Transformation):
    """
    TODO: This transformation does stuff!

    References
    ----------
        [1] J. Kronqvist, R. Misener, and C. Tsay, "Between Steps: Intermediate 
            Relaxations between big-M and Convex Hull Reformulations," 2021.
    """
    CONFIG = ConfigBlock("gdp.partition_disjuncts")
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
        default=arbitrary_partition,
        domain=_to_dict,
        description="""Method to partition the variables. By default, the 
        partitioning will be done arbitrarily.""",
        doc="""
        A function which takes a Disjunction object and a number P and return 
        a valid partitioning of the variables that appear in the disjunction 
        into P partitions. 

        Note that you must give a value for 'P' if you are using this method
        to calculate partitions.

        Note that if any constraints contain partially additively separable
        functions, the partitions for the Disjunctions cannot be calculated
        automatically. Please specify the paritions for the Disjunctions with 
        these Constraints in the variable_partitions argument.
        """
    ))
    CONFIG.declare('assume_fixed_vars_permanent', ConfigValue(
        default=False,
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
    CONFIG.declare('compute_bounds_method', ConfigValue(
        default=compute_fbbt_bounds,
        description="""Function which takes an expression, the instance, 
        the partition_disjuncts transformation block, and a configured solver 
        and returns both a lower and upper bound for the expression.""",
        doc="""Callback for computing bounds on expressions, in order to bound
        the auxilary variables created by the transformation. Some 
        pre-implemented options include
            * compute_optimal_bounds, and
            * compute_fbbt_bounds (the default),
        or you can write your own callback which accepts an Expression object, 
        the original instance, a model containing the variables and global 
        constraints of the original isntance, and a configured solver and 
        returns a tuple (LB, UB) where either element can be None if no valid 
        bound could be found.
        """
    ))
    CONFIG.declare('compute_bounds_solver', ConfigValue(
        default=None,
        description="""Solver object to pass to compute_bounds_method. 
        This is required if you are using 'compute_optimal_bounds'.""",
        doc="""
        Configured solver object for use in the compute_bounds_method.

        In particular, if compute_bounds_method is 'compute_optimal_bounds', 
        this will be used to solve the subproblems, so needs to handle non-convex
        problems if any Disjunctions contain nonlinear constraints.
        """
    ))
    def __init__(self):
        super(PartitionDisjuncts_Transformation, self).__init__()
        self.handlers = {
            Constraint:  self._transform_constraint,
            Var:         self._add_var_reference,
            BooleanVar:  False,
            Connector:   False,
            Expression:  False,
            Suffix:      False,
            Param:       False,
            Set:         False,
            SetOf:       False,
            RangeSet:    False,
            Disjunction: False, # this actually isn't possible in this
                                # transformation because we transform them when
                                # we find them
            Disjunct:    self._warn_for_active_disjunct,
            Block:       False,
            LogicalConstraint: self._warn_for_active_logical_constraint,
            ExternalFunction: False,
            Port:        False, # not Arcs, because those are deactivated after
                                # the network.expand_arcs transformation
        }

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

            if not self._config.assume_fixed_vars_permanent:
                fixed_vars = ComponentMap()
                for v in get_vars_from_constraints(instance, include_fixed=True,
                                                   active=True,
                                                   descend_into=(Block,
                                                                 Disjunct)):
                    if v.fixed:
                        fixed_vars[v] = value(v)
                        v.fixed = False

                fixed_indicator_vars = ComponentMap()
                for disjunct in instance.component_data_objects(
                        Disjunct,
                        descend_into=(Block,
                                      Disjunct),
                        active=True):
                    ind_var = disjunct.indicator_var
                    if ind_var.fixed:
                        fixed_indicator_vars[disjunct] = value(ind_var)
                        ind_var.fixed = False

            self._apply_to_impl(instance)

        finally:
            # restore fixed variables
            if not self._config.assume_fixed_vars_permanent:
                for v, val in fixed_vars.items():
                    v.fix(val)
                for disjunct, val in fixed_indicator_vars.items():
                    disjunct.transformation_block().indicator_var.fix(val)

            del self._config
            # clear the global name buffer
            NAME_BUFFER.clear()
            # restore logging level
            logger.setLevel(original_log_level)

    def _apply_to_impl(self, instance):
        self.variable_partitions = self._config.variable_partitions if \
                                   self._config.variable_partitions is not None \
                                   else {}
        self.partitioning_method = self._config.variable_partitioning_method

        # create a model to store the global constraints on that we will pass to
        # the compute_bounds_method, for if it wants them. We're making it a
        # separate model because we don't need it again and because I was
        # hitting #2130 by making it a block and attaching it to the instance.
        global_constraints = ConcreteModel()
        for cons in instance.component_objects( 
                Constraint, active=True, descend_into=Block,
                sort=SortComponents.deterministic):
            global_constraints.add_component(unique_component_name(
                global_constraints, cons.getname(fully_qualified=True,
                                                 name_buffer=NAME_BUFFER)), 
                                             Reference(cons))
        for var in instance.component_objects(Var, descend_into=(Block,
                                                                 Disjunct),
                                              sort=SortComponents.deterministic):
            global_constraints.add_component(unique_component_name(
                global_constraints, var.getname(fully_qualified=True,
                                                name_buffer=NAME_BUFFER)), 
                                             Reference(var))
        self._global_constraints = global_constraints

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
                    self._transform_disjunction(t)
                else:
                    self._transform_disjunctionData(t, t.index())
            elif t.ctype in (Block, Disjunct):
                if t.is_indexed():
                    self._transform_block(t)
                else:
                    self._transform_blockData(t)
            else:
                raise GDP_Error(
                    "Target '%s' was not a Block, Disjunct, or Disjunction. "
                    "It was of type %s and can't be transformed."
                    % (t.name, type(t)) )

    def _add_transformation_block(self, block):
        # create a transformation block on which we will create the reformulated
        # GDP...
        transformation_block = Block()
        block.add_component(
            unique_component_name( 
                block,
                '_pyomo_gdp_partition_disjuncts_reformulation'),
            transformation_block)

        transformation_block.indicator_var_equalities = LogicalConstraint(
            NonNegativeIntegers)

        return transformation_block

    def _add_var_reference(self, var, disjunct, transformed_disjunct,
                           transBlock, partition):
        # When there are variables that were declared on the Disjunct we are
        # transforming, we need to make references to them on the new Disjunct
        # so that the writers will be able to find them.
        transformed_disjunct.add_component(unique_component_name(
            transformed_disjunct, var.getname(fully_qualified=True,
                                              name_buffer=NAME_BUFFER)), 
                                           Reference(var))

    def _transform_block(self, obj):
        for i in sorted(obj.keys()):
            self._transform_blockData(obj[i])

    def _transform_blockData(self, obj):
        # compute the list of Disjunctions to transform *once*, then do it. Else
        # we will pick up the Disjunctions we create!
        to_transform = []
        # Transform every (active) disjunction in the block. Don't descend into
        # Disjuncts because we'll transform what's on them recursively.
        for disjunction in obj.component_data_objects(
                Disjunction,
                active=True,
                sort=SortComponents.deterministic,
                descend_into=Block):
            to_transform.append(disjunction)

        for disjunction in to_transform:
            self._transform_disjunctionData(disjunction, disjunction.index())

    def _transform_disjunction(self, obj):
        if not obj.active:
            return

        transBlock = self._add_transformation_block(obj.parent_block())
            
        # relax each of the disjunctionDatas
        for i in sorted(obj.keys()):
            self._transform_disjunctionData(obj[i], i, transBlock)

        obj.deactivate()

    def _transform_disjunctionData(self, obj, idx, transBlock=None,
                                   transformed_parent_disjunct=None):
        if not obj.active:
            return

        # Just because it's unlikely this is what someone meant to do...
        if len(obj.disjuncts) == 0:
            raise GDP_Error("Disjunction '%s' is empty. This is "
                            "likely indicative of a modeling error."  %
                            obj.getname(fully_qualified=True,
                                        name_buffer=NAME_BUFFER))

        if transBlock is None and transformed_parent_disjunct is not None:
            transBlock = self._add_transformation_block(
                transformed_parent_disjunct)
        if transBlock is None:
            transBlock = self._add_transformation_block(obj.parent_block())

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
                method = method if method is not None else arbitrary_partition

                # now figure out P
                if self._config.P is None:
                    # This will just end in failure below. (We're checking here
                    # because we don't need a value of P if the partitions were
                    # specified for every Disjunction.)
                    P = None
                else:
                    P = self._config.P.get(obj)
                    if P is None:
                        P = self._config.P.get(None)
                if P is None:
                    raise GDP_Error("No value for P was given for disjunction "
                                    "%s! Please specify a value of P "
                                    "(number of "
                                    "partitions), if you do not specify the "
                                    "partitions directly." % obj.name)
                # it's this method's job to scream if it can't handle what's
                # here, we can only assume it worked for now, since it's a
                # callback.
                partition = method(obj, P)
        # these have to be ComponentSets
        partition = [ComponentSet(var_list) for var_list in partition]

        transformed_disjuncts = []
        for disjunct in obj.disjuncts:
            transformed_disjunct = self._transform_disjunct(disjunct, partition,
                                                            transBlock)
            if transformed_disjunct is not None:
                transformed_disjuncts.append(transformed_disjunct)
                # These require transformation, but that's okay because we are
                # going to a GDP
                transBlock.indicator_var_equalities[
                    len(transBlock.indicator_var_equalities)] = \
                        disjunct.indicator_var.equivalent_to(
                            transformed_disjunct.indicator_var)

        # make a new disjunction with the transformed guys
        transformed_disjunction = Disjunction(expr=[disj for disj in
                                                    transformed_disjuncts])
        transBlock.add_component(
            unique_component_name(transBlock, 
                                  obj.getname(fully_qualified=True, 
                                              name_buffer=NAME_BUFFER)),
            transformed_disjunction)
        obj._algebraic_constraint = weakref_ref(transformed_disjunction)

        obj.deactivate()

    def _get_leq_constraints(self, cons):
        constraints = []
        if cons.lower is not None:
            constraints.append((-cons.body, -cons.lower))
        if cons.upper is not None:
            constraints.append((cons.body, cons.upper))
        return constraints

    def _transform_disjunct(self, disjunct, partition, transBlock):
        # deactivated -> either we've already transformed or user deactivated
        if not disjunct.active:
            if disjunct.indicator_var.is_fixed():
                if not value(disjunct.indicator_var):
                    # The user cleanly deactivated the disjunct: there
                    # is nothing for us to do here.
                    return
                else:
                    raise GDP_Error(
                        "The disjunct '%s' is deactivated, but the "
                        "indicator_var is fixed to %s. This makes no sense."
                        % ( disjunct.name, value(disjunct.indicator_var) ))
            if disjunct._transformation_block is None:
                raise GDP_Error(
                    "The disjunct '%s' is deactivated, but the "
                    "indicator_var is not fixed and the disjunct does not "
                    "appear to have been relaxed. This makes no sense. "
                    "(If the intent is to deactivate the disjunct, fix its "
                    "indicator_var to False.)"
                    % ( disjunct.name, ))

        if disjunct._transformation_block is not None:
            # we've transformed it, which means this is the second time it's
            # appearing in a Disjunction
            raise GDP_Error(
                    "The disjunct '%s' has been transformed, but a disjunction "
                    "it appears in has not. Putting the same disjunct in "
                    "multiple disjunctions is not supported." % disjunct.name)

        transformed_disjunct = Disjunct()
        disjunct._transformation_block = weakref_ref(transformed_disjunct)
        transBlock.add_component(unique_component_name(
            transBlock,
            disjunct.getname(fully_qualified=True, name_buffer=NAME_BUFFER)),
                                 transformed_disjunct)

        # need to transform inner Disjunctions first (before we complain about
        # active Disjuncts)
        for disjunction in disjunct.component_data_objects(
                Disjunction,
                active=True,
                sort=SortComponents.deterministic,
                descend_into=Block):
            self._transform_disjunctionData(disjunction, disjunction.index(),
                                            None, transformed_disjunct)
    
        # transform everything else
        for obj in disjunct.component_data_objects(
                active=True,
                sort=SortComponents.deterministic,
                descend_into=Block):
            handler = self.handlers.get(obj.ctype, None)
            if not handler:
                if handler is None:
                    raise GDP_Error(
                        "No partition_disjuncts transformation handler "
                        "registered "
                        "for modeling components of type %s. If your "
                        "disjuncts contain non-GDP Pyomo components that "
                        "require transformation, please transform them first."
                        % obj.ctype)
                continue
            # we are really only transforming constraints and checking for
            # anything nutty (active Disjuncts, etc) here, so pass through what
            # is necessary for transforming Constraints
            handler(obj, disjunct, transformed_disjunct, transBlock, partition)

        disjunct._deactivate_without_fixing_indicator()
        return transformed_disjunct

    def _transform_constraint(self, cons, disjunct, transformed_disjunct,
                              transBlock, partition):
        instance = disjunct.model()
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
            repn = generate_standard_repn(body, compute_values=True)
            nonlinear_repn = None
            if repn.nonlinear_expr is not None:
                nonlinear_repn = _generate_additively_separable_repn(
                    repn.nonlinear_expr)
            split_exprs = []
            split_aux_vars = []
            vars_not_accounted_for = ComponentSet(v for v in
                                                  EXPR.identify_variables(
                                                      body,
                                                      include_fixed=False))
            vars_accounted_for = ComponentSet()
            for idx, var_list in enumerate(partition):
                # we are going to recreate the piece of the expression
                # involving the vars in var_list
                split_exprs.append(0)
                expr = split_exprs[-1]
                for i, v in enumerate(repn.linear_vars):
                    if v in var_list:
                        expr += repn.linear_coefs[i]*v
                        vars_accounted_for.add(v)
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
                        vars_accounted_for.add(v1)
                        vars_accounted_for.add(v2)
                if nonlinear_repn is not None:
                    for i, expr_var_set in enumerate(
                            nonlinear_repn['nonlinear_vars']):
                        # check if v_list is a subset of var_list. If it is
                        # not and there is no intersection, we move on. If
                        # it is not and there is an intersection, we raise
                        # an error: It's not a valid partition. If it is,
                        # then we add this piece of the expression.
                        # subset?
                        if all(v in var_list for v in list(expr_var_set)):
                            expr += nonlinear_repn['nonlinear_exprs'][i]
                            for var in expr_var_set:
                                vars_accounted_for.add(var)
                        # intersection?
                        elif len(ComponentSet(expr_var_set) & var_list) != 0:
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
                                            "partition. If you did not "
                                            "specify a partition, only "
                                            "a value of P, note that to "
                                            "automatically partition the "
                                            "variables, we assume all the "
                                            "expressions are additively "
                                            "separable." % 
                                            nonlinear_repn[
                                                'nonlinear_exprs'][i])

                expr_lb, expr_ub = self._config.compute_bounds_method( 
                    expr, instance, self._global_constraints, 
                    self._config.compute_bounds_solver)
                if expr_lb is None or expr_ub is None:
                    raise GDP_Error("Expression %s from constraint '%s' "
                                    "is unbounded! Please ensure all "
                                    "variables that appear "
                                    "in the constraint are bounded or "
                                    "specify compute_bounds_method="
                                    "compute_optimal_bounds"
                                    " if the expression is bounded by the "
                                    "global constraints." % 
                                    (expr, cons.name))
                # if the expression was empty wrt the partition, we don't
                # need to bother with any of this. The aux_var doesn't need
                # to exist because it would be 0.
                if type(expr) is not int or expr != 0: 
                    aux_var = aux_vars[len(aux_vars)]
                    aux_var.setlb(expr_lb)
                    aux_var.setub(expr_ub)  
                    split_aux_vars.append(aux_var)
                    split_constraints[
                        len(split_constraints)] = expr <= aux_var

            if len(vars_accounted_for) < len(vars_not_accounted_for):
                    orphans = vars_not_accounted_for - vars_accounted_for
                    orphan_string = ""
                    for v in orphans:
                        orphan_string += "'%s', " % v.name
                    orphan_string = orphan_string[:-2]
                    raise GDP_Error("Partition specified for disjunction "
                                    "containing Disjunct '%s' does not "
                                    "include all the variables that appear "
                                    "in the disjunction. The following "
                                    "variables are not assigned to any part "
                                    "of the partition: %s" % (disjunct.name, 
                                                              orphan_string))
            transformed_constraint[
                len(transformed_constraint)] = sum(v for v in
                                                   split_aux_vars) <= \
                rhs - repn.constant
        # deactivate the constraint since we've transformed it
        cons.deactivate()                             

    def _warn_for_active_disjunct(self, disjunct, parent_disjunct,
                                  transformed_parent_disjunct, transBlock,
                                  partition):
        _warn_for_active_disjunct(disjunct, parent_disjunct, NAME_BUFFER)

    def _warn_for_active_logical_constraint(self, constraint, parent_disjunct,
                                            transformed_parent_disjunct,
                                            transBlock, partition):
        _warn_for_active_logical_constraint(constraint, parent_disjunct,
                                            NAME_BUFFER)

# ESJ: TODO: Add a test for the old indicator variables appearing in logical
# constraints.

# Also, I'm not confident of nested and targets... 

# TODO: Indexed Var declared on Disjunct
