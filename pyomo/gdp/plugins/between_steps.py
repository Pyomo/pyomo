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

from pyomo.common.config import (ConfigBlock, ConfigValue)#,
#                                 NonNegativeFloat, PositiveInt, In)
from pyomo.common.modeling import unique_component_name
from pyomo.core import ( Block, Constraint, Var, SortComponents, Transformation,
                         TransformationFactory, TraversalStrategy,
                         NonNegativeIntegers)#,
#                         value, Reals, NonNegativeReals,
#                         Suffix, ComponentMap )
from pyomo.common.collections import ComponentSet
from pyomo.repn import generate_standard_repn

from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.gdp.util import preprocess_targets, is_child_of, target_list, _to_dict

from pyomo.contrib.fbbt.fbbt import compute_bounds_on_expr

import logging
logger = logging.getLogger('pyomo.gdp.between_steps')

from nose.tools import set_trace

NAME_BUFFER = {}

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
    CONFIG.declare('verbose', ConfigValue(
        default=False,
        domain=bool,
        description="""Enable verbose output.""",
        doc="""
        Set to True for verbose output, False otherwise.
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

            self._apply_to_impl(instance)

        finally:
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
                # It not, see what method to use to calculate one
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
                    for i, v in enumerate(repn.nonlinear_vars):
                        # TODO I have no idea what to do with these
                        raise NotImplementedError("I don't know what these "
                                                  "can look like!")
                
                    # now we have the piece of the expression, we need bounds on
                    # it. We first check if they were specified via args. If not
                    # we use fbbt.

                    # TODO: need to implement a bounds arg and try to retrieve
                    # them here. Only do the below if that fails.

                    expr_lb, expr_ub = compute_bounds_on_expr(expr)
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
