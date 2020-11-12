#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import six
import logging

from pyomo.core.base import Block, VarList, ConstraintList, Objective, Var, Constraint, maximize, ComponentUID, Set, TransformationFactory
from pyomo.repn import generate_standard_repn
from pyomo.mpec import ComplementarityList, complements
from pyomo.bilevel.plugins.transform import Base_BilevelTransformation


logger = logging.getLogger('pyomo.core')



@TransformationFactory.register('bilevel.linear_mpec', doc="Generate a linear MPEC from the optimality conditions of the submodel")
class LinearComplementarity_BilevelTransformation(Base_BilevelTransformation):

    def __init__(self):
        super(LinearComplementarity_BilevelTransformation, self).__init__()

    def _apply_to(self, instance, **kwds):
        self._deterministic = kwds.get('deterministic',False)
        #
        # Process options
        #
        submodel = self._preprocess('bilevel.linear_mpec', instance, **kwds)
        instance.reclassify_component_type(submodel, Block)
        #
        # Create a block with optimality conditions
        #
        setattr(instance, self._submodel+'_kkt', self._add_optimality_conditions(instance, submodel))
        instance._transformation_data['bilevel.linear_mpec'].submodel_cuid = ComponentUID(submodel)
        instance._transformation_data['bilevel.linear_mpec'].block_cuid = ComponentUID(getattr(instance,self._submodel+'_kkt'))
        #-------------------------------------------------------------------------------
        #
        # Disable the original submodel and
        #
        #instance.reclassify_component_type(submodel, SubModel)
        #submodel.deactivate()
        # TODO: Cache the list of components that were deactivated
        for (name, data) in submodel.component_map(active=True).items():
            if not isinstance(data,Var) and not isinstance(data,Set):
                data.deactivate()


    def _add_optimality_conditions(self, instance, submodel):
        """
        Add optimality conditions for the submodel

        This assumes that the original model has the form:

            min c1*x + d1*y
                A3*x <= b3
                A1*x + B1*y <= b1
                min c2*x + d2*y + x'*Q*y
                    A2*x + B2*y + x'*E2*y <= b2
                    y >= 0

        NOTE THE VARIABLE BOUNDS!
        """
        #
        # Populate the block with the linear constraints.
        # Note that we don't simply clone the current block.
        # We need to collect a single set of equations that
        # can be easily expressed.
        #
        d2 = {}
        B2 = {}
        vtmp = {}
        utmp = {}
        sids_set = set()
        sids_list = []
        #
        block = Block(concrete=True)
        block.u = VarList()
        block.v = VarList()
        block.c1 = ConstraintList()
        block.c2 = ComplementarityList()
        block.c3 = ComplementarityList()
        #
        # Collect submodel objective terms
        #
        # TODO: detect fixed variables
        #
        for odata in submodel.component_data_objects(Objective, active=True):
            if odata.sense == maximize:
                d_sense = -1
            else:
                d_sense = 1
            #
            # Iterate through the variables in the representation
            #
            o_terms = generate_standard_repn(odata.expr, compute_values=False)
            #
            # Linear terms
            #
            for i, var in enumerate(o_terms.linear_vars):
                if var.parent_component().local_name in self._fixed_upper_vars:
                    #
                    # Skip fixed upper variables
                    #
                    continue
                #
                # Store the coefficient for the variable.  The coefficient is
                # negated if the objective is maximized.
                #
                id_ = id(var)
                d2[id_] = d_sense * o_terms.linear_coefs[i]
                if not id_ in sids_set:
                    sids_set.add(id_)
                    sids_list.append(id_)
            #
            # Quadratic terms
            #
            for i, var in enumerate(o_terms.quadratic_vars):
                if var[0].parent_component().local_name in self._fixed_upper_vars:
                    if var[1].parent_component().local_name in self._fixed_upper_vars:
                        #
                        # Skip fixed upper variables
                        #
                        continue
                    #
                    # Add the linear term
                    #
                    id_ = id(var[1])
                    d2[id_] = d2.get(id_,0) + d_sense * o_terms.quadratic_coefs[i] * var[0]
                    if not id_ in sids_set:
                        sids_set.add(id_)
                        sids_list.append(id_)
                elif var[1].parent_component().local_name in self._fixed_upper_vars:
                    #
                    # Add the linear term
                    #
                    id_ = id(var[0])
                    d2[id_] = d2.get(id_,0) + d_sense * o_terms.quadratic_coefs[i] * var[1]
                    if not id_ in sids_set:
                        sids_set.add(id_)
                        sids_list.append(id_)
                else:
                    raise RuntimeError("Cannot apply this transformation to a problem with quadratic terms where both variables are in the lower level.")
            #
            # Stop after the first objective
            #
            break
        #
        # Iterate through all lower level variables, adding dual variables
        # and complementarity slackness conditions for y bound constraints
        #
        for vcomponent in instance.component_objects(Var, active=True):
            if vcomponent.local_name in self._fixed_upper_vars:
                #
                # Skip fixed upper variables
                #
                continue
            for ndx in vcomponent:
                #
                # For each index, get the bounds for the variable
                #
                lb, ub = vcomponent[ndx].bounds
                if not lb is None:
                    #
                    # Add the complementarity slackness condition for a lower bound
                    #
                    v = block.v.add()
                    block.c3.add( complements(vcomponent[ndx] >= lb, v >= 0) )
                else:
                    v = None
                if not ub is None:
                    #
                    # Add the complementarity slackness condition for an upper bound
                    #
                    w = block.v.add()
                    vtmp[id(vcomponent[ndx])] = w
                    block.c3.add( complements(vcomponent[ndx] <= ub, w >= 0) )
                else:
                    w = None
                if not (v is None and w is None):
                    #
                    # Record the variables for which complementarity slackness conditions
                    # were created.
                    #
                    id_ = id(vcomponent[ndx])
                    vtmp[id_] = (v,w)
                    if not id_ in sids_set:
                        sids_set.add(id_)
                        sids_list.append(id_)
        #
        # Iterate through all constraints, adding dual variables and
        # complementary slackness conditions (for inequality constraints)
        #
        for cdata in submodel.component_data_objects(Constraint, active=True):
            if cdata.equality:
                # Don't add a complementary slackness condition for an equality constraint
                u = block.u.add()
                utmp[id(cdata)] = (None,u)
            else:
                if not cdata.lower is None:
                    #
                    # Add the complementarity slackness condition for a greater-than inequality
                    #
                    u = block.u.add()
                    block.c2.add( complements(- cdata.body <= - cdata.lower, u >= 0) )
                else:
                    u = None
                if not cdata.upper is None:
                    #
                    # Add the complementarity slackness condition for a less-than inequality
                    #
                    w = block.u.add()
                    block.c2.add( complements(cdata.body <= cdata.upper, w >= 0) )
                else:
                    w = None
                if not (u is None and w is None):
                    utmp[id(cdata)] = (u,w)
            #
            # Store the coefficients for the constraint variables that are not fixed
            #
            c_terms = generate_standard_repn(cdata.body, compute_values=False)
            #
            # Linear terms
            #
            for i, var in enumerate(c_terms.linear_vars):
                if var.parent_component().local_name in self._fixed_upper_vars:
                    continue
                id_ = id(var)
                B2.setdefault(id_,{}).setdefault(id(cdata),c_terms.linear_coefs[i])
                if not id_ in sids_set:
                    sids_set.add(id_)
                    sids_list.append(id_)
            #
            # Quadratic terms
            #
            for i, var in enumerate(c_terms.quadratic_vars):
                if var[0].parent_component().local_name in self._fixed_upper_vars:
                    if var[1].parent_component().local_name in self._fixed_upper_vars:
                        continue
                    id_ = id(var[1])
                    if id_ in B2:
                        B2[id_][id(cdata)] = c_terms.quadratic_coefs[i] * var[0]
                    else:
                        B2.setdefault(id_,{}).setdefault(id(cdata),c_terms.quadratic_coefs[i] * var[0])
                    if not id_ in sids_set:
                        sids_set.add(id_)
                        sids_list.append(id_)
                elif var[1].parent_component().local_name in self._fixed_upper_vars:
                    id_ = id(var[0])
                    if id_ in B2:
                        B2[id_][id(cdata)] = c_terms.quadratic_coefs[i] * var[1]
                    else:
                        B2.setdefault(id_,{}).setdefault(id(cdata),c_terms.quadratic_coefs[i] * var[1])
                    if not id_ in sids_set:
                        sids_set.add(id_)
                        sids_list.append(id_)
                else:
                    raise RuntimeError("Cannot apply this transformation to a problem with quadratic terms where both variables are in the lower level.")
        #
        # Generate stationarity equations
        #
        tmp__ = (None, None)
        for vid in sids_list:
            exp = d2.get(vid,0)
            #
            lb_dual, ub_dual = vtmp.get(vid, tmp__)
            if vid in vtmp:
                if not lb_dual is None:
                    exp -= lb_dual             # dual for variable lower bound
                if not ub_dual is None:
                    exp += ub_dual             # dual for variable upper bound
            #
            B2_ = B2.get(vid,{})
            utmp_keys = list(utmp.keys())
            if self._deterministic:
                utmp_keys.sort(key=lambda x:utmp[x][0].local_name if utmp[x][1] is None else utmp[x][1].local_name)
            for uid in utmp_keys:
                if uid in B2_:
                    lb_dual, ub_dual = utmp[uid]
                    if not lb_dual is None:
                        exp -= B2_[uid] * lb_dual
                    if not ub_dual is None:
                        exp += B2_[uid] * ub_dual
            if type(exp) in six.integer_types or type(exp) is float:
                # TODO: Annotate the model as unbounded
                raise IOError("Unbounded variable without side constraints")
            else:
                block.c1.add( exp == 0 )
        #
        # Return block
        #
        return block

