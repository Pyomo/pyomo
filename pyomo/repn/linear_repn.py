#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2010 Sandia Corporation.
#  This software is distributed under the BSD License.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  For more information, see the FAST README.txt file.
#  _________________________________________________________________________

__all__ = ['linearize_model_expressions', 'is_linear_expression_constant']

# a prototype utility to translate from a "normal" model instance containing
# constraints and objectives with expresssions for their bodies, into a
# more compact linearized verson of the problem. eliminates the need for
# canonical expressions and the associated overhead (memory and speed).

# NOTE: Currently only works for constraints - leaving the expressions alone,
#       mainly because PH doesn't modify those and I don't want to deal with
#       that aspect for the moment. In particular, this only works for
#       immutable parameters due to the reliance on the canonical expression
#       generator to extract the linear and constant terms.

from six import iteritems, itervalues
from pyomo.core import *
from pyomo.repn import generate_canonical_repn, canonical_is_linear, canonical_is_constant

def linearize_model_expressions(instance):
    var_id_map = {}

    for block in instance.all_blocks():

        block_canonical_repn = getattr(block,"canonical_repn",None)

        # Just overwrite any existing component that be called
        # lin_body even if it is a component
        lin_body_map = block.lin_body = ComponentMap()

        # TBD: Should we really be doing this for all components, and
        # not just active ones?
        for constraint_data in active_components_data(block,Constraint):

            delete = True
            canonical_encoding = None
            if block_canonical_repn is not None:
                canonical_encoding = block_canonical_repn.get(constraint_data)
            if canonical_encoding is None:
                canonical_encoding = generate_canonical_repn(constraint_data.body, var_id_map)
                delete = False

            # we obviously can't linearize an expression if it has
            # higher-order terms!
            if canonical_is_linear(canonical_encoding) or \
               canonical_is_constant(canonical_encoding):

                variable_map = canonical_encoding.variables

                constant_term = 0.0
                linear_terms = [] # a list of coefficient, _VarData pairs.

                if canonical_encoding.constant != None:
                    try:
                        # LinearCanonicalRepn
                        constant_term = canonical_encoding.constant
                    except AttributeError:
                        # GeneralCanonicalRepn
                        constant_term = canonical_encoding[0]

                for i in xrange(0, len(canonical_encoding.linear)):
                    var_coefficient = canonical_encoding.linear[i]
                    var_value = canonical_encoding.variables[i]
                    linear_terms.append((var_coefficient, var_value))

                # eliminate the expression tree - we don't need it any
                # longer.  ditto the canonical representation.
                constraint_data.body = None
                if delete:
                    del block_canonical_repn[constraint_data]
                lin_body_map[constraint_data] = [constant_term, linear_terms]

def is_linear_expression_constant(lin_body):
    """Return True if the linear expression is a constant expression, due either to a lack of variables or fixed variables"""
    return (len(lin_body[1]) == 0)
