#  _________________________________________________________________________
#
#  Coopr: A COmmon Optimization Python Repository
#  Copyright (c) 2008 Sandia Corporation.
#  This software is distributed under the BSD License.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  For more information, see the FAST README.txt file.
#  _________________________________________________________________________
from __future__ import division

__all__ = [ 'AmplRepn', 'generate_ampl_repn']

import weakref

from coopr.pyomo.base import expr as Expr
from coopr.pyomo.base import _ExpressionData, Expression
from coopr.pyomo.base.var import _VarData, value
from coopr.pyomo.base import external
from coopr.pyomo.expr.canonical_repn import collect_linear_canonical_repn

import six
from six import itervalues, iteritems, StringIO
from six.moves import xrange, zip

import logging

logger = logging.getLogger('coopr.pyomo')

using_py3 = six.PY3

class AmplRepn(object):

    __slots__ = ('_constant',
                 '_linear_terms_coef',
                 '_linear_vars',
                 '_nonlinear_expr',
                 '_nonlinear_vars')

    def __getstate__(self):
        """
        This method is required because this class uses slots.
        """ 
        return  (self._constant,
                 self._linear_terms_coef,
                 self._linear_vars,
                 self._nonlinear_expr,
                 self._nonlinear_vars)

    def __setstate__(self, state):
        """
        This method is required because this class uses slots.
        """
        self._constant, \
        self._linear_terms_coef, \
        self._linear_vars, \
        self._nonlinear_expr, \
        self._nonlinear_vars = state
        
    def __init__(self):
        self._constant = 0
        self._linear_vars = {}
        self._linear_terms_coef = {}
        self._nonlinear_expr = None
        self._nonlinear_vars = {}

    # 
    # Although it is convenient to have the dictionaries hashed by some
    # variable id when generating the ampl representation, we can 
    # compress these to lists after generation is complete
    #
    def compress(self):
        if  self._linear_vars.__class__ is dict:
            linear_keys = self._linear_vars.keys()
            self._linear_vars = tuple(self._linear_vars[key] for key in linear_keys)
            self._linear_terms_coef = tuple(self._linear_terms_coef[key] for key in linear_keys)
        if self._nonlinear_vars.__class__ is dict:
            self._nonlinear_vars = tuple(itervalues(self._nonlinear_vars))

    # GAH: I'm not sure why this funciton is here and I think it is misleading. Calling clone on 
    #      a model will deepcopy the AmplRepn (e.g. new expressions and variables). Calling clone
    #      directly on the AmplRepn (as defined below) does nothing close to that.
    def Xclone(self):
        clone = AmplRepn()
        clone._constant = self._constant
        clone._linear_vars = self._linear_vars
        clone._linear_terms_coef = self._linear_terms_coef
        clone._nonlinear_expr = self._nonlinear_expr
        clone._nonlinear_vars = self._nonlinear_vars
        return clone

    # for debugging
    def __str__(self):
        output = StringIO()
        output.write("\n")
        output.write("constant:       "+str(self._constant)+"\n")
        output.write("linear vars:    "+str(self._linear_vars)+"\n")
        output.write("linear coef:    "+str(self._linear_terms_coef)+"\n")
        output.write("nonlinear expr:\n")
        try:
            self._nonlinear_expr.pprint(ostream=output)
        except AttributeError:
            output.write(str(self._nonlinear_expr)+"\n")
        output.write("nonlinear vars: "+str(self._nonlinear_vars)+"\n")
        output.write("\n")
        ret_str = output.getvalue()
        output.close()
        return ret_str

    def __eq__(self, other):
        # Can only be equal to other AmplRepn instances
        if not isinstance(other, AmplRepn):
            return False
        
        # Immediately check constants
        if self._constant != other._constant:
            return False

        # Optimization: check linear term lengths before iterating
        if len(self._linear_vars) != len(other._linear_vars):
            return False

        if self._linear_vars.__class__ is dict:
            self_linear_vars = list(itervalues(self._linear_vars))
            self_linear_terms = self._linear_terms_coef
        else:
            self_linear_vars = self._linear_vars
            self_linear_terms = dict((id(var),coef) for var,coef in zip(self._linear_vars,self._linear_terms_coef))

        if other._linear_vars.__class__ is dict:
            other_linear_vars = list(itervalues(other._linear_vars))
            other_linear_terms = other._linear_terms_coef
        else:
            other_linear_vars = other._linear_vars
            other_linear_terms = dict((id(var),coef) for var,coef in zip(other._linear_vars,other._linear_terms_coef))

        # Establish a mapping between self's linear terms and other's
        found_match = []
        for var in self_linear_vars:
            match = False
            for other_var in other_linear_vars:
                if var is other_var:
                    match = True
                    break
            found_match.append(match)
                    
        # We have to have found all our own vars
        if not all(found_match):
            return False

        # Check that linear term coefficients are equal
        for var_ID in self_linear_terms:
            if self_linear_terms[var_ID] != other_linear_terms[var_ID]:
                return False

        if self._nonlinear_expr is not None:
            if other._nonlinear_expr is None:
                return False

            if type(self._nonlinear_expr) is list:
                if type(other._nonlinear_expr) is not list:
                    return False

                # Check that nonlinear expressions are the same length
                if len(self._nonlinear_expr) != len(other._nonlinear_expr):
                    return False

                # Check that nonlinear expression have the same terms
                for i in xrange(len(self._nonlinear_expr)):
                    if self._nonlinear_expr[i][0] != other._nonlinear_expr[i][0]:
                        return False
                    if not self._nonlinear_expr[i][1] is other._nonlinear_expr[i][1]:
                        return False
            else:
                if not self._nonlinear_expr is other._nonlinear_expr:
                    return False
        else:
            if other._nonlinear_expr is not None:
                return False

        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def is_fixed(self):
        if len(self._linear_vars) == 0 and len(self._nonlinear_vars) == 0:
            return True
        return False

    def is_linear(self):
        if self._nonlinear_expr is None:
            return True
        return False

    def is_nonlinear(self):
        if self._nonlinear_expr is None:
            return False
        return True


def _generate_ampl_repn(exp):
    ampl_repn = AmplRepn()

    #
    # Expression
    #
    if exp.is_expression():
        exp_type = type(exp)

        #
        # Sum
        #
        if exp_type is Expr._SumExpression:
            ampl_repn._constant = exp._const #float(exp._const)
            ampl_repn._nonlinear_expr = None
            for i in xrange(len(exp._args)):
                exp_coef = exp._coef[i]
                child_repn = _generate_ampl_repn(exp._args[i])
                # adjust the constant
                ampl_repn._constant += exp_coef*child_repn._constant

                # adjust the linear terms
                for var_ID in child_repn._linear_vars:
                    if var_ID in ampl_repn._linear_terms_coef:
                        ampl_repn._linear_terms_coef[var_ID] += exp_coef*child_repn._linear_terms_coef[var_ID]
                    else:
                        ampl_repn._linear_terms_coef[var_ID] = exp_coef*child_repn._linear_terms_coef[var_ID]
                # adjust the linear vars
                ampl_repn._linear_vars.update(child_repn._linear_vars)

                # adjust the nonlinear terms
                if not child_repn._nonlinear_expr is None:
                    if ampl_repn._nonlinear_expr is None:
                        ampl_repn._nonlinear_expr = [(exp_coef, child_repn._nonlinear_expr)]
                    else:
                        ampl_repn._nonlinear_expr.append((exp_coef, child_repn._nonlinear_expr))
                # adjust the nonlinear vars
                ampl_repn._nonlinear_vars.update(child_repn._nonlinear_vars)
            return ampl_repn

        #
        # Product
        #
        elif exp_type is Expr._ProductExpression:
            #
            # Iterate through the denominator.  If they aren't all constants, then
            # simply return this expresion.
            #
            denom=1.0
            for e in exp._denominator:
                if e.is_fixed():
                    denom *= value(e)
                else:
                    ampl_repn._nonlinear_expr = exp
                    break
                if denom == 0.0:
                    print("Divide-by-zero error - offending sub-expression:")
                    e.pprint()
                    raise ZeroDivisionError

            if not ampl_repn._nonlinear_expr is None:
                # we have a nonlinear expression ... build up all the vars
                for e in exp._denominator:
                    arg_repn = _generate_ampl_repn(e)
                    ampl_repn._nonlinear_vars.update(arg_repn._linear_vars)
                    ampl_repn._nonlinear_vars.update(arg_repn._nonlinear_vars)

                for e in exp._numerator:
                    arg_repn = _generate_ampl_repn(e)
                    ampl_repn._nonlinear_vars.update(arg_repn._linear_vars)
                    ampl_repn._nonlinear_vars.update(arg_repn._nonlinear_vars)
                return ampl_repn

            #
            # OK, the denominator is a constant.
            #
            # build up the ampl_repns for the numerator
            
            n_linear_args = 0
            n_nonlinear_args = 0
            arg_repns = list()
            for i in xrange(len(exp._numerator)):
                e = exp._numerator[i]
                e_repn = _generate_ampl_repn(e)
                arg_repns.append(e_repn)
                # check if the expression is not nonlinear else it is nonlinear
                if e_repn._nonlinear_expr is not None:
                    n_nonlinear_args += 1
                # Check whether the expression is constant or else it is linear
                elif len(e_repn._linear_vars) > 0:
                    n_linear_args += 1
                # At this point we do not have a nonlinear expression and there are
                # no linear terms. If the expression constant is zero, then we
                # have a zero term in the product expression, so the entire product 
                # expression becomes trivial. 
                elif e_repn._constant == 0.0:
                    ampl_repn = e_repn
                    return ampl_repn

            is_nonlinear = False
            if n_linear_args > 1 or n_nonlinear_args > 0:
                is_nonlinear = True


            if is_nonlinear is True:
                # do like AMPL and simply return the expression
                # without extracting the potentially linear part
                ampl_repn = AmplRepn()
                ampl_repn._nonlinear_expr = exp
                for repn in arg_repns:
                    ampl_repn._nonlinear_vars.update(repn._linear_vars)
                    ampl_repn._nonlinear_vars.update(repn._nonlinear_vars)
                return ampl_repn

            else: # is linear or constant
                ampl_repn = current_repn = arg_repns[0]
                for i in xrange(1,len(arg_repns)):
                    e_repn = arg_repns[i]
                    ampl_repn = AmplRepn()

                    # const_c * const_e
                    ampl_repn._constant = current_repn._constant * e_repn._constant

                    # const_e * L_c
                    if e_repn._constant != 0.0:
                        for (var_ID, var) in iteritems(current_repn._linear_vars):
                            ampl_repn._linear_terms_coef[var_ID] = current_repn._linear_terms_coef[var_ID] * e_repn._constant
                        ampl_repn._linear_vars.update(current_repn._linear_vars)

                    # const_c * L_e
                    if current_repn._constant != 0.0:
                        for (e_var_ID,e_var) in iteritems(e_repn._linear_vars):
                            if e_var_ID in ampl_repn._linear_vars:
                                ampl_repn._linear_terms_coef[e_var_ID] += current_repn._constant * e_repn._linear_terms_coef[e_var_ID]
                            else:
                                ampl_repn._linear_terms_coef[e_var_ID] = current_repn._constant * e_repn._linear_terms_coef[e_var_ID]
                        ampl_repn._linear_vars.update(e_repn._linear_vars)
                    current_repn = ampl_repn

            # now deal with the product expression's coefficient that needs
            # to be applied to all parts of the ampl_repn
            ampl_repn._constant *= exp._coef/denom
            for var_ID in ampl_repn._linear_terms_coef:
                ampl_repn._linear_terms_coef[var_ID] *= exp._coef/denom

            return ampl_repn

        #
        # Power Expressions
        #
        elif exp_type is Expr._PowExpression:
            assert(len(exp._args) == 2)
            base_repn = _generate_ampl_repn(exp._args[0])
            base_repn_fixed = base_repn.is_fixed()
            exponent_repn = _generate_ampl_repn(exp._args[1])
            exponent_repn_fixed = exponent_repn.is_fixed()

            if base_repn_fixed and exponent_repn_fixed:
                ampl_repn._constant = base_repn._constant**exponent_repn._constant
            elif exponent_repn_fixed and exponent_repn._constant == 1.0:
                ampl_repn = base_repn
            elif exponent_repn_fixed and exponent_repn._constant == 0.0:
                ampl_repn._constant = 1.0
            else:
                # instead, let's just return the expression we are given and only
                # use the ampl_repn for the vars
                ampl_repn._nonlinear_expr = exp
                ampl_repn._nonlinear_vars = base_repn._nonlinear_vars
                ampl_repn._nonlinear_vars.update(exponent_repn._nonlinear_vars)
                ampl_repn._nonlinear_vars.update(base_repn._linear_vars)
                ampl_repn._nonlinear_vars.update(exponent_repn._linear_vars)
            return ampl_repn
        #
        # External Functions
        #
        elif exp_type is external._ExternalFunctionExpression:
            # In theory, the argument are fixed, we can simply evaluate this expression
            if exp.is_fixed():
                ampl_repn._constant = value(exp)
                return ampl_repn

            # this is inefficient since it is using much more than what we need
            ampl_repn._nonlinear_expr = exp
            for arg in exp._args:
                if isinstance(arg, basestring):
                    continue
                child_repn = _generate_ampl_repn(arg)
                ampl_repn._nonlinear_vars.update(child_repn._nonlinear_vars)
                ampl_repn._nonlinear_vars.update(child_repn._linear_vars)
            return ampl_repn
        #
        # Intrinsic Functions
        #
        elif isinstance(exp,Expr._IntrinsicFunctionExpression):
            # Checking isinstance above accounts for the fact that _AbsExpression
            # is a subclass of _IntrinsicFunctionExpression
            assert(len(exp._args) == 1)
            
            # the argument is fixed, we can simply evaluate this expression
            if exp._args[0].is_fixed():
                ampl_repn._constant = value(exp)
                return ampl_repn

            # this is inefficient since it is using much more than what we need
            child_repn = _generate_ampl_repn(exp._args[0])

            ampl_repn._nonlinear_expr = exp
            ampl_repn._nonlinear_vars = child_repn._nonlinear_vars
            ampl_repn._nonlinear_vars.update(child_repn._linear_vars)
            return ampl_repn
        #
        # AMPL-style If-Then-Else expression
        #
        elif exp_type is Expr.Expr_if:
            if exp._if.is_fixed():
                if exp._if():
                    ampl_repn = _generate_ampl_repn(exp._then)
                else:
                    ampl_repn = _generate_ampl_repn(exp._else)
            else:
                if_repn = _generate_ampl_repn(exp._if)
                then_repn = _generate_ampl_repn(exp._then)
                else_repn = _generate_ampl_repn(exp._else)
                ampl_repn._nonlinear_expr = exp
                ampl_repn._nonlinear_vars = if_repn._nonlinear_vars
                ampl_repn._nonlinear_vars.update(then_repn._nonlinear_vars)
                ampl_repn._nonlinear_vars.update(else_repn._nonlinear_vars)
                ampl_repn._nonlinear_vars.update(if_repn._linear_vars)
                ampl_repn._nonlinear_vars.update(then_repn._linear_vars)
                ampl_repn._nonlinear_vars.update(else_repn._linear_vars)
            return ampl_repn
        elif (exp_type is Expr._InequalityExpression) or (exp_type is Expr._EqualityExpression):
            for arg in exp._args:
                arg_repn = _generate_ampl_repn(arg)
                ampl_repn._nonlinear_vars.update(arg_repn._nonlinear_vars)
                ampl_repn._nonlinear_vars.update(arg_repn._linear_vars)
            ampl_repn._nonlinear_expr = exp
            return ampl_repn
        elif exp.is_fixed():
            ampl_repn._constant = value(exp)
            return ampl_repn
        #
        # Expression (the component)
        #
        elif type(exp) is _ExpressionData or exp.type() is Expression:
            ampl_repn = _generate_ampl_repn(exp.value)
            return ampl_repn
        #
        # ERROR
        #
        else:
            raise ValueError("Unsupported expression type: "+str(exp))
    #
    # Constant
    #
    elif exp.is_fixed():
        ### GAH: Why were we even checking this
        #if not exp.value.__class__ in native_numeric_types:
        #    ampl_repn = _generate_ampl_repn(exp.value) 
        #    return ampl_repn
        ampl_repn._constant = value(exp)
        return ampl_repn
    #
    # Variable
    #
    elif isinstance(exp, _VarData):
        if exp.fixed:
            ampl_repn._constant = exp.value
            return ampl_repn
        var_ID = id(exp)
        ampl_repn._linear_terms_coef[var_ID] = 1.0
        ampl_repn._linear_vars[var_ID] = exp
        return ampl_repn

    #
    # ERROR
    #
    else:
        raise ValueError("Unexpected expression type: "+str(exp))

def generate_ampl_repn(exp):
    degree = exp.polynomial_degree()
    if (degree is None) or (degree > 1):
        repn = _generate_ampl_repn(exp)
        repn.compress()
    elif degree == 0:
        repn = AmplRepn()
        repn._constant = value(exp)
        # compress
        repn._linear_vars = tuple()
        repn._linear_terms_coef = tuple()
        repn._nonlinear_vars = tuple()
    else: # degree == 1
        repn = AmplRepn()
        # compress
        repn._linear_vars = tuple()
        repn._linear_terms_coef = tuple()
        repn._nonlinear_vars = tuple()
        coef, varmap = collect_linear_canonical_repn(exp, {})
        if None in coef:
            val = coef.pop(None)
            if val:
                repn._constant = val
        # the six module is inefficient in terms of wrapping iterkeys and itervalues,
        # in the context of Python 2.7. use the native dictionary methods where possible.
        if using_py3 is False:
            repn._linear_terms_coef = tuple(val for val in coef.itervalues() if val)
            repn._linear_vars = tuple((varmap[var_hash] for var_hash,val in coef.iteritems() if val))
        else:
            repn._linear_terms_coef = tuple(val for val in coef.values() if val)
            repn._linear_vars = tuple((varmap[var_hash] for var_hash,val in coef.items() if val))
    return repn
