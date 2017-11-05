#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

#
# non-recursive implementations of generate_standard_repn
#

def OLD_nonrecursive_generate_standard_repn(expr, idMap=None, compute_values=True, verbose=False, quadratic=True, repn=None, _multiplier=None):
    ##
    ## Recurse through the expression tree, collecting variables and linear terms, etc
    ##
    linear = True
    #
    # The stack starts with the current expression
    #
    _stack = [ (expr, expr._args, 0, expr.nargs(), False, [])]
    #
    # Iterate until the stack is empty
    #
    # Note: 1 is faster than True for Python 2.x
    #
    while 1:
        #
        # Get the top of the stack
        #   _obj        Current expression object
        #   _argList    The arguments for this expression objet
        #   _idx        The current argument being considered
        #   _len        The number of arguments
        #
        # Note: expressions pushed onto the stack are guaranteed to 
        # be potentially variable.
        #
        _obj, _argList, _idx, _len, _compute_value, _result = _stack.pop()
        if verbose: #pragma:nocover
            print("*"*10 + " POP  " + "*"*10)

        #
        # Iterate through the arguments
        #
        while _idx < _len:
            if verbose: #pragma:nocover
                print("-"*30)
                print(type(_obj))
                print(_obj.to_string())
                print(_argList)
                print(_idx)
                print(_len)
                print(_compute_value)
                print(_result)

            ##
            ## Process context based on _obj type
            ##

            # No special processing for *Sum* objects

            # No special processing for _ProductExpression

            if _obj.__class__ is EXPR._PowExpression:
                if _idx == 0:
                    #
                    # Evaluate the RHS (_args[1]) first, and compute its value
                    #
                    _argList = (_argList[1], _argList[0])
                    _compute_value = True
                elif _idx == 1:
                    _compute_value = False
                    if -999 in _result[0]:
                        #
                        # If the RHS (_args[1]) is variable, then
                        # treat the entire subexpression as a nonlinear expression
                        #
                        _result = [{None:_obj}]
                        linear = False
                        break
                    else:
                        val = _result[0][0]
                        if val == 0:
                            #
                            # If the exponent is zero, then the value of this expression is 1
                            #
                            _result = [{0:1}]
                            break
                        elif val == 1:
                            #
                            # If the exponent is one, then simply return 
                            # the value of the LHS (_args[0])
                            #
                            _result = []
                        elif val == 2 and quadratic:
                            #
                            # If the exponent is two, then set the value of the exponent and continue
                            # processing the value of the LHS (_args[0])
                            #
                            _result = [{0:2}]
                        else:
                            #
                            # Otherwise, we treat this as a nonlinear expression
                            #
                            _result = [{None:_obj}]
                            linear = False
                            break

            elif _obj.__class__ is EXPR.Expr_if:
                if _idx == 0:
                    #
                    # Compute the value of the condition argument
                    #
                    _compute_value = True
                elif _idx == 1:
                    _compute_value = False
                    if -999 in _result[0]:
                        #
                        # If the condition argument is variable, then
                        # treat the entire subexpression as a nonlinear expression
                        #
                        _result = [{None:_obj}]
                        linear = False
                        break
                    else:
                        val = _result[0][0]
                        _idx = 0
                        _len = 1
                        _result = []
                        if val:
                            _argList = [_argList[1]]
                        else:
                            _argList = [_argList[2]]
            
            ##
            ## Process the next current _obj object
            ##

            _sub = _argList[_idx]
            _idx += 1

            if _sub.__class__ in native_numeric_types:
                #
                # Store a native object
                #
                _result.append( {0:_sub} )

            elif _compute_value:
                val = EXPR.evaluate_expression(_sub, only_fixed_vars=True, exception=False)
                if val is None:
                    _result = [{-999: "Error evaluating expression: %s" % str(_sub)}] 
                else:
                    _result.append( {0:val} )

            elif (_sub.__class__ is _GeneralVarData) or isinstance(_sub, (_VarData, IVariable)):
                #
                # Process a single variable
                #
                if not _sub.fixed:
                    #
                    # Store a variable 
                    #
                    id_ = id(_sub)
                    if id_ in idMap[None]:
                        key = idMap[None][id_]
                    else:
                        key = len(idMap) - 1
                        idMap[None][id_] = key
                        idMap[key] = _sub

                    _result.append( {1:{key:1}} )
                else:
                    if compute_values:
                        _result.append( {0:_sub.value} )
                    else:
                        _result.append( {0:_sub} )

            elif not _sub._potentially_variable():
                #
                # Store a non-variable expression
                #
                if compute_values:
                    val = EXPR.evaluate_expression(_sub, only_fixed_vars=True, exception=False)
                    if val is None:
                        _result = [{-999: "Error evaluating expression: %s" % str(_sub)}] 
                    else:
                        _result.append( {0:val} )
                else:
                    _result.append( {0:_sub} )

            elif _sub.__class__ is EXPR._StaticLinearExpression:
                #
                # Extract data from the linear expression
                #
                val = {}
                constant = _sub.constant
                if len(_sub.linear_vars) > 0:
                    ans = {}
                    for c,v in zip(_sub.linear_coefs, _sub.linear_vars):
                        if v.fixed:
                            if compute_values:
                                constant += EXPR.evaluate_expression(c)*v.value
                            else:
                                constant += c*v
                        else:
                            #
                            # Store a variable 
                            #
                            id_ = id(v)
                            if id_ in idMap[None]:
                                key = idMap[None][id_]
                            else:
                                key = len(idMap) - 1
                                idMap[None][id_] = key
                                idMap[key] = v
                            if compute_values:
                                ans[key] = EXPR.evaluate_expression(c)
                            else:
                                ans[key] = c
                    val[1] = ans
                if not isclose(constant, 0):
                    val[0] = constant
                _result.append( val )
            else:
                #
                # Push an expression onto the stack
                #
                if verbose: #pragma:nocover
                    print("*"*10 + " PUSH " + "*"*10)

                _stack.append( (_obj, _argList, _idx, _len, _compute_value, _result) )

                _obj     = _sub
                _argList = _sub._args
                _idx     = 0
                _len     = _sub.nargs()
                _result  = []

        #
        # POST-DIVE
        #
        if verbose: #pragma:nocover
            print("="*30)
            print(type(_obj))
            print(_obj.to_string())
            print(_argList)
            print(_idx)
            print(_len)
            print(_compute_value)
            print(_result)
            print("STACK LEN %d" % len(_stack))

        if -999 in _result[-1]:
            #
            # "return" the recursion by putting the return value on the end of the results stack
            #
            if _stack:
                _stack[-1][-1].append( {-999:_result[-1][-999]} )
                continue
            else:
                ans = {}
                break

        if _obj.__class__ is EXPR._ViewSumExpression:
            ans = {}
            # Add nonlinear terms
            # Do some extra work to combine the arguments of 'Sum' expressions
            nonl = []
            if not linear:
                for res in _result:
                    if None in res:
                        if res[None].__class__ is EXPR._SumExpression or res[None].__class__ is EXPR._ViewSumExpression:
                            for arg in itertools.islice(res[None]._args, res[None].nargs()):
                                nonl.append(arg)
                        else:
                            nonl.append(res[None])
                if len(nonl) > 0:
                    nonl = Sum(x for x in nonl)
                    if not (nonl.__class__ in native_numeric_types and isclose(nonl,0)):
                        ans[None] = nonl
                        linear = False
            # Add constant terms
            cons = 0
            cons = 0 + sum(res[0] for res in _result if 0 in res)
            if not cons is 0:
                ans[0] = cons

            for res in _result:
                # Add linear terms
                if 1 in res:
                    if not 1 in ans:
                        ans[1] = {}
                    for key in res[1]:
                        if key in ans[1]:
                            coef = ans[1][key] + res[1][key]
                            if not (coef.__class__ in native_numeric_types and isclose(coef, 0.0)):     # coef != 0.0
                                ans[1][key] = coef
                            else:
                                del ans[1][key]
                        else:
                            ans[1][key] = res[1][key]           # We shouldn't need to check if this is zero
                # Add quadratic terms
                if quadratic and 2 in res:
                    if not 2 in ans:
                        ans[2] = {}
                    for key in res[2]:
                        if key in ans[2]:
                            coef = ans[2][key] + res[2][key]
                            if not (coef.__class__ in native_numeric_types and isclose(coef, 0.0)):     # coef != 0.0
                                ans[2][key] = coef
                            else:
                                del ans[2][key]
                        else:
                            ans[2][key] = res[2][key]           # We shouldn't need to check if this is zero

        elif _obj.__class__ is EXPR._ProductExpression or (_obj.__class__ is EXPR._PowExpression and len(_result) == 2):
            #
            # The POW expression is a special case.  This the length==2 indicates that this is a quadratic.
            #
            if _obj.__class__ is EXPR._PowExpression:
                _tmp, _l = _result
                _r = _l
            else:
                _l, _r = _result
            #print("_l")
            #print(_l)
            #print("_r")
            #print(_r)
            ans = {}
            #
            # Compute the product
            #
            # l\r   None    0       1       2
            # None  None    None    None    None
            # 0     None    0       1       2
            # 1     None    1       2       None
            # 2     None    2       None    None
            #

            #
            # GENERATING A NONLINEAR TERM
            #
            # Products that include a nonlinear term
            nonl = []
            if None in _l:
                rhs = 0
                if None in _r:
                    rhs += _r[None]
                if 0 in _r and \
                   not (_r[0].__class__ in native_numeric_types and isclose(_r[0], 0.0)):    # _r[0] != 0.0
                    rhs += _r[0]
                if 1 in _r:
                    rhs += Sum(_r[1][key]*idMap[key] for key in _r[1])
                if 2 in _r:
                    rhs += Sum(_r[2][key]*idMap[key[0]]*idMap[key[1]] for key in _r[2])
                nonl.append(_l[None]*rhs)
            if None in _r:
                lhs = 0
                if 0 in _l and \
                   not (_l[0].__class__ in native_numeric_types and isclose(_l[0], 0.0)):        # _l[0] != 0.0
                    lhs += _l[0]
                if 1 in _l:
                    lhs += Sum(_l[1][key]*idMap[key] for key in _l[1])
                if 2 in _l:
                    lhs += Sum(_l[2][key]*idMap[key[0]]*idMap[key[1]] for key in _l[2])
                nonl.append(lhs*_r[None])
            if quadratic:
                # Products that generate term with degree > 2
                if 2 in _l:
                    if 1 in _r:
                        for lkey in _l[2]:
                            v1_, v2_ = lkey
                            for rkey in _r[1]:
                                nonl.append(_l[2][lkey]*_r[1][rkey]*idMap[v1_]*idMap[v2_]*idMap[rkey])
                    if 2 in _r:
                        for lkey in _l[2]:
                            lv1_, lv2_ = lkey
                            for rkey in _r[2]:
                                rv1_, rv2_ = rkey
                                nonl.append(_l[2][lkey]*_r[2][rkey]*idMap[lv1_]*idMap[lv2_]*idMap[rv1_]*idMap[rv2_])
                if 1 in _l and 2 in _r:
                        for lkey in _l[1]:
                            for rkey in _r[2]:
                                v1_, v2_ = rkey
                                nonl.append(_l[1][lkey]*_r[2][rkey]*idMap[lkey]*idMap[v1_]*idMap[v2_])
            else:
                # Products that generate term with degree = 2
                if 1 in _l and 1 in _r:
                    # TODO: Consider creating Multsum objects here with the Sum() function
                    nonl.append( Sum(_l[1][i]*idMap[i] for i in _l[1]) * Sum(_r[1][i]*idMap[i] for i in _r[1]) )
            if len(nonl) > 0:
                nonl = Sum(x for x in nonl)
                if not (nonl.__class__ in native_numeric_types and isclose(nonl,0)):
                    ans[None] = nonl
                    linear = False

            #
            # GENERATING A CONSTANT TERM
            #
            if 0 in _l and 0 in _r:
                ans[0] = _l[0]*_r[0]

            #
            # GENERATING LINEAR TERMS
            #
            if (0 in _l and 1 in _r) or (1 in _l and 0 in _r):
                ans[1] = {}
                if 0 in _l and 1 in _r and \
                   not (_l[0].__class__ in native_numeric_types and isclose(_l[0], 0.0)):    # _l[0] != 0.0
                    for key in _r[1]:
                        ans[1][key] = _l[0]*_r[1][key]
                if 1 in _l and 0 in _r and \
                   not (_r[0].__class__ in native_numeric_types and isclose(_r[0], 0.0)):    # _r[0] != 0.0
                    for key in _l[1]:
                        if key in ans[1]:
                            ans[1][key] += _l[1][key]*_r[0]
                        else:
                            ans[1][key] = _l[1][key]*_r[0]

            #
            # GENERATING QUADRATIC TERMS
            #
            if quadratic:
                if (0 in _l and 2 in _r) or (2 in _l and 0 in _r) or (1 in _l and 1 in _r):
                    ans[2] = {}
                    if 0 in _l and 2 in _r and \
                       not (_l[0].__class__ in native_numeric_types and isclose(_l[0], 0.0)):
                        for key in _r[2]:
                            ans[2][key] = _l[0]*_r[2][key]
                    if 2 in _l and 0 in _r and \
                       not (_r[0].__class__ in native_numeric_types and isclose(_r[0], 0.0)):
                        for key in _l[2]:
                            if key in ans[2]:
                                ans[2][key] += _l[2][key]*_r[0]
                            else:
                                ans[2][key] = _l[2][key]*_r[0]
                    if 1 in _l and 1 in _r:
                        for lkey in _l[1]:
                            for rkey in _r[1]:
                                if id(idMap[lkey]) <= id(idMap[rkey]):
                                    key_ = (lkey,rkey)
                                else:
                                    key_ = (rkey,lkey)
                                if key_ in ans[2]:
                                    ans[2][key_] += _l[1][lkey]*_r[1][rkey]
                                else:
                                    ans[2][key_] = _l[1][lkey]*_r[1][rkey]

        elif _obj.__class__ is EXPR._NegationExpression:
            ans = _result[0]
            if None in ans:
                ans[None] *= -1
            if 0 in ans:
                ans[0] *= -1
            if 1 in ans:
                for i in ans[1]:
                    ans[1][i] *= -1
            if 2 in ans:
                for i in ans[2]:
                    ans[2][i] *= -1

        elif _obj.__class__ is EXPR._ReciprocalExpression:
            if None in _result[0] or 1 in _result[0] or 2 in _result[0]:
                ans = {None:_obj}
                linear = False
            else:
                ans = {0:1/_result[0][0]}

        elif _obj.__class__ is EXPR._AbsExpression or _obj.__class__ is EXPR._UnaryFunctionExpression:
            if None in _result[0] or 1 in _result[0] or 2 in _result[0]:
                ans = {None:_obj}
                linear = False
            else:
                ans = {0:_obj(_result[0][0])}

        elif _obj.__class__ is EXPR.Expr_if:
            ans = _result[0]

        else:
            try:
                assert(len(_result) == 1)
            except Exception as e:
                print("ERROR: "+str(type(_obj)))
                raise
            ans = _result[0]

        #print("ans")
        #print(ans)
        if verbose: #pragma:nocover
            print("*"*10 + " RETURN  " + "*"*10)
            print("."*30)
            print(type(_obj))
            print(_obj.to_string())
            print(_argList)
            print(_idx)
            print(_len)
            print(_compute_value)
            print(_result)
            print("STACK LEN %d" % len(_stack))

        if _stack:
            #
            # "return" the recursion by putting the return value on the end of the results stack
            #
            _stack[-1][-1].append( ans )
        else:
            break

    #
    # Create the final object here from 'ans'
    #
    repn.constant = _multiplier*ans.get(0,0)
    if 1 in ans:
        keys = list(ans[1].keys())
        repn.linear_vars  = tuple(idMap[i] for i in keys)
        repn.linear_coefs = tuple(_multiplier*ans[1][i] for i in keys)
    if 2 in ans:
        keys = list(ans[2].keys())
        repn.quadratic_vars  = tuple((idMap[v1_],idMap[v2_]) for v1_, v2_ in keys)
        repn.quadratic_coefs = tuple(_multiplier*ans[2][i] for i in keys)
    repn.nonlinear_expr = ans.get(None,None)
    if not repn.nonlinear_expr is None:
        repn.nonlinear_expr *= _multiplier
    repn.nonlinear_vars = {}
    if not repn.nonlinear_expr is None:
        repn.nonlinear_vars = []
        for v_ in EXPR.identify_variables(repn.nonlinear_expr, include_fixed=False, include_potentially_variable=False):
            repn.nonlinear_vars.append(v_)
            #
            # Update idMap in case we skipped nonlinear sub-expressions
            #
            # Q: Should we skip nonlinear sub-expressions?
            #
            id_ = id(v_)
            if not id_ in idMap[None]:
                key = len(idMap) - 1
                idMap[None][id_] = key
                idMap[key] = v_
        repn.nonlinear_vars = tuple(repn.nonlinear_vars)
    return repn


#@profile
def nonrecursive_generate_standard_repn(expr, idMap=None, compute_values=True, verbose=False, quadratic=True, repn=None):
    if quadratic:
        class Results(object):
            __slot__ = ('const', 'nonl', 'linear', 'quadratic')

            def __init__(self, const=0, nonl=0, linear=None, quadratic=None):
                self.const = const
                self.nonl = nonl
                if linear is None:
                    self.linear = {}
                else:
                    self.linear = linear
                if quadratic is None:
                    self.quadratic = {}
                else:
                    self.quadratic = quadratic

            def __str__(self):
                return "Const:\t%f\nLinear:\t%s\nQuadratic:\t%s\nNonlinear:\t%s" % (self.const, str(self.linear), str(self.quadratic), str(self.nonl))
    else:
        class Results(object):
            __slot__ = ('const', 'nonl', 'linear')

            def __init__(self, const=0, nonl=0, linear=None):
                self.const = const
                self.nonl = nonl
                if linear is None:
                    self.linear = {}
                else:
                    self.linear = linear

            def __str__(self):
                return "Const:\t%f\nLinear:\t%s\nNonlinear:\t%s" % (self.const, str(self.linear), str(self.nonl))

    ##
    ## Recurse through the expression tree, collecting variables and linear terms, etc
    ##
    linear = True
    varkeys = idMap[None]
    #
    # The stack starts with the current expression
    #
    _stack = [ [[]], [expr, expr._args, 0, expr.nargs(), []]]
    #
    # Iterate until the stack is empty
    #
    # Note: 1 is faster than True for Python 2.x
    #
    while 1:
        #
        # Get the top of the stack
        #   _obj        Current expression object
        #   _argList    The arguments for this expression objet
        #   _idx        The current argument being considered
        #   _len        The number of arguments
        #
        # Note: expressions pushed onto the stack are guaranteed to 
        # be potentially variable.
        #
        if len(_stack) == 1:
            break
        print("")
        print("STACK")
        for i in range(len(_stack)):
            print("%d %s" % (i, str(_stack[i])))
        _obj, _argList, _idx, _len, _result = _stack[-1]
        if verbose: #pragma:nocover
            print("*"*10 + " POP  " + "*"*10)
        print("TYPE")
        print(_obj.__class__)
        print("RESULT")
        print(_result)
    
        input("Hit enter...")

        # Products
        if _obj.__class__ is EXPR._ProductExpression:
            if _idx == 0:
                if _obj._args[0].__class__ in native_numeric_types or not _obj._args[0]._potentially_variable():
                    ans_ = Results()
                    _result.append(ans_)
                    ans_.const += value(_obj._args[0])
                    _idx = 1
                elif _obj._args[0].__class__ in pyomo5_variable_types:
                    _stack[-1][2] = 1
                    _stack.append( [_obj._args[0], None, None, None, []] )
                    continue
                else:
                    _stack[-1][2] = 1
                    _stack.append( [_obj._args[0], _obj._args[0]._args, 0, _obj._args[0].nargs(), []] )
                    continue
            if _idx == 1:
                if _obj._args[1].__class__ in native_numeric_types or not _obj._args[1]._potentially_variable():
                    ans_ = Results()
                    _result.append(ans_)
                    ans_.const += value(_obj._args[1])
                    _idx = 2
                elif _obj._args[1].__class__ in pyomo5_variable_types:
                    _stack[-1][2] = 2
                    _stack.append( [_obj._args[1], None, None, None, []] )
                    continue
                else:
                    _stack[-1][2] = 2
                    _stack.append( [_obj._args[1], _obj._args[1]._args, 0, _obj._args[1].nargs(), []] )
                    continue
            #
            # Multiply term "returned" from recursion
            #
            lhs, rhs = _result
            #print("LHS")
            #print(lhs)
            #print("RHS")
            #print(rhs)
            if lhs.nonl != 0 or lhs.nonl != 0:
                ans = Results(nonl=_obj)
            else:
                ans = Results()
                ans.const = lhs.const * rhs.const
                if lhs.const != 0:
                    for key, coef in six.iteritems(rhs.linear):
                        ans.linear[key] = coef*lhs.const
                if rhs.const != 0:
                    for key, coef in six.iteritems(lhs.linear):
                        if key in ans.linear:
                            ans.linear[key] += coef*rhs.const
                        else:
                            ans.linear[key] = coef*rhs.const

                if quadratic:
                    if lhs.const != 0:
                        for key, coef in six.iteritems(rhs.quadratic):
                            ans.quadratic[key] = coef*lhs.const
                    if rhs.const != 0:
                        for key, coef in six.iteritems(lhs.quadratic):
                            if key in ans.quadratic:
                                ans.quadratic[key] += coef*rhs.const
                            else:
                                ans.quadratic[key] = coef*rhs.const
                    for lkey, lcoef in six.iteritems(lhs.linear):
                        for rkey, rcoef in six.iteritems(rhs.linear):
                            ans.quadratic[lkey,rkey] = lcoef*rcoef
                    el_linear = sum(coef*idMap[key] for key, coef in six.iteritems(lhs.linear))
                    er_linear = sum(coef*idMap[key] for key, coef in six.iteritems(rhs.linear))
                    el_quadratic = sum(coef*idMap[key[0]]*idMap[key[1]] for key, coef in six.iteritems(lhs.quadratic))
                    er_quadratic = sum(coef*idMap[key[0]]*idMap[key[1]] for key, coef in six.iteritems(rhs.quadratic))
                    ans.nonl += el_linear*er_quadratic + el_quadratic*er_linear
                elif len(lhs.linear) + len(rhs.linear) > 1:
                    el_linear = sum(coef*idMap[key] for key, coef in six.iteritems(lhs.linear))
                    er_linear = sum(coef*idMap[key] for key, coef in six.iteritems(rhs.linear))
                    ans.nonl += el_linear*er_linear

            #print("HERE - prod ends")
            _stack[-2][-1].append( ans )
            _stack.pop()

        # Summation
        elif _obj.__class__ is EXPR._ViewSumExpression:
            if _idx == 0:
                ans_ = Results()
                _stack[-2][-1].append(ans_)
            else:
                #
                # Add term "returned" from recursion
                #
                ans_ = _stack[-2][-1][-1]
                res_ = _result[-1]
                ans_.const += res_.const
                ans_.nonl += res_.nonl
                for i, val in six.iteritems(res_.linear):
                    ans_.linear[i] = ans_.linear.get(i, 0) + val
                if quadratic:
                    for i, val in six.iteritems(res_.quadratic):
                        ans_.quadratic[i] = ans_.quadratic.get(i, 0) + val
            #
            # Loop through remaining terms
            #
            for i in range(_idx,_len):
                e_ = _obj._args[i]
                if e_.__class__ in pyomo5_variable_types:
                    if e_.fixed:
                        if compute_values:
                            ans_.const += e_.value
                        else:
                            ans_.const += e_
                    else:
                        id_ = id(e_)
                        if not id_ in varkeys:
                            key = len(idMap) - 1
                            varkeys[id_] = key
                            idMap[key] = e_
                        if key in ans_.linear:
                            ans_.linear[key] += 1
                        else:
                            ans_.linear[key] = 1
                elif e_.__class__ is EXPR._ProductExpression:
                    c_ = True
                    if e_._args[1].__class__ in pyomo5_variable_types:
                        if e_._args[1].fixed:
                            v1 = e_._args[1].value
                            v_ = e_._args[0]
                            if v_.__class__ in native_numeric_types:
                                ans_.const += v_ * v1
                            elif not v_._potentially_variable():
                                ans_.const += value(v_) * v1
                            elif v_.__class__ in pyomo5_variable_types:
                                if v_.fixed:
                                    ans_.const += v_.value * v1
                                else:
                                    c_ = v1
                            else:
                                c_=False
                        elif e_._args[0].__class__ in native_numeric_types:
                            c_ = e_._args[0]
                            v_ = e_._args[1]
                        elif not e_._args[0]._potentially_variable() or \
                             (e_._args[0].__class__ in pyomo5_variable_types and e_._args[0].fixed):
                            c_ = value(e_._args[0])
                            v_ = e_._args[1]
                        else:
                            c_=False
                    else:
                        c_=False
                    #
                    # Add the variable
                    #
                    if c_ is False:
                        _stack[-1][2] = i+1
                        _stack.append( [e_, e_._args, 0, e_.nargs(), []] )
                        break
                    elif not c_ is True:
                        id_ = id(v_)
                        if not id_ in varkeys:
                            key = len(idMap) - 1
                            varkeys[id_] = key
                            idMap[key] = v_
                        if key in ans_.linear:
                            ans_.linear[key] += c_
                        else:
                            ans_.linear[key] = c_
                elif e_.__class__ in native_numeric_types:
                    ans_.const += e_
                elif not e_._potentially_variable():
                    ans_.const += value(e_)
                else:
                    #print("HERE?")
                    #print(type(e_))
                    _stack[-1][2] = i+1
                    _stack.append( [e_, e_._args, 0, e_.nargs(), []] )
                    break
            else:
                #print("HERE - sum ends")
                _stack.pop()

        # Variables
        elif _obj.__class__ in pyomo5_variable_types:
            ans = Results()
            if _obj.fixed:
                if compute_values:
                    ans.const = _obj.value
                else:
                    ans.const = _obj
            else:
                id_ = id(_obj)
                key = varkeys.get(id_, None)
                if key is None:
                    key = len(idMap) - 1
                    varkeys[id_] = key
                    idMap[key] = _obj
                ans.linear[key] = 1
            _stack[-2][-1].append( ans )
            _stack.pop()

        # Constant
        elif _obj.__class__ in native_numeric_types or not _obj._potentially_variable():
            ans = Results(const = value(_obj))
            _stack[-2][-1].append( ans )
            _stack.pop()

        else:
            raise RuntimeError("Unknown expression %s" % str(_obj))
                    
    ans = _stack[-1][-1][-1]
    #
    # Create the final object here from 'ans'
    #
    repn.constant = ans.const

    keys = list(ans.linear.keys())
    repn.linear_vars = tuple(idMap[key] for key in keys)
    repn.linear_coefs = tuple(ans.linear[key] for key in keys)

    if quadratic:
        keys = list(ans.quadratic.keys())
        repn.quadratic_vars = tuple((idMap[key[0]],idMap[key[1]]) for key in keys)
        repn.quadratic_coefs = tuple(ans.quadratic[key] for key in keys)

    if not ans.nonl is 0:
        repn.nonlinear_expr = ans.nonl
        repn.nonlinear_vars = []
        for v_ in EXPR.identify_variables(repn.nonlinear_expr, include_fixed=False, include_potentially_variable=False):
            repn.nonlinear_vars.append(v_)
            #
            # Update idMap in case we skipped nonlinear sub-expressions
            #
            # Q: Should we skip nonlinear sub-expressions?
            #
            id_ = id(v_)
            if not id_ in idMap[None]:
                key = len(idMap) - 1
                idMap[None][id_] = key
                idMap[key] = v_
        repn.nonlinear_vars = tuple(repn.nonlinear_vars)

    return repn

