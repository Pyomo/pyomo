#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

__all__ = ['Pyomo2FuncDesigner']

import logging
import six
try:
    import FuncDesigner
    FD_available=True
except ImportError:
    FD_available=False

from pyomo.core.base import SymbolMap, NumericLabeler
from pyomo.core.base import Constraint, Objective, Var
from pyomo.core.base import expr, var
from pyomo.core.base import param
from pyomo.core.base import numvalue
from pyomo.core.base import _ExpressionData

try:
    long
    intlist = [int, float, long]
except:
    intlist = [int, float]

logger = logging.getLogger('pyomo.core')

labeler = NumericLabeler("x")

if FD_available:

    try:
        tanh = FuncDesigner.tanh
        arcsinh = FuncDesigner.arcsinh
        arccosh = FuncDesigner.arccosh
        arctanh = FuncDesigner.arctanh
    except:
        import FuncDesignerExt
        tanh = FuncDesignerExt.tanh
        arcsinh = FuncDesignerExt.arcsinh
        arccosh = FuncDesignerExt.arccosh
        arctanh = FuncDesignerExt.arctanh

    def fd_pow(x,y):
        return x**y

    intrinsic_function_expressions = {
        'log':FuncDesigner.log,
        'log10':FuncDesigner.log10,
        'sin':FuncDesigner.sin,
        'cos':FuncDesigner.cos,
        'tan':FuncDesigner.tan,
        'sinh':FuncDesigner.sinh,
        'cosh':FuncDesigner.cosh,
        'tanh':FuncDesigner.tanh,
        'asin':FuncDesigner.arcsin,
        'acos':FuncDesigner.arccos,
        'atan':FuncDesigner.arctan,
        'exp':FuncDesigner.exp,
        'sqrt':FuncDesigner.sqrt,
        'asinh':FuncDesigner.arcsinh,
        'acosh':arccosh,
        'atanh':arctanh,
        'pow':fd_pow,
        'abs':FuncDesigner.abs,
        'ceil':FuncDesigner.ceil,
        'floor':FuncDesigner.floor
    }


id_counter=0

def Pyomo2FD_expression(exp, ipoint, vars, symbol_map):
    if isinstance(exp, expr._IntrinsicFunctionExpression):
        if not exp.cname() in intrinsic_function_expressions:
            logger.error("Unsupported intrinsic function (%s)", exp.cname(True))
            raise TypeError("FuncDesigner does not support '{0}' expressions".format(exp.cname(True)))

        args = []
        for child_exp in exp._args:
            args.append( Pyomo2FD_expression(child_exp, ipoint, vars, symbol_map) )

        fn = intrinsic_function_expressions[exp.cname()]
        return fn(*tuple(args))

    elif isinstance(exp, expr._SumExpression):
        args = []
        for child_exp in exp._args:
            args.append( Pyomo2FD_expression(child_exp, ipoint, vars, symbol_map) )

        iargs = args.__iter__()
        #
        # NOTE: this call to FuncDesigner.sum() _must_ be passed a list.  If a 
        # generator is passed to this function, then an unbalanced expression tree will
        # be generated that is not well-suited for large models!
        #
        if six.PY2:
            return FuncDesigner.sum([c*iargs.next() for c in exp._coef]) + exp._const
        else:
            return FuncDesigner.sum([c*next(iargs) for c in exp._coef]) + exp._const

    elif isinstance(exp, expr._ProductExpression):
        ans = exp._coef
        for n in exp._numerator:
            ans *= Pyomo2FD_expression(n, ipoint, vars, symbol_map)
        for n in exp._denominator:
            ans /= Pyomo2FD_expression(n, ipoint, vars, symbol_map)
        return ans

    #elif isinstance(exp, expr._InequalityExpression):
        #args = []
        #for child_exp in exp._args:
            #args.append( Pyomo2FD_expression(child_exp, ipoint, vars, symbol_map) )
#
        #ans = args[0]
        #for i in xrange(len(args)-1):
            ## FD doesn't care whether the inequality is strict
            #ans = ans < args[i+1]
        #return ans

    #elif isinstance(exp, expr._InequalityExpression):
        #return Pyomo2FD_expression(exp._args[0], ipoint, vars) == Pyomo2FD_expression(exp._args[1], ipoint, vars, symbol_map)

    elif isinstance(exp, _ExpressionData):
        return Pyomo2FD_expression(exp._args[0], ipoint, vars, symbol_map)

    elif (isinstance(exp,var._VarData) or isinstance(exp,var.Var)) and not exp.is_fixed():
        vname = symbol_map.getSymbol(exp, labeler)
        if not vname in vars:
            vars[vname] = FuncDesigner.oovar(vname)
            ipoint[vars[vname]] = 0.0 if exp.value is None else exp.value
            #symbol_map.getSymbol(exp, lambda obj,x: x, vname)
        return vars[vname]

    elif isinstance(exp,param._ParamData):
        return exp.value

    elif type(exp) in intlist:
        return exp

    elif isinstance(exp,numvalue.NumericConstant) or exp.is_fixed():
        return exp.value

    else:
        raise ValueError("Unsupported expression type in Pyomo2FD_expression: "+str(type(exp)))


def Pyomo2FuncDesigner(instance):
    if not FD_available:
        return None

    ipoint = {}
    vars = {}
    sense = None
    nobj = 0
    smap = SymbolMap()

    _f_name = []
    _f = []
    _c = []
    for con in instance.component_data_objects(Constraint, active=True):
        body = Pyomo2FD_expression(con.body, ipoint, vars, smap)
        if not con.lower is None:
            lower = Pyomo2FD_expression(con.lower, ipoint, vars, smap)
            _c.append( body > lower )
        if not con.upper is None:
            upper = Pyomo2FD_expression(con.upper, ipoint, vars, smap)
            _c.append( body < upper )

    for var in instance.component_data_objects(Var, active=True):
        body = Pyomo2FD_expression(var, ipoint, vars, smap)
        if not var.lb is None:
            lower = Pyomo2FD_expression(var.lb, ipoint, vars, smap)
            _c.append( body > lower )
        if not var.ub is None:
            upper = Pyomo2FD_expression(var.ub, ipoint, vars, smap)
            _c.append( body < upper )


    for obj in instance.component_data_objects(Objective, active=True):
        nobj += 1
        if obj.is_minimizing():
            _f.append( Pyomo2FD_expression(obj.expr, ipoint, vars, smap) )
        else:
            _f.append( - Pyomo2FD_expression(obj.expr, ipoint, vars, smap) )
        _f_name.append( obj.cname(True) )
        smap.getSymbol(obj, lambda objective: objective.cname(True))

    # TODO - use 0.0 for default values???
    # TODO - create results map
    S = FuncDesigner.oosystem()
    S._symbol_map = smap
    S.f = _f[0]
    S._f_name = _f_name
    S.constraints.update(_c)
    S.initial_point = ipoint
    S.sense = sense
    return S

