#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________


import pyomo.opt
from pyomo.core.base import expr, Var, Constraint, Objective
from pyomo.core.base.var import _VarData
from pyomo.core.base.param import _ParamData
from pyomo.core.base.expr import *
from pyomo.core.base.numvalue import *
from pyomo.core.base import _ExpressionData

from six.moves import xrange


class ProblemWriterUtils(pyomo.opt.AbstractProblemWriter):
    """
    Class that contains utilities for organizing model data before
    the problem is written.

    WEH: Perhaps these should be moved into AbstractProblemWriter.
    We'll see how this code evolves...

    """

    def __init__(self, problem_format):
        pyomo.opt.AbstractProblemWriter.__init__(self,problem_format)
        self._varmap={}

    def _name_fix(self, name):
        name = name.replace("\"","")
        name = name.replace("%","_")
        name = name.replace("[","_")
        name = name.replace("]","_")
        name = name.replace("(","_")
        name = name.replace(")","_")
        name = name.replace(" ","")
        name = name.replace(",","_")
        name = name.replace("-","_")
        name = name.replace("'","")
        name = name.replace("`","")
        return name

    #
    # Identify variables and confirm that the expression is linear
    #
    def _Collect1(self,exp):
        #
        # Expression
        #
        if isinstance(exp,expr._ExpressionBase):
            #
            # SumExpression
            #
            if isinstance(exp,expr._SumExpression):
                for i in xrange(len(exp._args)):
                    self._Collect1(exp._args[i])
            #
            # Identity
            #
            elif isinstance(exp,_ExpressionData):
                self._Collect1(exp.value)
            #
            # Product
            #
            elif isinstance(exp,expr._ProductExpression):
                v = "0"
                for i in xrange(len(exp._args)):
                    e = exp._args[i]
                    if isinstance(e,Var):
                        if v != "0":
                            raise ValueError("Two variables in ProductExpression: "+str(e))
                        e._data[None]._sno = 0
                    elif isinstance(e,_VarData):
                        if v != "0":
                            raise ValueError("Two variables in ProductExpression: "+str(e))
                        e._sno = 0
                    elif not isinstance(e,NumericConstant) and not isinstance(e,_ParamData):
                        print("ERROR: Unexpected item of type=" + e.__class__.__name__ + " encountered in product expression during simplification (_Collect1)")
                        print("Parent expression: ")
                        exp.pprint()
                        print("Offending item: ")
                        e.pprint()
                        raise ValueError("Unexpected item in ProductExpression - failed to simplify expression")

            elif isinstance(exp,expr._MinusExpression):
                self._Collect1(exp._args[0])
                self._Collect1(exp._args[1])
            #
            # ERROR
            #
            else:
                raise ValueError("Unsupported expression type: "+str(type(exp)))
        #
        # Variable Value
        #
        elif isinstance(exp,_VarData):
            exp._sno = 0

        # "pure" (non-indexed) variables must be handled a bit differently,
        # forcing the "_sno" on the variable value explicitly.
        elif isinstance(exp,Var):
            exp._data[None]._sno = 0
        #
        # If not a constant, then this is an error
        #
        elif not isinstance(exp,NumericConstant) and not isinstance(exp,_ParamData):
            raise ValueError("ERROR: Unexpected expression type in _Collect1: " + str(exp))


    #
    # Identify variables and confirm that the expression is linear
    #
    def _Collect2(self, exp, x, scale=1.0):
        #
        # Expression
        #
        if isinstance(exp,expr._ExpressionBase):
                #
                # Sum
                #
            if isinstance(exp,expr._SumExpression):
                for i in xrange(len(exp._args)):
                    x = self._Collect2(exp._args[i], x, scale)
            #
            # Identity
            #
            elif isinstance(exp,expr.IdentityExpression):
                x = self._Collect2(exp._args[0], x, scale)
            #
            # Product
            #
            elif isinstance(exp,expr._ProductExpression):
                c = scale
                ve = v = "0"
                for i in xrange(len(exp._args)):
                    e = exp._args[i]
                    if isinstance(e,NumericConstant) or isinstance(e,_ParamData):
                        c *= e.value
                    elif isinstance(e,Var):
                        if v != "0":
                            raise ValueError("ERROR: two variables in ProductExpression: "+str(e))
                        v = self._name_fix(e._data[None].name)
                        ve = e._data[None]
                    elif isinstance(e,_VarData):
                        if v != "0":
                            raise ValueError("ERROR: two variables in ProductExpression: "+str(e))
                        v = self._name_fix(e.name)
                        ve = e
                    else:
                        raise ValueError("ERROR: unexpected item in ProductExpression: "+str(e))
                if x.has_key(v):
                    xv = x[v]
                    x[v] = (xv[0]+c,xv[1])
                else:
                    x[v] = (c,ve)
            #
            # Minus
            #
            elif isinstance(exp,expr._MinusExpression):
                self._Collect2(exp._args[0], x, scale)
                self._Collect2(exp._args[1], x, -scale)
            #
            # ERROR
            #
            else:
                raise ValueError("Unsupported expression type: "+str(exp))
        #
        # Constant
        #
        elif isinstance(exp,NumericConstant) or isinstance(exp, _ParamData):
            c = exp.value * scale
            if x.has_key("0"):
                xv = x["0"]
                x["0"] = (xv[0]+c,xv[1])
            else:
                x["0"] = (c,"0")
        #
        # Variable
        #
        elif isinstance(exp,_VarData) or isinstance(exp,Var):
            v = self._name_fix(exp.name)
            if x.has_key(v):
                xv = x[v]
                x[v] = (xv[0] + scale, xv[1])
            else:
                x[v] = (scale, exp)
        #
        # ERROR
        #
        else:
            raise ValueError("Unexpected expression type in _Collect2: "+str(exp))
        return x


    def _Collect3(self, exp):
        x = self._Collect2(exp,{})
        y = {}
        for i in x:
            if x[i][0] != 0.:
                y[i] = x[i]
        return y


    def _Collect(self,model):
        Vars = model.component_map(Var, active=True)
        Con = model.component_map(Constraint, active=True)
        Obj = model.component_map(Objective, active=True)
        Con1 = []
        #
        # Indicate that all variables are unused
        #
        for var in Vars.values():
            for V in var._data.keys():
                var._data[V]._sno = -1
        #
        # Call _Collect1 to find the variables that are used in
        # the objective and constraints
        #
        for key in Obj.keys():
            for ondx in Obj[key]._expr:
                try:
                    self._Collect1(Obj[key]._expr[ondx])
                except ValueError:
                    e = sys.exc_info()[1]
                    print("ERROR: Detected issue during simplification of objective (method=ProblemWriterUtils::_Collect1)")
                    print("ISSUE: " + str(e))
                    if ondx == None:
                        print("")
                    else:
                        print("   Index="+str(ondx))
                    print("Expression=")
                    print(Obj[key]._expr[ondx].pprint())
                    raise ValueError
        for key in Con.keys():
            C = Con[key]
            for cndx in C.keys():
                try:
                    self._Collect1(C._body[cndx])
                except ValueError:
                    e = sys.exc_inf()[1]
                    print("ERROR: Detected issue during simplification of constraint="+C._name)
                    if cndx == None:
                        print("")
                    else:
                        print(", Index="+str(cndx))
                    print( "Expression=")
                    print( C._body[cndx].pprint())
                    raise ValueError
        #
        # Count the number of variables, and order them
        #
        sno = 0
        for var in Vars.values():
            Vv = var._data
            for V in Vv:
                if Vv[V]._sno != -1:
                    Vv[V]._sno = sno
                    self._varmap[self._name_fix(Vv[V].name)] = Vv[V]
                    sno += 1
        model.nsno = sno
        #
        # Collect the linear terms
        #
        for key in Obj.keys():
            Obj[key]._linterm = {}
            for ondx in Obj[key]._expr:
                Obj[key]._linterm[ondx] = self._Collect3(Obj[key]._expr[ondx])
        for key in Con.keys():
            C = Con[key]
            C._linterm = {}
            Cnz = []
            nt = 0
            for cndx in C.keys():
                t = C._linterm[cndx] = self._Collect3(C._body[cndx])
                lt = len(t)
                if lt > 0:
                    Cnz.append(cndx)
                    nt += 1
            if nt > 0:
                Con1.append(key)
            C._Cnz = Cnz
        model.Cnontriv = Con1
