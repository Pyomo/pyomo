#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

__all__ = ['tanh', 'arcsinh', 'arccosh', 'arctanh']

try:
    import numpy as np
except:
    pass
try:
    from pyomo.openopt.FuncDesigner import oofun, ooarray, sqrt, FDmisc
except:
    pass

def tanh(inp):
    if isinstance(inp, ooarray) and inp.dtype == object:
        return ooarray([tanh(elem) for elem in inp])
    if not isinstance(inp, oofun):
        return np.tanh(inp)
    # TODO: move it outside of tanh definition
    def interval(arg_inf, arg_sup):
        raise 'interval for tanh is unimplemented yet'
    r = oofun(np.tanh, inp, d = lambda x: FDmisc.Diag(1.0/np.cosh(x) ** 2), vectorized = True, interval = interval)
    return r

def arcsinh(inp):
    if isinstance(inp, ooarray) and inp.dtype == object:
        return ooarray([arcsinh(elem) for elem in inp])
    if not isinstance(inp, oofun):
        return np.arcsinh(inp)
    # TODO: move it outside of arcsinh definition
    def interval(arg_inf, arg_sup):
        raise 'interval for arcsinh is unimplemented yet'
    r = oofun(np.arcsinh, inp, d = lambda x: FDmisc.Diag(1.0/sqrt(x**2 + 1)), vectorized = True, interval = interval)
    return r

def arccosh(inp):
    if isinstance(inp, ooarray) and inp.dtype == object:
        return ooarray([arccosh(elem) for elem in inp])
    if not isinstance(inp, oofun):
        return np.arccosh(inp)
    # TODO: move it outside of arccosh definition
    def interval(arg_inf, arg_sup):
        raise 'interval for arccosh is unimplemented yet'
    r = oofun(np.arccosh, inp, d = lambda x: FDmisc.Diag(1.0/sqrt(x**2 - 1)), vectorized = True, interval = interval)
    return r

def arctanh(inp):
    if isinstance(inp, ooarray) and inp.dtype == object:
        return ooarray([arctanh(elem) for elem in inp])
    if not isinstance(inp, oofun):
        return np.arctanh(inp)
    # TODO: move it outside of arctanh definition
    def interval(arg_inf, arg_sup):
        raise 'interval for arctanh is unimplemented yet'
    r = oofun(np.arctanh, inp, d = lambda x: FDmisc.Diag(1.0/(1 - x**2)), vectorized = True, interval = interval)
    return r

