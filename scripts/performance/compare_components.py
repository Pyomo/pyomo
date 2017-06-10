import gc
import time
import pickle

from pyomo.kernel import (block,
                          tiny_block,
                          block_list,
                          variable,
                          variable_list,
                          variable_dict,
                          constraint,
                          objective)
from pyomo.core.kernel.component_variable import IVariable

from pyomo.core.base import Integers, RangeSet, Constraint, Objective
from pyomo.core.base.var import _GeneralVarData, Var
from pyomo.core.base.block import _BlockData, Block


pympler_available = True
try:
    import pympler.asizeof
except:
    pympler_available = False

import six
from six.moves import xrange

# set the size of various experiments
# (I think they scale with N^2)
N = 50

pympler_kwds = {}

def _fmt(num, suffix='B'):
    """format memory output"""
    if num is None:
        return "<unknown>"
    for unit in ['','K','M','G','T','P','E','Z']:
        if abs(num) < 1000.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1000.0
    return "%.1f %s%s" % (num, 'Yi', suffix)

def measure(f, n=25):
    """measure average execution time over n trials"""
    gc.collect()
    time_seconds = 0
    for i in range(n):
        gc.collect()
        start = time.time()
        obj = f()
        stop = time.time()
        if hasattr(f, "reset_for_test"):
            f.reset_for_test()
        gc.collect()
        time_seconds += stop - start
    time_seconds /= float(n)
    if pympler_available:
        mem_bytes = pympler.asizeof.asizeof(obj, **pympler_kwds)
    else:
        mem_bytes = None
    gc.collect()
    return mem_bytes, time_seconds

def summarize(results):
    """neatly summarize output for comparison of several tests"""
    line = "%50s %12s %7s %12s %7s"
    print(line % ("Label", "Bytes", "", "Seconds", ""))
    _, (initial_mem_b, initial_time_s) = results[0]
    line = "%50s %12s %7s %12.6f %7s"
    for i, (label, (mem_b, time_s)) in enumerate(results):
        mem_factor = ""
        time_factor = ""
        if i > 0:
            if initial_mem_b is not None:
                mem_factor = "(%4.2fx)" % (float(mem_b)/initial_mem_b)
            else:
                mem_factory = ""
            time_factor = "(%4.2fx)" % (float(time_s)/initial_time_s)
        print(line % (label,
                      _fmt(mem_b),
                      mem_factor,
                      time_s,
                      time_factor))

def build_Var():
    """Build a Var and delete any references to external
    objects so its size can be computed."""
    obj = Var()
    obj.construct()
    obj._domain = None
    return obj

def build_GeneralVarData():
    """Build a _GeneralVarData and delete any references to
    external objects so its size can be computed."""
    obj = _GeneralVarData()
    obj._domain = None
    return obj

def build_variable():
    """Build a variable with no references to external
    objects so its size can be computed."""
    return variable(domain_type=None,
                    lb=None,
                    ub=None)

class _staticvariable(IVariable):
    """A more lightweight variable."""
    _ctype = Var
    domain_type = None
    lb = None
    ub = None
    fixed = False
    stale = False
    __slots__ = ("value","_parent")
    def __init__(self):
        self.value = None
        self._parent = None

def build_staticvariable():
    """Build a static variable with no references to
    external objects so its size can be computed."""
    return _staticvariable()

def _bounds_rule(m, i, j):
    return (None, None)
def _initialize_rule(m, i, j):
    return None
def _reset():
    build_indexed_Var.model = Block(concrete=True)
    build_indexed_Var.model.ndx = RangeSet(0, N-1)
    build_indexed_Var.bounds_rule = _bounds_rule
    build_indexed_Var.initialize_rule = _initialize_rule
def build_indexed_Var():
    """Build an indexed Var with no references to external
    objects so its size can be computed."""
    model = build_indexed_Var.model
    model.indexed_Var = Var(model.ndx*model.ndx,
                            domain=Integers,
                            bounds=build_indexed_Var.bounds_rule,
                            initialize=build_indexed_Var.initialize_rule)
    model.indexed_Var._domain = None
    model.indexed_Var._component = None
    return model.indexed_Var
build_indexed_Var.reset_for_test = _reset
build_indexed_Var.reset_for_test()

def build_variable_dict():
    """Build a variable_dict with no references to external
    objects so its size can be computed."""
    return variable_dict(
        ((i,j), variable(domain_type=None, lb=None, ub=None, value=None))
        for i in xrange(N)
        for j in xrange(N))

def build_variable_list():
    """Build a variable_list with no references to external
    objects so its size can be computed."""
    return variable_list(
        variable(domain_type=None, lb=None, ub=None, value=None)
        for i in xrange(N)
        for j in xrange(N))

def build_staticvariable_list():
    """Build a variable_list of static variables with no
    references to external objects so its size can be
    computed."""
    return variable_list(_staticvariable()
                         for i in xrange(N)
                         for j in xrange(N))

def build_Block():
    """Build a Block with a few components."""
    obj = Block(concrete=True)
    obj.construct()
    return obj

def build_BlockData():
    """Build a _BlockData with a few components."""
    obj = _BlockData(build_BlockData.owner)
    obj._component = None
    return obj
build_BlockData.owner = Block()

build_block = block

build_tiny_block = tiny_block

def build_Block_with_objects():
    """Build an empty Block"""
    obj = Block(concrete=True)
    obj.construct()
    obj.x = Var()
    obj.x._domain = None
    obj.c = Constraint()
    obj.o = Objective()
    return obj

def build_BlockData_with_objects():
    """Build an empty _BlockData"""
    obj = _BlockData(build_BlockData_with_objects.owner)
    obj.x = Var()
    obj.x._domain = None
    obj.c = Constraint()
    obj.o = Objective()
    obj._component = None
    return obj
build_BlockData_with_objects.owner = Block()

def build_block_with_objects():
    """Build an empty block."""
    b = block()
    b.x = build_variable()
    b.c = constraint()
    b.o = objective()
    return b

def build_tiny_block_with_objects():
    """Build an empty block."""
    b = tiny_block()
    b.x = build_variable()
    b.c = constraint()
    b.o = objective()
    return b

def _indexed_Block_rule(b, i, j):
    b.x1 = Var()
    b.x1._domain = None
    b.x2 = Var()
    b.x2._domain = None
    b.x3 = Var()
    b.x3._domain = None
    b.x4 = Var()
    b.x4._domain = None
    b.x5 = Var()
    b.x5._domain = None
    b.x6 = Var()
    b.x6._domain = None
    b.x7 = Var()
    b.x7._domain = None
    b.x8 = Var()
    b.x8._domain = None
    return b
def _reset():
    build_indexed_BlockWVars.model = Block(concrete=True)
    build_indexed_BlockWVars.model.ndx = RangeSet(0, N-1)
    build_indexed_BlockWVars.indexed_Block_rule = _indexed_Block_rule
def build_indexed_BlockWVars():
    model = build_indexed_BlockWVars.model
    model.indexed_Block = Block(model.ndx,
                                model.ndx,
                                rule=build_indexed_BlockWVars.indexed_Block_rule)
    return model.indexed_Block
build_indexed_BlockWVars.reset_for_test = _reset
build_indexed_BlockWVars.reset_for_test()

def build_block_list_with_variables():
    blist = block_list()
    for i in xrange(N):
        for j in xrange(N):
            b = block()
            b.x1 = variable(domain_type=None, lb=None, ub=None)
            b.x2 = variable(domain_type=None, lb=None, ub=None)
            b.x3 = variable(domain_type=None, lb=None, ub=None)
            b.x4 = variable(domain_type=None, lb=None, ub=None)
            b.x5 = variable(domain_type=None, lb=None, ub=None)
            b.x6 = variable(domain_type=None, lb=None, ub=None)
            b.x7 = variable(domain_type=None, lb=None, ub=None)
            b.x8 = variable(domain_type=None, lb=None, ub=None)
            blist.append(b)
    return blist

def _get_tiny_block():
    b = tiny_block()
    b.x1 = variable(domain_type=None, lb=None, ub=None)
    b.x2 = variable(domain_type=None, lb=None, ub=None)
    b.x3 = variable(domain_type=None, lb=None, ub=None)
    b.x4 = variable(domain_type=None, lb=None, ub=None)
    b.x5 = variable(domain_type=None, lb=None, ub=None)
    b.x6 = variable(domain_type=None, lb=None, ub=None)
    b.x7 = variable(domain_type=None, lb=None, ub=None)
    b.x8 = variable(domain_type=None, lb=None, ub=None)
    return b

def build_tiny_block_list_with_variables():
    return block_list(
        build_tiny_block_list_with_variables.myblock()
        for i in xrange(N)
        for j in xrange(N))
build_tiny_block_list_with_variables.myblock = _get_tiny_block

def _get_tiny_block_wstaticvars():
    myvar = _staticvariable
    b = tiny_block()
    b.x1 = myvar()
    b.x2 = myvar()
    b.x3 = myvar()
    b.x4 = myvar()
    b.x5 = myvar()
    b.x6 = myvar()
    b.x7 = myvar()
    b.x8 = myvar()
    return b

def build_tiny_block_list_with_staticvariables():
    return block_list(
        build_tiny_block_list_with_staticvariables.myblock()
        for i in xrange(N)
        for j in xrange(N))
build_tiny_block_list_with_staticvariables.myblock = _get_tiny_block_wstaticvars


if __name__ == "__main__":

    #
    # Compare construction time of different variable
    # implementations
    #
    results = []
    results.append(("Var", measure(build_Var)))
    results.append(("_GeneralVarData", measure(build_GeneralVarData)))
    results.append(("variable", measure(build_variable)))
    results.append(("staticvariable", measure(build_staticvariable)))
    summarize(results)
    print("")

    #
    # Compare construction time of different variable
    # container implementations
    #
    results = []
    results.append(("Indexed Var (%s)" % (N*N), measure(build_indexed_Var)))
    results.append(("variable_dict (%s)" % (N*N), measure(build_variable_dict)))
    results.append(("variable_list (%s)" % (N*N), measure(build_variable_list)))
    results.append(("staticvariable_list (%s)" % (N*N), measure(build_staticvariable_list)))
    summarize(results)
    print("")

    #
    # Compare construction time of different block
    # container implementations
    #
    results = []
    results.append(("Block", measure(build_Block)))
    results.append(("_BlockData", measure(build_BlockData)))
    results.append(("block", measure(build_block)))
    results.append(("tiny_block", measure(build_tiny_block)))
    summarize(results)
    print("")

    results = []
    results.append(("Block w/ 3 components", measure(build_Block_with_objects)))
    results.append(("_BlockData w/ 3 components", measure(build_BlockData_with_objects)))
    results.append(("block w/ 3 components", measure(build_block_with_objects)))
    results.append(("tiny_block w/ 3 components", measure(build_tiny_block_with_objects)))
    summarize(results)
    print("")

    #
    # Compare construction time of different block
    # container implementations
    #
    results = []
    results.append(("Indexed Block (%s) w/ Vars (8)" % (N*N),
                    measure(build_indexed_BlockWVars)))
    results.append(("block_list (%s) w/ variables (8)" % (N*N),
                    measure(build_block_list_with_variables)))
    results.append(("tiny_block_list (%s) w/ variables (8)" % (N*N),
                    measure(build_tiny_block_list_with_variables)))
    results.append(("tiny_block_list (%s) w/ staticvariables (8)" % (N*N),
                    measure(build_tiny_block_list_with_staticvariables)))
    summarize(results)
