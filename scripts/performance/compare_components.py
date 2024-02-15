#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2024
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

#
# This script compares build time and memory usage for
# various modeling objects. The output is organized into
# three columns. The first column is a description of the
# modeling object. The second column is the memory used by
# the modeling object (as computed using the
# pympler.asizeof.asizeof function). The third column is the
# average time to create the modeling object averaged over a
# specified number of trials.
#

# set the size of indexed objects in the experiments
N = 500

import gc
import time
import pickle

from pyomo.kernel import (
    block,
    block_list,
    variable,
    variable_list,
    variable_dict,
    constraint,
    linear_constraint,
    constraint_dict,
    constraint_list,
    matrix_constraint,
    objective,
)
from pyomo.core.kernel.variable import IVariable

from pyomo.core.base import Integers, RangeSet, Objective
from pyomo.core.base.constraint import _GeneralConstraintData, Constraint
from pyomo.core.base.var import _GeneralVarData, Var
from pyomo.core.base.block import _BlockData, Block

import numpy
import scipy
import scipy.sparse

pympler_available = True
try:
    import pympler.asizeof
except:
    pympler_available = False


pympler_kwds = {}


def _fmt(num, suffix='B'):
    """format memory output"""
    if num is None:
        return "<unknown>"
    for unit in ['', 'K', 'M', 'G', 'T', 'P', 'E', 'Z']:
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
    line = "%9s %50s %12s %7s %12s %7s"
    print(line % ("Library", "Label", "Memory", "", "Build Time", ""))
    _, _, (initial_mem_b, initial_time_s) = results[0]
    line = "%9s %50s %12s %7s %12.6f %7s"
    for i, (libname, label, (mem_b, time_s)) in enumerate(results):
        mem_factor = ""
        time_factor = ""
        if i > 0:
            if initial_mem_b is not None:
                mem_factor = "(%4.2fx)" % (float(mem_b) / initial_mem_b)
            else:
                mem_factory = ""
            time_factor = "(%4.2fx)" % (float(time_s) / initial_time_s)
        print(line % (libname, label, _fmt(mem_b), mem_factor, time_s, time_factor))


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
    return variable(domain_type=None, lb=None, ub=None)


class _staticvariable(IVariable):
    """An _example_ of a more lightweight variable."""

    _ctype = IVariable
    domain_type = None
    lb = None
    ub = None
    fixed = False
    stale = False
    __slots__ = ("value", "_parent", "_storage_key", "_active")

    def __init__(self):
        self.value = None
        self._parent = None
        self._storage_key = None
        self._active = True


def build_staticvariable():
    """Build a static variable with no references to
    external objects so its size can be computed."""
    return _staticvariable()


def build_Constraint():
    """Build a Constraint and delete any references to external
    objects so its size can be computed."""
    expr = sum(x * c for x, c in zip(build_Constraint.xlist, build_Constraint.clist))
    obj = Constraint(expr=(0, expr, 1))
    obj._parent = build_Constraint.dummy_parent
    obj.construct()
    obj._parent = None
    return obj


build_Constraint.xlist = [build_GeneralVarData() for i in range(5)]
build_Constraint.clist = [1.1 for i in range(5)]
build_Constraint.dummy_parent = lambda: None


def build_GeneralConstraintData():
    """Build a _GeneralConstraintData and delete any references to external
    objects so its size can be computed."""
    expr = sum(x * c for x, c in zip(build_Constraint.xlist, build_Constraint.clist))
    return _GeneralConstraintData(expr=(0, expr, 1))


build_Constraint.xlist = [build_GeneralVarData() for i in range(5)]
build_Constraint.clist = [1.1 for i in range(5)]


def build_constraint():
    """Build a constraint with no references to external
    objects so its size can be computed."""
    expr = sum(x * c for x, c in zip(build_constraint.xlist, build_constraint.clist))
    return constraint(lb=0, body=expr, ub=1)


build_constraint.xlist = [build_variable() for i in range(5)]
build_constraint.clist = [1.1 for i in range(5)]


def build_linear_constraint():
    """Build a linear_constraint with no references to external
    objects so its size can be computed."""
    return linear_constraint(
        variables=build_linear_constraint.xlist,
        coefficients=build_linear_constraint.clist,
        lb=0,
        ub=1,
    )


build_linear_constraint.xlist = [build_variable() for i in range(5)]
build_linear_constraint.clist = [1.1 for i in range(5)]


def _bounds_rule(m, i):
    return (None, None)


def _initialize_rule(m, i):
    return None


def _reset():
    build_indexed_Var.model = Block(concrete=True)
    build_indexed_Var.model.ndx = RangeSet(0, N - 1)
    build_indexed_Var.bounds_rule = _bounds_rule
    build_indexed_Var.initialize_rule = _initialize_rule


def build_indexed_Var():
    """Build an indexed Var with no references to external
    objects so its size can be computed."""
    model = build_indexed_Var.model
    model.indexed_Var = Var(
        model.ndx,
        domain=Integers,
        bounds=build_indexed_Var.bounds_rule,
        initialize=build_indexed_Var.initialize_rule,
    )
    model.indexed_Var._domain = None
    model.indexed_Var._component = None
    return model.indexed_Var


build_indexed_Var.reset_for_test = _reset
build_indexed_Var.reset_for_test()


def build_variable_dict():
    """Build a variable_dict with no references to external
    objects so its size can be computed."""
    return variable_dict(
        (
            (i, variable(domain_type=None, lb=None, ub=None, value=None))
            for i in range(N)
        )
    )


def build_variable_list():
    """Build a variable_list with no references to external
    objects so its size can be computed."""
    return variable_list(
        variable(domain_type=None, lb=None, ub=None, value=None) for i in range(N)
    )


def build_staticvariable_list():
    """Build a variable_list of static variables with no
    references to external objects so its size can be
    computed."""
    return variable_list(_staticvariable() for i in range(N))


A = scipy.sparse.random(N, N, density=0.2, format='csr', dtype=float)
b = numpy.ones(N)
# as lists
A_data = A.data.tolist()
A_indices = A.indices.tolist()
A_indptr = A.indptr.tolist()
# vars
X_aml = [build_GeneralVarData() for i in range(N)]
X_kernel = [build_variable() for i in range(N)]


def _con_rule(m, i):
    # expr == rhs
    return (
        sum(
            A_data[p] * X_aml[A_indices[p]] for p in range(A_indptr[i], A_indptr[i + 1])
        ),
        1,
    )


def _reset():
    build_indexed_Constraint.model = Block(concrete=True)
    build_indexed_Constraint.model.ndx = RangeSet(0, N - 1)
    build_indexed_Constraint.rule = _con_rule


def build_indexed_Constraint():
    """Build an indexed Constraint with no references to external
    objects so its size can be computed."""
    model = build_indexed_Constraint.model
    model.indexed_Constraint = Constraint(model.ndx, rule=build_indexed_Constraint.rule)
    model.indexed_Constraint._component = None
    return model.indexed_Constraint


build_indexed_Constraint.reset_for_test = _reset
build_indexed_Constraint.reset_for_test()


def build_constraint_dict():
    """Build a constraint_dict with no references to external
    objects so its size can be computed."""
    return constraint_dict(
        (
            (
                i,
                constraint(
                    rhs=1,
                    body=sum(
                        A_data[p] * X_kernel[A_indices[p]]
                        for p in range(A_indptr[i], A_indptr[i + 1])
                    ),
                ),
            )
            for i in range(N)
        )
    )


def build_constraint_list():
    """Build a constraint_list with no references to external
    objects so its size can be computed."""
    return constraint_list(
        constraint(
            rhs=1,
            body=sum(
                A_data[p] * X_kernel[A_indices[p]]
                for p in range(A_indptr[i], A_indptr[i + 1])
            ),
        )
        for i in range(N)
    )


def build_linear_constraint_list():
    """Build a constraint_list of linear_constraints with no references to external
    objects so its size can be computed."""
    return constraint_list(
        linear_constraint(
            variables=(
                X_kernel[A_indices[p]] for p in range(A_indptr[i], A_indptr[i + 1])
            ),
            coefficients=(A_data[p] for p in range(A_indptr[i], A_indptr[i + 1])),
            rhs=1,
        )
        for i in range(N)
    )


def build_matrix_constraint():
    """Build a constraint_list with no references to external
    objects so its size can be computed."""
    return matrix_constraint(A, rhs=b, x=X_kernel)


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


def build_block():
    b = block()
    b._activate_large_storage_mode()
    return b


build_small_block = block


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
    b._activate_large_storage_mode()
    b.x = build_variable()
    b.c = constraint()
    b.o = objective()
    return b


def build_small_block_with_objects():
    """Build an empty block."""
    b = block()
    b.x = build_variable()
    b.c = constraint()
    b.o = objective()
    return b


def _indexed_Block_rule(b, i):
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
    return b


def _reset():
    build_indexed_BlockWVars.model = Block(concrete=True)
    build_indexed_BlockWVars.model.ndx = RangeSet(0, N - 1)
    build_indexed_BlockWVars.indexed_Block_rule = _indexed_Block_rule


def build_indexed_BlockWVars():
    model = build_indexed_BlockWVars.model
    model.indexed_Block = Block(
        model.ndx, rule=build_indexed_BlockWVars.indexed_Block_rule
    )
    return model.indexed_Block


build_indexed_BlockWVars.reset_for_test = _reset
build_indexed_BlockWVars.reset_for_test()


def build_block_list_with_variables():
    blist = block_list()
    for i in range(N):
        b = block()
        b._activate_large_storage_mode()
        b.x1 = variable(domain_type=None, lb=None, ub=None)
        b.x2 = variable(domain_type=None, lb=None, ub=None)
        b.x3 = variable(domain_type=None, lb=None, ub=None)
        b.x4 = variable(domain_type=None, lb=None, ub=None)
        b.x5 = variable(domain_type=None, lb=None, ub=None)
        blist.append(b)
    return blist


def _get_small_block():
    b = block()
    b.x1 = variable(domain_type=None, lb=None, ub=None)
    b.x2 = variable(domain_type=None, lb=None, ub=None)
    b.x3 = variable(domain_type=None, lb=None, ub=None)
    b.x4 = variable(domain_type=None, lb=None, ub=None)
    b.x5 = variable(domain_type=None, lb=None, ub=None)
    return b


def build_small_block_list_with_variables():
    return block_list(build_small_block_list_with_variables.myblock() for i in range(N))


build_small_block_list_with_variables.myblock = _get_small_block


def _get_small_block_wstaticvars():
    myvar = _staticvariable
    b = block()
    b.x1 = myvar()
    b.x2 = myvar()
    b.x3 = myvar()
    b.x4 = myvar()
    b.x5 = myvar()
    return b


def build_small_block_list_with_staticvariables():
    return block_list(
        build_small_block_list_with_staticvariables.myblock() for i in range(N)
    )


build_small_block_list_with_staticvariables.myblock = _get_small_block_wstaticvars


if __name__ == "__main__":
    #
    # Compare construction time of different variable
    # implementations
    #
    results = []
    results.append(("AML", "Var", measure(build_Var)))
    results.append(("AML", "_GeneralVarData", measure(build_GeneralVarData)))
    results.append(("Kernel", "variable", measure(build_variable)))
    results.append(("<locals>", "staticvariable", measure(build_staticvariable)))
    summarize(results)
    print("")

    #
    # Compare construction time of different variable
    # container implementations
    #
    results = []
    results.append(("AML", "Indexed Var (%s)" % N, measure(build_indexed_Var)))
    results.append(("Kernel", "variable_dict (%s)" % N, measure(build_variable_dict)))
    results.append(("Kernel", "variable_list (%s)" % N, measure(build_variable_list)))
    results.append(
        ("<locals>", "staticvariable_list (%s)" % N, measure(build_staticvariable_list))
    )
    summarize(results)
    print("")

    #
    # Compare construction time of different constraint
    # implementations
    #
    results = []
    results.append(
        ("AML", "Constraint(<linear_expression>)", measure(build_Constraint))
    )
    results.append(
        (
            "AML",
            "_GeneralConstraintData(<linear_expression>)",
            measure(build_GeneralConstraintData),
        )
    )
    results.append(
        ("Kernel", "constraint(<linear_expression>)", measure(build_constraint))
    )
    results.append(("Kernel", "linear_constraint", measure(build_linear_constraint)))
    summarize(results)
    print("")

    #
    # Compare construction time of different constraint
    # container implementations
    #
    results = []
    results.append(
        ("AML", "Indexed Constraint (%s)" % N, measure(build_indexed_Constraint))
    )
    results.append(
        ("Kernel", "constraint_dict (%s)" % N, measure(build_constraint_dict))
    )
    results.append(
        ("Kernel", "constraint_list (%s)" % N, measure(build_constraint_list))
    )
    results.append(
        (
            "Kernel",
            "linear_constraint_list (%s)" % N,
            measure(build_linear_constraint_list),
        )
    )
    results.append(
        ("Kernel", "matrix_constraint (%s)" % N, measure(build_matrix_constraint))
    )
    summarize(results)
    print("")

    #
    # Compare construction time of different block
    # container implementations
    #
    results = []
    results.append(("AML", "Block", measure(build_Block)))
    results.append(("AML", "_BlockData", measure(build_BlockData)))
    results.append(("Kernel", "block", measure(build_block)))
    results.append(("Kernel", "small_block", measure(build_small_block)))
    summarize(results)
    print("")

    results = []
    results.append(("AML", "Block w/ 3 components", measure(build_Block_with_objects)))
    results.append(
        ("AML", "_BlockData w/ 3 components", measure(build_BlockData_with_objects))
    )
    results.append(
        ("Kernel", "block w/ 3 components", measure(build_block_with_objects))
    )
    results.append(
        (
            "Kernel",
            "small_block w/ 3 components",
            measure(build_small_block_with_objects),
        )
    )
    summarize(results)
    print("")

    #
    # Compare construction time of different block
    # container implementations
    #
    results = []
    results.append(
        ("AML", "Indexed Block (%s) w/ Vars (5)" % N, measure(build_indexed_BlockWVars))
    )
    results.append(
        (
            "Kernel",
            "block_list (%s) w/ variables (5)" % N,
            measure(build_block_list_with_variables),
        )
    )
    results.append(
        (
            "Kernel",
            "small_block_list (%s) w/ variables (5)" % N,
            measure(build_small_block_list_with_variables),
        )
    )
    results.append(
        (
            "<locals>",
            "small_block_list (%s) w/ staticvariables (5)" % N,
            measure(build_small_block_list_with_staticvariables),
        )
    )
    summarize(results)
