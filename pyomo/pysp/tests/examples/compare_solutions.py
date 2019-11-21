#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import math
import sys

from pyutilib.misc import compare_repn

from pyomo.pysp.plugins.phhistoryextension import load_solution

import six

if six.PY3:
    from collections.abc import MutableMapping as collections_MutableMapping
else:
    from collections import MutableMapping as collections_MutableMapping


assert len(sys.argv) == 3

_diff_tolerance = 1e-6
fail_flag = False

def flatten(d, parent_key=''):
    items = []
    for k, v in d.items():
        new_key = parent_key + '_' + k if parent_key else k
        if v and isinstance(v, collections_MutableMapping):
            items.extend(flatten(v, new_key).items())
        else:
            items.append((new_key, v))
    return dict(items)

st1, r1 = load_solution(sys.argv[1])

st2, r2 = load_solution(sys.argv[2])

if st1 != st2:
    print("Scenario Tree Mismatch")
    fail_flag = True

try:
    compare_repn(r1, r2, tolerance=_diff_tolerance)
except (ValueError, AssertionError):
    print("Differences found in solutions. Message:\n")
    print(sys.exc_info()[1])
    fail_flag = True

if fail_flag:
    print("Inspecting flattened solutions")
    res1 = flatten(r1)
    res2 = flatten(r2)
    all_keys = set(res1.keys()).union(set(res2.keys()))
    res1_missing_keys = all_keys-set(res1.keys())
    res2_missing_keys = all_keys-set(res2.keys())
    if len(res1_missing_keys):
        print("Missing keys from results: %s"
              % (res1_missing_keys))
    if len(res2_missing_keys):
        print("Missing keys from results: %s"
              % (res2_missing_keys))
    for key in sorted(all_keys):
        val1 = res1[key]
        val2 = res2[key]
        if (type(val1) is float or type(val2) is float) and \
             type(val1) in [int,float] \
             and type(val2) in [int,float]:
            if math.fabs(val1-val2) > _diff_tolerance:
                print(key)
                print("\t "+str(val1))
                print("\t "+str(val2))
        else:
            if val1 != val2:
                print(key)
                print("\t "+str(val1))
                print("\t "+str(val2))

if fail_flag:
    print("\n")
    print("THERE WAS A FAILURE")
    print("\n")
else:
    print("\n")
    print("ALL CHECKS PASS")
    print("\n")
