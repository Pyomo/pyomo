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

from pyomo.pysp.plugins.phhistoryextension import load_history

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

f1 = sys.argv[1]
scenariotree1, history1, iter_keys1 = load_history(f1)
f2 = sys.argv[2]
scenariotree2, history2, iter_keys2 = load_history(f2)

if scenariotree1 != scenariotree2:
    print("Scenario Tree Mismatch")
    fail_flag = True

if iter_keys1 != iter_keys2:
    print("Total PH Iteration Mismatch: %s != %s"
          % (len(iter_keys1), len(iter_keys2)))
    fail_flag = True

valid_iters = set(iter_keys1).intersection(iter_keys2)
valid_iters = sorted([int(v) for v in valid_iters])
valid_iters = [str(k) for k in valid_iters]

no_key = object()
start_diff_key = no_key
for key in valid_iters:
    res1 = history1[key]
    res2 = history2[key]
    try:
        compare_repn(res1, res2, tolerance=_diff_tolerance)
    except:
        print("Differences begin at iteration %s"
              % (key))
        start_diff_key = key
        fail_flag = True

    if start_diff_key is not no_key:
        break

keys_to_compare = []
if start_diff_key is not no_key:
    start_idx = valid_iters.index(start_diff_key)
    keys_to_compare = valid_iters[start_idx:min(len(valid_iters),start_idx+5)]
for diff_key in keys_to_compare:
    print("\n")
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("Inspecting results at iteration %s"
          % (diff_key))
    res1 = flatten(history1[diff_key])
    res2 = flatten(history2[diff_key])
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
