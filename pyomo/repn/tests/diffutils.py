#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import os

import pyomo.core.expr as EXPR


def compare_floats(base, test, abstol=1e-14, reltol=1e-14):
    base = base.split()
    test = test.split()
    if len(base) != len(test):
        return False
    for i, b in enumerate(base):
        if b.strip() == test[i].strip():
            continue
        try:
            b = float(b)
            t = float(test[i])
        except:
            return False
        if abs(b - t) < abstol:
            continue
        if abs((b - t) / max(abs(b), abs(t))) < reltol:
            continue
        return False
    return True


def load_baseline(baseline, testfile, extension, version):
    with open(testfile, 'r') as FILE:
        test = FILE.read()
    if baseline.endswith(f'.{extension}'):
        _tmp = [baseline[:-3]]
    else:
        _tmp = baseline.split(f'.{extension}.', 1)
    _tmp.insert(1, f'expr{int(EXPR.Mode.CURRENT)}')
    _tmp.insert(2, version)
    if not os.path.exists('.'.join(_tmp)):
        _tmp.pop(1)
        if not os.path.exists('.'.join(_tmp)):
            _tmp = []
    if _tmp:
        baseline = '.'.join(_tmp)
    with open(baseline, 'r') as FILE:
        base = FILE.read()
    return base, test, baseline, testfile
