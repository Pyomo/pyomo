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
import re

from difflib import SequenceMatcher, unified_diff

import pyomo.core.expr.current as EXPR

_strip_comment = re.compile(r'\s*\\.*')


def _compare_floats(base, test, abstol=1e-14, reltol=1e-14):
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


def _update_subsets(subset, base, test):
    for i, j in zip(*subset):
        if _compare_floats(base[i], test[j]):
            base[i] = test[j]


def _preprocess_data(data):
    for line in data.splitlines():
        fields = line.split()
        for i, f in enumerate(fields):
            try:
                if int(f) == float(f):
                    fields[i] = str(int(f))
            except:
                pass
        yield ' '.join(fields)


def lp_diff(base, test, baseline='baseline', testfile='testfile'):
    if test == base:
        return [], []

    test = list(_preprocess_data(test))
    base = list(_preprocess_data(base))
    if test == base:
        return [], []

    for group in SequenceMatcher(None, base, test).get_grouped_opcodes(3):
        for tag, i1, i2, j1, j2 in group:
            if tag != 'replace':
                continue
            _update_subsets((range(i1, i2), range(j1, j2)), base, test)

    if test == base:
        return [], []

    print(
        ''.join(
            unified_diff(
                [_ + "\n" for _ in base],
                [_ + "\n" for _ in test],
                fromfile=baseline,
                tofile=testfile,
            )
        )
    )
    return base, test


def load_lp_baseline(baseline, testfile, version='lp'):
    with open(testfile, 'r') as FILE:
        test = FILE.read()
    if baseline.endswith('.lp'):
        _tmp = [baseline[:-3]]
    else:
        _tmp = baseline.split('.lp.', 1)
    _tmp.insert(1, f'expr{int(EXPR._mode)}')
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


def load_and_compare_lp_baseline(baseline, testfile, version='lp'):
    return lp_diff(*load_lp_baseline(baseline, testfile, version))


if __name__ == '__main__':
    import sys

    base, test = load_and_compare_lp_baseline(sys.argv[1], sys.argv[2])
    sys.exit(1 if base or test else 0)
