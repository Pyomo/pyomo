#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import re

from difflib import SequenceMatcher, unified_diff

from pyomo.repn.tests.diffutils import compare_floats, load_baseline

_strip_comment = re.compile(r'\s*\\.*')


def _update_subsets(subset, base, test):
    for i, j in zip(*subset):
        if compare_floats(base[i], test[j]):
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
    return load_baseline(baseline, testfile, 'lp', version)


def load_and_compare_lp_baseline(baseline, testfile, version='lp'):
    return lp_diff(*load_lp_baseline(baseline, testfile, version))


if __name__ == '__main__':
    import sys

    base, test = load_and_compare_lp_baseline(sys.argv[1], sys.argv[2])
    sys.exit(1 if base or test else 0)
