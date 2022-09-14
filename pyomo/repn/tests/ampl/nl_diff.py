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

import itertools
import os
import re

from difflib import SequenceMatcher, unified_diff

import pyomo.repn.plugins.nl_writer as nl_writer

template = nl_writer.text_nl_debug_template

_norm_whitespace = re.compile(r'[^\S\n]+')
_norm_integers = re.compile(r'(?m)\.0+$')
_norm_comment = re.compile(r'\s*#\s*')
_strip_comment = re.compile(r'\s*#.*')
_norm_negation = re.compile(r'(?m)^o2(\s*#\s*\*)?\nn-1(.0)?\s*\n')
_norm_timesone = re.compile(r'(?m)^o2(\s*#\s*\*)?\nn1(.0)?\s*\n')

def _compare_floats(base, test, abstol=1e-14, reltol=1e-14):
    base = base.split()
    test = test.split()
    if len(base) != len(test):
        return False
    for i, b in enumerate(base):
        if b == test[i]:
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
        # Try checking for numbers
        if base[i][0] == 'n' and test[j][0] == 'n':
            if _compare_floats(base[i][1:], test[j][1:]):
                test[j] = base[i]
        elif _compare_floats(base[i], test[j]):
            test[j] = base[i]
        else:
            # try stripping comments, but only if it results in a match
            base_nc = _strip_comment.sub('', base[i])
            test_nc = _strip_comment.sub('', test[j])
            if _compare_floats(base_nc, test_nc):
                if len(base_nc) > len(test_nc):
                    test[j] = base[i]
                else:
                    base[i] = test[j]

def _preprocess_data(data):
    # Normalize negation (convert " * -1" to the negation operator)
    data = _norm_negation.sub(template.negation, data)
    # Remove multiplication by 1
    data = _norm_timesone.sub('', data)
    # Normalize consecutive whitespace to a single space
    data = _norm_whitespace.sub(' ', data)
    # preface all comments with a single tab character
    data = _norm_comment.sub('\t#', data)
    # Normalize floating point integers to integers
    data = _norm_integers.sub('', data)
    # return the sequence of lines
    return data.splitlines()

def nl_diff(base, test, baseline='baseline', testfile='testfile'):
    if test == base:
        return [], []

    test = _preprocess_data(test)
    base = _preprocess_data(base)
    if test == base:
        return [], []

    # First do a quick pass to check / standardize embedded numbers.
    # This is a little fragile (it requires that the embedded constants
    # appear in the same order in the base and test files), but we see
    # cases where differences within numerical tolerances lead to huge
    # add / delete chunks (instead of small replace chunks) from the
    # SequenceMatcher (because it is not as fast / aggressive as Unix
    # diff).  Those add/remove chunks are ignored by the _update_subsets
    # code below, leading to unnecessary test failures.
    test_nlines = list(x for x in enumerate(test) if x[1] and x[1][0] == 'n')
    base_nlines = list(x for x in enumerate(base) if x[1] and x[1][0] == 'n')
    if len(test_nlines) == len(base_nlines):
        for t_line, b_line in zip(test_nlines, base_nlines):
            if _compare_floats(t_line[1][1:], b_line[1][1:]):
                test[t_line[0]] = base[b_line[0]]

    for group in SequenceMatcher(None, base, test).get_grouped_opcodes(3):
        for tag, i1, i2, j1, j2 in group:
            if tag != 'replace':
                continue
            _update_subsets((range(i1, i2), range(j1, j2)), base, test)

    if test == base:
        return [], []

    print(''.join(unified_diff(
        [_+"\n" for _ in base],
        [_+"\n" for _ in test],
        fromfile=baseline,
        tofile=testfile)))
    return base, test

def load_nl_baseline(baseline, testfile, version='nl'):
    with open(testfile, 'r') as FILE:
        test = FILE.read()
    if baseline.endswith('.nl'):
        _tmp = baseline[:-2] + version
    else:
        _tmp = baseline.replace('.nl.', f'.{version}.')
    if os.path.exists(_tmp):
        baseline = _tmp
    with open(baseline, 'r') as FILE:
        base = FILE.read()
    return base, test

def load_and_compare_nl_baseline(baseline, testfile, version='nl'):
    return nl_diff(
        *load_nl_baseline(baseline, testfile, version), baseline, testfile
    )
