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

_norm_whitespace = re.compile(r'\s+')
_norm_comment = re.compile(r'\s*#\s*')
_strip_comment = re.compile(r'\s*#.*')
_norm_negation = re.compile(r'(?m)^o2(\s*#\s*\*)?$n-1(.0)?$')

def _to_float_list(line):
    ans = []
    for field in line.split():
        try:
            ans.append(float(field))
        except:
            ans.append(field)
    return ans

def _update_subsets(subset, base, test):
    for i, j in zip(*subset):
        # Try checking for numbers
        if base[i][0] == 'n' and test[j][0] == 'n':
            if float(base[i][1:]) == float(test[j][1:]):
                test[j] = base[i]
        elif _to_float_list(base[i]) == _to_float_list(test[j]):
            test[j] = base[i]
        else:
            # try stripping comments, but only if it results in a match
            base_nc = _strip_comment.sub('', base[i])
            test_nc = _strip_comment.sub('', test[j])
            if base_nc == test_nc or \
               _to_float_list(base_nc) == _to_float_list(test_nc):
                if len(base_nc) > len(test_nc):
                    test[j] = base[i]
                else:
                    base[i] = test[j]

def nl_diff(base, test):
    if test == base:
        return [], []
    test = _norm_negation.sub(test, template.negation)
    base = _norm_negation.sub(base, template.negation)
    test = test.splitlines()
    base = base.splitlines()

    for i in range(min(len(test), len(base))):
        if test[i] == base[i]:
            continue
        # normalize comment whitespace
        base[i] = _norm_comment.sub(
            '\t#', _norm_whitespace.sub(' ', base[i]))
        test[i] = _norm_comment.sub(
            '\t#', _norm_whitespace.sub(' ', test[i]))
    if test == base:
        return [], []

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
    return test, base

def load_and_compare_nl_baseline(baseline, testfile, version='nl'):
    return nl_diff(*load_nl_baseline(baseline, testfile, version))
