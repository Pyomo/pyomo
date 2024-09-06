#!/usr/bin/env python
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

import sys

try:
    import ujson as json
except ImportError:
    import json

from math import sqrt, log10, floor
from statistics import stdev, mean

# import scipy.stats as st
# scipy.stats.norm.ppf(0.9)
# 0.95 = 1.6448536269514722
# 0.90 = 1.2815515655446004
# 0.85 = 1.0364333894937898
# 0.80 = 0.8416212335729143
# 0.75 = 0.6744897501960817
# Z-score:  (mean(x) - mean(y)) / sqrt(
#      stdev(x)**2 / card(x)  +  stdev(y)**2 / card(y) )


class Result(object):
    z_threshold = 1.6448536269514722  # 95%
    # z_threshold = 1.2815515655446004  # 90%
    # z_threshold = 0.8416212335729143  # 80%
    # z_threshold = 0.6744897501960817  # 75%

    def __init__(self, test, base=None, relative=False, precision=None):
        self.test = test
        self.base = base
        if precision is None:
            precision = 2 if relative else 3
        self.precision = precision
        self.relative = relative

    def value(self):
        _test, _base = self.test_base_value()
        if not _test:
            return '--', None
        if isinstance(self.test, list) and len(self.test) > 1:
            t_stdev = stdev(self.test)
        else:
            t_stdev = 0
        if isinstance(self.base, list) and len(self.base) > 1:
            b_stdev = stdev(self.base)
        else:
            b_stdev = 0
        if self.relative:
            if not _base:
                return '**', None
            else:
                val = 100 * (_test - _base) / _base
                dev = min(t_stdev, b_stdev) / _base
        elif not self.base:
            val = _test
            dev = None
        else:
            val = _test - _base
            dev = min(t_stdev, b_stdev)
        return val, dev

    def test_base_value(self):
        if isinstance(self.test, list):
            _test = min(self.test)
        else:
            _test = self.test or 0
        if isinstance(self.base, list):
            _base = min(self.base)
        else:
            _base = self.base or 0
        return _test, _base

    def __float__(self):
        val, dev = self.value()
        if isinstance(val, str):
            return 0.0
        return float(val)

    def __lt__(self, other):
        try:
            return self.value() < other.value()
        except:
            return float(self) < float(other)

    def tostr(self, width=0):
        val, dev = self.value()
        if isinstance(val, str):
            return val
        try:
            t_stdev = stdev(self.test) if len(self.test) > 1 else 0
            b_stdev = stdev(self.base) if len(self.base) > 1 else 0
            z = abs(mean(self.test) - mean(self.base)) / sqrt(
                t_stdev**2 / len(self.test) + b_stdev**2 / len(self.base)
            )
        except (ZeroDivisionError, TypeError):
            z = 0

        precision = self.precision
        if width:
            precision = max(
                0,
                min(
                    precision,
                    width
                    - (2 if val >= 0 else 3)
                    - (1 if not val else floor(log10(abs(val)))),
                ),
            )
        val_str = ('%%%d.%df' % (width, precision)) % val
        if z > Result.z_threshold:
            if val < 0:
                return '\033[92m' + val_str + '\033[0m'
            else:
                return '\033[91m' + val_str + '\033[0m'
        else:
            return val_str

    def __str__(self):
        return self.tostr(0)


def combine(*results):
    """Combine multiple performance JSON results into a single result

    Individual metrics are combined using `min()`

    Returns a dict mapping test names to dicts of {metric: value}
    """
    ans = {}
    for result in results:
        for dataset in result[1:]:
            for test, result in dataset.items():
                if "::" not in test:
                    # Convert nosetests results in to pytest format
                    path, test = test.split(':')
                    test = (
                        '/'.join(path.split('.')) + '.py::' + '::'.join(test.split('.'))
                    )
                if test not in ans:
                    ans[test] = {}
                testdata = ans[test]
                for metric, value in result.items():
                    if type(value) is dict:
                        continue
                    testdata.setdefault(metric, []).append(value)
    # Nosetests and pytest would give different test names (based on
    # where they started including path elements).  We will assume that
    # tests should be uniquely declared by the test file, class, and
    # test name.  So, any two tests where one name ends with the
    # complete name of the other will be assumed to be the same test and
    # combined.
    for base in list(ans):
        for test in ans:
            if test == base:
                break
            if test.endswith(base):
                testdata = ans[test]
                otherdata = ans.pop(base)
                for metric, value in otherdata.items():
                    testdata.setdefault(metric, []).append(value)
                break
    return ans


def compare(base_data, test_data):
    """Compare two data sets (generated by compare())"""
    # Nosetests and pytest would give different test names (based on
    # where they started including path elements).  We will assume that
    # tests should be uniquely declared by the test file, class, and
    # test name.  So, any two tests where one name ends with the
    # complete name of the other will be assumed to be the same test.
    # We will make both to the "more specific" (longer) name before
    # comparing.
    for base in list(base_data):
        for test in test_data:
            if base == test:
                break
            if test.endswith(base):
                base_data[test] = base_data.pop(base)
                break
            if base.endswith(test):
                test_data[base] = test_data.pop(test)
                break

    fields = set()
    for testname, base in base_data.items():
        if testname not in test_data:
            continue
        fields.update(set(base).intersection(test_data[testname]))
    fields = sorted(fields - {'test_time'})

    lines = []
    for testname, base in base_data.items():
        if testname not in test_data:
            continue
        test = test_data[testname]
        lines.append(
            [
                [Result(testname)],
                [Result(test['test_time'])],
                [
                    Result(test['test_time'], base['test_time']),
                    Result(test['test_time'], base['test_time'], relative=True),
                ],
                [
                    Result(test.get(field, None), base.get(field, None))
                    for field in fields
                ],
                [Result(test.get(field, None)) for field in fields],
            ]
        )
    lines.sort()
    return (
        [['test_name'], ['test_time'], ['time(\u0394)', 'time(%)'], fields, fields],
        lines,
    )


def print_comparison(os, data):
    """Print the 'comparison' table from the data to os

    Parameters
    ----------
    os: TextIO
        Stream to write the table to
    data: list
        Data structure generated by compare()

    """
    _printer([2, 1, 3, 0], os, data)


def print_test_result(os, data):
    """Print the 'test result' table from the data to os

    Parameters
    ----------
    os: TextIO
        Stream to write the table to
    data: list
        Data structure generated by compare()

    """
    _printer([1, 4, 0], os, data)


def _printer(arglist, os, data):
    fields = sum((data[0][i] for i in arglist), [])
    lines = [sum((line[i] for i in arglist), []) for line in data[1]]

    field_w = [max(len(field), 7) for field in fields]
    os.write(' '.join(('%%%ds' % w) % fields[i] for i, w in enumerate(field_w)) + '\n')
    os.write('-' * (len(field_w) + sum(field_w) - 1) + '\n')
    cumul = [Result(0, 0, relative=v.relative) for i, v in enumerate(lines[0])]
    for line in sorted(lines):
        os.write(
            ' '.join(
                ('%%%ds' % width) % line[i].tostr(width)
                for i, width in enumerate(field_w)
            )
            + '\n'
        )
        for i, v in enumerate(line):
            _test, _base = v.test_base_value()
            if isinstance(_test, str):
                continue
            cumul[i].test += _test
            cumul[i].base += _base
    os.write('-' * (len(field_w) + sum(field_w) - 1) + '\n')
    cumul[-1].test = "[ TOTAL ]"
    os.write(
        ' '.join(
            ('%%%ds' % width) % cumul[i].tostr(width) for i, width in enumerate(field_w)
        )
        + '\n'
    )
    if not any(c.base for c in cumul):
        return
    for c in cumul[:-1]:
        if c.relative:
            c.base = 0
        c.relative = True
    cumul[-1].test = "[ %diff ]"
    os.write(
        ' '.join(
            ('%%%ds' % width) % cumul[i].tostr(width) for i, width in enumerate(field_w)
        )
        + '\n'
    )


if __name__ == '__main__':
    clean = '--clean' in sys.argv
    if clean:
        sys.argv.remove('--clean')
    if len(sys.argv) != 3:
        print("Usage: %s <base> <test> [--clean]" % (sys.argv[0],))
        print("    <base>: comma-separated list of performance JSON files")
        print("    <test>: comma-separated list of performance JSON files")
        print("    --clean: remove test names for 'nonpublic' test results")
        sys.exit(1)
    jsons = []
    for fname in sys.argv[1].split(','):
        with open(fname) as F:
            jsons.append(json.load(F))
        base = combine(*jsons)
    jsons = []
    for fname in sys.argv[2].split(','):
        with open(fname) as F:
            jsons.append(json.load(F))
        test = combine(*jsons)
    data = compare(base, test)
    if clean:
        n = 0
        for line in data[1]:
            name = line[0][0].test
            if 'nonpublic' in name:
                line[0][0].test = name[: name.find('.', name.find('nonpublic'))] + (
                    ".%s" % n
                )
                n += 1
    print_test_result(sys.stdout, data)
    sys.stdout.write('\n')
    print_comparison(sys.stdout, data)
