# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________
#
# Unit Tests for pyomo.base.misc
#

import math
import os
import re
import sys
import subprocess

import pyomo.common.unittest as unittest


class ImportData:
    def __init__(self):
        self.tpl = {}
        self.module = {}

    def update(self, other):
        self.tpl.update(other.tpl)
        self.module.update(other.module)

    def imax(self, other):
        for k, v in other.tpl.items():
            if k in self.tpl:
                v = max(v, self.tpl[k])
            self.tpl[k] = v
        self.module.update(other.module)


def collect_import_time(module, preimport=""):
    basemodule = module.split('.')[0]
    if preimport:
        cmd = f"{preimport}; import {module}"
    else:
        cmd = f"import {module}"
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join(filter(None, sys.path))
    env.pop('COVERAGE_PROCESS_START', None)
    output = subprocess.check_output(
        [sys.executable, '-S', '-X', 'importtime', '-c', cmd],
        stderr=subprocess.STDOUT,
        env=env,
    )
    # Note: test only runs in PY3
    output = output.decode()
    line_re = re.compile(r'.*:\s*(\d+) \|\s*(\d+) \| ( *)([^ ]+)')
    header_re = re.compile(r'.*:\s*(.*)')
    results = []
    data = None
    for line in output.splitlines():
        g = line_re.match(line)
        if not g:
            g = header_re.match(line)
            if g:
                data = []
                results.append(data)
            else:
                raise RuntimeError(f"Unrecognized line: '{line}'")
            continue
        _self = int(g.group(1))
        _cumul = int(g.group(2))
        _level = len(g.group(3)) // 2
        _module = g.group(4)
        # print("%6d %8d %2d %s" % (_self, _cumul, _level, _module))
        while len(data) < _level + 1:
            data.append(ImportData())
        if len(data) > _level + 1:
            if len(data) != _level + 2:
                raise RuntimeError(
                    f"Error processing line '{line}': unexpected unindent"
                )
            inner = data.pop()
            inner.tpl = {
                (k if '(from' in k else "%s (from %s)" % (k, _module), v)
                for k, v in inner.tpl.items()
            }
            if _module.startswith(basemodule):
                data[_level].update(inner)
                data[_level].module[_module] = _self
            else:  # _level > 0:
                data[_level].tpl[_module] = _cumul
        elif _module.startswith(basemodule):
            data[_level].module[_module] = _self
        else:  # _level > 0:
            data[_level].tpl[_module] = _self
    ans = None
    for d in results:
        assert len(d) == 1
        d = d[0]
        if not d.module:
            continue
        if ans:
            raise RuntimeError(
                "Multiple timing results imported target module '{module}'"
            )
        ans = d
    return ans, output


def summarize_import_time(module, data, raw_output):
    print(raw_output)
    print("\n")

    modname = module.split('.')[0]

    N = int(math.log10(max(max(data.module.values()), max(data.tpl.values())))) + 4
    print(f"{modname.title()} (by module time):")
    print(
        "\n".join(
            f"%{N}d: %s" % (v, k)
            for k, v in sorted(data.module.items(), key=lambda x: x[1])
        )
    )
    tpls = sorted(
        (*_mod.split(' ', maxsplit=1), '', _time) for _mod, _time in data.tpl.items()
    )
    print("TPLS:")
    _line_fmt = f"   %{max(len(l[0]) for l in tpls)}s: %6d %s"
    print("\n".join(_line_fmt % (l[0], l[-1], l[1]) for l in tpls))
    tpl = {}
    for k, v in data.tpl.items():
        _mod = k.split()[0].split('.')[0]
        _base_time, _base_cat = tpl.get(_mod, (0, 0))
        tpl[_mod] = _base_time + v, _base_cat | (1 if ' ' in k else 2)
    tpl_by_time = sorted(tpl.items(), key=lambda x: x[1])

    pyomo_time = sum(data.module.values())
    tpl_time = sum(data.tpl.values())
    total = float(pyomo_time + tpl_time)
    python_time = sum(t for m, (t, s) in tpl_by_time if s & 1 == 0)
    module_tpl_time = sum(t for m, (t, s) in tpl_by_time if s & 1)
    assert abs(python_time + module_tpl_time - tpl_time) < 1

    print("TPLS (by package time):")
    _line_fmt = f"   %{max(len(k) for k in tpl)}s: %6d (%4.1f%%)%s"
    source = {1: '', 2: ' *', 3: ' *+'}
    print(
        "\n".join(
            _line_fmt % (m, t, 100 * t / total, source[s]) for m, (t, s) in tpl_by_time
        )
    )
    N = len(modname) + 8
    print(
        f"\n%-{N}s %6d (%4.1f%%)"
        % (f"{modname.title()}:", pyomo_time, 100 * pyomo_time / total)
    )
    print(
        f"%-{N}s %6d (%4.1f%%)"
        % (f"TPL ({modname}):", module_tpl_time, 100 * module_tpl_time / total)
    )
    print(
        f"%-{N}s %6d (%4.1f%%)"
        % ("TPL (python):", python_time, 100 * python_time / total)
    )

    return python_time, module_tpl_time, pyomo_time, tpl_by_time


class TestPyomoEnviron(unittest.TestCase):
    def test_not_auto_imported(self):
        rc = subprocess.call(
            [
                sys.executable,
                '-c',
                'import pyomo.core, sys; '
                'sys.exit( 1 if "pyomo.environ" in sys.modules else 0 )',
            ]
        )
        if rc:
            self.fail(
                "Importing pyomo.core automatically imports "
                "pyomo.environ and it should not."
            )

    @unittest.skipIf(
        'pypy_version_info' in dir(sys), "PyPy does not support '-X importtime'"
    )
    def test_tpl_import_time(self):
        data, output = collect_import_time(
            'pyomo.environ',
            # We used to pre-load and pre-start multiprocessing so that the
            # asynchronous task triggered by creating a Lock will not be
            # interleaved in the importtime report:
            #
            ##'import time, multiprocessing; multiprocessing.Lock(); time.sleep(0.25)',
            #
            # This is no longer needed as we have removed
            # multiprocessing from the list of required modules for
            # pyomo.environ.
        )
        python_time, module_tpl_time, pyomo_time, tpl_by_time = summarize_import_time(
            'pyomo.environ', data, output
        )

        # Arbitrarily choose a threshold 10% more than the expected
        # value (at time of writing, TPL imports were 52-57% of the
        # import time on a development machine)
        self.assertLess(module_tpl_time / (module_tpl_time + pyomo_time), 0.33)
        # Spot-check the (known) worst offenders.  The following are
        # modules from the "standard" library.  Their order in the list
        # of slow-loading TPLs can vary from platform to platform.
        ref = {
            '__future__',
            'argparse',
            'ast',  # Imported on Windows
            'backports_abc',  # Imported by cython on Linux
            'base64',  # Imported on Windows
            'bisect',  # Imported by dae, dataportal, contrib/mpc
            'cPickle',
            'copy',  # Imported by ply, et al.
            'csv',
            'ctypes',  # mandatory import in core/base/external.py; TODO: fix this
            'datetime',  # imported by contrib.solver
            'decimal',
            'encodings',  # We tabulate modules imported by python
            'gc',  # Imported on MacOS, Windows; Linux in 3.10
            'glob',
            'heapq',  # Added in Python 3.10
            'importlib',
            'inspect',
            'io',
            'json',  # Imported on Windows
            'locale',  # Added in Python 3.9
            'logging',
            'pickle',
            'platform',
            'shlex',
            'socket',  # Imported on MacOS, Windows; Linux in 3.10
            'subprocess',
            'tempfile',  # Imported on MacOS, Windows
            'textwrap',
            'typing',
            'win32file',  # Imported on Windows
            'win32pipe',  # Imported on Windows
        }
        # Non-standard-library TPLs that Pyomo will load unconditionally:
        # ref.add('ply')  # PLY removed as a dependency in 6.10.0
        diff = set(_[0] for _ in tpl_by_time[-5:]).difference(ref)
        self.assertEqual(
            diff, set(), "Unexpected module found in 5 slowest-loading TPL modules"
        )


if __name__ == "__main__":
    # Running this file as a script will print out the package timing
    # information from test_tpl_import_time()
    unittest.main()
