# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

import os

import pyomo.common.unittest as unittest
import pyomo.version as pyomo_ver
import pyomo.version.info as info

from pyomo.common.tempfiles import TempfileManager


class Tests(unittest.TestCase):
    def test_releaselevel(self):
        _relLevel = pyomo_ver.version_info[3].split('{')[0].strip()
        self.assertIn(_relLevel, ('devel', 'VOTD', 'final'))

    def test_version(self):
        try:
            from importlib.metadata import version

            pyomo_version = version('pyomo')
        except:
            self.skipTest('importlib.metadata is not available')

        if pyomo_ver.version_info[3] == 'final':
            self.assertEqual(pyomo_ver.version, pyomo_version)

        else:
            tmp_ = pyomo_version.split('.')
            self.assertEqual(str(tmp_[0]), str(pyomo_ver.version_info[0]))
            self.assertEqual(str(tmp_[1]), str(pyomo_ver.version_info[1]))
            if tmp_[-1].startswith('dev'):
                import pyomo.version.info as info

                self.assertEqual(int(tmp_[-1][3:]), info.serial)
                tmp_.pop()
            if len(tmp_) > 2:
                self.assertEqual(str(tmp_[2]), str(pyomo_ver.version_info[2]))

    def test_estimate_release_level(self):
        with TempfileManager as tmp:
            dname = tmp.mkdtemp()
            orig = info.__file__
            try:
                info.__file__ = os.path.join(dname, 'setup.py')
                self.assertEqual('VOTD', info._estimate_release_level())
                os.mkdir(os.path.join(dname, '.git'))
                self.assertEqual('devel', info._estimate_release_level())
                with open(os.path.join(dname, '.git', 'HEAD'), 'w') as F:
                    F.write('12345\n')
                self.assertEqual('devel {12345}', info._estimate_release_level())
                with open(os.path.join(dname, '.git', 'HEAD'), 'w') as F:
                    F.write('ref: refs/heads/main\n')
                self.assertEqual('devel {main}', info._estimate_release_level())
            finally:
                info.__file__ = orig

    def test_finalize_version(self):
        vi = [1, 2, 3, 'final', 0]
        self.assertEqual(('1.2.3', '1.2.3'), info._finalize_version(vi))
        vi = [1, 2, 3, 'VOTD', 0]
        self.assertEqual(('1.2.3.a0', '1.2.3.a0 (VOTD)'), info._finalize_version(vi))
        vi = [1, 2, 3, 'devel', 0]
        self.assertEqual(
            ('1.2.3.dev0', '1.2.3.dev0 (devel)'), info._finalize_version(vi)
        )
        vi = [1, 2, 3, 'devel {main}', 0]
        self.assertEqual(
            ('1.2.3.dev0', '1.2.3.dev0 (devel {main})'), info._finalize_version(vi)
        )


if __name__ == "__main__":
    unittest.main()
