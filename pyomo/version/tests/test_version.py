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

import pyomo.common.unittest as unittest
import pyomo.version as pyomo_ver


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


if __name__ == "__main__":
    unittest.main()
