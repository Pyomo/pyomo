#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyutilib.th as unittest
import pyomo.version as pyomo


class Tests(unittest.TestCase):

    def test_releaselevel(self):
        _relLevel = pyomo.version_info[3].split('{')[0].strip()
        self.assertIn( _relLevel,('devel','VOTD','final') )

    def test_version(self):
        try:
            import pkg_resources
            version = pkg_resources.get_distribution('pyomo').version
        except:
            self.skipTest('pkg_resources is not available')

        if pyomo.version_info[3] == 'final':
            self.assertEqual(pyomo.version, version)

        else:
            tmp_ = version.split('.')
            self.assertEqual(str(tmp_[0]), str(pyomo.version_info[0]))
            self.assertEqual(str(tmp_[1]), str(pyomo.version_info[1]))
            if len(tmp_) > 2:
                self.assertEqual(str(tmp_[2]), str(pyomo.version_info[2]))


if __name__ == "__main__":
    unittest.main()
