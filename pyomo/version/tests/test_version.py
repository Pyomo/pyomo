#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

import pyutilib.th as unittest
import pyomo.version as pyomo


class Tests(unittest.TestCase):

    def test_releaselevel(self):
        self.assertTrue( pyomo.version_info[3]
                         in ('trunk','trunk {git}','VOTD','final') )

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
