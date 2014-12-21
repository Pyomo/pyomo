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
        self.assertTrue(pyomo.version_info[3] in ('trunk','VOTD','final'))

    def test_version(self):
        try:
            import pkg_resources
            version = pkg_resources.get_distribution('pyomo').version
        except:
            self.skipTest('pkg_resources is not available')

        if pyomo.version_info[3] == 'final':
            self.assertEquals(pyomo.version, version)

        elif pyomo.version_info[3] == 'trunk':
            self.assertEquals( tuple(int(x) for x in version.split('.')),
                               pyomo.version_info[:2] )
            self.assertEquals( pyomo.version_info[2], 0 )
        else:
            self.assertEquals( tuple(int(x) for x in version.split('.')),
                               pyomo.version_info[:2] )
            self.assertNotEquals( pyomo.version_info[2], 0 )


if __name__ == "__main__":
    unittest.main()
            
