"""Testing for deprecated function."""
import pyutilib.th as unittest

import logging
logger = logging.getLogger('pyomo.common')


class Test(unittest.TestCase):

    def test_bilevel(self):
        try:
            import pyomo.ext.bilevel
        except:
            self.fail("Error importing pyomo.ext.bilevel")
        self.assertTrue('SubModel' in dir(pyomo.ext.bilevel))

    def test_dae(self):
        try:
            import pyomo.ext.dae
        except:
            self.fail("Error importing pyomo.ext.dae")
        self.assertTrue('ContinuousSet' in dir(pyomo.ext.dae))

    def test_gdp(self):
        try:
            import pyomo.ext.gdp
        except:
            self.fail("Error importing pyomo.ext.gdp")
        self.assertTrue('Disjunct' in dir(pyomo.ext.gdp))

    def test_mpec(self):
        try:
            import pyomo.ext.mpec
        except:
            self.fail("Error importing pyomo.ext.mpec")
        self.assertTrue('Complementarity' in dir(pyomo.ext.mpec))

    def test_pysp(self):
        try:
            import pyomo.ext.pysp
        except:
            self.fail("Error importing pyomo.ext.pysp")
        self.assertTrue('ph' in dir(pyomo.ext.pysp))

    def test_simplemodel(self):
        available=False
        try:
            import pyomo.ext.simplemodel
            available=False
        except:
            pass
        if available:
            self.assertTrue('SimpleModel' in dir(pyomo.ext.simplemodel))
        else:
            self.skipTest("Skipping test of pyomo.ext.simplemodel")


if __name__ == '__main__':
    unittest.main()
