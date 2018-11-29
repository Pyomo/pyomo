#!/usr/python/env python

import pyutilib.th as unittest

from pyomo.common.config import ( 
    PositiveInt, PositiveFloat, NonNegativeFloat, In)
from pyomo.core import Var, value

class UseConfig():

    CONFIG = ConfigBlock('Use Config')

    CONFIG.declare('trust radius', ConfigValue(
        default = 1.0,
        domain = PositiveFloat,
        description = '',
        doc = ''))

    CONFIGFULL = ConfigBlock('A')
    CONFIGFULL.declare('B', ConfigValue(default=1.0))
    CONFIGFULL.declare('C', ConfigBlock('D'))
    CONFIGFULL.declare('E', ConfigList([25,50,75]))


    def __init__(self, **kwds):
        self.config = self.CONFIG(kwds)


    def solve(self, **kwds):
        self.config.display()
        local_config = self.config(kwds)
        self._local_config = local_config



class TestConfigBlock(unittest.TestCase):
    def setUp(self):
        
        self.params1 = UseConfig()
        self.params2 = UseConfig()
        self.params3 = UseConfig(trust_radius = 3.0)


    def test1(self):

        # Initialized with 1.0
        self.assertEqual(self.params1.config.trust_radius, 1.0)
        
        # Both persistent and local values should be 1.0
        self.params1.solve()
        self.assertEqual(self.params1.config.trust_radius, 1.0)
        self.assertEqual(self.params1._local_config.trust_radius, 1.0)

        # Persistent should be 1.0, local should be 20.0
        self.assertEqual(self.params1.config.trust_radius, 1.0)
        self.params1.solve(trust_radius=20.0)
        self.assertEqual(self.params1.config.trust_radius, 1.0)
        self.assertEqual(self.params1._local_config.trust_radius, 20.0)

    def test2(self):

        # Initialized with 1.0
        self.assertEqual(self.params2.config.trust_radius, 1.0)
        
        # Set persistent value to 4.0; local value should also be set to 4.0
        self.params2.config.trust_radius = 4.0
        self.params2.solve()
        self.assertEqual(self.params2.config.trust_radius, 4.0)
        self.assertEqual(self.params2._local_config.trust_radius, 4.0)

        # Persistent should be 4.0, local should be 20.0
        self.params2.solve(trust_radius=20.0)
        self.assertEqual(self.params2.config.trust_radius, 4.0)
        self.assertEqual(self.params2._local_config.trust_radius, 20.0)

    def test3(self):

        # Initialized with 3.0
        self.assertEqual(self.params3.config.trust_radius, 3.0)
        
        # Both persistent and local values should be set to 3.0
        self.params3.solve()
        self.assertEqual(self.params3.config.trust_radius, 3.0)
        self.assertEqual(self.params3._local_config.trust_radius, 3.0)

        # Persistent should be 3.0, local should be 20.0
        self.params3.solve(trust_radius=20.0)
        self.assertEqual(self.params3.config.trust_radius, 3.0)
        self.assertEqual(self.params3._local_config.trust_radius, 20.0)

            
if __name__ =='__main__':
    unittest.main()
