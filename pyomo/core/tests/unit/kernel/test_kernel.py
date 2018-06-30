import pickle

import pyutilib.th as unittest
import pyomo.kernel as pmo

import six
from six import StringIO

class Test_kernel(unittest.TestCase):

    def test_no_ctype_collisions(self):
        hash_set = set()
        hash_list = list()
        for cls in [pmo.variable,
                    pmo.constraint,
                    pmo.objective,
                    pmo.expression,
                    pmo.parameter,
                    pmo.suffix,
                    pmo.sos,
                    pmo.block]:
            hash_set.add(hash(cls))
            hash_list.append(hash(cls))
        self.assertEqual(len(hash_set),
                         len(hash_list))

if __name__ == "__main__":
    unittest.main()
