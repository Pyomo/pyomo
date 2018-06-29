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

    def test_component_objects_hack(self):
        m = pmo.block()
        m.v = pmo.variable()
        m.c = pmo.constraint()
        m.B = pmo.block_list()
        m.B.append(pmo.block())
        m.B[0].v = pmo.variable()
        m.B[0].c = pmo.constraint()
        m.b = pmo.block()
        m.b.v = pmo.variable()
        m.b.c = pmo.constraint()
        for obj1, obj2 in zip(m.components(),
                              m.component_objects()):
            self.assertIs(obj1, obj2)

if __name__ == "__main__":
    unittest.main()
