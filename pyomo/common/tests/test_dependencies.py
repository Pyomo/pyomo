import pyutilib.th as unittest
from pyomo.common.dependencies import attempt_import, DeferredImportError

from six import StringIO

import logging
logger = logging.getLogger('pyomo.common')

class TestDependencies(unittest.TestCase):
    def test_import_error(self):
        module_obj, module_available = attempt_import('__there_is_no_module_named_this__', 'Testing import of a non-existant module')
        self.assertFalse(module_available)
        with self.assertRaises(DeferredImportError):
            module_obj.try_to_call_a_method()
                
    def test_import_success(self):
        module_obj, module_available = attempt_import('pyutilib','Testing import of PyUtilib')
        self.assertTrue(module_available)
        import pyutilib
        self.assertTrue(module_obj is pyutilib)

if __name__ == '__main__':
    unittest.main()
