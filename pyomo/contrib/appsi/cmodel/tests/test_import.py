from pyomo.common import unittest
from pyomo.common.fileutils import find_library, this_file_dir
import os
from pyomo.common.envvar import PYOMO_CONFIG_DIR
import sys
from pyomo.contrib.appsi.cmodel import cmodel_available


class TestCmodelImport(unittest.TestCase):
    def test_import(self):
        pyomo_config_dir = os.path.join(
            PYOMO_CONFIG_DIR,
            "lib",
            "python%s.%s" % sys.version_info[:2],
            "site-packages",
        )
        cmodel_dir = this_file_dir()
        cmodel_dir = os.path.join(cmodel_dir, os.pardir)
        lib = find_library("appsi_cmodel.*", pathlist=pyomo_config_dir)
        if lib is None:
            lib = find_library("appsi_cmodel.*", pathlist=cmodel_dir)
        if lib is not None:
            self.assertTrue(cmodel_available)
        else:
            raise unittest.SkipTest('appsi library file not found')
