#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
import os
import sys
import shutil
from os.path import join, dirname, abspath, exists

import pyutilib.th as unittest

from pyomo.common.dependencies import yaml_available
from pyomo.pysp.scenariotree.instance_factory import \
    ScenarioTreeInstanceFactory
from pyomo.pysp.scenariotree.tree_structure_model import \
    CreateAbstractScenarioTreeModel
from pyomo.pysp.scenariotree.tree_structure import \
    ScenarioTree
from pyomo.pysp.util.misc import load_external_module

try:
    import networkx
    has_networkx = True
except:
    has_networkx = False

thisfile = abspath(__file__)
thisdir = dirname(thisfile)
testdatadir = join(thisdir, "testdata")

reference_test_model = None
def setUpModule():
    global reference_test_model
    reference_test_model = load_external_module(
        join(testdatadir, "reference_test_model.py"))[0].model


class Test(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import pyomo.environ

    def setUp(self):
        if "ReferenceModel" in sys.modules:
            del sys.modules["ReferenceModel"]
        if "reference_test_model" in sys.modules:
            del sys.modules["reference_test_model"]
        if "reference_test_model_with_callback" in sys.modules:
            del sys.modules["reference_test_model_with_callback"]
        if "reference_test_scenario_tree_model" in sys.modules:
            del sys.modules["reference_test_scenario_tree_model"]
        if "ScenarioStructure" in sys.modules:
            del sys.modules["ScenarioStructure"]
        if "both_callbacks" in sys.modules:
            del sys.modules["both_callbacks"]

    def _get_testfname_prefix(self):
        class_name, test_name = self.id().split('.')[-2:]
        return join(thisdir, class_name+"."+test_name)

    def _check_factory(self, factory):
        scenario_tree = factory.generate_scenario_tree()
        instances = factory.construct_instances_for_scenario_tree(scenario_tree,
                                                                  verbose=True)
        self.assertEqual(len(instances), 3)
        self.assertEqual(instances["s1"].p(), 1)
        self.assertEqual(instances["s2"].p(), 2)
        self.assertEqual(instances["s3"].p(), 3)
        instances = factory.construct_instances_for_scenario_tree(scenario_tree,
                                                                  compile_scenario_instances=True,
                                                                  verbose=True)
        self.assertEqual(len(instances), 3)
        self.assertEqual(instances["s1"].p(), 1)
        self.assertEqual(instances["s2"].p(), 2)
        self.assertEqual(instances["s3"].p(), 3)
        self.assertEqual(factory.construct_scenario_instance("s1", scenario_tree, verbose=True).p(), 1)
        with self.assertRaises(ValueError):
            factory.construct_scenario_instance("s0", scenario_tree, verbose=True)

        with self.assertRaises(ValueError):
            scenario_tree = factory.generate_scenario_tree(random_bundles=1000, verbose=True)
        with self.assertRaises(ValueError):
            scenario_tree = factory.generate_scenario_tree(random_bundles=2, bundles={}, verbose=True)

        scenario_tree = factory.generate_scenario_tree(random_bundles=2, verbose=True)
        self.assertEqual(scenario_tree.contains_bundles(), True)
        self.assertEqual(len(scenario_tree._scenario_bundles), 2)
        scenario_tree = factory.generate_scenario_tree(downsample_fraction=0.1, verbose=True)
        scenario_tree = factory.generate_scenario_tree(bundles={'b1': ['s1'],
                                                                'b2': ['s2'],
                                                                'b3': ['s3']},
                                                       verbose=True)
        self.assertEqual(scenario_tree.contains_bundles(), True)
        self.assertEqual(len(scenario_tree.bundles), 3)
        if factory._scenario_tree is not None:
            self.assertTrue(factory._scenario_tree_model is None)
            with self.assertRaises(ValueError):
                scenario_tree = factory.generate_scenario_tree(
                    bundles=join(testdatadir, "bundles.dat"),
                    verbose=True)
        else:
            scenario_tree = factory.generate_scenario_tree(
                bundles=join(testdatadir, "bundles.dat"),
                verbose=True)
            self.assertEqual(scenario_tree.contains_bundles(), True)
            self.assertEqual(len(scenario_tree.bundles), 3)
            scenario_tree = factory.generate_scenario_tree(
                bundles=join(testdatadir, "bundles"),
                verbose=True)
            self.assertEqual(scenario_tree.contains_bundles(), True)
            self.assertEqual(len(scenario_tree.bundles), 3)
            if (factory.data_directory() is not None) and \
               exists(join(factory.data_directory(), "bundles.dat")):
                scenario_tree = factory.generate_scenario_tree(
                    bundles="bundles.dat",
                    verbose=True)
                self.assertEqual(scenario_tree.contains_bundles(), True)
                self.assertEqual(len(scenario_tree.bundles), 3)
                scenario_tree = factory.generate_scenario_tree(
                    bundles="bundles",
                    verbose=True)
                self.assertEqual(scenario_tree.contains_bundles(), True)
                self.assertEqual(len(scenario_tree.bundles), 3)
            with self.assertRaises(ValueError):
                scenario_tree = factory.generate_scenario_tree(
                    bundles="bundles.notexists",
                    verbose=True)

    # model: name of .py file with model
    # scenario_tree: name of .dat file
    def test_init1(self):
        self.assertTrue("reference_test_model" not in sys.modules)
        with ScenarioTreeInstanceFactory(
                model=join(testdatadir,
                           "reference_test_model.py"),
                scenario_tree=join(testdatadir,
                                   "reference_test_scenario_tree.dat")) as factory:
            self.assertEqual(len(factory._archives), 0)
            self.assertTrue(factory.model_directory() is not None)
            self.assertTrue(factory.scenario_tree_directory() is not None)
            self._check_factory(factory)
        self.assertTrue("reference_test_model" in sys.modules)

    # model: name of directory with ReferenceModel.py with model
    # scenario_tree: name of directory with ScenarioStructure.dat
    def test_init1_default(self):
        self.assertTrue("ReferenceModel" not in sys.modules)
        with ScenarioTreeInstanceFactory(
                model=testdatadir,
                scenario_tree=testdatadir) as factory:
            self.assertEqual(len(factory._archives), 0)
            self.assertTrue(factory.model_directory() is not None)
            self.assertTrue(factory.scenario_tree_directory() is not None)
            self._check_factory(factory)
        self.assertTrue("ReferenceModel" in sys.modules)

    # model: name of archive with .py file with model
    # scenario_tree: name of .dat file
    def test_init2(self):
        self.assertTrue("reference_test_model" not in sys.modules)
        with ScenarioTreeInstanceFactory(
                model=join(testdatadir,
                           "archive_test.tgz,reference_test_model.py"),
                scenario_tree=join(testdatadir,
                                   "reference_test_scenario_tree.dat")) as factory:
            self.assertEqual(len(factory._archives), 1)
            self.assertTrue(factory.model_directory() is not None)
            self.assertTrue(factory.scenario_tree_directory() is not None)
            self._check_factory(factory)
        self.assertTrue("reference_test_model" in sys.modules)

    # model: name of archive containing ReferenceModel.py with model
    # scenario_tree: name of directory with ScenarioStructure.dat file
    def test_init2_default(self):
        self.assertTrue("ReferenceModel" not in sys.modules)
        with ScenarioTreeInstanceFactory(
                model=join(testdatadir, "archive_test.tgz,"),
                scenario_tree=testdatadir) as factory:
            self.assertEqual(len(factory._archives), 1)
            self.assertTrue(factory.model_directory() is not None)
            self.assertTrue(factory.scenario_tree_directory() is not None)
            self._check_factory(factory)
        self.assertTrue("ReferenceModel" in sys.modules)

    # model: name of .py file with model
    # scenario_tree: name of archive with .dat file
    def test_init3(self):
        self.assertTrue("reference_test_model" not in sys.modules)
        with ScenarioTreeInstanceFactory(
                model=join(testdatadir,
                           "reference_test_model.py"),
                scenario_tree=join(testdatadir,
                                   "archive_test.tgz,reference_test_scenario_tree.dat")) as factory:
            self.assertEqual(len(factory._archives), 1)
            self.assertTrue(factory.model_directory() is not None)
            self.assertTrue(factory.scenario_tree_directory() is not None)
            self._check_factory(factory)
        self.assertTrue("reference_test_model" in sys.modules)

    # model: name of directory with ReferenceModel.py file with model
    # scenario_tree: name of archive with ScenarioStructure.dat file
    def test_init3_default(self):
        self.assertTrue("ReferenceModel" not in sys.modules)
        with ScenarioTreeInstanceFactory(
                model=testdatadir,
                scenario_tree=join(testdatadir,
                                   "archive_test.tgz")) as factory:
            self.assertEqual(len(factory._archives), 1)
            self.assertTrue(factory.model_directory() is not None)
            self.assertTrue(factory.scenario_tree_directory() is not None)
            self._check_factory(factory)
        self.assertTrue("ReferenceModel" in sys.modules)

    # model: name of archive with .py file with model
    # scenario_tree: name of copy of archive with .dat file
    def test_init4(self):
        self.assertTrue("reference_test_model" not in sys.modules)
        archive_copy = self._get_testfname_prefix()+".archive_copy.tgz"
        shutil.copyfile(join(testdatadir, "archive_test.tgz"),
                        archive_copy)
        with ScenarioTreeInstanceFactory(
                model=join(testdatadir,
                           "archive_test.tgz,reference_test_model.py"),
                scenario_tree=join(testdatadir,
                                   archive_copy+",reference_test_scenario_tree.dat")) as factory:
            self.assertEqual(len(factory._archives), 2)
            self.assertTrue(factory.model_directory() is not None)
            self.assertTrue(factory.scenario_tree_directory() is not None)
            self._check_factory(factory)
        self.assertEqual(len(factory._archives), 0)
        os.remove(archive_copy)
        self.assertTrue("reference_test_model" in sys.modules)

    # model: name of archive with ReferenceModel.py file with model
    # scenario_tree: name of copy of archive with ScenarioStructure.dat file
    def test_init4_default(self):
        self.assertTrue("ReferenceModel" not in sys.modules)
        archive_copy = self._get_testfname_prefix()+".archive_copy.tgz"
        shutil.copyfile(join(testdatadir, "archive_test.tgz"),
                        archive_copy)
        with ScenarioTreeInstanceFactory(
                model=join(testdatadir, "archive_test.tgz,"),
                scenario_tree=join(testdatadir,
                                   archive_copy+",")) as factory:
            self.assertEqual(len(factory._archives), 2)
            self.assertTrue(factory.model_directory() is not None)
            self.assertTrue(factory.scenario_tree_directory() is not None)
            self._check_factory(factory)
        self.assertEqual(len(factory._archives), 0)
        os.remove(archive_copy)
        self.assertTrue("ReferenceModel" in sys.modules)

    # model: name of archive with .py file with model
    # scenario_tree: name of same archive with .dat file
    def test_init5(self):
        self.assertTrue("reference_test_model" not in sys.modules)
        with ScenarioTreeInstanceFactory(
                model=join(testdatadir,
                           "archive_test.tgz,reference_test_model.py"),
                scenario_tree=join(testdatadir,
                                   "archive_test.tgz,reference_test_scenario_tree.dat")) as factory:
            self.assertEqual(len(factory._archives), 1)
            self.assertTrue(factory.model_directory() is not None)
            self.assertTrue(factory.scenario_tree_directory() is not None)
            self._check_factory(factory)
        self.assertEqual(len(factory._archives), 0)
        self.assertTrue("reference_test_model" in sys.modules)

    # model: name of archive with ReferenceModel.py file with model
    # scenario_tree: name of same archive with ScenarioStructure.dat file
    def test_init5_default(self):
        self.assertTrue("ReferenceModel" not in sys.modules)
        with ScenarioTreeInstanceFactory(
                model=join(testdatadir, "archive_test.tgz,"),
                scenario_tree=join(testdatadir, "archive_test.tgz,")) as factory:
            self.assertEqual(len(factory._archives), 1)
            self.assertTrue(factory.model_directory() is not None)
            self.assertTrue(factory.scenario_tree_directory() is not None)
            self._check_factory(factory)
        self.assertEqual(len(factory._archives), 0)
        self.assertTrue("ReferenceModel" in sys.modules)

    # model: name of .py file with model
    # scenario_tree: Pyomo scenario tree model
    def test_init6(self):
        self.assertTrue("reference_test_model" not in sys.modules)
        scenario_tree_model = CreateAbstractScenarioTreeModel().\
            create_instance(
                join(testdatadir, "reference_test_scenario_tree.dat"))
        with ScenarioTreeInstanceFactory(
                model=join(testdatadir, "reference_test_model.py"),
                scenario_tree=scenario_tree_model) as factory:
            self.assertTrue(factory.model_directory() is not None)
            self.assertTrue(factory.scenario_tree_directory() is None)
            self._check_factory(factory)
        with self.assertRaises(TypeError):
            with ScenarioTreeInstanceFactory(
                    model=join(testdatadir, "reference_test_model.py"),
                    scenario_tree=int) as f:
                pass
        with self.assertRaises(ValueError):
            with ScenarioTreeInstanceFactory(
                    model=join(testdatadir, "reference_test_model.py"),
                    scenario_tree=None) as f:
                pass
        with self.assertRaises(TypeError):
            with ScenarioTreeInstanceFactory(
                    model=None,
                    scenario_tree=scenario_tree_model) as f:
                pass
        with self.assertRaises(IOError):
            with ScenarioTreeInstanceFactory(
                    model=join(testdatadir, "reference_test_model_does_not_exist.py"),
                    scenario_tree=scenario_tree_model) as f:
                pass
        with self.assertRaises(ValueError):
            with ScenarioTreeInstanceFactory(
                    model=join(testdatadir, "reference_test_model.py"),
                    scenario_tree=CreateAbstractScenarioTreeModel()) as f:
                pass
        self.assertEqual(len(factory._archives), 0)
        self.assertTrue("reference_test_model" in sys.modules)

    # model: Pyomo model
    # scenario_tree: Pyomo scenario tree model
    def test_init7(self):
        self.assertTrue("reference_test_model" not in sys.modules)
        scenario_tree_model = CreateAbstractScenarioTreeModel().\
            create_instance(
                join(testdatadir, "reference_test_scenario_tree.dat"))
        with self.assertRaises(ValueError):
            with ScenarioTreeInstanceFactory(
                    model=reference_test_model,
                    scenario_tree=scenario_tree_model) as factory:
                pass
        with ScenarioTreeInstanceFactory(
                model=reference_test_model,
                scenario_tree=scenario_tree_model,
                data=testdatadir) as factory:
            self.assertTrue(factory.model_directory() is None)
            self.assertTrue(factory.scenario_tree_directory() is None)
            self._check_factory(factory)
        self.assertEqual(factory._closed, True)
        with ScenarioTreeInstanceFactory(
                model=reference_test_model,
                scenario_tree=join(testdatadir,
                                   "reference_test_scenario_tree.dat")) as factory:
            self.assertTrue(factory.model_directory() is None)
            self.assertTrue(factory.scenario_tree_directory() is not None)
            self._check_factory(factory)
        self.assertEqual(len(factory._archives), 0)

    # model: Pyomo model
    # scenario_tree: name of .py file with Pyomo scenario tree model
    def test_init8(self):
        self.assertTrue("reference_test_model" not in sys.modules)
        self.assertTrue("reference_test_scenario_tree_model" not in sys.modules)
        with ScenarioTreeInstanceFactory(
                model=reference_test_model,
                scenario_tree=join(testdatadir,
                                   "reference_test_scenario_tree_model.py"),
                data=testdatadir) as factory:
            self.assertTrue(factory.model_directory() is None)
            self.assertTrue(factory.scenario_tree_directory() is not None)
            self.assertTrue(factory._scenario_tree_module is not None)
            self._check_factory(factory)
        self.assertEqual(factory._closed, True)
        self.assertEqual(len(factory._archives), 0)
        self.assertTrue("reference_test_model" not in sys.modules)
        self.assertTrue("reference_test_scenario_tree_model" in sys.modules)

    # model: name of .py file with model callback
    # scenario_tree: name of .dat file
    def test_init9(self):
        self.assertTrue("reference_test_model_with_callback" not in sys.modules)
        with ScenarioTreeInstanceFactory(
                model=join(testdatadir,
                           "reference_test_model_with_callback.py"),
                scenario_tree=join(testdatadir,
                                   "reference_test_scenario_tree.dat")) as factory:
            self.assertEqual(len(factory._archives), 0)
            self.assertTrue(factory.model_directory() is not None)
            self.assertTrue(factory.scenario_tree_directory() is not None)
            self._check_factory(factory)
        self.assertEqual(len(factory._archives), 0)
        self.assertTrue("reference_test_model_with_callback" in sys.modules)

    # model: name of directory with ReferenceModel.py file with model
    # scenario_tree: name of .dat file
    # data: name of directory with yaml files
    @unittest.skipIf(not yaml_available, "PyYAML is not available")
    def test_init10(self):
        with ScenarioTreeInstanceFactory(
                model=testdatadir,
                scenario_tree=join(testdatadir,
                                   "reference_test_scenario_tree.dat"),
                data=join(testdatadir, "yaml_data")) as factory:
            self.assertEqual(len(factory._archives), 0)
            self.assertTrue(factory.model_directory() is not None)
            self.assertTrue(factory.scenario_tree_directory() is not None)
            self.assertTrue(factory.data_directory(),
                            join(testdatadir, "yaml_data"))
            self._check_factory(factory)
        self.assertEqual(len(factory._archives), 0)

    # model: Pyomo model
    # scenario_tree: Pyomo scenario tree model
    # data: name of directory with .dat files
    def test_init11(self):
        self.assertTrue("reference_test_model" not in sys.modules)
        scenario_tree_model = CreateAbstractScenarioTreeModel().\
            create_instance(
                join(testdatadir, "reference_test_scenario_tree.dat"))
        scenario_tree_model.ScenarioBasedData = False
        with ScenarioTreeInstanceFactory(
                model=reference_test_model,
                scenario_tree=scenario_tree_model,
                data=testdatadir) as factory:
            self.assertTrue(factory.model_directory() is None)
            self.assertTrue(factory.scenario_tree_directory() is None)
            self._check_factory(factory)
        self.assertEqual(factory._closed, True)
        self.assertEqual(len(factory._archives), 0)

    # model: model callback
    # scenario_tree: Pyomo scenario tree model
    def test_init12(self):
        self.assertTrue("reference_test_model" not in sys.modules)
        scenario_tree_model = CreateAbstractScenarioTreeModel().\
            create_instance(
                join(testdatadir, "reference_test_scenario_tree.dat"))
        def scenario_model_callback(scenario_tree, scenario_name, node_list):
            self.assertIs(scenario_tree, scenario_tree_model)
            instance = reference_test_model.create_instance()
            if scenario_name == "s1":
                instance.p = 1.0
            elif scenario_name == "s2":
                instance.p = 2.0
            else:
                assert scenario_name == "s3"
                instance.p = 3.0
            return instance
        with ScenarioTreeInstanceFactory(
                model=scenario_model_callback,
                scenario_tree=scenario_tree_model) as factory:
            self.assertTrue(factory.model_directory() is None)
            self.assertTrue(factory.scenario_tree_directory() is None)
            self._check_factory(factory)
        self.assertEqual(factory._closed, True)
        self.assertEqual(len(factory._archives), 0)

    # model: Pyomo model
    # scenario_tree: Pyomo scenario tree model
    def test_init13(self):
        model = reference_test_model.create_instance()
        scenario_tree_model = CreateAbstractScenarioTreeModel().\
            create_instance(
                join(testdatadir, "reference_test_scenario_tree.dat"))
        with ScenarioTreeInstanceFactory(
                model=model,
                scenario_tree=scenario_tree_model) as factory:
            self.assertTrue(factory.model_directory() is None)
            self.assertTrue(factory.scenario_tree_directory() is None)

            scenario_tree = factory.generate_scenario_tree()
            instances = factory.construct_instances_for_scenario_tree(
                scenario_tree,
                verbose=True)
            self.assertEqual(len(instances), 3)
            self.assertEqual(instances["s1"].p(), model.p())
            self.assertEqual(instances["s2"].p(), model.p())
            self.assertEqual(instances["s3"].p(), model.p())

        self.assertEqual(factory._closed, True)
        self.assertEqual(len(factory._archives), 0)

    # model: Pyomo model
    # scenario_tree: Pyomo scenario tree model
    # data: name of a directory with .dat files
    def test_init14(self):
        self.assertTrue("reference_test_model" not in sys.modules)
        scenario_tree_model = CreateAbstractScenarioTreeModel().\
            create_instance(
                join(testdatadir, "reference_test_scenario_tree.dat"))
        with ScenarioTreeInstanceFactory(
                model=reference_test_model,
                scenario_tree=scenario_tree_model,
                data=testdatadir) as factory:
            scenario_tree = factory.generate_scenario_tree()
        self.assertEqual(factory._closed, True)
        self.assertEqual(len(factory._archives), 0)
        # start with a scenario tree (not a scenario tree model)
        with ScenarioTreeInstanceFactory(
                model=reference_test_model,
                scenario_tree=scenario_tree,
                data=testdatadir) as factory:
            self._check_factory(factory)
        self.assertEqual(factory._closed, True)
        self.assertEqual(len(factory._archives), 0)

    # model: name of a .py file with model
    # scenario_tree: name of a .py file
    # data: name of a directory with .dat files
    @unittest.skipIf(not has_networkx, "Requires networkx module")
    def test_init15(self):
        self.assertTrue("reference_test_model" not in sys.modules)
        with ScenarioTreeInstanceFactory(
                model=join(testdatadir,
                           "reference_test_model.py"),
                scenario_tree=join(testdatadir,
                                   "ScenarioStructure.py")) as factory:
            self.assertEqual(len(factory._archives), 0)
            self.assertTrue(factory.model_directory() is not None)
            self.assertTrue(factory.scenario_tree_directory() is not None)
            self._check_factory(factory)
        self.assertTrue("reference_test_model" in sys.modules)

    # model: name of a .py file with model
    # scenario_tree: networkx tree
    @unittest.skipIf(not has_networkx, "Requires networkx module")
    def test_init16(self):
        self.assertTrue("reference_test_model" not in sys.modules)
        self.assertTrue("ScenarioStructure" not in sys.modules)
        nx_tree = load_external_module(os.path.join(testdatadir,
                                                    "ScenarioStructure.py"))[0].G
        with ScenarioTreeInstanceFactory(
                model=join(testdatadir,
                           "reference_test_model.py"),
                scenario_tree=nx_tree) as factory:
            self.assertEqual(len(factory._archives), 0)
            self.assertTrue(factory.model_directory() is not None)
            self.assertTrue(factory.scenario_tree_directory() is None)
            self._check_factory(factory)
        self.assertTrue("reference_test_model" in sys.modules)
        self.assertTrue("ScenarioStructure" in sys.modules)

    # model: name of a .py file with model
    # scenario_tree: networkx tree
    @unittest.skipIf(not has_networkx, "Requires networkx module")
    def test_init17(self):
        self.assertTrue("reference_test_model" not in sys.modules)
        self.assertTrue("ScenarioStructure" not in sys.modules)
        nx_tree = load_external_module(os.path.join(testdatadir,
                                                    "ScenarioStructure.py"))[0].G
        with ScenarioTreeInstanceFactory(
                model=join(testdatadir,
                           "reference_test_model.py"),
                scenario_tree=nx_tree) as factory:
            self.assertEqual(len(factory._archives), 0)
            self.assertTrue(factory.model_directory() is not None)
            self.assertTrue(factory.scenario_tree_directory() is None)
            self._check_factory(factory)

            scenario_tree = factory.generate_scenario_tree()
            self.assertEqual(scenario_tree.contains_bundles(), False)
            # check that we can modify the networkx tree to redefine
            # bundles
            nx_tree.nodes["s1"]["bundle"] = 0
            nx_tree.nodes["s2"]["bundle"] = 0
            nx_tree.nodes["s3"]["bundle"] = 0
            scenario_tree = factory.generate_scenario_tree()
            self.assertEqual(scenario_tree.contains_bundles(), True)
            self.assertEqual(len(scenario_tree.bundles), 1)
            nx_tree.nodes["s1"]["bundle"] = 0
            nx_tree.nodes["s2"]["bundle"] = 1
            nx_tree.nodes["s3"]["bundle"] = 2
            scenario_tree = factory.generate_scenario_tree()
            self.assertEqual(scenario_tree.contains_bundles(), True)
            self.assertEqual(len(scenario_tree.bundles), 3)
            nx_tree.nodes["s1"]["bundle"] = None
            nx_tree.nodes["s2"]["bundle"] = None
            nx_tree.nodes["s3"]["bundle"] = None
            scenario_tree = factory.generate_scenario_tree()
            self.assertEqual(scenario_tree.contains_bundles(), False)

        self.assertTrue("reference_test_model" in sys.modules)
        self.assertTrue("ScenarioStructure" in sys.modules)

    # model: model callback
    # scenario_tree: networkx tree
    @unittest.skipIf(not has_networkx, "Requires networkx module")
    def test_init18(self):
        self.assertTrue("reference_test_model" not in sys.modules)
        self.assertTrue("ScenarioStructure" not in sys.modules)
        nx_tree = load_external_module(os.path.join(testdatadir,
                                                    "ScenarioStructure.py"))[0].G
        def scenario_model_callback(scenario_tree, scenario_name, node_list):
            self.assertIs(scenario_tree, nx_tree)
            instance = reference_test_model.create_instance()
            if scenario_name == "s1":
                instance.p = 1.0
            elif scenario_name == "s2":
                instance.p = 2.0
            else:
                assert scenario_name == "s3"
                instance.p = 3.0
            return instance
        with ScenarioTreeInstanceFactory(
                model=scenario_model_callback,
                scenario_tree=nx_tree) as factory:
            self.assertTrue(factory.model_directory() is None)
            self.assertTrue(factory.scenario_tree_directory() is None)
            self._check_factory(factory)
        self.assertEqual(factory._closed, True)
        self.assertEqual(len(factory._archives), 0)

    # model: model callback
    # scenario_tree: name of .dat file
    def test_init19(self):
        self.assertTrue("reference_test_model" not in sys.modules)
        def scenario_model_callback(scenario_tree, scenario_name, node_list):
            self.assertTrue(isinstance(scenario_tree, ScenarioTree))
            instance = reference_test_model.create_instance()
            if scenario_name == "s1":
                instance.p = 1.0
            elif scenario_name == "s2":
                instance.p = 2.0
            else:
                assert scenario_name == "s3"
                instance.p = 3.0
            return instance
        with ScenarioTreeInstanceFactory(
                model=scenario_model_callback,
                scenario_tree=join(testdatadir,
                                   "reference_test_scenario_tree.dat")) as factory:
            self.assertTrue(factory.model_directory() is None)
            self.assertTrue(factory.scenario_tree_directory() is not None)
            self._check_factory(factory)
        self.assertEqual(factory._closed, True)
        self.assertEqual(len(factory._archives), 0)

    # model: model callback
    # scenario_tree: PySP scenario tree
    def test_init20(self):
        self.assertTrue("reference_test_model" not in sys.modules)
        def scenario_model_callback(scenario_tree, scenario_name, node_list):
            self.assertTrue(isinstance(scenario_tree, ScenarioTree))
            instance = reference_test_model.create_instance()
            if scenario_name == "s1":
                instance.p = 1.0
            elif scenario_name == "s2":
                instance.p = 2.0
            else:
                assert scenario_name == "s3"
                instance.p = 3.0
            return instance
        with ScenarioTreeInstanceFactory(
                model=scenario_model_callback,
                scenario_tree=join(testdatadir,
                                   "reference_test_scenario_tree.dat")) as factory:
            my_scenario_tree = factory.generate_scenario_tree()
        with ScenarioTreeInstanceFactory(
                model=scenario_model_callback,
                scenario_tree=my_scenario_tree) as factory:
            self.assertTrue(factory.model_directory() is None)
            self.assertTrue(factory.scenario_tree_directory() is None)
            self._check_factory(factory)
        self.assertEqual(factory._closed, True)
        self.assertEqual(len(factory._archives), 0)

    # model: name of .py file with model callback
    # scenario_tree: name of same .py file with scenario tree callback
    @unittest.skipIf(not has_networkx, "Requires networkx module")
    def test_init21(self):
        self.assertTrue("both_callbacks" not in sys.modules)
        with ScenarioTreeInstanceFactory(
                model=join(testdatadir,
                           "both_callbacks.py"),
                scenario_tree=join(testdatadir,
                                   "both_callbacks.py")) as factory:
            self.assertEqual(len(factory._archives), 0)
            self.assertTrue(factory.model_directory() is not None)
            self.assertTrue(factory.scenario_tree_directory() is not None)
            self._check_factory(factory)
        self.assertEqual(len(factory._archives), 0)
        self.assertTrue("both_callbacks" in sys.modules)

    # model: name of .py file with model callback
    # scenario_tree: None (which implies the model filename)
    @unittest.skipIf(not has_networkx, "Requires networkx module")
    def test_init22(self):
        self.assertTrue("both_callbacks" not in sys.modules)
        with ScenarioTreeInstanceFactory(
                model=join(testdatadir,
                           "both_callbacks.py"),
                scenario_tree=None) as factory:
            self.assertEqual(len(factory._archives), 0)
            self.assertTrue(factory.model_directory() is not None)
            self.assertTrue(factory.scenario_tree_directory() is not None)
            self._check_factory(factory)
        self.assertEqual(len(factory._archives), 0)
        self.assertTrue("both_callbacks" in sys.modules)

if __name__ == "__main__":
    unittest.main()
