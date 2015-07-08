#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

__all__ = ('ScenarioTreeInstanceFactory',)

import os
import gc
import posixpath
import tempfile
import shutil
import logging

from pyutilib.misc import (ArchiveReaderFactory,
                           ArchiveReader,
                           import_file)

from pyomo.core import (Block,
                        IPyomoScriptModifyInstance,
                        DataPortal)
from pyomo.core.base.block import _BlockData
from pyomo.util.plugin import ExtensionPoint
from pyomo.pysp.phutils import (load_external_module,
                                _OLD_OUTPUT)

from pyomo.pysp.scenariotree.tree_structure_model import CreateAbstractScenarioTreeModel
from pyomo.pysp.scenariotree import ScenarioTree

logger = logging.getLogger('pyomo.pysp')

class ScenarioTreeInstanceFactory(object):

    def __init__(self, model, data=None, verbose=False):

        self._model_spec = model
        self._data_spec = data
        self._verbose = verbose

        self._model_directory = None
        self._model_filename = None
        self._model_archive = None
        self._data_directory = None
        self._data_filename = None
        self._data_archive = None
        self._tmpdirs = []
        # Define the above by inspecting model_spec and data_spec
        try:
            self._extract_model_and_data_locations()
        except:
            self.close()
            raise

        self._model_object = None
        self._model_callback = None
        self._scenario_tree_instance = None
        self._scenario_tree = None
        # Define the above by inspecting self._model_filename
        try:
            self._import_model_and_data()
        except:
            self.close()
            raise

    def __getstate__(self):
        self.close()
        raise NotImplementedError("Do not deepcopy or serialize this class")

    def __setstate__(self,d):
        self.close()
        raise NotImplementedError("Do not deepcopy or serialize this class")

    def close(self):
        for dirname in self._tmpdirs:
            if os.path.exists(dirname):
                shutil.rmtree(dirname, True)
        if self._model_archive is not None:
            self._model_archive.close()
        if self._data_archive is not None:
            self._data_archive.close()

    #
    # Support "with" statements. Forgetting to call close()
    # on this class can result in temporary unarchived
    # directories being left sitting around
    #
    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def model_directory(self):
        return self._model_directory

    def data_directory(self):
        return self._data_directory

    #
    # construct a scenario instance - just like it sounds!
    #
    def _construct_scenario_instance(self,
                                     scenario_name,
                                     scenario_tree,
                                     report_timing=False):

        if not scenario_tree.contains_scenario(scenario_name):
            raise ValueError("ScenarioTree does not contain scenario "
                             "with name %s." % (scenario_name))

        scenario = scenario_tree.get_scenario(scenario_name)
        node_name_list = [n._name for n in scenario._node_list]

        if self._verbose:
            print("Creating instance for scenario=%s" % (scenario_name))

        scenario_instance = None

        try:

            if self._model_callback is not None:

                assert self._model_object is None
                scenario_instance = self._model_callback(scenario_name, node_name_list)

            elif self._model_object is not None:

                if scenario_tree._scenario_based_data:

                    scenario_data_filename = \
                        os.path.join(self._data_directory,
                                     str(scenario_name))
                    # JPW: The following is a hack to support initialization
                    #      of block instances, which don't work with .dat
                    #      files at the moment. Actually, it's not that bad of
                    #      a hack - it just needs to be extended a bit, and
                    #      expanded into the node-based data read logic (where
                    #      yaml is completely ignored at the moment.
                    if os.path.exists(scenario_data_filename+'.dat'):
                        scenario_data_filename = scenario_data_filename + ".dat"
                        data = None
                    elif os.path.exists(scenario_data_filename+'.yaml'):
                        import yaml
                        scenario_data_filename = scenario_data_filename + ".yaml"
                        yaml_input_file=open(scenario_data_filename,"r")
                        data = yaml.load(yaml_input_file)
                        yaml_input_file.close()
                    else:
                        raise RuntimeError("Cannot find the scenario data for "
                                           + scenario_data_filename)
                    if self._verbose:
                        print("Data for scenario=%s loads from file=%s"
                              % (scenario_name, scenario_data_filename))
                    if data is None:
                        scenario_instance = \
                            self._model_object.create_instance(filename=scenario_data_filename,
                                                               preprocess=False,
                                                               report_timing=report_timing)
                    else:
                        scenario_instance = \
                            self._model_object.create_instance(data,
                                                               preprocess=False,
                                                               report_timing=report_timing)
                else:

                    data_files = []
                    for node_name in node_name_list:
                        node_data_filename = \
                            os.path.join(self._data_directory,
                                         str(node_name)+".dat")
                        if not os.path.exists(node_data_filename):
                            raise RuntimeError("Node data file="
                                               +node_data_filename+
                                               " does not exist or cannot be accessed")
                        data_files.append(node_data_filename)

                    scenario_data = DataPortal(model=self._model_object)
                    for data_file in data_files:
                        if self._verbose:
                            print("Node data for scenario=%s partially "
                                  "loading from file=%s"
                                  % (scenario_name, data_file))
                        scenario_data.load(filename=data_file)

                    scenario_instance = self._model_object.create_instance(scenario_data,
                                                                           preprocess=False,
                                                                           report_timing=report_timing)
            else:
                raise RuntimeError("Unable to construct scenario instance. "
                                   "Neither a reference model or callback "
                                   "is defined.")

            # name each instance with the scenario name
            scenario_instance.name = scenario_name

            # apply each of the post-instance creation plugins. this
            # really shouldn't be associated (in terms of naming) with the
            # pyomo script - this should be rectified with a workflow
            # re-work. it is unclear how this interacts, or doesn't, with
            # the preprocessors.
            ep = ExtensionPoint(IPyomoScriptModifyInstance)
            for ep in ExtensionPoint(IPyomoScriptModifyInstance):
                logger.warn("DEPRECATION WARNING: IPyomoScriptModifyInstance extension "
                            "point callbacks will be ignored by PySP in the future")
                ep.apply(options=None,
                         model=reference_model,
                         instance=scenario_instance)

        except Exception as exc:
            msg = ("Failed to create model instance "
                   "for scenario=%s"
                   % (scenario_name))
            print(msg)
            raise

        return scenario_instance

    def construct_instances_for_scenario_tree(self,
                                              scenario_tree,
                                              report_timing=False):

        if scenario_tree._scenario_instance_factory is not self:
            raise RuntimeError("Can not construct scenario tree instances. "
                               "The scenario tree was not generated by this "
                               "instance factory.")

        # the construction of instances takes little overhead in terms
        # of memory potentially lost in the garbage-collection sense
        # (mainly only that due to parsing and instance
        # simplification/prep-processing).  to speed things along,
        # disable garbage collection if it enabled in the first place
        # through the instance construction process.
        # IDEA: If this becomes too much for truly large numbers of
        #       scenarios, we could manually collect every time X
        #       instances have been created.
        re_enable_gc = False
        if gc.isenabled():
            re_enable_gc = True
            gc.disable()

        if scenario_tree._scenario_based_data:
            if self._verbose is True:
                print("Scenario-based instance initialization enabled")
        else:
            if self._verbose is True:
                print("Node-based instance initialization enabled")

        scenario_instances = {}
        for scenario in scenario_tree._scenarios:

            scenario_instance = \
                self._construct_scenario_instance(
                    scenario._name,
                    scenario_tree,
                    report_timing=report_timing)

            scenario_instances[scenario._name] = scenario_instance
            assert scenario_instance.name == scenario._name

        if re_enable_gc:
            gc.enable()

        return scenario_instances

    def _extract_model_and_data_locations(self):

        model_filename = None
        model_archive = None
        model_archive_inputs = (None,None)
        model_unarchived_dir = None
        try:
            # un-archive the model directory if necessary
            normalized_model_spec = None
            model_archive_subdir = None
            modelname = "ReferenceModel.py"
            if not os.path.exists(self._model_spec):
                normalized_model_spec, _, model_archive_subdir = \
                    ArchiveReader.normalize_name(self._model_spec).rpartition(',')
                if model_archive_subdir.endswith('.py') or \
                   model_archive_subdir.endswith('.pyc'):
                    modelname = os.path.basename(model_archive_subdir)
                    model_archive_subdir = os.path.dirname(model_archive_subdir)
                model_archive_subdir = model_archive_subdir.strip()
                if model_archive_subdir == '':
                    model_archive_subdir = None
            else:
                normalized_model_spec = ArchiveReader.normalize_name(self._model_spec)

            if ArchiveReader.isArchivedFile(normalized_model_spec):
                model_archive = ArchiveReaderFactory(normalized_model_spec,
                                               subdir=model_archive_subdir)
                self._model_archive = model_archive
                model_archive_inputs = (normalized_model_spec,model_archive_subdir)
                model_unarchived_dir = model_archive.normalize_name(
                    tempfile.mkdtemp(prefix='pysp_unarchived',
                                     dir=os.path.dirname(normalized_model_spec)))
                self._tmpdirs.append(model_unarchived_dir)
                print("Model directory unarchiving to: %s" % model_unarchived_dir)
                model_archive.extractall(path=model_unarchived_dir)
                model_filename = \
                    posixpath.join(model_unarchived_dir, modelname)
            else:
                if model_archive_subdir is not None:
                    model_unarchived_dir = posixpath.join(normalized_model_spec,
                                                        model_archive_subdir)
                else:
                    model_unarchived_dir = normalized_model_spec

                if not os.path.isfile(model_unarchived_dir):
                    model_filename = \
                        posixpath.join(model_unarchived_dir, modelname)
                else:
                    model_filename = model_unarchived_dir
            if not os.path.exists(model_filename):
                raise RuntimeError("Model input does not exist: "
                                   +str(model_filename))
        except:
            print("***ERROR: Failed to locate reference "
                  "model file with specification string: "
                  +self._model_spec)
            raise

        self._model_filename = model_filename
        self._model_directory = os.path.dirname(model_filename)

        if self._data_spec is None:
            return

        data_filename = None
        data_archive = None
        try:
            # un-archive the data directory if necessary
            normalized_data_spec = None
            data_archive_subdir = None
            data_unarchived_dir = None
            dataname = "ScenarioStructure.dat"
            if not os.path.exists(self._data_spec):
                normalized_data_spec, _, data_archive_subdir = \
                    ArchiveReader.normalize_name(self._data_spec).rpartition(',')
                if data_archive_subdir.endswith('.dat'):
                    dataname = os.path.basename(data_archive_subdir)
                    data_archive_subdir = os.path.dirname(data_archive_subdir)
                data_archive_subdir = data_archive_subdir.strip()
                if data_archive_subdir == '':
                    data_archive_subdir = None
            else:
                normalized_data_spec = ArchiveReader.normalize_name(self._data_spec)

            if ArchiveReader.isArchivedFile(normalized_data_spec):
                if (normalized_data_spec == model_archive_inputs[0]) and \
                   ((model_archive_inputs[1] is None) or \
                    ((data_archive_subdir is not None) and \
                     (data_archive_subdir.startswith(model_archive_inputs[1]+'/')))):
                    # The scenario tree data has already been extracted with the
                    # model archive, no need to extract again
                    print("Data directory found in unarchived model directory")
                    data_unarchived_dir = model_unarchived_dir
                    if data_archive_subdir is not None:
                        if model_archive_inputs[1] is not None:
                            data_unarchived_dir = \
                                posixpath.join(data_unarchived_dir,
                                             os.path.relpath(data_archive_subdir,
                                                             start=model_archive_inputs[1]))
                        else:
                            data_unarchived_dir = posixpath.join(data_unarchived_dir,
                                                               data_archive_subdir)
                else:
                    data_archive = ArchiveReaderFactory(normalized_data_spec,
                                                        subdir=data_archive_subdir)
                    data_unarchived_dir = \
                        data_archive.normalize_name(
                            tempfile.mkdtemp(prefix='pysp_unarchived',
                                             dir=os.path.dirname(normalized_data_spec)))
                    self._tmpdirs.append(data_unarchived_dir)
                    print("Data directory unarchiving to: %s" % data_unarchived_dir)
                    data_archive.extractall(path=data_unarchived_dir)

                data_filename = \
                    posixpath.join(data_unarchived_dir, dataname)
            else:
                if data_archive_subdir is not None:
                    data_unarchived_dir = posixpath.join(normalized_data_spec,
                                                        data_archive_subdir)
                else:
                    data_unarchived_dir = normalized_data_spec

                if not os.path.isfile(data_unarchived_dir):
                    data_filename = \
                        posixpath.join(data_unarchived_dir, dataname)
                else:
                    data_filename = data_unarchived_dir
            if not os.path.exists(data_filename):
                raise RuntimeError("Scenario data input does not exist: "
                                   +str(data_filename))
        except:
            print("***ERROR: Failed to locate scenario tree structure "
                  "file with specification string: "
                  +self._data_spec)
            raise

        self._data_filename = data_filename
        self._data_directory = os.path.dirname(data_filename)

    def _import_model_and_data(self):

        #if not _OLD_OUTPUT:
        #    module_name, model_import = load_external_module(self._model_filename)
        #else:
        model_import = import_file(self._model_filename, clear_cache=True)

        self._model_object = None
        self._model_callback = None
        if "pysp_instance_creation_callback" in dir(model_import):
            callback = model_import.pysp_instance_creation_callback
            if not hasattr(callback,"__call__"):
                raise TypeError("'pysp_instance_creation_callback' object is "
                                "not callable in model file: %s"
                                % (self._model_filename))
            self._model_callback = callback
        elif "model" in dir(model_import):
            model = model_import.model
            if not isinstance(model,(_BlockData, Block)):
                raise TypeError("'model' object has incorrect type "
                                "in model file: "+self._model_filename)
            self._model_object = model
        else:
            raise AttributeError("No 'model' or 'pysp_instance_creation_callback' "
                                 "object found in model file: "+self._model_filename)

        if self._data_filename is None:
            assert self._data_spec is None
            if "pysp_scenario_tree_model_callback" in dir(model_import):
                callback = model_import.pysp_scenario_tree_model_callback
                if not hasattr(callback,"__call__"):
                    raise TypeError("'pysp_scenario_tree_model_callback' object is "
                                    "not callable in model file: %s"
                                    % (self._model_filename))
                self._scenario_tree_instance = callback()
                if not isinstance(self._scenario_tree_instance, (_BlockData, Block)):
                    raise TypeError("'pysp_scenario_tree_model_callback' returned "
                                    "an object that is not of the correct type for "
                                    "a Pyomo model (e.g, _BockData, Block): %s"
                                    % (type(self._scenario_tree_instance)))
            else:
                raise ValueError("No scenario tree file was given but no function "
                                 "named 'pysp_scenario_tree_model_callback' was "
                                 "found in the model file.")
        else:
            self._scenario_tree_instance = \
                CreateAbstractScenarioTreeModel().\
                create_instance(filename=self._data_filename)

    def generate_scenario_tree(self,
                               downsample_fraction=1.0,
                               include_scenarios=None,
                               bundles_file=None,
                               random_bundles=None,
                               random_seed=None):

        scenario_tree_instance = self._scenario_tree_instance
        bundles_file_path = None
        if bundles_file is not None:
            # we interpret the scenario bundle specification in one of
            # two ways. if the supplied name is a file, it is used
            # directly. otherwise, it is interpreted as the root of a
            # file with a .dat suffix to be found in the instance
            # directory.
            if os.path.exists(os.path.expanduser(bundles_file)):
                bundles_file_path = \
                    os.path.expanduser(bundles_file)
            else:
                bundles_file_path = \
                    os.path.join(self._data_directory,
                                 bundles_file+".dat")

            if self._verbose:
                if bundles_file_path is not None:
                    print("Scenario tree bundle specification filename="
                          +bundles_file_path)

            scenario_tree_instance = scenario_tree_instance.clone()
            scenario_tree_instance.Bundling._constructed = False
            scenario_tree_instance.Bundles._constructed = False
            scenario_tree_instance.BundleScenarios._constructed = False
            scenario_tree_instance.load(bundles_file_path)

        #
        # construct the scenario tree
        #
        scenario_tree = ScenarioTree(scenariotreeinstance=scenario_tree_instance,
                                     scenariobundlelist=include_scenarios)

        # compress/down-sample the scenario tree, if operation is
        # required. and the\ option exists!
        if (downsample_fraction is not None) and \
           (downsample_fraction < 1.0):
            scenario_tree.downsample(downsample_fraction,
                                     random_seed,
                                     self._verbose)

        #
        # create random bundles, if the user has specified such.
        #
        if (random_bundles is not None) and \
           (random_bundles > 0):
            if bundles_file is not None:
                raise ValueError("Cannot specify both random "
                                 "bundles and a bundles filename")

            num_scenarios = len(scenario_tree._scenarios)
            if random_bundles > num_scenarios:
                raise ValueError("Cannot create more random bundles "
                                 "than there are scenarios!")

            print("Creating "+str(random_bundles)+
                  " random bundles using seed="
                  +str(random_seed))

            scenario_tree.create_random_bundles(self._scenario_tree_instance,
                                                random_bundles,
                                                random_seed)

        scenario_tree._scenario_instance_factory = self

        return scenario_tree
