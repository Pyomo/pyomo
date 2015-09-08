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
import time
import posixpath
import tempfile
import shutil
import logging

from pyutilib.misc import (ArchiveReaderFactory,
                           ArchiveReader,
                           import_file,
                           PauseGC)

from pyomo.core import (Block,
                        IPyomoScriptModifyInstance,
                        DataPortal)
from pyomo.core.base.block import _BlockData
from pyomo.util.plugin import ExtensionPoint
from pyomo.repn.beta.matrix import \
    compile_block_linear_constraints
from pyomo.pysp.phutils import _OLD_OUTPUT
from pyomo.pysp.util.misc import load_external_module
from pyomo.pysp.scenariotree.tree_structure_model import \
    CreateAbstractScenarioTreeModel
from pyomo.pysp.scenariotree.tree_structure import \
    ScenarioTree

logger = logging.getLogger('pyomo.pysp')

class ScenarioTreeInstanceFactory(object):

    def __init__(self, model_location, scenario_tree_location=None, verbose=False):

        self._model_location = model_location
        self._scenario_tree_location = scenario_tree_location
        self._verbose = verbose

        self._model_directory = None
        self._model_filename = None
        self._model_archive = None
        self._scenario_tree_directory = None
        self._scenario_tree_filename = None
        self._scenario_tree_archive = None
        self._tmpdirs = []
        # Define the above by inspecting model_location and scenario_tree_location
        try:
            self._extract_model_and_scenario_tree_locations()
        except:
            self.close()
            raise

        self._model_object = None
        self._model_callback = None
        self._scenario_tree_instance = None
        # Define the above by inspecting self._model_filename
        try:
            self._import_model_and_scenario_tree()
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
        if self._scenario_tree_archive is not None:
            self._scenario_tree_archive.close()

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

    def scenario_tree_directory(self):
        return self._scenario_tree_directory

    #
    # construct a scenario instance - just like it sounds!
    #
    def construct_scenario_instance(self,
                                    scenario_name,
                                    scenario_tree,
                                    profile_memory=False,
                                    output_instance_construction_time=False,
                                    compile_instance=False):

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
                scenario_instance = self._model_callback(scenario_name,
                                                         node_name_list)

            elif self._model_object is not None:

                if scenario_tree._scenario_based_data:

                    scenario_data_filename = \
                        os.path.join(self._scenario_tree_directory,
                                     str(scenario_name))
                    # JPW: The following is a hack to support
                    #      initialization of block instances, which
                    #      don't work with .dat files at the
                    #      moment. Actually, it's not that bad of a
                    #      hack - it just needs to be extended a bit,
                    #      and expanded into the node-based data read
                    #      logic (where yaml is completely ignored at
                    #      the moment.
                    if os.path.exists(scenario_data_filename+'.dat'):
                        scenario_data_filename = \
                            scenario_data_filename + ".dat"
                        data = None
                    elif os.path.exists(scenario_data_filename+'.yaml'):
                        import yaml
                        scenario_data_filename = \
                            scenario_data_filename + ".yaml"
                        yaml_input_file=open(scenario_data_filename,"r")
                        data = yaml.load(yaml_input_file)
                        yaml_input_file.close()
                    else:
                        raise RuntimeError(
                            "Cannot find the scenario data for "
                            + scenario_data_filename)
                    if self._verbose:
                        print("Data for scenario=%s loads from file=%s"
                              % (scenario_name, scenario_data_filename))
                    if data is None:
                        scenario_instance = \
                            self._model_object.create_instance(
                                filename=scenario_data_filename,
                                preprocess=False,
                                profile_memory=profile_memory,
                                report_timing=output_instance_construction_time)
                    else:
                        scenario_instance = \
                            self._model_object.create_instance(
                                data,
                                preprocess=False,
                                profile_memory=profile_memory,
                                report_timing=output_instance_construction_time)
                else:

                    data_files = []
                    for node_name in node_name_list:
                        node_data_filename = \
                            os.path.join(self._scenario_tree_directory,
                                         str(node_name)+".dat")
                        if not os.path.exists(node_data_filename):
                            raise RuntimeError(
                                "Node data file="+node_data_filename+
                                " does not exist or cannot be accessed")
                        data_files.append(node_data_filename)

                    scenario_data = DataPortal(model=self._model_object)
                    for data_file in data_files:
                        if self._verbose:
                            print("Node data for scenario=%s partially "
                                  "loading from file=%s"
                                  % (scenario_name, data_file))
                        scenario_data.load(filename=data_file)

                    scenario_instance = self._model_object.create_instance(
                        scenario_data,
                        preprocess=False,
                        profile_memory=profile_memory,
                        report_timing=output_instance_construction_time)
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
                logger.warn(
                    "DEPRECATED: IPyomoScriptModifyInstance extension "
                    "point callbacks will be ignored by PySP in the future")
                ep.apply(options=None,
                         model=reference_model,
                         instance=scenario_instance)

            if compile_instance:
                compile_block_linear_constraints(
                    scenario_instance,
                    "_PySP_compiled_linear_constraints",
                    verbose=self._verbose)

        except Exception as exc:
            msg = ("Failed to create model instance for scenario=%s"
                   % (scenario_name))
            print(msg)
            raise

        return scenario_instance

    def construct_instances_for_scenario_tree(
            self,
            scenario_tree,
            profile_memory=False,
            output_instance_construction_time=False,
            compile_scenario_instances=False):

        if scenario_tree._scenario_based_data:
            if self._verbose is True:
                print("Scenario-based instance initialization enabled")
        else:
            if self._verbose is True:
                print("Node-based instance initialization enabled")

        scenario_instances = {}
        for scenario in scenario_tree._scenarios:

            # the construction of instances takes little overhead in terms
            # of memory potentially lost in the garbage-collection sense
            # (mainly only that due to parsing and instance
            # simplification/prep-processing).  to speed things along,
            # disable garbage collection if it enabled in the first place
            # through the instance construction process.
            # IDEA: If this becomes too much for truly large numbers of
            #       scenarios, we could manually collect every time X
            #       instances have been created.
            scenario_instance = None
            with PauseGC() as pgc:
                scenario_instance = \
                    self.construct_scenario_instance(
                        scenario._name,
                        scenario_tree,
                        profile_memory=profile_memory,
                        output_instance_construction_time=output_instance_construction_time,
                        compile_instance=compile_scenario_instances)

            scenario_instances[scenario._name] = scenario_instance
            assert scenario_instance.name == scenario._name

        return scenario_instances

    def _extract_model_and_scenario_tree_locations(self):

        model_filename = None
        model_archive = None
        model_archive_inputs = (None,None)
        model_unarchived_dir = None
        try:
            # un-archive the model directory if necessary
            normalized_model_location = None
            model_archive_subdir = None
            model_basename = "ReferenceModel.py"
            if not os.path.exists(self._model_location):
                normalized_model_location, _, model_archive_subdir = \
                    ArchiveReader.normalize_name(self._model_location).rpartition(',')
                if model_archive_subdir.endswith('.py') or \
                   model_archive_subdir.endswith('.pyc'):
                    model_basename = os.path.basename(model_archive_subdir)
                    model_archive_subdir = os.path.dirname(model_archive_subdir)
                model_archive_subdir = model_archive_subdir.strip()
                if model_archive_subdir == '':
                    model_archive_subdir = None
            else:
                normalized_model_location = \
                    ArchiveReader.normalize_name(self._model_location)

            if ArchiveReader.isArchivedFile(normalized_model_location):
                model_archive = ArchiveReaderFactory(normalized_model_location,
                                               subdir=model_archive_subdir)
                self._model_archive = model_archive
                model_archive_inputs = (normalized_model_location,model_archive_subdir)
                model_unarchived_dir = model_archive.normalize_name(
                    tempfile.mkdtemp(prefix='pysp_unarchived',
                                     dir=os.path.dirname(normalized_model_location)))
                self._tmpdirs.append(model_unarchived_dir)
                print("Model directory unarchiving to: %s" % model_unarchived_dir)
                model_archive.extractall(path=model_unarchived_dir)
                model_filename = \
                    posixpath.join(model_unarchived_dir, model_basename)
            else:
                if model_archive_subdir is not None:
                    model_unarchived_dir = posixpath.join(normalized_model_location,
                                                        model_archive_subdir)
                else:
                    model_unarchived_dir = normalized_model_location

                if not os.path.isfile(model_unarchived_dir):
                    model_filename = \
                        posixpath.join(model_unarchived_dir, model_basename)
                else:
                    model_filename = model_unarchived_dir
            if not os.path.exists(model_filename):
                raise IOError("Model input does not exist: "+str(model_filename))
        except:
            print("***ERROR: Failed to locate reference "
                  "model file with location string: "
                  +self._model_location)
            raise

        self._model_filename = model_filename
        self._model_directory = os.path.dirname(model_filename)

        if self._scenario_tree_location is None:
            return

        scenario_tree_filename = None
        scenario_tree_archive = None
        try:
            # un-archive the scenario tree directory if necessary
            normalized_scenario_tree_location = None
            scenario_tree_archive_subdir = None
            scenario_tree_unarchived_dir = None
            scenario_tree_basename = "ScenarioStructure.dat"
            if not os.path.exists(self._scenario_tree_location):
                normalized_scenario_tree_location, _, scenario_tree_archive_subdir = \
                    ArchiveReader.normalize_name(
                        self._scenario_tree_location).rpartition(',')
                if scenario_tree_archive_subdir.endswith('.dat'):
                    scenario_tree_basename = \
                        os.path.basename(scenario_tree_archive_subdir)
                    scenario_tree_archive_subdir = \
                        os.path.dirname(scenario_tree_archive_subdir)
                scenario_tree_archive_subdir = scenario_tree_archive_subdir.strip()
                if scenario_tree_archive_subdir == '':
                    scenario_tree_archive_subdir = None
            else:
                normalized_scenario_tree_location = \
                    ArchiveReader.normalize_name(self._scenario_tree_location)

            if ArchiveReader.isArchivedFile(normalized_scenario_tree_location):
                if (normalized_scenario_tree_location == model_archive_inputs[0]) and \
                   ((model_archive_inputs[1] is None) or \
                    ((scenario_tree_archive_subdir is not None) and \
                     (scenario_tree_archive_subdir.startswith(model_archive_inputs[1]+'/')))):
                    # The scenario tree has already been extracted with the
                    # model archive, no need to extract again
                    print("Scenario tree directory found in unarchived model directory")
                    scenario_tree_unarchived_dir = model_unarchived_dir
                    if scenario_tree_archive_subdir is not None:
                        if model_archive_inputs[1] is not None:
                            scenario_tree_unarchived_dir = \
                                posixpath.join(
                                    scenario_tree_unarchived_dir,
                                    os.path.relpath(scenario_tree_archive_subdir,
                                                    start=model_archive_inputs[1]))
                        else:
                            scenario_tree_unarchived_dir = posixpath.join(
                                scenario_tree_unarchived_dir,
                                scenario_tree_archive_subdir)
                else:
                    scenario_tree_archive = ArchiveReaderFactory(
                        normalized_scenario_tree_location,
                        subdir=scenario_tree_archive_subdir)
                    scenario_tree_unarchived_dir = \
                        scenario_tree_archive.normalize_name(
                            tempfile.mkdtemp(
                                prefix='pysp_unarchived',
                                dir=os.path.dirname(normalized_scenario_tree_location)))
                    self._tmpdirs.append(scenario_tree_unarchived_dir)
                    print("Scenario tree directory unarchiving to: %s"
                          % scenario_tree_unarchived_dir)
                    scenario_tree_archive.extractall(path=scenario_tree_unarchived_dir)

                scenario_tree_filename = \
                    posixpath.join(scenario_tree_unarchived_dir,
                                   scenario_tree_basename)
            else:
                if scenario_tree_archive_subdir is not None:
                    scenario_tree_unarchived_dir = posixpath.join(
                        normalized_scenario_tree_location,
                        scenario_tree_archive_subdir)
                else:
                    scenario_tree_unarchived_dir = normalized_scenario_tree_location

                if not os.path.isfile(scenario_tree_unarchived_dir):
                    scenario_tree_filename = \
                        posixpath.join(scenario_tree_unarchived_dir,
                                       scenario_tree_basename)
                else:
                    scenario_tree_filename = scenario_tree_unarchived_dir
            if not os.path.exists(scenario_tree_filename):
                raise IOError("Input does not exist: "+str(scenario_tree_filename))
        except:
            print("***ERROR: Failed to locate scenario tree structure "
                  "file with location string: "
                  +self._scenario_tree_location)
            raise

        self._scenario_tree_filename = scenario_tree_filename
        self._scenario_tree_directory = os.path.dirname(scenario_tree_filename)

    def _import_model_and_scenario_tree(self):

        #if not _OLD_OUTPUT:
        #    module_name, model_import = load_external_module(self._model_filename)
        #else:

        model_import = import_file(self._model_filename, clear_cache=True)

        dir_model_import = dir(model_import)
        self._model_object = None
        self._model_callback = None
        if "pysp_instance_creation_callback" in dir_model_import:
            callback = model_import.pysp_instance_creation_callback
            if not hasattr(callback,"__call__"):
                raise TypeError(
                    "'pysp_instance_creation_callback' object is "
                    "not callable in model file: %s"
                    % (self._model_filename))
            self._model_callback = callback
        elif "model" in dir_model_import:
            model = model_import.model
            if not isinstance(model,(_BlockData, Block)):
                raise TypeError(
                    "'model' object has incorrect type "
                    "in model file: %s"
                    % (self._model_filename))
            self._model_object = model
        else:
            raise AttributeError(
                "No 'model' or 'pysp_instance_creation_callback' "
                "object found in model file: %s"
                % (self._model_filename))

        if self._scenario_tree_filename is None:
            assert self._scenario_tree_location is None
            if "pysp_scenario_tree_model_callback" in dir_model_import:
                callback = model_import.pysp_scenario_tree_model_callback
                if not hasattr(callback,"__call__"):
                    raise TypeError(
                        "'pysp_scenario_tree_model_callback' object is "
                        "not callable in model file: %s"
                        % (self._model_filename))
                self._scenario_tree_instance = callback()
                if not isinstance(self._scenario_tree_instance,
                                  (_BlockData, Block)):
                    raise TypeError(
                        "'pysp_scenario_tree_model_callback' returned "
                        "an object that is not of the correct type for "
                        "a Pyomo model (e.g, _BockData, Block): %s"
                        % (type(self._scenario_tree_instance)))
            else:
                raise ValueError(
                    "No scenario tree file was given but no function "
                    "named 'pysp_scenario_tree_model_callback' was "
                    "found in the model file.")
        else:
            self._scenario_tree_instance = \
                CreateAbstractScenarioTreeModel().\
                create_instance(filename=self._scenario_tree_filename)

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
                    os.path.join(self._scenario_tree_directory,
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
