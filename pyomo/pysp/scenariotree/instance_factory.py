#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

__all__ = ('ScenarioTreeInstanceFactory',)

import os
import posixpath
import tempfile
import shutil
import copy
import logging

from pyutilib.misc import (ArchiveReaderFactory,
                           ArchiveReader)

from pyomo.dataportal import DataPortal
from pyomo.core import (Block,
                        IPyomoScriptModifyInstance,
                        AbstractModel)
from pyomo.core.base.block import _BlockData
from pyomo.common.dependencies import yaml, yaml_available, yaml_load_args
from pyomo.common.gc_manager import PauseGC
from pyomo.common.plugin import ExtensionPoint
from pyomo.pysp.util.misc import load_external_module
from pyomo.pysp.scenariotree.tree_structure_model import \
    (CreateAbstractScenarioTreeModel,
     ScenarioTreeModelFromNetworkX)
from pyomo.pysp.scenariotree.tree_structure import \
    ScenarioTree

import six

from pyomo.common.dependencies import (
    networkx, networkx_available as has_networkx
)

logger = logging.getLogger('pyomo.pysp')

def _extract_pathspec(
        pathspec,
        default_basename,
        archives=None):
    """Obtain a file location from a pathspec.

    Extracts a file location from the provided input
    path specification by normalizing the name or by
    opening an archive reader.

    Args:
        pathspec (str): The path specification. This can
            be a standard path to a file or represent a
            file contained within an archive. In the
            case of an archived file, the input string
            consist of two parts separated by a comma,
            where the first part represents the path to
            the archive and the second part represents
            the relative path to a file or directory
            within that archive.
        default_basename (str): The default filename to
            search for when the pathspec represents a
            directory (or a directory within an
            archive). This name must have an extension,
            which is used by this function to interpret
            whether the pathspec ends in a filename or a
            directory name. If this argument is None, the
            function will attempt to extract a directory
            name instead of a file.
        archives (list): A list of currently open
            archive readers to check before opening a
            new archive. If a new archive is opened, it will
            be appended to this list.

    Returns:
        A tuple consisting of the normalized absolute
        path to the file followed by the current list of
        open archives that can be passed into this function
        the next time it is called.
    """

    logger.debug("expanding pathspec %s to %s"
                 % (pathspec, os.path.expanduser(pathspec)))
    pathspec = os.path.expanduser(pathspec)

    if archives is None:
        archives = []

    filename = None
    normalized_location = None
    archive = None
    archive_subdir = None
    unarchived_dir = None
    basename = None

    if not os.path.exists(pathspec):
        logger.debug("pathspec does not exist, normalizing name")
        (normalized_location, _, archive_subdir) = \
            ArchiveReader.normalize_name(pathspec).rpartition(',')
        if default_basename is not None:
            extension = os.path.splitext(default_basename)[1].strip()
            assert extension != ''
            if archive_subdir.endswith(extension):
                logger.debug("recognized extension type '%s' appears "
                             "after comma, treating as file" % (extension))
                basename = os.path.basename(archive_subdir)
                archive_subdir = os.path.dirname(archive_subdir).strip()
        if archive_subdir == '':
            archive_subdir = None
    else:
        logger.debug("pathspec exists, normalizing name")
        normalized_location = \
            ArchiveReader.normalize_name(pathspec)

    logger.debug("normalized pathspec: (%s, %s, %s)"
                 % (normalized_location, archive_subdir, basename))
    if ArchiveReader.isArchivedFile(normalized_location):
        logger.debug("pathspec defines a recognized archive type")
        for prev_archive_inputs, prev_archive, prev_unarchived_dir \
              in archives:
            if (normalized_location == \
                prev_archive_inputs[0]) and \
                ((prev_archive_inputs[1] is None) or \
                 ((archive_subdir is not None) and \
                  (archive_subdir.startswith(prev_archive_inputs[1]+'/')))):
                logger.debug("pathspec matches previous archive")
                unarchived_dir = prev_unarchived_dir
                if archive_subdir is not None:
                    if prev_archive_inputs[1] is not None:
                        unarchived_dir = posixpath.join(
                            unarchived_dir,
                            os.path.relpath(archive_subdir,
                                            start=prev_archive_inputs[1]))
                    else:
                        unarchived_dir = posixpath.join(unarchived_dir,
                                                        archive_subdir)
                logger.debug("unarchived directory: %s" % (unarchived_dir))
                break
        else: # if no break occurs in previous for-loop
            archive = ArchiveReaderFactory(
                normalized_location,
                subdir=archive_subdir)
            unarchived_dir = archive.normalize_name(
                tempfile.mkdtemp(prefix='pysp_unarchived'))
            archives.append(((normalized_location, archive_subdir),
                             archive,
                             unarchived_dir))
            logger.debug("New archive opened. Temporary archive "
                         "extraction directory: %s" % (unarchived_dir))
            archive.extractall(path=unarchived_dir)
        if basename is not None:
            filename = posixpath.join(unarchived_dir, basename)
        elif default_basename is not None:
            filename = posixpath.join(unarchived_dir, default_basename)
        else:
            filename = unarchived_dir
        logger.debug("extracted filename: %s" % (filename))
    else:
        logger.debug("pathspec defines a standard path")
        if archive_subdir is not None:
            unarchived_dir = posixpath.join(normalized_location,
                                            archive_subdir)
        else:
            unarchived_dir = normalized_location

        if not os.path.isfile(unarchived_dir):
            if basename is not None:
                filename = posixpath.join(unarchived_dir, basename)
            elif default_basename is not None:
                filename = posixpath.join(unarchived_dir, default_basename)
            else:
                filename = unarchived_dir
        else:
            filename = unarchived_dir

    return filename, archives

def _find_reference_model_or_callback(src):
    """Tries to find a single reference model or callback for
    generating scenario models."""
    module, _ = load_external_module(src, clear_cache=True)
    reference_model = None
    callback = None
    dir_module = dir(module)
    if "pysp_instance_creation_callback" in dir_module:
        callback = getattr(module, "pysp_instance_creation_callback")
        if not hasattr(callback,"__call__"):
            raise TypeError("'pysp_instance_creation_callback' "
                            "object found in source '%s' is not "
                            "callable" % (src))
    else:
        matching_names = []
        for attr_name in dir_module:
            obj = getattr(module, attr_name)
            if isinstance(obj, (_BlockData, Block)):
                reference_model = obj
                matching_names.append(attr_name)
        if len(matching_names) > 1:
            raise ValueError("Multiple objects found in source '%s' "
                             "that could be a reference model. Make "
                             "sure there is only one Pyomo model in "
                             "the source file. Object names: %s"
                             % (str(matching_names)))

    return module, reference_model, callback

def _find_scenariotree(src=None, module=None):
    """Tries to find a single reference model or callback for
    generating scenario models."""
    if module is None:
        assert src is not None
        module, _ = load_external_module(src, clear_cache=True)
    else:
        assert src is None
    scenario_tree_object = None
    scenario_tree_model = None
    callback = None
    dir_module = dir(module)
    if "pysp_scenario_tree_model_callback" in dir_module:
        callback = getattr(module, "pysp_scenario_tree_model_callback")
        if not hasattr(callback,"__call__"):
            raise TypeError("'pysp_scenario_tree_model_callback' "
                            "object found in source '%s' is not "
                            "callable" % (src))
        attrs = [(None, callback())]
    else:
        attrs = [(attr_name,getattr(module, attr_name))
                 for attr_name in dir_module]
    matching_names = []
    for attr_name, obj in attrs:
        if isinstance(obj, ScenarioTree):
            scenario_tree_object = obj
            matching_names.append(attr_name)
        elif isinstance(obj, (_BlockData, Block)):
            scenario_tree_model = obj
            matching_names.append(attr_name)
        elif has_networkx and \
             isinstance(obj, networkx.DiGraph):
            scenario_tree_model = obj
            matching_names.append(attr_name)
    if len(matching_names) > 1:
        raise ValueError("Multiple objects found in source '%s' "
                         "that could act as a scenario tree "
                         "specification. Make sure there is only "
                         "one Pyomo model, ScenarioTree, or "
                         "networkx.DiGraph object in the source "
                         "file. Object names: %s"
                         % (str(matching_names)))

    return module, scenario_tree_object, scenario_tree_model

class ScenarioTreeInstanceFactory(object):

    def __init__(self,
                 model,
                 scenario_tree,
                 data=None):
        """Class to help manage construction of scenario tree models.

        This class is designed to help manage the various input formats
        that that are accepted by PySP and provide a unified interface
        for building scenario trees that are paired with a set of
        concrete Pyomo models.

        Args:
            model: The reference scenario model. Can be set
                to Pyomo model or the name of a file
                containing a Pyomo model. For historical
                reasons, this argument can also be set to a
                directory name where it is assumed a file
                named ReferenceModel.py exists.
            scenario_tree: The scenario tree. Can be set to
                a Pyomo model, a file containing a Pyomo
                model, or a .dat file containing data for an
                abstract scenario tree model representation,
                which defines the structure of the scenario
                tree. It can also be a .py file that
                contains a networkx scenario tree or a
                networkx scenario tree object.  For
                historical reasons, this argument can also
                be set to a directory name where it is
                assumed a file named ScenarioStructure.dat
                exists.
            data: Directory containing .dat files necessary
                for building the scenario instances
                associated with the scenario tree. This
                argument is required if no directory
                information can be extracted from the first
                two arguments and the reference model is an
                abstract Pyomo model. Otherwise, it is not
                required or the location will be inferred
                from the scenario tree location (first) or
                from the reference model location (second),
                where it is assumed the data files reside in
                the same directory.
        """
        self._closed = True

        self._archives = []

        self._model_filename = None
        self._model_module = None
        self._model_object = None
        self._model_callback = None
        self._scenario_tree_filename = None
        self._scenario_tree_module = None
        self._scenario_tree_model = None
        self._scenario_tree = None
        self._data_directory = None
        try:
            self._init(model, scenario_tree, data)
        except:
            self.close()
            raise
        self._closed = False

    def _init(self, model, scenario_tree, data):

        self._model_filename = None
        self._model_module = None
        self._model_object = None
        self._model_callback = None
        if isinstance(model, six.string_types):
            logger.debug("A model filename was provided.")
            self._model_filename, self._archives = \
                _extract_pathspec(model,
                                  "ReferenceModel.py",
                                  archives=self._archives)
            if not os.path.exists(self._model_filename):
                logger.error("Failed to extract reference model python file "
                             "from path specification: %s"
                             % (model))
                raise IOError("path does not exist: %s"
                              % (self._model_filename))
            assert self._model_filename is not None
            assert self._model_filename.endswith(".py")
            (self._model_module,
             self._model_object,
             self._model_callback) = \
                _find_reference_model_or_callback(self._model_filename)
            if (self._model_object is None) and \
               (self._model_callback is None):
                raise AttributeError(
                    "No reference Pyomo model or "
                    "'pysp_instance_creation_callback' "
                    "function object found in src: %s"
                    % (self._model_filename))
        elif hasattr(model, "__call__"):
            logger.debug("A model callback function was provided.")
            self._model_callback = model
        else:
            if not isinstance(model, (_BlockData, Block)):
                raise TypeError(
                    "model argument object has incorrect type: %s. "
                    "Must be a string type, a callback, or a Pyomo model."
                    % (type(model)))
            logger.debug("A model object was provided.")
            self._model_object = model

        self._scenario_tree_filename = None
        self._scenario_tree_model = None
        self._scenario_tree = None
        if isinstance(scenario_tree, ScenarioTree):
            for scenario in scenario_tree.scenarios:
                if scenario.instance is not None:
                    raise ValueError(
                        "The scenario tree can not be linked with instances")
            if hasattr(scenario_tree, "_scenario_instance_factory"):
                del scenario_tree._scenario_instance_factory
            self._scenario_tree = scenario_tree
        elif has_networkx and \
             isinstance(scenario_tree, networkx.DiGraph):
            self._scenario_tree_model = scenario_tree
        elif isinstance(scenario_tree, six.string_types):
            logger.debug("scenario tree input is a string, attempting "
                         "to load file specification: %s"
                         % (scenario_tree))
            self._scenario_tree_filename = None
            if not scenario_tree.endswith(".py"):
                self._scenario_tree_filename, self._archives = \
                    _extract_pathspec(scenario_tree,
                                      "ScenarioStructure.dat",
                                      archives=self._archives)
                if not os.path.exists(self._scenario_tree_filename):
                    logger.debug("Failed to extract scenario tree structure "
                                 ".dat file from path specification: %s"
                                 % (scenario_tree))
                    self._scenario_tree_filename = None
            if self._scenario_tree_filename is None:
                self._scenario_tree_filename, self._archives = \
                    _extract_pathspec(scenario_tree,
                                      "ScenarioStructure.py",
                                      archives=self._archives)
                if not os.path.exists(self._scenario_tree_filename):
                    logger.debug("Failed to locate scenario tree structure "
                                 ".py file with path specification: %s"
                                 % (scenario_tree))
                    self._scenario_tree_filename = None
            if self._scenario_tree_filename is None:
                raise ValueError("Failed to extract scenario tree structure "
                                 "file with .dat or .py extension from path "
                                 "specification: %s" % (scenario_tree))
            elif self._scenario_tree_filename.endswith(".py"):
                if self._scenario_tree_filename == self._model_filename:
                    # try not to clobber the model import
                    (self._scenario_tree_module,
                     self._scenario_tree,
                     self._scenario_tree_model) = \
                        _find_scenariotree(module=self._model_module)
                else:
                    (self._scenario_tree_module,
                     self._scenario_tree,
                     self._scenario_tree_model) = \
                        _find_scenariotree(src=self._scenario_tree_filename)
                if (self._scenario_tree is None) and \
                   (self._scenario_tree_model is None):
                    raise AttributeError(
                        "No scenario tree or "
                        "'pysp_scenario_tree_model_callback' "
                        "function found in src: %s"
                        % (self._scenario_tree_filename))
            elif self._scenario_tree_filename.endswith(".dat"):
                self._scenario_tree_model = \
                    CreateAbstractScenarioTreeModel().\
                    create_instance(filename=self._scenario_tree_filename)
            else:
                assert False
        elif scenario_tree is None:
            if self._model_module is not None:
                self._scenario_tree_filename = self._model_filename
                (self._scenario_tree_module,
                 self._scenario_tree,
                 self._scenario_tree_model) = \
                    _find_scenariotree(module=self._model_module)
                if (self._scenario_tree is None) and \
                   (self._scenario_tree_model is None):
                    raise ValueError(
                        "No input was provided for the scenario tree "
                        "and no callback or scenario tree object was "
                        "found with the model")
            else:
                raise ValueError(
                    "No input was provided for the scenario tree "
                    "but there is no module to search for a "
                    "'pysp_scenario_tree_model_callback' function "
                    "or a ScenarioTree object.")
        else:
            self._scenario_tree_model = scenario_tree

        if self._scenario_tree is None:
            if (not isinstance(self._scenario_tree_model,
                               (_BlockData, Block))) and \
               ((not has_networkx) or \
                (not isinstance(self._scenario_tree_model,
                                networkx.DiGraph))):
                raise TypeError(
                    "scenario tree model object has incorrect type: %s. "
                    "Must be a string type,  Pyomo model, or a "
                    "networkx.DiGraph object." % (type(scenario_tree)))
            if isinstance(self._scenario_tree_model, (_BlockData, Block)):
                if not self._scenario_tree_model.is_constructed():
                    raise ValueError(
                        "scenario tree model is not constructed")

        self._data_directory = None
        if data is None:
            if self.scenario_tree_directory() is not None:
                logger.debug("data directory is set to the scenario tree "
                             "directory: %s"
                             % (self.scenario_tree_directory()))
                self._data_directory = self.scenario_tree_directory()
            elif self.model_directory() is not None:
                logger.debug("data directory is set to the reference model "
                             "directory: %s"
                             % (self.model_directory()))
                self._data_directory = self.model_directory()
            else:
                if (self._model_callback is None) and \
                   isinstance(self._model_object, AbstractModel) and \
                   (not self._model_object.is_constructed()):
                    raise ValueError(
                        "A data location is required since no model "
                        "callback was provided and no other location could "
                        "be inferred.")
                logger.debug("no data directory is required")
        else:
            logger.debug("data location is provided, attempting "
                         "to load specification: %s"
                         % (data))
            self._data_directory, self._archives = \
                _extract_pathspec(data,
                                  None,
                                  archives=self._archives)
            if not os.path.exists(self._data_directory):
                logger.error("Failed to extract data directory "
                             "from path specification: %s"
                             % (data))
                raise IOError("path does not exist: %s"
                              % (self._data_directory))

    def __getstate__(self):
        self.close()
        raise NotImplementedError("Do not deepcopy or serialize this class")

    def __setstate__(self,d):
        self.close()
        raise NotImplementedError("Do not deepcopy or serialize this class")

    def close(self):
        for _,archive,tmpdir in self._archives:
            if os.path.exists(tmpdir):
                shutil.rmtree(tmpdir, True)
            archive.close()
        self._archives = []
        self._closed = True

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
        if self._model_filename is not None:
            return os.path.dirname(self._model_filename)
        else:
            return None

    def scenario_tree_directory(self):
        if self._scenario_tree_filename is not None:
            return os.path.dirname(self._scenario_tree_filename)
        else:
            return None

    def data_directory(self):
        return self._data_directory

    #
    # construct a scenario instance - just like it sounds!
    #
    def construct_scenario_instance(self,
                                    scenario_name,
                                    scenario_tree,
                                    profile_memory=False,
                                    output_instance_construction_time=False,
                                    compile_instance=False,
                                    verbose=False):
        assert not self._closed
        if not scenario_tree.contains_scenario(scenario_name):
            raise ValueError("ScenarioTree does not contain scenario "
                             "with name %s." % (scenario_name))

        scenario = scenario_tree.get_scenario(scenario_name)
        node_name_list = [n._name for n in scenario._node_list]

        if verbose:
            print("Creating instance for scenario=%s" % (scenario_name))

        scenario_instance = None

        try:

            if self._model_callback is not None:

                assert self._model_object is None
                try:
                    _scenario_tree_arg = None
                    # new callback signature
                    if (self._scenario_tree_filename is not None) and \
                       self._scenario_tree_filename.endswith('.dat'):
                        # we started with a .dat file, so
                        # send the PySP scenario tree
                        _scenario_tree_arg = scenario_tree
                    elif self._scenario_tree_model is not None:
                        # We started from a Pyomo
                        # scenario tree model instance, or a
                        # networkx tree.
                        _scenario_tree_arg = self._scenario_tree_model
                    else:
                        # send the PySP scenario tree
                        _scenario_tree_arg = scenario_tree
                    scenario_instance = self._model_callback(_scenario_tree_arg,
                                                             scenario_name,
                                                             node_name_list)
                except TypeError:
                    # old callback signature
                    # TODO:
                    #logger.warning(
                    #    "DEPRECATED: The 'pysp_instance_creation_callback' function "
                    #    "signature has changed. An additional argument should be "
                    #    "added to the beginning of the arguments list that will be "
                    #    "set to the user provided scenario tree object when called "
                    #    "by PySP (e.g., a Pyomo scenario tree model instance, "
                    #    "a networkx tree, or a PySP ScenarioTree object.")
                    scenario_instance = self._model_callback(scenario_name,
                                                             node_name_list)

            elif self._model_object is not None:

                if (not isinstance(self._model_object, AbstractModel)) or \
                   (self._model_object.is_constructed()):
                    scenario_instance = self._model_object.clone()
                elif scenario_tree._scenario_based_data:
                    assert self.data_directory() is not None
                    scenario_data_filename = \
                        os.path.join(self.data_directory(),
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
                        if not yaml_available:
                            raise ValueError(
                                "Found yaml data file for scenario '%s' "
                                "but he PyYAML module is not available"
                                % (scenario_name))
                        scenario_data_filename = \
                            scenario_data_filename+".yaml"
                        with open(scenario_data_filename) as f:
                            data = yaml.load(f, **yaml_load_args)
                    else:
                        raise RuntimeError(
                            "Cannot find a data file for scenario '%s' "
                            "in directory: %s\nRecognized formats: .dat, "
                            ".yaml" % (scenario_name, self.data_directory()))
                    if verbose:
                        print("Data for scenario=%s loads from file=%s"
                              % (scenario_name, scenario_data_filename))
                    if data is None:
                        scenario_instance = \
                            self._model_object.create_instance(
                                filename=scenario_data_filename,
                                profile_memory=profile_memory,
                                report_timing=output_instance_construction_time)
                    else:
                        scenario_instance = \
                            self._model_object.create_instance(
                                data,
                                profile_memory=profile_memory,
                                report_timing=output_instance_construction_time)
                else:
                    assert self.data_directory() is not None
                    data_files = []
                    for node_name in node_name_list:
                        node_data_filename = \
                            os.path.join(self.data_directory(),
                                         str(node_name)+".dat")
                        if not os.path.exists(node_data_filename):
                            raise RuntimeError(
                                "Cannot find a data file for scenario tree "
                                "node '%s' in directory: %s\nRecognized "
                                "formats: .dat" % (node_name,
                                                   self.data_directory()))
                        data_files.append(node_data_filename)

                    scenario_data = DataPortal(model=self._model_object)
                    for data_file in data_files:
                        if verbose:
                            print("Node data for scenario=%s partially "
                                  "loading from file=%s"
                                  % (scenario_name, data_file))
                        scenario_data.load(filename=data_file)

                    scenario_instance = self._model_object.create_instance(
                        scenario_data,
                        profile_memory=profile_memory,
                        report_timing=output_instance_construction_time)
            else:
                raise RuntimeError("Unable to construct scenario instance. "
                                   "Neither a reference model or callback "
                                   "is defined.")

            # name each instance with the scenario name
            scenario_instance._name = scenario_name

            # apply each of the post-instance creation plugins. this
            # really shouldn't be associated (in terms of naming) with the
            # pyomo script - this should be rectified with a workflow
            # re-work. it is unclear how this interacts, or doesn't, with
            # the preprocessors.
            ep = ExtensionPoint(IPyomoScriptModifyInstance)
            for ep in ExtensionPoint(IPyomoScriptModifyInstance):
                logger.warning(
                    "DEPRECATED: IPyomoScriptModifyInstance extension "
                    "point callbacks will be ignored by PySP in the future")
                ep.apply(options=None,
                         model=reference_model,
                         instance=scenario_instance)

            if compile_instance:
                from pyomo.repn.beta.matrix import \
                    compile_block_linear_constraints
                compile_block_linear_constraints(
                    scenario_instance,
                    "_PySP_compiled_linear_constraints",
                    verbose=verbose)

        except:
            logger.error("Failed to create model instance for scenario=%s"
                         % (scenario_name))
            raise

        return scenario_instance

    def construct_instances_for_scenario_tree(
            self,
            scenario_tree,
            profile_memory=False,
            output_instance_construction_time=False,
            compile_scenario_instances=False,
            verbose=False):
        assert not self._closed

        if scenario_tree._scenario_based_data:
            if verbose:
                print("Scenario-based instance initialization enabled")
        else:
            if verbose:
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
                        compile_instance=compile_scenario_instances,
                        verbose=verbose)

            scenario_instances[scenario._name] = scenario_instance
            assert scenario_instance.local_name == scenario.name

        return scenario_instances

    def generate_scenario_tree(self,
                               downsample_fraction=1.0,
                               include_scenarios=None,
                               bundles=None,
                               random_bundles=None,
                               random_seed=None,
                               verbose=True):

        scenario_tree_model = self._scenario_tree_model
        if scenario_tree_model is not None:
            if has_networkx and \
               isinstance(scenario_tree_model, networkx.DiGraph):
                scenario_tree_model = \
                    ScenarioTreeModelFromNetworkX(scenario_tree_model)
            else:
                assert isinstance(scenario_tree_model, (_BlockData, Block)), \
                    str(scenario_tree_model)+" "+str(type(scenario_tree_model))

        if bundles is not None:
            if isinstance(bundles, six.string_types):
                if scenario_tree_model is None:
                    raise ValueError(
                        "A bundles file can not be used when the "
                        "scenario tree input was not a Pyomo "
                        "model or ScenarioStructure.dat file.")
                logger.debug("attempting to locate bundle file for input: %s"
                             % (bundles))
                # we interpret the scenario bundle specification in one of
                # two ways. if the supplied name is a file, it is used
                # directly. otherwise, it is interpreted as the root of a
                # file with a .dat suffix to be found in the instance
                # directory.
                orig_input = bundles
                if not bundles.endswith(".dat"):
                    bundles = bundles+".dat"
                bundles = os.path.expanduser(bundles)
                if not os.path.exists(bundles):
                    if self.data_directory() is None:
                        raise ValueError(
                            "Could not locate bundle .dat file from input "
                            "'%s'. Path does not exist and there is no data "
                            "directory to search in." % (orig_input))
                    bundles = os.path.join(self.data_directory(), bundles)
                if not os.path.exists(bundles):
                    raise ValueError("Could not locate bundle .dat file "
                                     "from input '%s' as absolute path or "
                                     "relative to data directory: %s"
                                     % (orig_input, self.data_directory()))

                if verbose:
                    print("Scenario tree bundle specification filename=%s"
                          % (bundles))

                scenario_tree_model = scenario_tree_model.clone()
                scenario_tree_model.Bundling = True
                scenario_tree_model.Bundling._constructed = False
                scenario_tree_model.Bundling._data.clear()
                scenario_tree_model.Bundles.clear()
                scenario_tree_model.Bundles._constructed = False
                scenario_tree_model.Bundles._data.clear()
                scenario_tree_model.BundleScenarios.clear()
                scenario_tree_model.BundleScenarios._constructed = False
                scenario_tree_model.BundleScenarios._data.clear()
                scenario_tree_model.load(bundles)

        #
        # construct the scenario tree
        #
        if scenario_tree_model is not None:
            scenario_tree = ScenarioTree(scenariotreeinstance=scenario_tree_model,
                                         scenariobundlelist=include_scenarios)
        else:
            assert self._scenario_tree is not None
            if include_scenarios is None:
                scenario_tree = copy.deepcopy(self._scenario_tree)
            else:
                # note: any bundles will be lost
                if self._scenario_tree.contains_bundles():
                    raise ValueError(
                        "Can not compress a scenario tree that "
                        "contains bundles")
                scenario_tree = self._scenario_tree.make_compressed(
                    include_scenarios,
                    normalize=True)

        # compress/down-sample the scenario tree, if requested
        if (downsample_fraction is not None) and \
           (downsample_fraction < 1.0):
            scenario_tree.downsample(downsample_fraction,
                                     random_seed,
                                     verbose)

        #
        # create bundles from a dict, if requested
        #
        if bundles is not None:
            if not isinstance(bundles, six.string_types):
                if verbose:
                    print("Adding bundles to scenario tree from "
                          "user-specified dict")
                if scenario_tree.contains_bundles():
                    if verbose:
                        print("Scenario tree already contains bundles. "
                              "All existing bundles will be removed.")
                    for bundle in list(scenario_tree.bundles):
                        scenario_tree.remove_bundle(bundle.name)
                for bundle_name in bundles:
                    scenario_tree.add_bundle(bundle_name,
                                             bundles[bundle_name])

        #
        # create random bundles, if requested
        #
        if (random_bundles is not None) and \
           (random_bundles > 0):
            if bundles is not None:
                raise ValueError("Cannot specify both random "
                                 "bundles and a bundles specification")

            num_scenarios = len(scenario_tree._scenarios)
            if random_bundles > num_scenarios:
                raise ValueError("Cannot create more random bundles "
                                 "than there are scenarios!")

            if verbose:
                print("Creating "+str(random_bundles)+
                      " random bundles using seed="
                      +str(random_seed))

            scenario_tree.create_random_bundles(random_bundles,
                                                random_seed)

        scenario_tree._scenario_instance_factory = self

        return scenario_tree
