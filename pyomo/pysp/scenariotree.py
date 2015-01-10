#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

import sys
import gc
import random
import weakref
import posixpath

from math import fabs, ceil
import copy

from pyomo.core import *
from pyomo.repn import GeneralCanonicalRepn
from pyomo.pysp.phutils import *
from pyomo.core.base import BasicSymbolMap, CounterLabeler

from six import iterkeys, iteritems, itervalues, advance_iterator, PY3
from six.moves import xrange
using_py3 = PY3

class ScenarioTreeInstanceFactory(object):

    def __init__(self, model, data, verbose=False):

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
                                     preprocess=False,
                                     flatten_expressions=False,
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
                            self._model_object.create(filename=scenario_data_filename,
                                                      preprocess=False,
                                                      report_timing=report_timing)
                    else:
                        scenario_instance = \
                            self._model_object.create(data,
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

                    scenario_instance = self._model_object.create(scenario_data,
                                                                  preprocess=False,
                                                                  report_timing=report_timing)
            else:
                raise RuntimeError("Unable to construct scenario instance. "
                                   "Neither a reference model or callback "
                                   "is defined.")

            if preprocess or flatten_expressions:
                scenario_instance.preprocess()

            if flatten_expressions:
                # IMPT: The model *must* be preprocessed in order for linearization to work. This is because
                #       linearization relies on the canonical expression representation extraction routine,
                #       which in turn relies on variables being identified/categorized (e.g., into "Used").
                scenario_instance.preprocess()
                linearize_model_expressions(scenario_instance)

            # apply each of the post-instance creation plugins. this
            # really shouldn't be associated (in terms of naming) with the
            # pyomo script - this should be rectified with a workflow
            # re-work. it is unclear how this interacts, or doesn't, with
            # the preprocessors.
            ep = ExtensionPoint(IPyomoScriptModifyInstance)
            for ep in ExtensionPoint(IPyomoScriptModifyInstance):
                ep.apply(options=None,
                         model=reference_model,
                         instance=scenario_instance)

        except:
            exception = sys.exc_info()[1]
            raise RuntimeError("Failed to create model instance "
                               "for scenario=%s; Source error=%s"
                               % (scenario_name, exception))

        return scenario_instance

    def construct_instances_for_scenario_tree(self,
                                              scenario_tree,
                                              preprocess=False,
                                              flatten_expressions=False,
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
        if gc.isenabled() is True:
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
                    preprocess=preprocess,
                    flatten_expressions=flatten_expressions)

            scenario_instances[scenario._name] = scenario_instance
            # name each instance with the scenario name
            scenario_instance.name = scenario._name

        if re_enable_gc is True:
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
        from pyomo.pysp.ph import _OLD_OUTPUT
        if not _OLD_OUTPUT:
            module_name, model_import = load_external_module(self._model_filename)
        else:
            model_import = import_file(self._model_filename)

        self._model_object = None
        self._model_callback = None
        if "pysp_instance_creation_callback" in dir(model_import):
            callback = model_import.pysp_instance_creation_callback
            if not hasattr(callback,"__call__"):
                raise TypeError("'pysp_instance_creation_callback' object is not callable "
                                "in model file: "+self._model_filename)
            self._model_callback = callback
        elif "model" in dir(model_import):
            model = model_import.model
            if not isinstance(model,Block):
                raise TypeError("'model' object has incorrect type "
                                "in model file: "+self._model_filename)
            self._model_object = model
        else:
            raise AttributeError("No 'model' or 'pysp_instance_creation_callback' "
                                 "object found in model file: "+self._model_filename)

        self._scenario_tree_instance = \
            scenario_tree_model.create(filename=self._data_filename)

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

class ScenarioTreeNode(object):

    """ Constructor
    """

    VARIABLE_FIXED = 0
    VARIABLE_FREED = 1

    def __init__(self, name, conditional_probability, stage):

        # self-explanatory!
        self._name = name

        # the stage to which this tree node belongs.
        self._stage = stage

        # defines the tree structure
        self._parent = None

        # a collection of ScenarioTreeNodes
        self._children = []

        # conditional on parent
        self._conditional_probability = conditional_probability

        # a collection of all Scenario objects passing through this node in the tree
        self._scenarios = []

        # the cumulative probability of scenarios at this node.
        # cached for efficiency.
        self._probability = 0.0

        # information relating to all variables blended at this node, whether
        # of the standard or derived varieties.
        self._variable_ids = {} # maps id -> (name, index)
        self._name_index_to_id = {} # maps (name,index) -> id
        self._variable_datas = {} # maps id -> list of (vardata,probability) across all scenarios

        # keep track of the variable indices at this node, independent of type.
        # this is useful for iterating. maps variable name to a list of indices.
        self._variable_indices = {}

        # variables are either standard or derived - but not both.
        # partition the ids into two sets, as we deal with these
        # differently in algorithmic and reporting contexts.
        self._standard_variable_ids = set()
        self._derived_variable_ids = set()
        # A temporary solution to help wwphextension and other code
        # for when pyomo instances no longer live on the master node
        # when using PHPyro
        self._discrete = set()

        # a list of _VarData objects, representing the cost variables
        # for each scenario passing through this tree node.
        # NOTE: This list actually contains tuples of
        #       (_VarData, scenario-probability) pairs.
        self._cost_variable_datas = []

        # general use statistics for the variables at each node.
        # each attribute is a map between the variable name and a
        # parameter (over the same index set) encoding the corresponding
        # statistic computed over all scenarios for that node. the
        # parameters are named as the source variable name suffixed
        # by one of: "NODEMIN", "NODEAVG", and "NODEMAX".
        # NOTE: the averages are probability_weighted - the min/max
        #       values are not.
        # NOTE: the parameter names are basically irrelevant, and the
        #       convention is assumed to be enforced by whoever populates
        #       these parameters.
        self._minimums = {}
        self._averages = {}
        self._maximums = {}
        # This gets pushed into PHXBAR on the instances
        self._xbars = {}
        # This gets pushed into PHBLEND on the instances
        self._blend = {}
        self._wbars = {}
        # node variables ids that are fixed (along with the value to fix)
        self._fixed = {}
        # variable ids currently out of sync with instance data
        # variable_id -> VARIABLE_FIXED | VARIABLE_FREED
        self._fix_queue = {}

        # solution (variable) values for this node. assumed to be distinct
        # from self._averages, as the latter are not necessarily feasible.
        # keys are variable ids.
        self._solution = {}

    #
    # given a set of scenario instances, compute the set of indices
    # for non-anticipative variables at this node, as defined by the
    # input match templates.
    #

    def updateVariableIndicesAndValues(self, variable_name, match_templates,
                                       derived=False, id_labeler=None, name_index_to_id_map=None):

        # ensure that the variable exists on each scenario instance,
        # and that there is at least one index match per template.

        # To avoid calling extractVariableIndices more than necessary
        # we take the last scenario in the next loop as our
        # "representative" scenario from which we use the
        # new_match_indices list
        new_match_indices = None
        var_component = {}
        symbolmap = {}
        scenario = None
        isVar = False
        for scenario in self._scenarios:

            scenario_instance = scenario._instance

            if scenario_instance is None:
                continue

            instance_variable = scenario_instance.find_component(variable_name)
            if instance_variable is None:
                raise RuntimeError("The component=%s associated with stage=%s "
                                   "is not present in instance=%s"
                                   % (variable_name,
                                      self._stage._name,
                                      scenario_instance.name))
            isVar = (instance_variable.type() is Var)

            if derived is False:
                if not isVar:
                    raise RuntimeError("The component=%s "
                                       "associated with stage=%s "
                                       "is present in instance=%s "
                                       "but is not a variable - type=%s"
                                       % (variable_name,
                                          self._stage._name,
                                          scenario_instance.name,
                                          type(instance_variable)))
            else:
                if not (isVar or (instance_variable.type() is Expression)):
                    raise RuntimeError("The derived component=%s "
                                       "associated with stage=%s "
                                       "is present in instance=%s "
                                       "but is not a Var or Expression "
                                       "- type=%s"
                                       % (variable_name,
                                          self._stage._name,
                                          scenario_instance.name,
                                          type(instance_variable)))

            new_match_indices = []

            for match_template in match_templates:

                indices = extractVariableIndices(instance_variable, match_template)

                # validate that at least one of the indices in the
                # variable matches to the template - otherwise, the
                # template is bogus.  with one exception: if the
                # variable is empty (the index set is empty), then
                # don't warn - the instance was designed this way.
                if (len(indices) == 0) and (len(instance_variable) > 0):
                    raise ValueError("No indices match template=%s "
                                     "for variable=%s in scenario=%s"
                                     % (match_template,
                                        variable_name,
                                        scenario._name))

                new_match_indices.extend(indices)

            var_component[scenario._name] = scenario_instance.find_component(variable_name)

            if (id_labeler is not None) or (name_index_to_id_map is not None):
                # Tag each instance with a ScenarioTreeSymbolMap. This
                # will allow us to identify common blended variables
                # within a node across scenario instances without
                # having to do an expensive name lookup each time.
                this_symbolmap = getattr(scenario_instance,"_ScenarioTreeSymbolMap", None)
                if this_symbolmap is None:
                    this_symbolmap = scenario_instance._ScenarioTreeSymbolMap = BasicSymbolMap()
                symbolmap[scenario._name] = this_symbolmap

        # find a representative scenario instance belonging to (or
        # passing through) this node in the tree. the first scenario
        # is as good as any.
        # NOTE: At some point we should check that the index sets
        #       across all scenarios at a node actually match for each
        #       variable.

        self._variable_indices.setdefault(variable_name, []).extend(new_match_indices)

        # cache some stuff up-front - we're accessing these attributes a lot in the loops below.
        if derived == False:
            variable_ids_to_update = self._standard_variable_ids
        else:
            variable_ids_to_update = self._derived_variable_ids

        self_variable_ids = self._variable_ids
        self_variable_datas = self._variable_datas

        if (id_labeler is not None) or (name_index_to_id_map is not None):

            for index in sorted(new_match_indices):

                # create the ScenarioTree integer id for this variable
                # across all scenario instances, or look it up if a
                # map has been provided.
                scenario_tree_id = None
                if id_labeler != None:
                    scenario_tree_id = id_labeler()
                elif name_index_to_id_map != None:
                    scenario_tree_id = name_index_to_id_map[variable_name, index]

                variable_ids_to_update.add(scenario_tree_id)

                self_variable_ids[scenario_tree_id] = (variable_name,index)
                self._name_index_to_id[(variable_name,index)] = scenario_tree_id
                self_variable_datas[scenario_tree_id] = []
                for scenario in self._scenarios:
                    vardata = var_component[scenario._name][index]
                    symbolmap[scenario._name].updateSymbol(vardata,scenario_tree_id)
                    self_variable_datas[scenario_tree_id].append((vardata, scenario._probability))
                # We are trusting that each instance variable has the same
                # domain (as we always do)
                if isVar:
                    rep_domain = self_variable_datas[scenario_tree_id][0][0].domain
                    if isinstance(rep_domain, IntegerSet) or \
                        isinstance(rep_domain, BooleanSet):
                        self._discrete.add(scenario_tree_id)

    #
    # same as the above, but specialized to cost variables.
    #

    def updateCostVariableIndexAndValue(self, cost_variable_name, cost_variable_index):

        # ensure that the cost variable exists on each scenario
        # instance, and that the index is valid.  if so, add it to the
        # list of _VarDatas for scenarios at this tree node.
        for scenario in self._scenarios:
            scenario_instance = scenario._instance
            cost_variable = scenario_instance.find_component(cost_variable_name)

            if cost_variable is None:
                raise ValueError("Cost variable=%s associated with "
                                 "stage=%s is not present in model=%s; "
                                 "scenario tree construction failed"
                                 % (cost_variable_name,
                                    self._stage._name,
                                    scenario_instance.name))
            if not cost_variable.type() in [Var,Expression]:
                raise RuntimeError("The component=%s associated with stage=%s "
                                   "is present in model=%s but is not a "
                                   "variable or expression - type=%s"
                                   % (cost_variable_name,
                                      self._stage._name,
                                      scenario_instance.name,
                                      cost_variable.type()))
            if cost_variable_index not in cost_variable:
                raise RuntimeError("The index %s is not defined for cost "
                                   "variable=%s on model=%s"
                                   % (cost_variable_index,
                                      cost_variable_name,
                                      scenario_instance.name))
            self._cost_variable_datas.append((cost_variable[cost_variable_index],scenario._probability))

    #
    # given a set of scenario instances, compute the set of indices being
    # blended for each variable at this node. populates the _variable_indices
    # and _variable_values attributes of a tree node.
    #

    def populateVariableIndicesAndValues(self,
                                         id_labeler=None,
                                         name_index_to_id_map=None,
                                         initialize_solution_data=True):

        self._variable_indices = {}
        self._variable_datas = {}
        self._standard_variable_ids = set()
        self._derived_variable_ids = set()

        stage_variables = self._stage._variables
        for variable_name in sorted(iterkeys(stage_variables)):
            self.updateVariableIndicesAndValues(variable_name,
                                                stage_variables[variable_name],
                                                derived=False,
                                                id_labeler=id_labeler,
                                                name_index_to_id_map=name_index_to_id_map)

        stage_derived_variables = self._stage._derived_variables
        for variable_name in sorted(iterkeys(stage_derived_variables)):
            self.updateVariableIndicesAndValues(variable_name,
                                                stage_derived_variables[variable_name],
                                                derived=True,
                                                id_labeler=id_labeler,
                                                name_index_to_id_map=name_index_to_id_map)

        self.updateCostVariableIndexAndValue(self._stage._cost_variable[0],
                                             self._stage._cost_variable[1])

        if not initialize_solution_data:
            return

        # Create a fully populated scenario tree node.
        if not self.is_leaf_node():
            self._minimums = dict.fromkeys(self._variable_ids,0)
            self._maximums = dict.fromkeys(self._variable_ids,0)
            # this is the true variable average at the node (unmodified)
            self._averages = dict.fromkeys(self._variable_ids,0)
            # this is the xbar used in the PH objective.
            self._xbars = dict.fromkeys(self._standard_variable_ids,None)
            # this is the blend used in the PH objective
            self._blend = dict.fromkeys(self._standard_variable_ids,None)
            # For the dual ph algorithm
            self._wbars = dict.fromkeys(self._standard_variable_ids,None)

            for scenario in self._scenarios:

                scenario._w[self._name] = \
                    dict.fromkeys(self._standard_variable_ids,None)
                scenario._rho[self._name] = \
                    dict.fromkeys(self._standard_variable_ids,None)

        for scenario in self._scenarios:
            scenario._x[self._name] = \
                dict.fromkeys(self._variable_ids,None)

    #
    # copies the parameter values values from the _averages attribute
    # into the _solution attribute - only for active variable values.
    # for leaf nodes, simply copies the values from the _VarValue objects
    # at that node - because there are no statistics maintained.
    #

    def snapshotSolutionFromAverages(self):

        self._solution = {}

        if self.is_leaf_node():

            self._solution.update(self._scenarios[0]._x[self._name])

        else:

            self._solution.update(self._averages)

    #
    # computes the solution values from the composite scenario
    # solutions at this tree node.
    #

    # Note: Trying to work this function out of the code. The only solution
    #       we should get used to working with is that stored on the scenario
    #       objects
    def XsnapshotSolutionFromInstances(self):

        self._solution = {}

        for variable_id in self._standard_variable_ids:

            var_datas = self._variable_datas[variable_id]
            # the following loop is just a sanity check.
            for var_data, scenario_probability in var_datas:
                # a variable that is fixed will be flagged as unused.
                if (var_data.stale) and (not var_data.fixed):
                    # Note: At this point the only way to get the name of the scenario
                    #       for this specific vardata in general is to print its full cname
                    # This will either be "MASTER", the bundle name, or the scenario name
                    # The important thing is that we always have the scenario name somewhere
                    # in the variable name we print
                    model_name = var_data.model().cname(True)
                    full_name = model_name+"."+var_data.cname(True)
                    if not self.is_leaf_node():
                        print("CAUTION: Encountered variable=%s "
                              "on node %s that is not in use within its "
                              "respective scenario instance but the scenario tree "
                              "specification indicates that non-anticipativity is to "
                              "be enforced; the variable should either be eliminated "
                              "from the model or from the scenario tree specification."
                              % (full_name, self._name))
                    else:
                        print("CAUTION: Encountered variable=%s "
                              "on leaf node %s that is not in use within "
                              "its respective scenario instance. This can be indicative "
                              "of a modeling error; the variable should either be "
                              "eliminated from the model or from the scenario tree "
                              "specification." % (full_name, self._name))

            # if a variable is stale, it could be because it is fixed - in which case, we want to snapshot the average value

            avg = sum(scenario_probability * value(var_data) for var_data, scenario_probability in var_datas if (var_data.stale is False) or (var_data.fixed is True))

            # the node probability is allowed to be zero in the scenario tree specification.
            # this is useful in cases where one wants to temporarily ignore certain scenarios.
            # in this case, just skip reporting of variables for that node.
            if self._probability > 0.0:
                avg /= self._probability

            self._solution[variable_id] = avg

        for variable_id in self._derived_variable_ids:

            var_datas = self._variable_datas[variable_id]

            avg = sum(scenario_probability * value(var_data) for var_data, scenario_probability in var_datas)

            # the node probability is allowed to be zero in the scenario tree specification.
            # this is useful in cases where one wants to temporarily ignore certain scenarios.
            # in this case, just skip reporting of variables for that node.
            if self._probability > 0.0:
                avg /= self._probability

            self._solution[variable_id] = avg

    def snapshotSolutionFromScenarios(self):

        self._solution = {}

        for variable_id in self._standard_variable_ids:

            var_values = [(scenario._x[self._name][variable_id],scenario._probability) \
                          for scenario in self._scenarios]

            avg = 0.0
            # the following loop is just a sanity check.
            for scenario in self._scenarios:
                scenario_probability = scenario._probability
                var_value = scenario._x[self._name][variable_id]
                is_fixed = scenario.is_variable_fixed(self, variable_id)
                is_stale = scenario.is_variable_stale(self, variable_id)
                # a variable that is fixed will be flagged as unused.
                if is_stale and (not is_fixed):
                    variable_name, index = self._variable_ids[variable_id]
                    full_name = variable_name+indexToString(index)
                    if not self.is_leaf_node():
                        print("CAUTION: Encountered variable=%s "
                              "on node %s that is not in use within its "
                              "respective scenario %s but the scenario tree "
                              "specification indicates that non-anticipativity is to "
                              "be enforced; the variable should either be eliminated "
                              "from the model or from the scenario tree specification."
                              % (full_name, self._name, scenario._name))
                    else:
                        print("CAUTION: Encountered variable=%s "
                              "on leaf node %s that is not in use within "
                              "its respective scenario %s. This can be indicative "
                              "of a modeling error; the variable should either be "
                              "eliminated from the model or from the scenario tree "
                              "specification." % (full_name, self._name, scenario._name))
                else:
                    avg += scenario_probability*var_value

            # the node probability is allowed to be zero in the scenario tree specification.
            # this is useful in cases where one wants to temporarily ignore certain scenarios.
            # in this case, just skip reporting of variables for that node.
            if self._probability > 0.0:
                avg /= self._probability

            self._solution[variable_id] = avg

        for variable_id in self._derived_variable_ids:

            # if any of the variable values are None (not reported), it will
            # trigger an exception. if this happens, trap it and simply remove
            # the solution from the tree node for this specific variable.
            # NOTE: This handling is a bit inconsistent relative to the above
            #       logic for handling non-derived variables, in terms of
            #       monitoring stale/fixed flags - for no good reason.
            try:
                avg = sum(scenario._probability * scenario._x[self._name][variable_id] \
                              for scenario in self._scenarios)

                # the node probability is allowed to be zero in the scenario tree specification.
                # this is useful in cases where one wants to temporarily ignore certain scenarios.
                # in this case, just skip reporting of variables for that node.
                if self._probability > 0.0:
                    avg /= self._probability

                self._solution[variable_id] = avg
            except:
                if variable_id in self._solution:
                    del self._solution[variable_id]

    #
    # a utility to compute the cost of the current node plus the expected costs of child nodes.
    #

    def computeExpectedNodeCost(self):

        stage_name = self._stage._name
        if any(scenario._stage_costs[stage_name] is None \
               for scenario in self._scenarios):
            return None

        my_cost = self._scenarios[0]._stage_costs[stage_name]
        # Don't assume the node has converged, this can
        # result in misleading output
        # UPDATE: It turns out this entire function is misleading
        #         it will be removed
        """
        my_cost = sum(scenario._stage_costs[stage_name] * scenario._probability \
                      for scenario in self._scenarios)
        my_cost /= sum(scenario._probability for scenario in self._scenarios)
        """
        # This version implicitely assumes convergence (which can be garbage for ph)

        children_cost = 0.0
        for child in self._children:
            child_cost = child.computeExpectedNodeCost()
            if child_cost is None:
                return None
            else:
                children_cost += (child._conditional_probability * child_cost)
        return my_cost + children_cost

    #
    # a simple predicate to check if this tree node belongs to the
    # last stage in the scenario tree.
    #
    def is_leaf_node(self):

        return self._stage.is_last_stage()

    #
    # a utility to determine if the input variable name/index pair is
    # a derived variable.
    #
    def is_derived_variable(self, variable_name, variable_index):
        return (variable_name, variable_index) in self._name_index_to_id

    #
    # a utility to extract the value for the input name/index pair.
    #
    def get_variable_value(self, name, index):

        try:
            variable_id = self._name_index_to_id[(name,index)]
        except KeyError:
            raise ValueError("No ID for variable=%s, index=%s "
                             "is defined for scenario tree "
                             "node=%s" % (name, index, self._name))

        try:
            return self._solution[variable_id]
        except KeyError:
            raise ValueError("No value for variable=%s, index=%s "
                             "is defined for scenario tree "
                             "node=%s" % (name, index, self._name))

    #
    # fix the indicated input variable / index pair to the input value.
    #
    def fix_variable(self, variable_id, fix_value):

        self._fix_queue[variable_id] = (self.VARIABLE_FIXED, fix_value)

    #
    # free the indicated input variable / index pair to the input value.
    #
    def free_variable(self, variable_id):

        self._fix_queue[variable_id] = (self.VARIABLE_FREED, None)

    def is_variable_discrete(self, variable_id):

        return variable_id in self._discrete

    def is_variable_fixed(self, variable_id):

        return variable_id in self._fixed

    def push_xbar_to_instances(self):
        arbitrary_instance = self._scenarios[0]._instance
        assert arbitrary_instance != None

        # Note: the PHXBAR Param is shared amongst the
        # scenarios in a tree node, so it's only
        # necessary to grab the Param from an arbitrary
        # scenario for each node and update once
        xbar_parameter_name = "PHXBAR_"+str(self._name)
        xbar_parameter = arbitrary_instance.find_component(xbar_parameter_name)
        xbar_parameter.store_values(self._xbars)

    def push_fix_queue_to_instances(self):
        have_instances = (self._scenarios[0]._instance != None)

        for variable_id, (fixed_status, new_value) in iteritems(self._fix_queue):
            if fixed_status == self.VARIABLE_FREED:
                assert new_value is None
                if have_instances:
                    for var_data, scenario_probability in \
                        self._variable_datas[variable_id]:
                        var_data.free()
                del self._fixed[variable_id]
            elif fixed_status == self.VARIABLE_FIXED:
                if have_instances:
                    for var_data, scenario_probability in \
                        self._variable_datas[variable_id]:
                        var_data.fix(new_value)
                self._fixed[variable_id] = new_value
            else:
                raise ValueError("Unexpected fixed status %s for variable with "
                                 "scenario tree id %s" % (fixed_status,
                                                          variable_id))

        self.clear_fix_queue()

    def push_all_fixed_to_instances(self):
        have_instances = (self._scenarios[0]._instance != None)

        for variable_id, fix_value in iteritems(self._fixed):
            if have_instances:
                for var_data, scenario_probability in \
                    self._variable_datas[variable_id]:
                    var_data.fix(fix_value)
            self._fixed[variable_id] = fix_value

        self.push_fix_queue_to_instances()

    def has_fixed_in_queue(self):
        return any((v[0] == self.VARIABLE_FIXED) \
                   for v in itervalues(self._fix_queue))

    def has_freed_in_queue(self):
        return any((v[0] == self.VARIABLE_FREED) \
                   for v in itervalues(self._fix_queue))

    def clear_fix_queue(self):

        self._fix_queue.clear()

class ScenarioTreeStage(object):

    """ Constructor
    """
    def __init__(self, *args, **kwds):

        self._name = ""

        # a collection of ScenarioTreeNode objects associated with this stage.
        self._tree_nodes = []

        # the parent scenario tree for this stage.
        self._scenario_tree = None

        # a map between a variable name and a list of original index
        # match templates, specified as strings.  we want to maintain
        # these for a variety of reasons, perhaps the most important
        # being that for output purposes. specific indices that match
        # belong to the tree node, as that may be specific to a tree
        # node.
        self._variables = {}

        # same as above, but for derived stage variables.
        self._derived_variables = {}

        # a tuple consisting of (1) the name of the variable that
        # stores the stage-specific cost in all scenarios and (2) the
        # corresponding index *string* - this is converted in the tree
        # node to a real index.
        self._cost_variable = (None, None)

    #
    # add a new variable to the stage, which will include updating the
    # solution maps for each associated ScenarioTreeNode.
    #
    def add_variable(self, variable_name, new_match_template, create_variable_ids=True):

        labeler = None
        if create_variable_ids is True:
            labeler = self._scenario_tree._id_labeler

        existing_match_templates = self._variables.setdefault(variable_name, [])
        existing_match_templates.append(new_match_template)

        for tree_node in self._tree_nodes:
            tree_node.updateVariableIndicesAndValues(variable_name, new_match_template,
                                                     derived=False,
                                                     id_labeler=labeler)

    #
    # a simple predicate to check if this stage is the last stage in
    # the scenario tree.
    #
    def is_last_stage(self):

        return self == self._scenario_tree._stages[-1]

class Scenario(object):

    """ Constructor
    """
    def __init__(self, *args, **kwds):

        self._name = None
        # allows for construction of node list
        self._leaf_node = None
        # sequence from parent to leaf of ScenarioTreeNodes
        self._node_list = []
        # the unconditional probability for this scenario, computed from the node list
        self._probability = 0.0
        # the Pyomo instance corresponding to this scenario.
        self._instance = None
        self._instance_cost_expression = None
        self._instance_objective = None
        self._objective_sense = None
        self._objective_name = None

        # The value of the (possibly augmented) objective function
        self._objective = None
        # The value of the original objective expression
        # (which should be the sum of the stage costs)
        self._cost = None
        # The individual stage cost values
        self._stage_costs = {}
        # The value of the ph weight term piece of the objective (if it exists)
        self._weight_term_cost = None
        # The value of the ph proximal term piece of the objective (if it exists)
        self._proximal_term_cost = None
        # The value of the scenariotree variables belonging to this scenario
        # (dictionary nested by node name)
        self._x = {}
        # The value of the weight terms belonging to this scenario
        # (dictionary nested by node name)
        self._w = {}
        # The value of the rho terms belonging to this scenario
        # (dictionary nested by node name)
        self._rho = {}

        # This set of fixed or reported stale variables
        # in each tree node
        self._fixed = {}
        self._stale = {}

    #
    # a utility to compute the stage index for the input tree node.
    # the returned index is 0-based.
    #

    def node_stage_index(self, tree_node):
        return self._node_list.index(tree_node)

    def is_variable_fixed(self, tree_node, variable_id):

        return variable_id in self._fixed[tree_node._name]

    def is_variable_stale(self, tree_node, variable_id):

        return variable_id in self._stale[tree_node._name]

    def update_solution_from_instance(self):

        results = {}
        scenario_instance = self._instance
        scenariotree_sm_bySymbol = \
            scenario_instance._ScenarioTreeSymbolMap.bySymbol
        self._objective = self._instance_objective(exception=False)
        self._cost = self._instance_cost_expression(exception=False)
        for tree_node in self._node_list:
            stage_name = tree_node._stage._name
            cost_variable_name, cost_variable_index = \
                tree_node._stage._cost_variable
            stage_cost_component = self._instance.find_component(cost_variable_name)
            self._stage_costs[stage_name] = \
                stage_cost_component[cost_variable_index](exception=False)

        self._weight_term_cost = \
            scenario_instance.PHWEIGHT_EXPRESSION(exception=False) \
            if (hasattr(scenario_instance,"PHWEIGHT_EXPRESSION") and \
                (scenario_instance.PHWEIGHT_EXPRESSION is not None)) \
            else None
        self._proximal_term_cost = \
            scenario_instance.PHPROXIMAL_EXPRESSION(exception=False) \
            if (hasattr(scenario_instance,"PHPROXIMAL_EXPRESSION") and \
                (scenario_instance.PHPROXIMAL_EXPRESSION is not None)) \
            else None

        for tree_node in self._node_list:
            # Some of these might be Expression objects so we use the
            # __call__ method rather than directly accessing .value
            # (since we want a number)
            self._x[tree_node._name].update(
                (variable_id,
                 scenariotree_sm_bySymbol[variable_id](exception=False)) \
                for variable_id in tree_node._variable_ids)
            scenario_fixed = self._fixed[tree_node._name] = set()
            scenario_stale = self._stale[tree_node._name] = set()
            for variable_id in tree_node._variable_ids:
                vardata = scenariotree_sm_bySymbol[variable_id]
                if vardata.fixed:
                    scenario_fixed.add(variable_id)
                if vardata.stale:
                    scenario_stale.add(variable_id)

    def push_solution_to_instance(self):

        scenario_instance = self._instance
        scenariotree_sm_bySymbol = \
            scenario_instance._ScenarioTreeSymbolMap.bySymbol
        for tree_node in self._node_list:
            stage_name = tree_node._stage._name
            cost_variable_name, cost_variable_index = \
                tree_node._stage._cost_variable
            stage_cost_component = \
                self._instance.find_component(cost_variable_name)[cost_variable_index]
            # Some of these might be Expression objects so we check
            # for is_expression before changing.value
            if not stage_cost_component.is_expression():
                stage_cost_component.value = self._stage_costs[stage_name]

        for tree_node in self._node_list:
            # Some of these might be Expression objects so we check
            # for is_expression before changing.value
            for variable_id, var_value in iteritems(self._x[tree_node._name]):
                compdata = scenariotree_sm_bySymbol[variable_id]
                if not compdata.is_expression():
                    compdata.value = var_value

            for variable_id in self._fixed[tree_node._name]:
                vardata = scenariotree_sm_bySymbol[variable_id]
                vardata.fix()

            for variable_id in self._stale[tree_node._name]:
                vardata = scenariotree_sm_bySymbol[variable_id]
                vardata.stale = True

    def package_current_solution(self, translate_ids=None, node_names=None):

        if node_names is None:
            node_names = [n._name for n in self._node_list]

        results = {}
        results['objective'] = self._objective
        results['cost'] = self._cost
        results['stage costs'] = copy.deepcopy(self._stage_costs)
        results['weight term cost'] = self._weight_term_cost
        results['proximal term cost'] = self._proximal_term_cost
        if translate_ids is None:
            results['x'] = copy.deepcopy(self._x)
            results['fixed'] = copy.deepcopy(self._fixed)
            results['stale'] = copy.deepcopy(self._stale)
        else:
            resx = results['x'] = {}
            for tree_node_name, tree_node_x in iteritems(self._x):
                if tree_node_name not in node_names:
                    continue
                tree_node_translate_ids = translate_ids[tree_node_name]
                resx[tree_node_name] = \
                    dict((tree_node_translate_ids[scenario_tree_id],val) \
                         for scenario_tree_id, val in \
                         iteritems(tree_node_x))
            resfixed = results['fixed'] = {}
            for tree_node_name, tree_node_fixed in iteritems(self._fixed):
                if tree_node_name not in node_names:
                    continue
                tree_node_translate_ids = translate_ids[tree_node_name]
                resfixed[tree_node_name] = \
                    set(tree_node_translate_ids[scenario_tree_id] \
                        for scenario_tree_id in tree_node_fixed)
            resstale = results['stale'] = {}
            for tree_node_name, tree_node_stale in iteritems(self._stale):
                if tree_node_name not in node_names:
                    continue
                tree_node_translate_ids = translate_ids[tree_node_name]
                resstale[tree_node_name] = \
                    set(tree_node_translate_ids[scenario_tree_id] \
                        for scenario_tree_id in tree_node_stale)
        return results

    def update_current_solution(self, results):

        self._objective = results['objective']
        self._cost = results['cost']
        assert len(results['stage costs']) == len(self._stage_costs)
        self._stage_costs.update(results['stage costs'])
        self._weight_term_cost = results['weight term cost']
        self._proximal_term_cost = results['proximal term cost']
        for node in self._node_list:
            if node._name in results['x']:
                node_x = results['x'][node._name]
                self._x[node._name].update(node_x)
            else:
                self._x[node._name].update((i,None) for i in self._x[node._name])

            self._fixed[node._name].clear()
            if node._name in results['fixed']:
                self._fixed[node._name].update(results['fixed'][node._name])

            self._stale[node._name].clear()
            if node._name in results['stale']:
                self._stale[node._name].update(results['stale'][node._name])

    def push_w_to_instance(self):
        assert self._instance != None
        for tree_node in self._node_list[:-1]:
            weight_parameter_name = "PHWEIGHT_"+str(tree_node._name)
            weight_parameter = self._instance.find_component(weight_parameter_name)
            weight_parameter.store_values(self._w[tree_node._name])

    def push_rho_to_instance(self):
        assert self._instance != None

        for tree_node in self._node_list[:-1]:
            rho_parameter_name = "PHRHO_"+str(tree_node._name)
            rho_parameter = self._instance.find_component(rho_parameter_name)
            rho_parameter.store_values(self._rho[tree_node._name])

    #
    # a utility to determine the stage to which the input variable belongs.
    #

    def variableNode(self, variable, index):

        tuple_to_check = (variable.cname(),index)

        for this_node in self._node_list:

            if tuple_to_check in this_node._name_index_to_id:
                return this_node

        raise RuntimeError("The variable="+str(variable.cname())+", index="+indexToString(index)+" does not belong to any stage in the scenario tree")

    #
    # a utility to determine the stage to which the input constraint "belongs".
    # a constraint belongs to the latest stage in which referenced variables
    # in the constraint appears in that stage.
    # input is a constraint is of type "Constraint", and an index of that
    # constraint - which might be None in the case of non-indexed constraints.
    # currently doesn't deal with SOS constraints, for no real good reason.
    # returns an instance of a ScenarioTreeStage object.
    # IMPT: this method works on the canonical representation ("repn" attribute)
    #       of a constraint. this implies that pre-processing of the instance
    #       has been performed.
    # NOTE: there is still the issue of whether the contained variables really
    #       belong to the same model, but that is a different issue we won't
    #       address right now (e.g., what does it mean for a constraint in an
    #       extensive form binding instance to belong to a stage?).
    #

    def constraintNode(self, constraint, index, repn=None):

        deepest_node_index = -1
        deepest_node = None

        vardata_list = None
        if isinstance(constraint, SOSConstraint):
            vardata_list = constraint[index].get_members()

        else:
            if repn is None:
                parent_instance = constraint.parent()
                repn = getattr(parent_instance,"canonical_repn",None)
                if (repn is None):
                    raise ValueError("Unable to find canonical_repn ComponentMap "
                                     "on constraint parent block %s for constraint %s"
                                     % (parent_instance.cname(True), constraint.cname(True)))

            canonical_repn = repn.get(constraint[index])
            if canonical_repn is None:
                raise RuntimeError("Method constraintStage in class "
                                   "ScenarioTree encountered a constraint "
                                   "with no canonical representation "
                                   "- was preprocessing performed?")

            if isinstance(canonical_repn, GeneralCanonicalRepn):
                raise RuntimeError("Method constraintStage in class "
                                   "ScenarioTree encountered a constraint "
                                   "with a general canonical encoding - "
                                   "only linear canonical encodings are expected!")

            vardata_list = canonical_repn.variables

        for var_data in vardata_list:

            var_node = self.variableNode(var_data.parent_component(), var_data.index())
            var_node_index = self._node_list.index(var_node)

            if var_node_index > deepest_node_index:
                deepest_node_index = var_node_index
                deepest_node = var_node

        return deepest_node

class ScenarioTreeBundle(object):

     def __init__(self, *args, **kwds):

         self._name = None
         self._scenario_names = []
         self._scenario_tree = None # This is a compressed scenario tree, just for the bundle.
         self._probability = 0.0 # the absolute probability of scenarios associated with this node in the scenario tree.

class ScenarioTree(object):

    # a utility to construct scenario bundles.
    def _construct_scenario_bundles(self, scenario_tree_instance):

        for bundle_name in scenario_tree_instance.Bundles:
           scenario_list = []
           bundle_probability = 0.0
           for scenario_name in scenario_tree_instance.BundleScenarios[bundle_name]:
              scenario_list.append(scenario_name)
              bundle_probability += self._scenario_map[scenario_name]._probability

           scenario_tree_instance.Bundling[None] = False # to stop recursion!

           scenario_tree_for_bundle = ScenarioTree(scenariotreeinstance=scenario_tree_instance,
                                                   scenariobundlelist=scenario_list)

           scenario_tree_instance.Bundling[None] = True

           if scenario_tree_for_bundle.validate() is False:
               raise RuntimeError("***ERROR: Bundled scenario tree is invalid!!!")

           new_bundle = ScenarioTreeBundle()
           new_bundle._name = bundle_name
           new_bundle._scenario_names = scenario_list
           new_bundle._scenario_tree = scenario_tree_for_bundle
           new_bundle._probability = bundle_probability

           self._scenario_bundles.append(new_bundle)
           self._scenario_bundle_map[new_bundle._name] = new_bundle

    #
    # a utility to construct the stage objects for this scenario tree.
    # operates strictly by side effects, initializing the self
    # _stages and _stage_map attributes.
    #

    def _construct_stages(self, stage_names, stage_variable_names, stage_cost_variable_names, stage_derived_variable_names):

        # construct the stage objects, which will leave them
        # largely uninitialized - no variable information, in particular.
        for stage_name in stage_names:

            new_stage = ScenarioTreeStage()
            new_stage._name = stage_name
            new_stage._scenario_tree = self

            for variable_string in stage_variable_names[stage_name]:
                if isVariableNameIndexed(variable_string) is True:
                    variable_name, match_template = extractVariableNameAndIndex(variable_string)
                else:
                    variable_name = variable_string
                    match_template = ""
                if variable_name not in new_stage._variables:
                    new_stage._variables[variable_name] = []
                new_stage._variables[variable_name].append(match_template)

            if stage_name in stage_derived_variable_names: # not all stages have derived variables defined
                for variable_string in stage_derived_variable_names[stage_name]:
                    if isVariableNameIndexed(variable_string) is True:
                        variable_name, match_template = extractVariableNameAndIndex(variable_string)
                    else:
                        variable_name = variable_string
                        match_template = ""
                    if variable_name not in new_stage._derived_variables:
                        new_stage._derived_variables[variable_name] = []
                    new_stage._derived_variables[variable_name].append(match_template)

            # de-reference is required to access the parameter value
            cost_variable_string = value(stage_cost_variable_names[stage_name])
            if isVariableNameIndexed(cost_variable_string) is True:
                cost_variable_name, cost_variable_index = extractVariableNameAndIndex(cost_variable_string)
            else:
                cost_variable_name = cost_variable_string
                cost_variable_index = None
            new_stage._cost_variable = (cost_variable_name, cost_variable_index)

            self._stages.append(new_stage)
            self._stage_map[stage_name] = new_stage


    """ Constructor
        Arguments:
            scenarioinstance     - the reference (deterministic) scenario instance.
            scenariotreeinstance - the pyomo model specifying all scenario tree (text) data.
            scenariobundlelist   - a list of scenario names to retain, i.e., cull the rest to create a reduced tree!
    """
    def __init__(self, *args, **kwds):

        self._name = None # some arbitrary identifier

        # should be called once for each variable blended across a node
        self._id_labeler = CounterLabeler()

        # the core objects defining the scenario tree.
        self._tree_nodes = [] # collection of ScenarioTreeNodes
        self._stages = [] # collection of ScenarioTreeStages - assumed to be in time-order. the set (provided by the user) itself *must* be ordered.
        self._scenarios = [] # collection of Scenarios
        self._scenario_bundles = [] # collection of ScenarioTreeBundles

        # dictionaries for the above.
        self._tree_node_map = {}
        self._stage_map = {}
        self._scenario_map = {}
        self._scenario_bundle_map = {}

        # a boolean indicating how data for scenario instances is specified.
        # possibly belongs elsewhere, e.g., in the PH algorithm.
        self._scenario_based_data = None

        scenario_tree_instance = kwds.pop( 'scenariotreeinstance', None )
        scenario_bundle_list = kwds.pop( 'scenariobundlelist', None )

        # process the keyword options
        for key in kwds:
            sys.stderr.write("Unknown option '%s' specified in call to ScenarioTree constructor\n" % key)

        if scenario_tree_instance is None:
            raise ValueError("A scenario tree instance must be supplied in the ScenarioTree constructor")

        node_ids = scenario_tree_instance.Nodes
        node_child_ids = scenario_tree_instance.Children
        node_stage_ids = scenario_tree_instance.NodeStage
        node_probability_map = scenario_tree_instance.ConditionalProbability
        stage_ids = scenario_tree_instance.Stages
        stage_variable_ids = scenario_tree_instance.StageVariables
        stage_cost_variable_ids = scenario_tree_instance.StageCostVariable
        stage_derived_variable_ids = scenario_tree_instance.StageDerivedVariables
        scenario_ids = scenario_tree_instance.Scenarios
        scenario_leaf_ids = scenario_tree_instance.ScenarioLeafNode
        scenario_based_data = scenario_tree_instance.ScenarioBasedData

        # save the method for instance data storage.
        self._scenario_based_data = scenario_based_data()

        # the input stages must be ordered, for both output purposes and knowledge of the final stage.
        if stage_ids.ordered is False:
            raise ValueError("An ordered set of stage IDs must be supplied in the ScenarioTree constructor")

        empty_nonleaf_stages = [stage for stage in stage_ids \
                                    if len(stage_variable_ids[stage])==0 \
                                    and stage != stage_ids.last()]
        if len(empty_nonleaf_stages) > 0:
            raise ValueError("A ScenarioTree has been declared with one"
                             " or more empty (non-leaf) stages. This must"
                             " be corrected by defining non-empty sets "
                             "for the following entries in "
                             "ScenarioStructure.dat: \n- %s" % \
                              ('\n- '.join('StageVariables[%s]'%(stage) \
                              for stage in empty_nonleaf_stages)))

        #
        # construct the actual tree objects
        #

        # construct the stage objects w/o any linkages first; link them up
        # with tree nodes after these have been fully constructed.
        self._construct_stages(stage_ids, stage_variable_ids, stage_cost_variable_ids, stage_derived_variable_ids)

        # construct the tree node objects themselves in a first pass,
        # and then link them up in a second pass to form the tree.
        # can't do a single pass because the objects may not exist.
        for tree_node_name in node_ids:

            if tree_node_name not in node_stage_ids:
                raise ValueError("No stage is assigned to tree node=%s" % (tree_node._name))

            stage_name = value(node_stage_ids[tree_node_name])
            if stage_name not in self._stage_map:
                raise ValueError("Unknown stage=%s assigned to tree node=%s"
                                 % (stage_name, tree_node._name))

            new_tree_node = ScenarioTreeNode(tree_node_name,
                                             value(node_probability_map[tree_node_name]),
                                             self._stage_map[stage_name])

            self._tree_nodes.append(new_tree_node)
            self._tree_node_map[tree_node_name] = new_tree_node
            self._stage_map[stage_name]._tree_nodes.append(new_tree_node)

        # link up the tree nodes objects based on the child id sets.
        for this_node in self._tree_nodes:
            this_node._children = []
            # otherwise, you're at a leaf and all is well.
            if this_node._name in node_child_ids:
                child_ids = node_child_ids[this_node._name]
                for child_id in child_ids:
                    if child_id in self._tree_node_map:
                        child_node = self._tree_node_map[child_id]
                        this_node._children.append(child_node)
                        if child_node._parent is None:
                            child_node._parent = this_node
                        else:
                            raise ValueError("Multiple parents specified for tree node=%s; "
                                             "existing parent node=%s; conflicting parent "
                                             "node=%s"
                                             % (child_id,
                                                child_node._parent._name,
                                                this_node._name))
                    else:
                        raise ValueError("Unknown child tree node=%s specified "
                                         "for tree node=%s"
                                         % (child_id, this_node._name))

        # at this point, the scenario tree nodes and the stages are set - no
        # two-pass logic necessary when constructing scenarios.
        for scenario_name in scenario_ids:

            new_scenario = Scenario()
            new_scenario._name=scenario_name

            if scenario_name not in scenario_leaf_ids:
                raise ValueError("No leaf tree node specified for scenario=%s"
                                 % (scenario_name))
            else:
                scenario_leaf_node_name = value(scenario_leaf_ids[scenario_name])
                if scenario_leaf_node_name not in self._tree_node_map:
                    raise ValueError("Uknown tree node=%s specified as leaf "
                                     "of scenario=%s"
                                     (scenario_leaf_node_name, scenario_name))
                else:
                    new_scenario._leaf_node = self._tree_node_map[scenario_leaf_node_name]

            current_node = new_scenario._leaf_node
            while current_node is not None:
                new_scenario._node_list.append(current_node)
                current_node._scenarios.append(new_scenario) # links the scenarios to the nodes to enforce necessary non-anticipativity
                current_node = current_node._parent
            new_scenario._node_list.reverse()
            # This now loops root -> leaf
            probability = 1.0
            for current_node in new_scenario._node_list:
                probability *= current_node._conditional_probability
                # NOTE: The line placement below is a little weird, in that
                #       it is embedded in a scenario loop - so the probabilities
                #       for some nodes will be redundantly computed. But this works.
                current_node._probability = probability

                new_scenario._stage_costs[current_node._stage._name] = None
                new_scenario._x[current_node._name] = {}
                new_scenario._w[current_node._name] = {}
                new_scenario._rho[current_node._name] = {}
                new_scenario._fixed[current_node._name] = set()
                new_scenario._stale[current_node._name] = set()

            new_scenario._probability = probability

            self._scenarios.append(new_scenario)
            self._scenario_map[scenario_name] = new_scenario

        # for output purposes, it is useful to known the maximal
        # length of identifiers in the scenario tree for any
        # particular category. I'm building these up incrementally, as
        # they are needed. 0 indicates unassigned.
        self._max_scenario_id_length = 0

        # does the actual traversal to populate the members.
        self.computeIdentifierMaxLengths()

        # if a sub-bundle of scenarios has been specified, mark the
        # active scenario tree components and compress the tree.
        if scenario_bundle_list is not None:
            self.compress(scenario_bundle_list)

        # NEW SCENARIO BUNDLING STARTS HERE
        if value(scenario_tree_instance.Bundling[None]) is True:
           self._construct_scenario_bundles(scenario_tree_instance)

    #
    # populate those portions of the scenario tree and associated
    # stages and tree nodes that reference the scenario instances
    # associated with the tree.
    #

    def linkInInstances(self,
                        scenario_instance_map,
                        objective_sense=None,
                        create_variable_ids=True,
                        master_scenario_tree=None,
                        initialize_solution_data=True):

        if objective_sense not in (minimize, maximize, None):
            raise ValueError("Invalid value (%r) for objective sense given to the linkInInstances method. "
                               "Choices are: [minimize, maximize, None]" % (objective_sense))

        if (create_variable_ids == True) and (master_scenario_tree is not None):
            raise RuntimeError("The linkInInstances method of ScenarioTree objects cannot be invoked with both create_variable_ids=True and master_scenario_tree!=None")

        # propagate the scenario instances to the scenario tree object
        # structure.
        # NOTE: The input scenario instances may be a super-set of the
        #       set of Scenario objects for this ScenarioTree.
        master_has_instance = {}
        for scenario_name, scenario_instance in iteritems(scenario_instance_map):
            if self.contains_scenario(scenario_name):
                master_has_instance[scenario_name] = False
                if master_scenario_tree is not None:
                    master_scenario = master_scenario_tree.get_scenario(scenario_name)
                    if master_scenario._instance is not None:
                        master_has_instance[scenario_name] = True
                _scenario = self.get_scenario(scenario_name)
                _scenario._instance = scenario_instance

        # link the scenario tree object structures to the instance components.
        self.populateVariableIndicesAndValues(create_variable_ids=create_variable_ids,
                                              master_scenario_tree=master_scenario_tree,
                                              initialize_solution_data=initialize_solution_data)

        # create the scenario cost expression to be used for the objective
        for scenario_name, scenario_instance in iteritems(scenario_instance_map):
            if self.contains_scenario(scenario_name):
                scenario = self.get_scenario(scenario_name)

                if master_has_instance[scenario_name]:
                    master_scenario = master_scenario_tree.get_scenario(scenario_name)
                    scenario._instance_cost_expression = master_scenario._instance_cost_expression
                    scenario._instance_objective = master_scenario._instance_objective
                    scenario._objective_sense = master_scenario._objective_sense
                    scenario._objective_name = master_scenario
                    continue

                user_objective = find_active_objective(scenario_instance, safety_checks=True)
                if objective_sense is None:
                    if user_objective is None:
                        raise RuntimeError("An active Objective could not "
                                           "be found on instance for "
                                           "scenario %s." % (scenario_name))
                    cost_expr_name = "_USER_COST_EXPRESSION_"+str(scenario_name)
                    cost_expr = Expression(name=cost_expr_name,initialize=user_objective.expr)
                    scenario_instance.add_component(cost_expr_name,cost_expr)
                    scenario._instance_cost_expression = cost_expr

                    user_objective_sense = minimize if (user_objective.is_minimizing()) else maximize
                    cost_obj_name = "_USER_COST_OBJECTIVE_"+str(scenario_name)
                    cost_obj = Objective(name=cost_obj_name,expr=cost_expr, sense=user_objective_sense)
                    scenario_instance.add_component(cost_obj_name,cost_obj)
                    scenario._instance_objective = cost_obj
                    scenario._objective_sense = user_objective_sense
                    scenario._objective_name = scenario._instance_objective.cname()
                    user_objective.deactivate()
                else:
                    if user_objective is not None:
                        #print("* Active Objective \"%s\" on scenario instance \"%s\" will not be used. "
                        #                       % (user_objective.cname(True),scenario_name))
                        user_objective.deactivate()

                    cost = 0.0
                    for stage in self._stages:
                        stage_cost_var = scenario_instance.find_component(stage._cost_variable[0])[stage._cost_variable[1]]
                        cost += stage_cost_var
                    cost_expr_name = "_PYSP_COST_EXPRESSION_"+str(scenario_name)
                    cost_expr = Expression(name=cost_expr_name,initialize=cost)
                    scenario_instance.add_component(cost_expr_name,cost_expr)
                    scenario._instance_cost_expression = cost_expr

                    cost_obj_name = "_PYSP_COST_OBJECTIVE_"+str(scenario_name)
                    cost_obj = Objective(name=cost_obj_name,expr=cost_expr, sense=objective_sense)
                    scenario_instance.add_component(cost_obj_name,cost_obj)
                    scenario._instance_objective = cost_obj
                    scenario._objective_sense = objective_sense
                    scenario._objective_name = scenario._instance_objective.cname()

    #
    # compute the set of variable indices being blended at each node. this can't be done
    # until all of the scenario instances are available, as different scenarios can have
    # different index sets.
    #

    def populateVariableIndicesAndValues(self,
                                         create_variable_ids=True,
                                         master_scenario_tree=None,
                                         initialize_solution_data=True):

        if (create_variable_ids == True) and (master_scenario_tree != None):
            raise RuntimeError("The populateVariableIndicesAndValues method of ScenarioTree objects cannot be invoked with both create_variable_ids=True and master_scenario_tree!=None")

        labeler = None
        if create_variable_ids is True:
            labeler = self._id_labeler

        for stage in self._stages:
            tree_node_list = sorted(stage._tree_nodes, key=lambda x: x._name)
            for tree_node in tree_node_list:
                name_index_to_id_map = None
                if master_scenario_tree is not None:
                    name_index_to_id_map = master_scenario_tree.get_node(tree_node._name)._name_index_to_id
                tree_node.populateVariableIndicesAndValues(id_labeler=labeler,
                                                           name_index_to_id_map=name_index_to_id_map,
                                                           initialize_solution_data=initialize_solution_data)

    #
    # is the indicated scenario / bundle in the tree?
    #

    def contains_scenario(self, name):
        return name in self._scenario_map

    def contains_bundles(self):
        return len(self._scenario_bundle_map) > 0

    def contains_bundle(self, name):
        return name in self._scenario_bundle_map

    #
    # get the scenario / bundle object from the tree.
    #

    def get_scenario(self, name):
        return self._scenario_map[name]

    def get_bundle(self, name):
        return self._scenario_bundle_map[name]

    # there are many contexts where manipulators of a scenario
    # tree simply need an arbitrary scenario to proceed...
    def get_arbitrary_scenario(self):
        return self._scenarios[0]

    def contains_node(self, name):
        return name in self._tree_node_map

    #
    # get the scenario tree node object from the tree
    #
    def get_node(self, name):
        return self._tree_node_map[name]

    #
    # utility for compressing or culling a scenario tree based on
    # a provided list of scenarios (specified by name) to retain -
    # all non-referenced components are eliminated. this particular
    # method compresses *in-place*, i.e., via direct modification
    # of the scenario tree structure.
    #

    def compress(self, scenario_bundle_list):

        # scan for and mark all referenced scenarios and
        # tree nodes in the bundle list - all stages will
        # obviously remain.
        for scenario_name in scenario_bundle_list:
            if scenario_name not in self._scenario_map:
                raise ValueError("Scenario=%s selected for "
                                 "bundling not present in "
                                 "scenario tree"
                                 % (scenario_name))
            scenario = self._scenario_map[scenario_name]
            scenario.retain = True

            # chase all nodes comprising this scenario,
            # marking them for retention.
            for node in scenario._node_list:
                node.retain = True

        # scan for any non-retained scenarios and tree nodes.
        scenarios_to_delete = []
        tree_nodes_to_delete = []
        for scenario in self._scenarios:
            if hasattr(scenario, "retain") is True:
                delattr(scenario, "retain")
            else:
                scenarios_to_delete.append(scenario)
                del self._scenario_map[scenario._name]

        for tree_node in self._tree_nodes:
            if hasattr(tree_node, "retain") is True:
                delattr(tree_node, "retain")
            else:
                tree_nodes_to_delete.append(tree_node)
                del self._tree_node_map[tree_node._name]

        # JPW does not claim the following routines are
        # the most efficient. rather, they get the job
        # done while avoiding serious issues with
        # attempting to remove elements from a list that
        # you are iterating over.

        # delete all references to unmarked scenarios
        # and child tree nodes in the scenario tree node
        # structures.
        for tree_node in self._tree_nodes:
            for scenario in scenarios_to_delete:
                if scenario in tree_node._scenarios:
                    tree_node._scenarios.remove(scenario)
            for node_to_delete in tree_nodes_to_delete:
                if node_to_delete in tree_node._children:
                    tree_node._children.remove(node_to_delete)

        # delete all references to unmarked tree nodes
        # in the scenario tree stage structures.
        for stage in self._stages:
            for tree_node in tree_nodes_to_delete:
                if tree_node in stage._tree_nodes:
                    stage._tree_nodes.remove(tree_node)

        # delete all unreferenced entries from the core scenario
        # tree data structures.
        for scenario in scenarios_to_delete:
            self._scenarios.remove(scenario)
        for tree_node in tree_nodes_to_delete:
            self._tree_nodes.remove(tree_node)

        # re-normalize the conditional probabilities of the
        # children at each tree node.
        for tree_node in self._tree_nodes:
            sum_child_probabilities = 0.0
            for child_node in tree_node._children:
                sum_child_probabilities += child_node._conditional_probability

            for child_node in tree_node._children:
                # the user may specify that the probability of a scenario is 0.0,
                # and while odd, we should allow the edge case.
                if sum_child_probabilities == 0.0:
                    child_node._conditional_probability = 0.0
                else:
                    child_node._conditional_probability = child_node._conditional_probability / sum_child_probabilities

        # re-compute the absolute scenario probabilities based
        # on the re-normalized conditional node probabilities.
        for scenario in self._scenarios:
            probability = 1.0
            for tree_node in scenario._node_list:
                probability = probability * tree_node._conditional_probability
            scenario._probability = probability

        # now that we've culled the scenarios, cull the bundles. do
        # this in two passes. in the first pass, we identify the names
        # of bundles to delete, by looking for bundles with deleted
        # scenarios. in the second pass, we delete the bundles from
        # the scenario tree, and normalize the probabilities of the
        # remaining bundles.

        # indices of the objects in the scenario tree bundle list
        bundles_to_delete = []
        for i in xrange(0,len(self._scenario_bundles)):
            scenario_bundle = self._scenario_bundles[i]
            for scenario_name in scenario_bundle._scenario_names:
                if scenario_name not in self._scenario_map:
                    bundles_to_delete.append(i)
                    break
        bundles_to_delete.reverse()
        for i in bundles_to_delete:
            deleted_bundle = self._scenario_bundles.pop(i)
            del self._scenario_bundle_map[deleted_bundle._name]

        sum_bundle_probabilities = sum(bundle._probability for bundle in self._scenario_bundles)
        for bundle in self._scenario_bundles:
            bundle._probability /= sum_bundle_probabilities

    #
    # utility for automatically selecting a proportion of scenarios from the
    # tree to retain, eliminating the rest.
    #

    def downsample(self, fraction_to_retain, random_seed, verbose=False):

        random.seed(random_seed)

        random_sequence=range(len(self._scenarios))
        random.shuffle(random_sequence)

        number_to_retain = max(int(round(float(len(random_sequence)*fraction_to_retain))), 1)

        scenario_bundle_list = []
        for i in xrange(number_to_retain):
            scenario_bundle_list.append(self._scenarios[random_sequence[i]]._name)

        if verbose is True:
            print("Downsampling scenario tree - retained %s "
                  "scenarios: %s"
                  % (len(scenario_bundle_list),
                     str(scenario_bundle_list)))

        self.compress(scenario_bundle_list)


    #
    # returns the root node of the scenario tree
    #

    def findRootNode(self):

        for tree_node in self._tree_nodes:
            if tree_node._parent is None:
                return tree_node
        return None

    #
    # a utility function to compute, based on the current scenario tree content,
    # the maximal length of identifiers in various categories.
    #

    def computeIdentifierMaxLengths(self):

        self._max_scenario_id_length = 0
        for scenario in self._scenarios:
            if len(str(scenario._name)) > self._max_scenario_id_length:
                self._max_scenario_id_length = len(str(scenario._name))

    #
    # a utility function to (partially, at the moment) validate a scenario tree
    #

    def validate(self):

        # for any node, the sum of conditional probabilities of the children should sum to 1.
        for tree_node in self._tree_nodes:
            sum_probabilities = 0.0
            if len(tree_node._children) > 0:
                for child in tree_node._children:
                    sum_probabilities += child._conditional_probability
                if abs(1.0 - sum_probabilities) > 0.000001:
                    print("The child conditional probabilities for tree node=%s "
                          " sum to %s" % (tree_node._name, sum_probabilities))
                    return False

        # ensure that there is only one root node in the tree
        num_roots = 0
        root_ids = []
        for tree_node in self._tree_nodes:
            if tree_node._parent is None:
                num_roots += 1
                root_ids.append(tree_node._name)

        if num_roots != 1:
            print("Illegal set of root nodes detected: " + str(root_ids))
            return False

        # there must be at least one scenario passing through each tree node.
        for tree_node in self._tree_nodes:
            if len(tree_node._scenarios) == 0:
                print("There are no scenarios associated with tree node=%s"
                      % (tree_node._name))
                return False

        return True

    #
    # copies the parameter values stored in any tree node _averages attribute
    # into any tree node _solution attribute - only for active variable values.
    #

    def snapshotSolutionFromAverages(self):

        for tree_node in self._tree_nodes:

            tree_node.snapshotSolutionFromAverages()

    #
    # assigns the variable values at each tree node based on the input
    # instances.
    #

    # Note: Trying to work this function out of the code. The only
    #       solution we should get used to working with is that stored
    #       on the scenario objects
    def XsnapshotSolutionFromInstances(self):

        for tree_node in self._tree_nodes:
            tree_node.snapshotSolutionFromInstances()

    def pullScenarioSolutionsFromInstances(self):

        for scenario in self._scenarios:
            scenario.update_solution_from_instance()

    def snapshotSolutionFromScenarios(self):
        for tree_node in self._tree_nodes:
            tree_node.snapshotSolutionFromScenarios()

    def create_random_bundles(self, scenario_tree_instance, num_bundles, random_seed):

        random.seed(random_seed)

        num_scenarios = len(self._scenarios)

        sequence = list(xrange(num_scenarios))
        random.shuffle(sequence)

        scenario_tree_instance.Bundling[None] = True

        next_scenario_index = 0

        # this is a hack-ish way to re-initialize the Bundles set of a
        # scenario tree instance, which should already be there
        # (because it is defined in the abstract model).  however, we
        # don't have a "clear" method on a set, so...
        scenario_tree_instance.del_component("Bundles")
        scenario_tree_instance.add_component("Bundles", Set())
        for i in xrange(1, num_bundles+1):
            bundle_name = "Bundle"+str(i)
            scenario_tree_instance.Bundles.add(bundle_name)

        # ditto above comment regarding del_component/add_component
        scenario_tree_instance.del_component("BundleScenarios")
        scenario_tree_instance.add_component("BundleScenarios",
                                             Set(scenario_tree_instance.Bundles))

        bundles = []
        for i in xrange(num_bundles):
            bundle_name = "Bundle"+str(i+1)
            scenario_tree_instance.BundleScenarios[bundle_name] = Set()
            bundles.append(scenario_tree_instance.BundleScenarios[bundle_name])

        scenario_index = 0
        while (scenario_index < num_scenarios):
            for bundle_index in xrange(num_bundles):
                if (scenario_index == num_scenarios):
                    break
                bundles[bundle_index].add(
                    self._scenarios[sequence[scenario_index]]._name)
                scenario_index += 1

        self._construct_scenario_bundles(scenario_tree_instance)

    #
    # a utility function to pretty-print the static/non-cost
    # information associated with a scenario tree
    #

    def pprint(self):

        print("Scenario Tree Detail")

        print("----------------------------------------------------")
        print("Tree Nodes:")
        print("")
        for tree_node_name in sorted(iterkeys(self._tree_node_map)):
            tree_node = self._tree_node_map[tree_node_name]
            print("\tName=%s" % (tree_node_name))
            if tree_node._stage is not None:
                print("\tStage=%s" % (tree_node._stage._name))
            else:
                print("\t Stage=None")
            if tree_node._parent is not None:
                print("\tParent=%s" % (tree_node._parent._name))
            else:
                print("\tParent=" + "None")
            if tree_node._conditional_probability is not None:
                print("\tConditional probability=%4.4f" % tree_node._conditional_probability)
            else:
                print("\tConditional probability=" + "***Undefined***")
            print("\tChildren:")
            if len(tree_node._children) > 0:
                for child_node in sorted(tree_node._children, key=lambda x: x._name):
                    print("\t\t%s" % (child_node._name))
            else:
                print("\t\tNone")
            print("\tScenarios:")
            if len(tree_node._scenarios) == 0:
                print("\t\tNone")
            else:
                for scenario in sorted(tree_node._scenarios, key=lambda x: x._name):
                    print("\t\t%s" % (scenario._name))
            print("")
        print("----------------------------------------------------")
        print("Stages:")
        for stage_name in sorted(iterkeys(self._stage_map)):
            stage = self._stage_map[stage_name]
            print("\tName=%s" % (stage_name))
            print("\tTree Nodes: ")
            for tree_node in sorted(stage._tree_nodes, key=lambda x: x._name):
                print("\t\t%s" % (tree_node._name))
            if len(stage._variables) > 0:
                print("\tVariables: ")
                for variable_name in sorted(iterkeys(stage._variables)):
                    match_templates = stage._variables[variable_name]
                    sys.stdout.write("\t\t "+variable_name+" : ")
                    for match_template in match_templates:
                       sys.stdout.write(indexToString(match_template)+' ')
                    print("")
            if len(stage._derived_variables) > 0:
                print("\tDerived Variables: ")
                for variable_name in sorted(iterkeys(stage._derived_variables)):
                    match_templates = stage._derived_variables[variable_name]
                    sys.stdout.write("\t\t "+variable_name+" : ")
                    for match_template in match_templates:
                       sys.stdout.write(indexToString(match_template)+' ')
                    print("")
            print("\tCost Variable: ")
            if stage._cost_variable[1] is None:
                print("\t\t" + stage._cost_variable[0])
            else:
                print("\t\t" + stage._cost_variable[0] + indexToString(stage._cost_variable[1]))
            print("")
        print("----------------------------------------------------")
        print("Scenarios:")
        for scenario_name in sorted(iterkeys(self._scenario_map)):
            scenario = self._scenario_map[scenario_name]
            print("\tName=%s" % (scenario_name))
            print("\tProbability=%4.4f" % scenario._probability)
            if scenario._leaf_node is None:
                print("\tLeaf node=None")
            else:
                print("\tLeaf node=%s" % (scenario._leaf_node._name))
            print("\tTree node sequence:")
            for tree_node in scenario._node_list:
                print("\t\t%s" % (tree_node._name))
            print("")
        print("----------------------------------------------------")
        if len(self._scenario_bundles) > 0:
            print("Scenario Bundles:")
            for bundle_name in sorted(iterkeys(self._scenario_bundle_map)):
                scenario_bundle = self._scenario_bundle_map[bundle_name]
                print("\tName=%s" % (bundle_name))
                print("\tProbability=%4.4f" % scenario_bundle._probability            )
                sys.stdout.write("\tScenarios:  ")
                for scenario_name in sorted(scenario_bundle._scenario_names):
                    sys.stdout.write(str(scenario_name)+' ')
                sys.stdout.write("\n")
                print("")
            print("----------------------------------------------------")

    #
    # a utility function to pretty-print the solution associated with a scenario tree
    #

    def pprintSolution(self, epsilon=1.0e-5):

        print("----------------------------------------------------")
        print("Tree Nodes:")
        print("")
        for tree_node_name in sorted(iterkeys(self._tree_node_map)):
            tree_node = self._tree_node_map[tree_node_name]
            print("\tName=%s" % (tree_node_name))
            if tree_node._stage is not None:
                print("\tStage=%s" % (tree_node._stage._name))
            else:
                print("\t Stage=None")
            if tree_node._parent is not None:
                print("\tParent=%s" % (tree_node._parent._name))
            else:
                print("\tParent=" + "None")
            if len(tree_node._stage._variables) > 0:
                print("\tVariables: ")
                for variable_name in sorted(iterkeys(tree_node._stage._variables)):
                    indices = sorted(tree_node._variable_indices[variable_name])
                    for index in indices:
                        id = tree_node._name_index_to_id[variable_name,index]
                        if id in tree_node._standard_variable_ids:
                            # if a solution has not yet been stored /
                            # snapshotted, then the value won't be in the solution map
                            try:
                                value = tree_node._solution[id]
                            except:
                                value = None
                            if (value is not None) and (fabs(value) > epsilon):
                                print("\t\t"+variable_name+indexToString(index)+"="+str(value))
            if len(tree_node._stage._derived_variables) > 0:
                print("\tDerived Variables: ")
                for variable_name in sorted(iterkeys(tree_node._stage._derived_variables)):
                    indices = sorted(tree_node._variable_indices[variable_name])
                    for index in indices:
                        id = tree_node._name_index_to_id[variable_name,index]
                        if id in tree_node._derived_variable_ids:
                            # if a solution has not yet been stored /
                            # snapshotted, then the value won't be in the solution map
                            try:
                                value = tree_node._solution[tree_node._name_index_to_id[variable_name,index]]
                            except:
                                value = None
                            if (value is not None) and (fabs(value) > epsilon):
                                print("\t\t"+variable_name+indexToString(index)+"="+str(value))
            print("")

    #
    # a utility function to pretty-print the cost information associated with a scenario tree
    #

    def pprintCosts(self):

        print("Scenario Tree Costs")
        print("***CAUTION***: Assumes full (or nearly so) convergence of scenario solutions at each node in the scenario tree - computed costs are invalid otherwise")

        print("----------------------------------------------------")
        print("Tree Nodes:")
        print("")
        for tree_node_name in sorted(iterkeys(self._tree_node_map)):
            tree_node = self._tree_node_map[tree_node_name]
            print("\tName=%s" % (tree_node_name))
            if tree_node._stage is not None:
                print("\tStage=%s" % (tree_node._stage._name))
            else:
                print("\t Stage=None")
            if tree_node._parent is not None:
                print("\tParent=%s" % (tree_node._parent._name))
            else:
                print("\tParent=" + "None")
            if tree_node._conditional_probability is not None:
                print("\tConditional probability=%4.4f" % tree_node._conditional_probability)
            else:
                print("\tConditional probability=" + "***Undefined***")
            print("\tChildren:")
            if len(tree_node._children) > 0:
                for child_node in sorted(tree_node._children, key=lambda x: x._name):
                    print("\t\t%s" % (child_node._name))
            else:
                print("\t\tNone")
            print("\tScenarios:")
            if len(tree_node._scenarios) == 0:
                print("\t\tNone")
            else:
                for scenario in sorted(tree_node._scenarios, key=lambda x: x._name):
                    print("\t\t%s" % (scenario._name))
            print("\tExpected cost of (sub)tree rooted at node=%10.4f" % tree_node.computeExpectedNodeCost())
            print("")

        print("----------------------------------------------------")
        print("Scenarios:")
        print("")
        for scenario_name in sorted(iterkeys(self._scenario_map)):
            scenario = self._scenario_map[scenario_name]

            print("\tName=%s" % (scenario_name))
            print("\tProbability=%4.4f" % scenario._probability)

            if scenario._leaf_node is None:
                print("\tLeaf Node=None")
            else:
                print("\tLeaf Node=%s" % (scenario._leaf_node._name))

            print("\tTree node sequence:")
            for tree_node in scenario._node_list:
                print("\t\t%s" % (tree_node._name))

            aggregate_cost = 0.0
            for stage in self._stages:
                # find the tree node for this scenario, representing this stage.
                tree_node = None
                for node in scenario._node_list:
                    if node._stage == stage:
                        tree_node = node
                        break

                cost_variable_value = scenario._stage_costs[stage._name]

                if cost_variable_value is not None:
                    print("\tStage=%20s     Cost=%10.4f"
                          % (stage._name, cost_variable_value))
                    cost = cost_variable_value
                else:
                    print("\tStage=%20s     Cost=%10s"
                          % (stage._name, "Not Rprted."))
                    cost = 0.0
                aggregate_cost += cost

            print("\tTotal scenario cost=%10.4f" % aggregate_cost)
            print("")
        print("----------------------------------------------------")
