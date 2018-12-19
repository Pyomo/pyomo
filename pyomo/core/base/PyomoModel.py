#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

__all__ = ['Model', 'ConcreteModel', 'AbstractModel', 'global_option']

import logging
import sys
from weakref import ref as weakref_ref
import gc
import time
import math
import functools

try:
    from collections import OrderedDict
except ImportError:                         #pragma:nocover
    from ordereddict import OrderedDict
try:
    from pympler import muppy
    from pympler import summary
    pympler_available = True
except ImportError:                         #pragma:nocover
    pympler_available = False
except AttributeError:                         #pragma:nocover
    pympler_available = False


from pyutilib.math import *
from pyutilib.misc import tuplize, Container, PauseGC, Bunch

import pyomo.common
from pyomo.common.deprecation import deprecation_warning
from pyomo.common.plugin import ExtensionPoint
from pyomo.common._task import pyomo_api
from pyomo.common.deprecation import deprecation_warning

from pyomo.core.expr import expr_common
from pyomo.core.expr.symbol_map import SymbolMap

from pyomo.core.base.var import _VarData, Var
from pyomo.core.base.constraint import Constraint
from pyomo.core.base.objective import Objective
from pyomo.core.base.set_types import *
from pyomo.core.base.suffix import active_import_suffix_generator
from pyomo.core.base.indexed_component import IndexedComponent
from pyomo.dataportal import DataPortal
from pyomo.core.base.plugin import *
from pyomo.core.base.numvalue import *
from pyomo.core.base.block import SimpleBlock
from pyomo.core.base.sets import Set
from pyomo.core.base.component import Component, ComponentUID
from pyomo.core.base.plugin import ModelComponentFactory, TransformationFactory
from pyomo.core.base.label import CNameLabeler, CuidLabeler

import pyomo.opt
from pyomo.opt.results import SolverResults, Solution, SolutionStatus, UndefinedData

from six import itervalues, iteritems, StringIO, string_types
from six.moves import xrange
try:
    unicode
except:
    basestring = unicode = str

logger = logging.getLogger('pyomo.core')
id_func = id


def global_option(function, name, value):
    """
    Declare the default value for a global Pyomo configuration option.

    Example use:

    @global_option('config.foo.bar', 1)
    def functor():
        ...
    """
    PyomoConfig._option[tuple(name.split('.'))] = value
    def wrapper_function(*args, **kwargs):
        return function(*args, **kwargs)
    return wrapper_function


class PyomoConfig(Container):
    """
    This is a pyomo-specific configuration object, which is a subclass of Container.
    """

    _option = {}

    def __init__(self, *args, **kw):
        Container.__init__(self, *args, **kw)
        self.set_name('PyomoConfig')
        #
        # Create the nested options specified by the the PyomoConfig._option
        # dictionary, which has been populated with the global_option decorator.
        #
        for item in PyomoConfig._option:
            d = self
            for attr in item[:-1]:
                if not attr in d:
                    d[attr] = Container()
                d = d[attr]
            d[item[-1]] = PyomoConfig._option[item]


class ModelSolution(object):

    def __init__(self):
        self._metadata = {}
        self._metadata['status'] = None
        self._metadata['message'] = None
        self._metadata['gap'] = None
        self._entry = {}
        #
        # entry[name]: id -> (object weakref, entry)
        #
        for name in ['objective', 'variable', 'constraint', 'problem']:
            self._entry[name] = {}

    def __getattr__(self, name):
        if name[0] == '_':
            if name in self.__dict__:
                return self.__dict__[name]
            else:
                raise AttributeError( "'%s' object has no attribute '%s'"
                                      % (self.__class__.__name__, name) )
        return self.__dict__['_metadata'][name]

    def __setattr__(self, name, val):
        if name[0] == '_':
            self.__dict__[name] = val
            return
        self.__dict__['_metadata'][name] = val

    def __getstate__(self):
        state = {
            '_metadata': self._metadata,
            '_entry': {}
        }
        for (name, data) in iteritems(self._entry):
            tmp = state['_entry'][name] = []
            # Note: We must convert all weakrefs to hard refs and
            # not indirect references like ComponentUIDs because
            # when it comes time to unpickle, we cannot count on the
            # model instance to have already been reconstructed --
            # so things like CUID.find_component will fail (return
            # None).
            for obj, entry in itervalues(data):
                if obj is None or obj() is None:
                    logger.warn(
                        "Solution component in '%s' no longer "
                        "accessible: %s!" % ( name, entry ))
                else:
                    tmp.append( ( obj(), entry ) )
        return state

    def __setstate__(self, state):
        self._metadata = state['_metadata']
        self._entry = {}
        for name, data in iteritems(state['_entry']):
            tmp = self._entry[name] = {}
            for obj, entry in data:
                tmp[ id(obj) ] = ( weakref_ref(obj), entry )


class ModelSolutions(object):

    def __init__(self, instance):
        self._instance = weakref_ref(instance)
        self.clear()

    def clear(self, clear_symbol_maps=True):
        # _symbol_map: smap_id -> SymbolMap
        if clear_symbol_maps:
            self.symbol_map = {}
        self.solutions = []
        self.index = None

    def __getstate__(self):
        state = {}
        state['index'] = self.index
        state['_instance'] = self._instance()
        state['solutions'] = self.solutions
        state['symbol_map'] = self.symbol_map
        return state

    def __setstate__(self, state):
        for key, val in iteritems(state):
            setattr(self, key, val)
        # Restore the instance weakref
        self._instance = weakref_ref(self._instance)

    def __len__(self):
        return len(self.solutions)

    def __getitem__(self, index):
        return self.solutions[index]

    def add_symbol_map(self, symbol_map):
        self.symbol_map[id(symbol_map)] = symbol_map

    def delete_symbol_map(self, smap_id):
        if not smap_id is None:
            del self.symbol_map[smap_id]

    def load_from(self,
                  results,
                  allow_consistent_values_for_fixed_vars=False,
                  comparison_tolerance_for_fixed_vars=1e-5,
                  ignore_invalid_labels=False,
                  id=None,
                  delete_symbol_map=True,
                  clear=True,
                  default_variable_value=None,
                  select=0,
                  ignore_fixed_vars=True):
        """
        Load solver results
        """
        instance = self._instance()
        #
        # If there is a warning, then print a warning message.
        #
        if (results.solver.status == pyomo.opt.SolverStatus.warning):
            logger.warning(
                'Loading a SolverResults object with a '
                'warning status into model=%s;\n'
                '    message from solver=%s'
                % (instance.name, results.solver.Message))
        #
        # If the solver status not one of either OK or Warning, then generate an error.
        #
        elif results.solver.status != pyomo.opt.SolverStatus.ok:
            if (results.solver.status == pyomo.opt.SolverStatus.aborted) and \
               (len(results.solution) > 0):
                logger.warning(
                    "Loading a SolverResults object with "
                    "an 'aborted' status, but containing a solution")
            else:
                raise ValueError("Cannot load a SolverResults object "
                                 "with bad status: %s"
                                 % str(results.solver.status))
        if clear:
            #
            # Clear the solutions, but not the symbol map
            #
            self.clear(clear_symbol_maps=False)
        #
        # Load all solutions
        #
        if len(results.solution) == 0:
            return
        smap = results.__dict__.get('_smap', None)
        if not smap is None:
            smap_id = id_func(smap)
            self.add_symbol_map(smap)
            results._smap = None
        else:
            smap_id = results.__dict__.get('_smap_id')
        cache = {}
        if not id is None:
            self.add_solution(results.solution(id),
                              smap_id,
                              delete_symbol_map=False,
                              cache=cache,
                              ignore_invalid_labels=ignore_invalid_labels,
                              default_variable_value=default_variable_value)
        else:
            for i in range(len(results.solution)):
                self.add_solution(results.solution(i),
                                  smap_id,
                                  delete_symbol_map=False,
                                  cache=cache,
                                  ignore_invalid_labels=ignore_invalid_labels,
                                  default_variable_value=default_variable_value)

        if delete_symbol_map:
            self.delete_symbol_map(smap_id)
        #
        # Load the first solution into the model
        #
        if not select is None:
            self.select(
                select,
                allow_consistent_values_for_fixed_vars=allow_consistent_values_for_fixed_vars,
                comparison_tolerance_for_fixed_vars=comparison_tolerance_for_fixed_vars,
                ignore_invalid_labels=ignore_invalid_labels,
                ignore_fixed_vars=ignore_fixed_vars)

    def store_to(self, results, cuid=False):
        """
        Return a Solution() object that is populated with the values in the model.
        """
        instance = self._instance()
        results.solution.clear()
        results._smap_id = None

        for soln_ in self.solutions:
            soln = Solution()
            soln._cuid = cuid
            for key, val in iteritems(soln_._metadata):
                setattr(soln, key, val)

            if cuid:
                labeler = CuidLabeler()
            else:
                labeler = CNameLabeler()
            sm = SymbolMap()

            entry = soln_._entry['objective']
            for obj in instance.component_data_objects(Objective, active=True):
                vals = entry.get(id(obj), None)
                if vals is None:
                    vals = {}
                else:
                    vals = vals[1]
                vals['Value'] = value(obj)
                soln.objective[ sm.getSymbol(obj, labeler) ] = vals
            entry = soln_._entry['variable']
            for obj in instance.component_data_objects(Var, active=True):
                if obj.stale:
                    continue
                vals = entry.get(id(obj), None)
                if vals is None:
                    vals = {}
                else:
                    vals = vals[1]
                vals['Value'] = value(obj)
                soln.variable[ sm.getSymbol(obj, labeler) ] = vals
            entry = soln_._entry['constraint']
            for obj in instance.component_data_objects(Constraint, active=True):
                vals = entry.get(id(obj), None)
                if vals is None:
                    continue
                else:
                    vals = vals[1]
                soln.constraint[ sm.getSymbol(obj, labeler) ] = vals
            results.solution.insert( soln )

    def add_solution(self,
                     solution,
                     smap_id,
                     delete_symbol_map=True,
                     cache=None,
                     ignore_invalid_labels=False,
                     ignore_missing_symbols=True,
                     default_variable_value=None):

        instance = self._instance()

        soln = ModelSolution()
        soln._metadata['status'] = solution.status
        if not type(solution.message) is UndefinedData:
            soln._metadata['message'] = solution.message
        if not type(solution.gap) is UndefinedData:
            soln._metadata['gap'] = solution.gap

        if smap_id is None:
            #
            # Cache symbol names, which might be re-used in subsequent
            # calls to add_solution()
            #
            if cache is None:
                cache = {}
            if solution._cuid:
                #
                # Loading a solution with CUID keys
                #
                if len(cache) == 0:
                    for obj in instance.component_data_objects(Var):
                        cache[ComponentUID(obj)] = obj
                    for obj in instance.component_data_objects(Objective, active=True):
                        cache[ComponentUID(obj)] = obj
                    for obj in instance.component_data_objects(Constraint, active=True):
                        cache[ComponentUID(obj)] = obj

                for name in ['problem', 'objective', 'variable', 'constraint']:
                    tmp = soln._entry[name]
                    for cuid, val in iteritems(getattr(solution, name)):
                        obj = cache.get(cuid, None)
                        if obj is None:
                            if ignore_invalid_labels:
                                continue
                            raise RuntimeError("CUID %s is missing from model %s"
                                               % (str(cuid), instance.name))
                        tmp[id(obj)] = (weakref_ref(obj), val)
            else:
                #
                # Loading a solution with string keys
                #
                if len(cache) == 0:
                    for obj in instance.component_data_objects(Var):
                        cache[obj.name] = obj
                    for obj in instance.component_data_objects(Objective, active=True):
                        cache[obj.name] = obj
                    for obj in instance.component_data_objects(Constraint, active=True):
                        cache[obj.name] = obj

                for name in ['problem', 'objective', 'variable', 'constraint']:
                    tmp = soln._entry[name]
                    for symb, val in iteritems(getattr(solution, name)):
                        obj = cache.get(symb, None)
                        if obj is None:
                            if ignore_invalid_labels:
                                continue
                            raise RuntimeError("Symbol %s is missing from model %s"
                                               % (symb, instance.name))
                        tmp[id(obj)] = (weakref_ref(obj), val)
        else:
            #
            # Map solution
            #
            smap = self.symbol_map[smap_id]
            for name in ['problem', 'objective', 'variable', 'constraint']:
                tmp = soln._entry[name]
                for symb, val in iteritems(getattr(solution, name)):
                    if symb in smap.bySymbol:
                        obj = smap.bySymbol[symb]
                    elif symb in smap.aliases:
                        obj = smap.aliases[symb]
                    elif ignore_missing_symbols:
                        continue
                    else:                                   #pragma:nocover
                        #
                        # This should never happen ...
                        #
                        raise RuntimeError(
                            "ERROR: Symbol %s is missing from "
                            "model %s when loading with a symbol map!"
                            % (symb, instance.name))

                    tmp[id(obj())] = (obj, val)
            #
            # Wrap up
            #
            if delete_symbol_map:
                self.delete_symbol_map(smap_id)

        #
        # Collect fixed variables
        #
        tmp = soln._entry['variable']
        for vdata in instance.component_data_objects(Var):
            id_ = id(vdata)
            if vdata.fixed:
                tmp[id_] = (weakref_ref(vdata), {'Value':value(vdata)})
            elif (default_variable_value is not None) and \
                 (smap_id is not None) and \
                 (id_ in smap.byObject) and \
                 (id_ not in tmp):
                tmp[id_] = (weakref_ref(vdata), {'Value':default_variable_value})

        self.solutions.append(soln)
        return len(self.solutions)-1

    def select(self,
               index=0,
               allow_consistent_values_for_fixed_vars=False,
               comparison_tolerance_for_fixed_vars=1e-5,
               ignore_invalid_labels=False,
               ignore_fixed_vars=True):
        """
        Select a solution from the model's solutions.

        allow_consistent_values_for_fixed_vars: a flag that
        indicates whether a solution can specify consistent
        values for variables in the model that are fixed.

        ignore_invalid_labels: a flag that indicates whether
        labels in the solution that don't appear in the model
        yield an error. This allows for loading a results object
        generated from one model into another related, but not
        identical, model.
        """
        instance = self._instance()
        #
        # Set the "stale" flag of each variable in the model prior to loading the
        # solution, so you known which variables have "real" values and which ones don't.
        #
        instance._flag_vars_as_stale()
        if not index is None:
            self.index = index
        soln = self.solutions[self.index]

        #
        # Generate the list of active import suffixes on this top level model
        #
        valid_import_suffixes = dict(active_import_suffix_generator(instance))
        #
        # To ensure that import suffix data gets properly overwritten (e.g.,
        # the case where nonzero dual values exist on the suffix and but only
        # sparse dual values exist in the results object) we clear all active
        # import suffixes.
        #
        for suffix in itervalues(valid_import_suffixes):
            suffix.clear_all_values()
        #
        # Load problem (model) level suffixes. These would only come from ampl
        # interfaced solution suffixes at this point in time.
        #
        for id_, (pobj,entry) in iteritems(soln._entry['problem']):
            for _attr_key, attr_value in iteritems(entry):
                attr_key = _attr_key[0].lower() + _attr_key[1:]
                if attr_key in valid_import_suffixes:
                    valid_import_suffixes[attr_key][pobj] = attr_value
        #
        # Load objective data (suffixes)
        #
        for id_, (odata, entry) in iteritems(soln._entry['objective']):
            odata = odata()
            for _attr_key, attr_value in iteritems(entry):
                attr_key = _attr_key[0].lower() + _attr_key[1:]
                if attr_key in valid_import_suffixes:
                    valid_import_suffixes[attr_key][odata] = attr_value
        #
        # Load variable data (suffixes and values)
        #
        for id_, (vdata, entry) in iteritems(soln._entry['variable']):
            vdata = vdata()
            val = entry['Value']
            if vdata.fixed is True:
                if ignore_fixed_vars:
                    continue
                if not allow_consistent_values_for_fixed_vars:
                    msg = "Variable '%s' in model '%s' is currently fixed - new" \
                          ' value is not expected in solution'
                    raise TypeError(msg % (vdata.name, instance.name))
                if math.fabs(val - vdata.value) > comparison_tolerance_for_fixed_vars:
                    raise TypeError("Variable '%s' in model '%s' is currently "
                                    "fixed - a value of '%s' in solution is "
                                    "not within tolerance=%s of the current "
                                    "value of '%s'"
                                    % (vdata.name,
                                       instance.name,
                                       str(val),
                                       str(comparison_tolerance_for_fixed_vars),
                                       str(vdata.value)))

            vdata.value = val
            vdata.stale = False

            for _attr_key, attr_value in iteritems(entry):
                attr_key = _attr_key[0].lower() + _attr_key[1:]
                if attr_key == 'value':
                    continue
                elif attr_key in valid_import_suffixes:
                    valid_import_suffixes[attr_key][vdata] = attr_value
        #
        # Load constraint data (suffixes)
        #
        for id_, (cdata, entry) in iteritems(soln._entry['constraint']):
            cdata = cdata()
            for _attr_key, attr_value in iteritems(entry):
                attr_key = _attr_key[0].lower() + _attr_key[1:]
                if attr_key in valid_import_suffixes:
                    valid_import_suffixes[attr_key][cdata] = attr_value


@ModelComponentFactory.register('Model objects can be used as a component of other models.')
class Model(SimpleBlock):
    """
    An optimization model.  By default, this defers construction of components
    until data is loaded.
    """

    preprocessor_ep = ExtensionPoint(IPyomoPresolver)

    _Block_reserved_words = set()

    def __new__(cls, *args, **kwds):
        if cls != Model:
            return super(Model, cls).__new__(cls)

        logger.warning(
"""DEPRECATION WARNING: Using the 'Model' class is deprecated.  Please
use the AbstractModel or ConcreteModel class instead.""")
        return AbstractModel.__new__(AbstractModel)

    def __init__(self, name='unknown', **kwargs):
        """Constructor"""
        #
        # NOTE: The 'ctype' keyword argument is not defined here.  Thus,
        # a model is treated as a 'Block' class type.  This simplifies
        # the definition of the block_data_objects() method, since we treat
        # Model and Block objects as the same.  Similarly, this avoids
        # the requirement to import PyomoModel.py in the block.py file.
        #
        SimpleBlock.__init__(self, **kwargs)
        self._name = name
        self.statistics = Container()
        self.config = PyomoConfig()
        self.solutions = ModelSolutions(self)
        self.config.preprocessor = 'pyomo.model.simple_preprocessor'

    def compute_statistics(self, active=True):
        """
        Compute model statistics
        """
        if len(self.statistics) > 0:
            return
        self.statistics.number_of_variables = 0
        self.statistics.number_of_constraints = 0
        self.statistics.number_of_objectives = 0
        for block in self.block_data_objects(active=active):
            for data in self.component_map(Var, active=active).itervalues():
                self.statistics.number_of_variables += len(data)
            for data in self.component_map(Objective, active=active).itervalues():
                self.statistics.number_of_objectives += len(data)
            for data in self.component_map(Constraint, active=active).itervalues():
                self.statistics.number_of_constraints += len(data)

    def nvariables(self):
        self.compute_statistics()
        return self.statistics.number_of_variables

    def nconstraints(self):
        self.compute_statistics()
        return self.statistics.number_of_constraints

    def nobjectives(self):
        self.compute_statistics()
        return self.statistics.number_of_objectives

    def create_instance( self, filename=None, data=None, name=None,
                         namespace=None, namespaces=None,
                         profile_memory=0, report_timing=False,
                         **kwds ):
        """
        Create a concrete instance of an abstract model, possibly using data
        read in from a file.

        Optional:
            filename:           The name of a Pyomo Data File that will be used
                                    to load data into the model.
            data:               A dictionary containing initialization data for
                                    the model to be used if there is no filename
            name:               The name given to the model.
            namespace:          A namespace used to select data.
            namespaces:         A list of namespaces used to select data.
            profile_memory:     A number that indicates the profiling level.
            report_timing:      Report timing statistics during construction.
        """
        #
        # Generate a warning if this is a concrete model but the
        # filename is specified.  A concrete model is already
        # constructed, so passing in a data file is a waste of time.
        #
        if self.is_constructed() and isinstance(filename, string_types):
            msg = "The filename=%s will not be loaded - supplied as an " \
                  "argument to the create_instance() method of a "\
                  "concrete instance with name=%s." % (filename, name)
            logger.warning(msg)

        if 'clone' in kwds:
            kwds.pop('clone')
            deprecation_warning(
                "Model.create_instance() no longer accepts the 'clone' "
                "argument: the base abstract model is always cloned.")
        if 'preprocess' in kwds:
            kwds.pop('preprocess')
            deprecation_warning(
                "Model.create_instance() no longer accepts the preprocess' "
                "argument: preprocessing is always deferred to when the "
                "model is sent to the solver")
        if kwds:
            msg = \
"""Model.create_instance() passed the following unrecognized keyword
arguments (which have been ignored):"""
            for k in kwds:
                msg = msg + "\n    '%s'" % (k,)
            logger.error(msg)

        if self.is_constructed():
            deprecation_warning(
                "Cannot call Model.create_instance() on a constructed "
                "model; returning a clone of the current model instance.")
            return self.clone()

        if report_timing:
            pyomo.common.timing.report_timing()

        if name is None:
            name = self.name
        if filename is not None:
            if data is not None:
                logger.warning("Model.create_instance() passed both 'filename' "
                               "and 'data' keyword arguments.  Ignoring the "
                               "'data' argument")
            data = filename
        if data is None:
            data = {}

        #
        # Clone the model and load the data
        #
        instance = self.clone()

        if name is not None:
            instance._name = name

        # If someone passed a rule for creating the instance, fire the
        # rule before constructing the components.
        if instance._rule is not None:
            instance._rule(instance)

        if namespaces:
            _namespaces = list(namespaces)
        else:
            _namespaces = []
        if namespace is not None:
            _namespaces.append(namespace)
        if None not in _namespaces:
            _namespaces.append(None)

        instance.load( data,
                       namespaces=_namespaces,
                       profile_memory=profile_memory )

        #
        # Preprocess the new model
        #

        if False and preprocess is True:

            if report_timing is True:
                start_time = time.time()

            instance.preprocess()

            if report_timing is True:
                total_time = time.time() - start_time
                print("      %6.2f seconds required for preprocessing" % total_time)

            if (pympler_available is True) and (profile_memory >= 2):
                mem_used = muppy.get_size(muppy.get_objects())
                print("      Total memory = %d bytes following instance preprocessing" % mem_used)
                print("")

            if (pympler_available is True) and (profile_memory >= 2):
                print("")
                print("      Summary of objects following instance preprocessing")
                post_preprocessing_summary = summary.summarize(muppy.get_objects())
                summary.print_(post_preprocessing_summary, limit=100)

        #
        # Indicate that the model is concrete/constructed
        #
        instance._constructed = True
        #
        # Change this class from "Abstract" to "Concrete".  It is
        # absolutely crazy that this is allowed in Python, but since the
        # AbstractModel and ConcreteModel are basically identical, we
        # can "reassign" the new concrete instance to be an instance of
        # ConcreteModel
        #
        instance.__class__ = ConcreteModel
        return instance


    def preprocess(self, preprocessor=None):
        """Apply the preprocess plugins defined by the user"""
        with PauseGC() as pgc:
            if preprocessor is None:
                preprocessor = self.config.preprocessor
            pyomo.common.PyomoAPIFactory(preprocessor)(self.config, model=self)

    def load(self, arg, namespaces=[None], profile_memory=0, report_timing=None):
        """
        Load the model with data from a file, dictionary or DataPortal object.
        """
        if report_timing is not None:
            deprecation_warning(
                "The report_timing argument to Model.load() is deprecated.  "
                "Use pyomo.common.timing.report_timing() to enable reporting "
                "construction timing")
        if arg is None or isinstance(arg, basestring):
            dp = DataPortal(filename=arg, model=self)
        elif type(arg) is DataPortal:
            dp = arg
        elif type(arg) is dict:
            dp = DataPortal(data_dict=arg, model=self)
        elif isinstance(arg, SolverResults):
            if len(arg.solution):
                logger.warning(
"""DEPRECATION WARNING: the Model.load() method is deprecated for
loading solutions stored in SolverResults objects.  Call
Model.solutions.load_from().""")
                self.solutions.load_from(arg)
            else:
                logger.warning(
"""DEPRECATION WARNING: the Model.load() method is deprecated for
loading solutions stored in SolverResults objects.  By default, results
from solvers are immediately loaded into the original model instance.""")
            return
        else:
            msg = "Cannot load model model data from with object of type '%s'"
            raise ValueError(msg % str( type(arg) ))
        self._load_model_data(dp,
                              namespaces,
                              profile_memory=profile_memory)

    def _tuplize(self, data, setobj):
        if data is None:            #pragma:nocover
            return None
        if setobj.dimen == 1:
            return data
        if len(list(data.keys())) == 1 and list(data.keys())[0] is None and len(data[None]) == 0: # dlw december 2017
            return None
        ans = {}
        for key in data:
            if type(data[key][0]) is tuple:
                return data
            ans[key] = tuplize(data[key], setobj.dimen, setobj.local_name)
        return ans

    def _load_model_data(self, modeldata, namespaces, **kwds):
        """
        Load declarations from a DataPortal object.
        """
        #
        # As we are primarily generating objects here (and acyclic ones
        # at that), there is no need to run the GC until the entire
        # model is created.  Simple reference-counting should be
        # sufficient to keep memory use under control.
        #
        with PauseGC() as pgc:

            #
            # Unlike the standard method in the pympler summary
            # module, the tracker doesn't print 0-byte entries to pad
            # out the limit.
            #
            profile_memory = kwds.get('profile_memory', 0)

            if (pympler_available is True) and (profile_memory >= 2):
                mem_used = muppy.get_size(muppy.get_objects())
                print("")
                print("      Total memory = %d bytes prior to model "
                      "construction" % mem_used)

            if (pympler_available is True) and (profile_memory >= 3):
                gc.collect()
                mem_used = muppy.get_size(muppy.get_objects())
                print("      Total memory = %d bytes prior to model "
                      "construction (after garbage collection)" % mem_used)

            #
            # Do some error checking
            #
            for namespace in namespaces:
                if not namespace is None and not namespace in modeldata._data:
                    msg = "Cannot access undefined namespace: '%s'"
                    raise IOError(msg % namespace)

            #
            # Initialize each component in order.
            #

            for component_name, component in iteritems(self.component_map()):

                if component.type() is Model:
                    continue

                self._initialize_component(modeldata, namespaces, component_name, profile_memory)
                if False:
                    total_time = time.time() - start_time
                    if isinstance(component, IndexedComponent):
                        clen = len(component)
                    else:
                        assert isinstance(component, Component)
                        clen = 1
                    print("    %%6.%df seconds required to construct component=%s; %d indicies total" \
                              % (total_time>=0.005 and 2 or 0, component_name, clen) \
                              % total_time)
                    tmp_clone_counter = expr_common.clone_counter
                    if clone_counter != tmp_clone_counter:
                        clone_counter = tmp_clone_counter
                        print("             Cloning detected! (clone count: %d)" % clone_counters)

            # Note: As is, connectors are expanded when using command-line pyomo but not calling model.create(...) in a Python script.
            # John says this has to do with extension points which are called from commandline but not when writing scripts.
            # Uncommenting the next two lines switches this (command-line fails because it tries to expand connectors twice)
            #connector_expander = ConnectorExpander()
            #connector_expander.apply(instance=self)

            if (pympler_available is True) and (profile_memory >= 2):
                print("")
                print("      Summary of objects following instance construction")
                post_construction_summary = summary.summarize(muppy.get_objects())
                summary.print_(post_construction_summary, limit=100)
                print("")

    def _initialize_component(self, modeldata, namespaces, component_name, profile_memory):
        declaration = self.component(component_name)

        if component_name in modeldata._default:
            if declaration.type() is not Set:
                declaration.set_default(modeldata._default[component_name])
        data = None

        for namespace in namespaces:
            if component_name in modeldata._data.get(namespace,{}):
                if declaration.type() is Set:
                    data = self._tuplize(modeldata._data[namespace][component_name],
                                         declaration)
                else:
                    data = modeldata._data[namespace][component_name]
            if not data is None:
                break

        if __debug__ and logger.isEnabledFor(logging.DEBUG):
            _blockName = "Model" if self.parent_block() is None \
                else "Block '%s'" % self.name
            logger.debug( "Constructing %s '%s' on %s from data=%s",
                          declaration.__class__.__name__,
                          declaration.name, _blockName, str(data) )
        try:
            declaration.construct(data)
        except:
            err = sys.exc_info()[1]
            logger.error(
                "Constructing component '%s' from data=%s failed:\n    %s: %s",
                str(declaration.name), str(data).strip(),
                type(err).__name__, err )
            raise

        if __debug__ and logger.isEnabledFor(logging.DEBUG):
                _out = StringIO()
                declaration.pprint(ostream=_out)
                logger.debug("Constructed component '%s':\n    %s"
                             % ( declaration.name, _out.getvalue()))

        if (pympler_available is True) and (profile_memory >= 2):
            mem_used = muppy.get_size(muppy.get_objects())
            print("      Total memory = %d bytes following construction of component=%s" % (mem_used, component_name))

        if (pympler_available is True) and (profile_memory >= 3):
            gc.collect()
            mem_used = muppy.get_size(muppy.get_objects())
            print("      Total memory = %d bytes following construction of component=%s (after garbage collection)" % (mem_used, component_name))


    def create(self, filename=None, **kwargs):
        """
        Create a concrete instance of this Model, possibly using data
        read in from a file.
        """
        logger.warning(
"""DEPRECATION WARNING: the Model.create() method is deprecated.  Call
Model.create_instance() to create a concrete instance from an abstract
model.  You do not need to call Model.create() for a concrete model.""")
        return self.create_instance(filename=filename, **kwargs)

    def transform(self, name=None, **kwds):
        if name is None:
            logger.warning(
"""DEPRECATION WARNING: Model.transform() is deprecated.  Use
the TransformationFactory iterator to get the list of known
transformations.""")
            return list(TransformationFactory)

        logger.warning(
"""DEPRECATION WARNING: Model.transform() is deprecated.  Use
TransformationFactory('%s') to construct a transformation object, or
TransformationFactory('%s').apply_to(model) to directly apply the
transformation to the model instance.""" % (name,name,) )

        xfrm = TransformationFactory(name)
        if xfrm is None:
            raise ValueError("Unknown model transformation '%s'" % name)
        return xfrm.apply_to(self, **kwds)


@ModelComponentFactory.register('A concrete optimization model that does not defer construction of components.')
class ConcreteModel(Model):
    """
    A concrete optimization model that does not defer construction of
    components.
    """

    def __init__(self, *args, **kwds):
        kwds['concrete'] = True
        Model.__init__(self, *args, **kwds)


@ModelComponentFactory.register('An abstract optimization model that defers construction of components.')
class AbstractModel(Model):
    """
    An abstract optimization model that defers construction of
    components.
    """

    def __init__(self, *args, **kwds):
        Model.__init__(self, *args, **kwds)


#
# Create a Model and record all the default attributes, methods, etc.
# These will be assumes to be the set of illegal component names.
#
# Note that creating a Model will result in a warning, so we will
# (arbitrarily) choose a ConcreteModel as the definitive list of
# reserved names.
#
Model._Block_reserved_words = set(dir(ConcreteModel()))

