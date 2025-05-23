#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import logging
import sys
from weakref import ref as weakref_ref
import gc
import math

from pyomo.common import timing
from pyomo.common.collections import Bunch
from pyomo.common.dependencies import pympler, pympler_available
from pyomo.common.deprecation import deprecated
from pyomo.common.gc_manager import PauseGC
from pyomo.common.log import is_debug_set
from pyomo.common.numeric_types import value
from pyomo.core.staleflag import StaleFlagManager
from pyomo.core.expr.symbol_map import SymbolMap
from pyomo.core.base.component import ModelComponentFactory
from pyomo.core.base.var import Var
from pyomo.core.base.constraint import Constraint
from pyomo.core.base.objective import Objective
from pyomo.core.base.suffix import active_import_suffix_generator
from pyomo.core.base.block import ScalarBlock
from pyomo.core.base.set import Set
from pyomo.core.base.componentuid import ComponentUID
from pyomo.core.base.label import CNameLabeler, CuidLabeler
from pyomo.dataportal.DataPortal import DataPortal

from pyomo.opt.results import Solution, SolverStatus, UndefinedData

from contextlib import nullcontext
from io import StringIO

logger = logging.getLogger('pyomo.core')
id_func = id


def global_option(function, name, value):
    """
    Declare the default value for a global Pyomo configuration option.

    Example use:

    .. code::

       @global_option('config.foo.bar', 1)
       def functor():
           # ...

    """
    PyomoConfig._option[tuple(name.split('.'))] = value

    def wrapper_function(*args, **kwargs):
        return function(*args, **kwargs)

    return wrapper_function


class PyomoConfig(Bunch):
    """
    This is a pyomo-specific configuration object, which is a subclass of Container.
    """

    _option = {}

    def __init__(self, *args, **kw):
        Bunch.__init__(self, *args, **kw)
        self.set_name('PyomoConfig')
        #
        # Create the nested options specified by the the PyomoConfig._option
        # dictionary, which has been populated with the global_option decorator.
        #
        for item in PyomoConfig._option:
            d = self
            for attr in item[:-1]:
                if not attr in d:
                    d[attr] = Bunch()
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
        # entry[name]: id -> (object, entry)
        #
        for name in ['objective', 'variable', 'constraint', 'problem']:
            self._entry[name] = {}

    def __getattr__(self, name):
        if name[0] == '_':
            if name in self.__dict__:
                return self.__dict__[name]
            else:
                raise AttributeError(
                    "'%s' object has no attribute '%s'"
                    % (self.__class__.__name__, name)
                )
        return self.__dict__['_metadata'][name]

    def __setattr__(self, name, val):
        if name[0] == '_':
            self.__dict__[name] = val
            return
        self.__dict__['_metadata'][name] = val

    def __getstate__(self):
        state = {'_metadata': self._metadata, '_entry': {}}
        for name, data in self._entry.items():
            tmp = state['_entry'][name] = []
            # Note: We must convert all weakrefs to hard refs and
            # not indirect references like ComponentUIDs because
            # when it comes time to unpickle, we cannot count on the
            # model instance to have already been reconstructed --
            # so things like CUID.find_component will fail (return
            # None).
            for obj, entry in data.values():
                if obj is None or obj is None:
                    logger.warning(
                        "Solution component in '%s' no longer "
                        "accessible: %s!" % (name, entry)
                    )
                else:
                    tmp.append((obj, entry))
        return state

    def __setstate__(self, state):
        self._metadata = state['_metadata']
        self._entry = {}
        for name, data in state['_entry'].items():
            tmp = self._entry[name] = {}
            for obj, entry in data:
                tmp[id(obj)] = (obj, entry)


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
        for key, val in state.items():
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

    def load_from(
        self,
        results,
        allow_consistent_values_for_fixed_vars=False,
        comparison_tolerance_for_fixed_vars=1e-5,
        ignore_invalid_labels=False,
        id=None,
        delete_symbol_map=True,
        clear=True,
        default_variable_value=None,
        select=0,
        ignore_fixed_vars=True,
    ):
        """
        Load solver results
        """
        instance = self._instance()
        #
        # If there is a warning, then print a warning message.
        #
        if results.solver.status == SolverStatus.warning:
            tc = getattr(results.solver, 'termination_condition', None)
            msg = getattr(results.solver, 'message', None)
            logger.warning(
                'Loading a SolverResults object with a '
                'warning status into model.name="%s";\n'
                '  - termination condition: %s\n'
                '  - message from solver: %s' % (instance.name, tc, msg)
            )
        #
        # If the solver status not one of either OK or Warning, then
        # generate an error.
        #
        elif results.solver.status != SolverStatus.ok:
            if (results.solver.status == SolverStatus.aborted) and (
                len(results.solution) > 0
            ):
                logger.warning(
                    "Loading a SolverResults object with "
                    "an 'aborted' status, but containing a solution"
                )
            else:
                raise ValueError(
                    "Cannot load a SolverResults object "
                    "with bad status: %s" % str(results.solver.status)
                )
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
            self.add_solution(
                results.solution(id),
                smap_id,
                delete_symbol_map=False,
                cache=cache,
                ignore_invalid_labels=ignore_invalid_labels,
                default_variable_value=default_variable_value,
            )
        else:
            for i in range(len(results.solution)):
                self.add_solution(
                    results.solution(i),
                    smap_id,
                    delete_symbol_map=False,
                    cache=cache,
                    ignore_invalid_labels=ignore_invalid_labels,
                    default_variable_value=default_variable_value,
                )

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
                ignore_fixed_vars=ignore_fixed_vars,
            )

    def store_to(self, results, cuid=False, skip_stale_vars=False):
        """
        Return a Solution() object that is populated with the values in the model.
        """
        instance = self._instance()
        results.solution.clear()
        results._smap_id = None

        for soln_ in self.solutions:
            soln = Solution()
            soln._cuid = cuid
            for key, val in soln_._metadata.items():
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
                soln.objective[sm.getSymbol(obj, labeler)] = vals
            entry = soln_._entry['variable']
            for obj in instance.component_data_objects(Var, active=True):
                if obj.stale and skip_stale_vars:
                    continue
                vals = entry.get(id(obj), None)
                if vals is None:
                    vals = {}
                else:
                    vals = vals[1]
                vals['Value'] = value(obj)
                soln.variable[sm.getSymbol(obj, labeler)] = vals
            entry = soln_._entry['constraint']
            for obj in instance.component_data_objects(Constraint, active=True):
                vals = entry.get(id(obj), None)
                if vals is None:
                    continue
                else:
                    vals = vals[1]
                soln.constraint[sm.getSymbol(obj, labeler)] = vals
            results.solution.insert(soln)

    def add_solution(
        self,
        solution,
        smap_id,
        delete_symbol_map=True,
        cache=None,
        ignore_invalid_labels=False,
        ignore_missing_symbols=True,
        default_variable_value=None,
    ):
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
                    for cuid, val in getattr(solution, name).items():
                        obj = cache.get(cuid, None)
                        if obj is None:
                            if ignore_invalid_labels:
                                continue
                            raise RuntimeError(
                                "CUID %s is missing from model %s"
                                % (str(cuid), instance.name)
                            )
                        tmp[id(obj)] = (obj, val)
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
                    for symb, val in getattr(solution, name).items():
                        obj = cache.get(symb, None)
                        if obj is None:
                            if ignore_invalid_labels:
                                continue
                            raise RuntimeError(
                                "Symbol %s is missing from model %s"
                                % (symb, instance.name)
                            )
                        tmp[id(obj)] = (obj, val)
        else:
            #
            # Map solution
            #
            smap = self.symbol_map[smap_id]
            for name in ['problem', 'objective', 'variable', 'constraint']:
                tmp = soln._entry[name]
                for symb, val in getattr(solution, name).items():
                    if symb in smap.bySymbol:
                        obj = smap.bySymbol[symb]
                    elif symb in smap.aliases:
                        obj = smap.aliases[symb]
                    elif ignore_missing_symbols:
                        continue
                    else:  # pragma:nocover
                        #
                        # This should never happen ...
                        #
                        raise RuntimeError(
                            "ERROR: Symbol %s is missing from "
                            "model %s when loading with a symbol map!"
                            % (symb, instance.name)
                        )

                    tmp[id(obj)] = (obj, val)
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
                tmp[id_] = (vdata, {'Value': vdata.value})
            elif (
                (default_variable_value is not None)
                and (smap_id is not None)
                and (id_ in smap.byObject)
                and (id_ not in tmp)
            ):
                tmp[id_] = (vdata, {'Value': default_variable_value})

        self.solutions.append(soln)
        return len(self.solutions) - 1

    def select(
        self,
        index=0,
        allow_consistent_values_for_fixed_vars=False,
        comparison_tolerance_for_fixed_vars=1e-5,
        ignore_invalid_labels=False,
        ignore_fixed_vars=True,
    ):
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
        # Set the "stale" flag of each variable in the model prior to
        # loading the solution, so you known which variables have "real"
        # values and which ones don't.
        #
        StaleFlagManager.mark_all_as_stale()

        if index is not None:
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
        for suffix in valid_import_suffixes.values():
            suffix.clear_all_values()
        #
        # Load problem (model) level suffixes. These would only come from ampl
        # interfaced solution suffixes at this point in time.
        #
        for id_, (pobj, entry) in soln._entry['problem'].items():
            for _attr_key, attr_value in entry.items():
                attr_key = _attr_key[0].lower() + _attr_key[1:]
                if attr_key in valid_import_suffixes:
                    valid_import_suffixes[attr_key][pobj] = attr_value
        #
        # Load objective data (suffixes)
        #
        for id_, (odata, entry) in soln._entry['objective'].items():
            for _attr_key, attr_value in entry.items():
                attr_key = _attr_key[0].lower() + _attr_key[1:]
                if attr_key in valid_import_suffixes:
                    valid_import_suffixes[attr_key][odata] = attr_value
        #
        # Load variable data (suffixes and values)
        #
        for id_, (vdata, entry) in soln._entry['variable'].items():
            val = entry['Value']
            if vdata.fixed is True:
                if ignore_fixed_vars:
                    continue
                if not allow_consistent_values_for_fixed_vars:
                    msg = (
                        "Variable '%s' in model '%s' is currently fixed - new"
                        ' value is not expected in solution'
                    )
                    raise TypeError(msg % (vdata.name, instance.name))
                if math.fabs(val - vdata.value) > comparison_tolerance_for_fixed_vars:
                    raise TypeError(
                        "Variable '%s' in model '%s' is currently "
                        "fixed - a value of '%s' in solution is "
                        "not within tolerance=%s of the current "
                        "value of '%s'"
                        % (
                            vdata.name,
                            instance.name,
                            str(val),
                            str(comparison_tolerance_for_fixed_vars),
                            str(vdata.value),
                        )
                    )

            vdata.set_value(val, skip_validation=True)

            for _attr_key, attr_value in entry.items():
                attr_key = _attr_key[0].lower() + _attr_key[1:]
                if attr_key == 'value':
                    continue
                elif attr_key in valid_import_suffixes:
                    valid_import_suffixes[attr_key][vdata] = attr_value
        #
        # Load constraint data (suffixes)
        #
        for id_, (cdata, entry) in soln._entry['constraint'].items():
            for _attr_key, attr_value in entry.items():
                attr_key = _attr_key[0].lower() + _attr_key[1:]
                if attr_key in valid_import_suffixes:
                    valid_import_suffixes[attr_key][cdata] = attr_value

        # Set the state flag to "delayed advance": it will auto-advance
        # if a non-stale variable is updated (causing all non-stale
        # variables to be marked as stale).
        StaleFlagManager.mark_all_as_stale(delayed=True)


@ModelComponentFactory.register(
    'Model objects can be used as a component of other models.'
)
class Model(ScalarBlock):
    """
    An optimization model.  By default, this defers construction of components
    until data is loaded.
    """

    _Block_reserved_words = set()

    def __new__(cls, *args, **kwds):
        if cls != Model:
            return super(Model, cls).__new__(cls)

        raise TypeError(
            "Directly creating the 'Model' class is not allowed.  Please use the "
            "AbstractModel or ConcreteModel class instead."
        )

    def __init__(self, name='unknown', **kwargs):
        """Constructor"""
        #
        # NOTE: The 'ctype' keyword argument is not defined here.  Thus,
        # a model is treated as a 'Block' class type.  This simplifies
        # the definition of the block_data_objects() method, since we treat
        # Model and Block objects as the same.  Similarly, this avoids
        # the requirement to import PyomoModel.py in the block.py file.
        #
        ScalarBlock.__init__(self, **kwargs)
        self._name = name
        self.statistics = Bunch()
        self.config = PyomoConfig()
        self.solutions = ModelSolutions(self)

    def compute_statistics(self, active=True):
        """
        Compute model statistics
        """
        self.statistics.number_of_variables = 0
        self.statistics.number_of_constraints = 0
        self.statistics.number_of_objectives = 0
        for block in self.block_data_objects(active=active):
            for data in block.component_map(Var, active=active).values():
                self.statistics.number_of_variables += len(data)
            for data in block.component_map(Objective, active=active).values():
                self.statistics.number_of_objectives += len(data)
            for data in block.component_map(Constraint, active=active).values():
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

    def create_instance(
        self,
        filename=None,
        data=None,
        name=None,
        namespace=None,
        namespaces=None,
        profile_memory=0,
        report_timing=False,
        **kwds,
    ):
        """
        Create a concrete instance of an abstract model, possibly using data
        read in from a file.

        Parameters
        ----------
        filename: `str`, optional
            The name of a Pyomo Data File that will be used to load data into
            the model.
        data: `dict`, optional
            A dictionary containing initialization data for the model to be
            used if there is no filename
        name: `str`, optional
            The name given to the model.
        namespace: `str`, optional
            A namespace used to select data.
        namespaces: `list`, optional
            A list of namespaces used to select data.
        profile_memory: `int`, optional
            A number that indicates the profiling level.
        report_timing: `bool`, optional
            Report timing statistics during construction.

        """
        #
        # Generate a warning if this is a concrete model but the
        # filename is specified.  A concrete model is already
        # constructed, so passing in a data file is a waste of time.
        #
        if self.is_constructed() and isinstance(filename, str):
            msg = (
                "The filename=%s will not be loaded - supplied as an "
                "argument to the create_instance() method of a "
                "concrete instance with name=%s." % (filename, name)
            )
            logger.warning(msg)

        if kwds:
            msg = """Model.create_instance() passed the following unrecognized keyword
arguments (which have been ignored):"""
            for k in kwds:
                msg = msg + "\n    '%s'" % (k,)
            logger.error(msg)

        if self.is_constructed():
            return self.clone()

        if name is None:
            # Preserve only the local name (not the FQ name, as that may
            # have been quoted or otherwise escaped)
            name = self.local_name
        if filename is not None:
            if data is not None:
                logger.warning(
                    "Model.create_instance() passed both 'filename' "
                    "and 'data' keyword arguments.  Ignoring the "
                    "'data' argument"
                )
            data = filename
        if data is None:
            data = {}

        reporting_context = timing.report_timing if report_timing else nullcontext
        with reporting_context():
            #
            # Clone the model and load the data
            #
            instance = self.clone()

            if name is not None:
                instance._name = name

            # If someone passed a rule for creating the instance, fire the
            # rule before constructing the components.
            if instance._rule is not None:
                instance._rule(instance, next(iter(self.index_set())))

            if namespaces:
                _namespaces = list(namespaces)
            else:
                _namespaces = []
            if namespace is not None:
                _namespaces.append(namespace)
            if None not in _namespaces:
                _namespaces.append(None)

            instance.load(data, namespaces=_namespaces, profile_memory=profile_memory)

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

    @deprecated(
        "The Model.preprocess() method is deprecated and no "
        "longer performs any actions",
        version='6.0',
    )
    def preprocess(self, preprocessor=None):
        return

    def load(self, arg, namespaces=[None], profile_memory=0):
        """
        Load the model with data from a file, dictionary or DataPortal object.
        """
        if arg is None or isinstance(arg, str):
            dp = DataPortal(filename=arg, model=self)
        elif type(arg) is DataPortal:
            dp = arg
        elif type(arg) is dict:
            dp = DataPortal(data_dict=arg, model=self)
        else:
            msg = "Cannot load model model data from with object of type '%s'"
            raise ValueError(msg % str(type(arg)))
        self._load_model_data(dp, namespaces, profile_memory=profile_memory)

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

            if profile_memory >= 2 and pympler_available:
                mem_used = pympler.muppy.get_size(pympler.muppy.get_objects())
                print("")
                print(
                    "      Total memory = %d bytes prior to model "
                    "construction" % mem_used
                )

                if profile_memory >= 3:
                    gc.collect()
                    mem_used = pympler.muppy.get_size(pympler.muppy.get_objects())
                    print(
                        "      Total memory = %d bytes prior to model "
                        "construction (after garbage collection)" % mem_used
                    )

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

            for component_name, component in self.component_map().items():
                if component.ctype is Model:
                    continue

                self._initialize_component(
                    modeldata, namespaces, component_name, profile_memory
                )

            # Note: As is, connectors are expanded when using command-line pyomo but not calling model.create(...) in a Python script.
            # John says this has to do with extension points which are called from commandline but not when writing scripts.
            # Uncommenting the next two lines switches this (command-line fails because it tries to expand connectors twice)
            # connector_expander = ConnectorExpander()
            # connector_expander.apply(instance=self)

            if profile_memory >= 2 and pympler_available:
                print("")
                print("      Summary of objects following instance construction")
                post_construction_summary = pympler.summary.summarize(
                    pympler.muppy.get_objects()
                )
                pympler.summary.print_(post_construction_summary, limit=100)
                print("")

    def _initialize_component(
        self, modeldata, namespaces, component_name, profile_memory
    ):
        declaration = self.component(component_name)

        if component_name in modeldata._default:
            if declaration.ctype is not Set:
                declaration.set_default(modeldata._default[component_name])
        data = None

        for namespace in namespaces:
            if component_name in modeldata._data.get(namespace, {}):
                data = modeldata._data[namespace][component_name]
            if data is not None:
                break

        generate_debug_messages = is_debug_set(logger)
        if generate_debug_messages:
            _blockName = (
                "Model" if self.parent_block() is None else "Block '%s'" % self.name
            )
            logger.debug(
                "Constructing %s '%s' on %s from data=%s",
                declaration.__class__.__name__,
                declaration.name,
                _blockName,
                str(data),
            )
        try:
            declaration.construct(data)
        except:
            err = sys.exc_info()[1]
            logger.error(
                "Constructing component '%s' from data=%s failed:\n    %s: %s",
                str(declaration.name),
                str(data).strip(),
                type(err).__name__,
                err,
                extra={'cleandoc': False},
            )
            raise

        if generate_debug_messages:
            _out = StringIO()
            declaration.pprint(ostream=_out)
            logger.debug(
                "Constructed component '%s':\n    %s"
                % (declaration.name, _out.getvalue())
            )

        if profile_memory >= 2 and pympler_available:
            mem_used = pympler.muppy.get_size(pympler.muppy.get_objects())
            print(
                "      Total memory = %d bytes following construction of component=%s"
                % (mem_used, component_name)
            )

            if profile_memory >= 3:
                gc.collect()
                mem_used = pympler.muppy.get_size(pympler.muppy.get_objects())
                print(
                    "      Total memory = %d bytes following construction of component=%s (after garbage collection)"
                    % (mem_used, component_name)
                )


@ModelComponentFactory.register(
    'A concrete optimization model that does not defer construction of components.'
)
class ConcreteModel(Model):
    """
    A concrete optimization model that does not defer construction of
    components.
    """

    def __init__(self, *args, **kwds):
        kwds['concrete'] = True
        Model.__init__(self, *args, **kwds)


@ModelComponentFactory.register(
    'An abstract optimization model that defers construction of components.'
)
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
