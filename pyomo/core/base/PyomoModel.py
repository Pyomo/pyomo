#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008 Sandia Corporation.
#  This software is distributed under the BSD License.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  For more information, see the Pyomo README.txt file.
#  _________________________________________________________________________

__all__ = ['Model', 'ConcreteModel', 'AbstractModel', 'global_option']

import array
import copy
import logging
import re
import sys
import traceback
import weakref
import gc
import time
import math
import functools

try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict
try:
    from pympler import muppy
    from pympler import summary
    pympler_available = True
except ImportError:
    pympler_available = False

from six import itervalues, iteritems, StringIO
from six.moves import xrange
try:
    unicode
except:
    basestring = unicode = str

from pyomo.misc.plugin import ExtensionPoint
from pyutilib.math import *
from pyutilib.misc import quote_split, tuplize, Container, PauseGC, Bunch

import pyomo.misc
import pyomo.opt
from pyomo.opt import ProblemFormat, ResultsFormat, guess_format
from pyomo.opt.results import SolutionMap, SolverResults, Solution, SolutionStatus
from pyomo.opt.results.container import MapContainer,UndefinedData

from pyomo.core.base.var import _VarData, Var
from pyomo.core.base.constraint import _ConstraintData
from pyomo.core.base.objective import Objective, _ObjectiveData
from pyomo.core.base.set_types import *
from pyomo.core.base.suffix import active_import_suffix_generator
from pyomo.core.base.symbol_map import SymbolMap
from pyomo.core.base.sparse_indexed_component import SparseIndexedComponent

from pyomo.core.base.connector import ConnectorExpander

from pyomo.core.base.DataPortal import *
from pyomo.core.base.plugin import *
from pyomo.core.base.numvalue import *
from pyomo.core.base.block import SimpleBlock
from pyomo.core.base.sets import Set
from pyomo.core.base.component import register_component, Component

from pyomo.core.base.plugin import IModelTransformation, TransformationFactory
logger = logging.getLogger('pyomo.core')


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


class ModelTransformationWrapper(object):

    def __init__(self, model):
        self._model = weakref.ref(model)

    def set_model(self, model):
        self._model = weakref.ref(model)

    def __call__(self, name, **kwds):
        return self.apply(name, **kwds)

    def apply(self, name, **kwds):
        xfrm = TransformationFactory(name)
        if xfrm is None:
            raise ValueError("Bad model transformation '%s'" % name)
        return xfrm(self._model(), **kwds)

    def __dir__(self):
        return TransformationFactory.services()

    def __getattr__(self, name):
        if name.startswith('_'):
            return self.__dict__[name]
        xfrm = TransformationFactory(name)
        if xfrm is None:
            raise ValueError("Bad model transformation '%s'" % name)
        return functools.partial(xfrm, self._model())

    def __setattr__(self, name, val):
        if name == '_model':
            self.__dict__[name] = val
        else:
            raise KeyError("Can only set _model attribute: "+name)

    def __getstate__(self):
        return { '_model': self._model() }

    def __setstate__(self, state):
        self._model = weakref.ref(state['_model'])

    def __copy__(self):
        return type(self)(self._model())

    def __deepcopy__(self, memo):
        return type(self)(self._model())


class Model(SimpleBlock):
    """
    An optimization model.  By default, this defers construction of components
    until data is loaded.
    """

    preprocessor_ep = ExtensionPoint(IPyomoPresolver)


    def __init__ ( self, name='unknown', _deprecate=True, **kwargs ):
        """Constructor"""
        if _deprecate:
            msg = "Using the 'Model' class is deprecated.  Please use the "    \
                  "'AbstractModel' class instead."
            logger.warning( msg )
        #
        # NOTE: The 'ctype' keyword argument is not defined here.  Thus,
        # a model is treated as a 'Block' class type.  This simplifies
        # the definition of the all_blocks() method, since we treat
        # Model and Block objects as the same.  Similarly, this avoids
        # the requirement to import PyomoModel.py in the block.py file.
        #
        SimpleBlock.__init__(self, **kwargs)
        self.name=name
        self.statistics = Container()
        self.config = PyomoConfig()
        self.config.preprocessor = 'pyomo.model.simple_preprocessor'
        self.transform = ModelTransformationWrapper(self)

    def model(self):
        # Special case: the "Model" is always the top-level block, so if
        # this is the top-level block, it must be the model
        if self.parent_block() is None:
            return self
        else:
            return super(Model, self).model()

    def nvariables(self):
        #if self.statistics.number_of_variables is None:
        #    self.statistics.number_of_variables = len(self.variables())
        return self.statistics.number_of_variables

    #def variables(self):
        #return self.components(Var)

    def nconstraints(self):
        #if self.statistics.number_of_constraints is None:
        #    self.statistics.number_of_constraints = len(self.constraints())
        return self.statistics.number_of_constraints

    #def constraints(self):
        #return self.components(Constraint)

    def nobjectives(self):
        #if self.statistics.number_of_objectives is None:
        #    self.statistics.number_of_objectives = len(self.objectives())
        return self.statistics.number_of_objectives

    #def objectives(self):
        #return self.components(Objective)

    def valid_problem_types(self):
        """This method allows the pyomo.opt convert function to work with a Model object."""
        return [ProblemFormat.pyomo]

    def create(self, filename=None, **kwargs):
        """
        Create a concrete instance of this Model, possibly using data
        read in from a file.
        """
        kwargs['filename'] = filename
        functor = kwargs.pop('functor', None)
        if functor is None:
            data = pyomo.misc.PyomoAPIFactory(self.config.create_functor)(self.config, model=self, **kwargs)
        else:
            data = pyomo.misc.PyomoAPIFactory(functor)(self.config, model=self, **kwargs)

        # Creating a model converts it from Abstract -> Concrete
        data.instance._constructed = True
        # Update transformation handle
        data.instance.transform.set_model(data.instance)
        return data.instance

    def reset(self):
        # TODO: check that this works recursively for nested models
        for obj in itervalues(self.components()):
            obj.reset()

    def preprocess(self, preprocessor=None):
        """Apply the preprocess plugins defined by the user"""
        suspend_gc = PauseGC()
        if preprocessor is None:
            preprocessor = self.config.preprocessor
        pyomo.misc.PyomoAPIFactory(preprocessor)(self.config, model=self)

    #
    # this method is a hack, used strictly by the pyomo command-line utility to
    # allow for user-readable names to be present in solver results objects.
    # this should be removed in the very near future, when labels are isolated
    # completely to solver plugins. in that situation, only human-readable
    # names will be present in a SolverResults object, and this method can
    # be removed.
    #
    # WEH - I think we need this method, even if the labeling method changes.
    #
    # The var/con/obj data is ordered by variable name, sorted by index.
    # This may differ from the declaration order, as well as the
    # instance order.  I haven't figured out how to efficiently generate
    # the declaration order.
    #
    # JDS - I think I agree with JP that this is no longer necessary
    #
    def update_results(self, results):
        results_symbol_map = results._symbol_map
        same_instance = results_symbol_map is not None and \
            results_symbol_map.instance() is self
        new_results = SolverResults()
        new_results.problem = results.problem
        new_results.solver = results.solver

        tmp_name_dict = {}

        for i in xrange(len(results.solution)):

            input_soln = results.solution(i+1)
            input_soln_variable = input_soln.variable
            input_soln_constraint = input_soln.constraint

            new_soln = Solution()
            new_soln.gap = input_soln.gap
            new_soln.status = input_soln.status
            #
            # Variables
            #
            vars = OrderedDict()
            var_id_index_map = dict()
            tmp = {}
            for label, entry in iteritems(input_soln_variable):
                # NOTE: the following is a hack, to handle the ONE_VAR_CONSTANT variable
                if label == "ONE_VAR_CONSTANT":
                    continue
                # translate the label first if there is a symbol map
                # associated with the input solution.
                if same_instance:
                    var_value = results_symbol_map.getObject(label)
                elif results_symbol_map is not None:
                    var_value = results_symbol_map.getEquivalentObject(label, self)
                else:
                    raise RuntimeError("Cannot update from results missing a symbol map")

                if var_value is SymbolMap.UnknownSymbol:
                    msg = "Variable with label '%s' is not in model '%s'."
                    raise KeyError(msg % ( label, self.name, ))

                var_value_id = id(var_value)
                if var_value_id not in var_id_index_map:
                    component = var_value.parent_component()
                    var_id_index_map.update(component.id_index_map())
                var_value_index = var_id_index_map[var_value_id]

                if var_value_index.__class__ is tuple:
                    tmp[(var_value.parent_component().cname(True),)+var_value_index] = (var_value.cname(False, tmp_name_dict), entry)
                else:
                    tmp[(var_value.parent_component().cname(True),var_value_index)] = (var_value.cname(False, tmp_name_dict), entry)
            for key in sorted(tmp.keys()):
                value = tmp[key]
                vars[value[0]] = value[1]
            new_soln.variable = vars
            #
            # Constraints
            #
            tmp = {}
            con_id_index_map = dict()
            for label, entry in iteritems(input_soln_constraint):
                # NOTE: the following is a hack, to handle the ONE_VAR_CONSTANT variable
                if label == "c_e_ONE_VAR_CONSTANT":
                    continue
                if same_instance:
                    con_value = results_symbol_map.getObject(label)
                elif results_symbol_map is not None:
                    con_value = results_symbol_map.getEquivalentObject(label, self)
                else:
                    raise RuntimeError("Cannot update from results missing a symbol map")

                if con_value is SymbolMap.UnknownSymbol:
                    msg = "Constraint with label '%s' is not in model '%s'."
                    raise KeyError(msg % ( label, self.name, ))

                con_value_id = id(con_value)
                if con_value_id not in con_id_index_map:
                    component = con_value.parent_component()
                    con_id_index_map.update(component.id_index_map())
                con_value_index = con_id_index_map[con_value_id]

                if con_value_index.__class__ is tuple:
                    tmp[(con_value.parent_component().cname(True),)+con_value_index] = (con_value.cname(False, tmp_name_dict), entry)
                else:
                    tmp[(con_value.parent_component().cname(True),con_value_index)] = (con_value.cname(False, tmp_name_dict), entry)
            for key in sorted(tmp.keys()):
                value = tmp[key]
                new_soln.constraint[value[0]] = value[1]
            #
            # Objectives
            #
            tmp = {}
            for label in input_soln.objective.keys():
                if same_instance:
                    obj_value = results_symbol_map.getObject(label)
                elif results_symbol_map is not None:
                    obj_value = results_symbol_map.getEquivalentObject(label, self)
                else:
                    raise RuntimeError("Cannot update from results missing a symbol map")

                if obj_value is SymbolMap.UnknownSymbol:
                    msg = "Objective with label '%s' is not in model '%s'."
                    raise KeyError(msg % ( label, self.name, ))

                entry = input_soln.objective[label]
                if obj_value.index().__class__ is tuple:
                    tmp[(obj_value.parent_component().cname(True),)+obj_value.index()] = (obj_value.cname(False, tmp_name_dict), entry)
                else:
                    tmp[(obj_value.parent_component().cname(True),obj_value.index())] = (obj_value.cname(False, tmp_name_dict), entry)
            for key in sorted(tmp.keys()):
                value = tmp[key]
                new_soln.objective.declare(value[0])
                dict.__setitem__(new_soln.objective, value[0], value[1])
            #
            new_results.solution.insert(new_soln)
        return new_results

    def get_solution(self):
        """
        Return a Solution() object that is populated with the values in the model.

        NOTE: this is a hack.  We need to do a better job with this!
        """
        soln = Solution()
        soln.status = SolutionStatus.optimal

        for block in self.all_blocks():
            for name_, index_, cdata_ in block.active_component_data(Objective):
                soln.objective[ cdata_.parent_component().cname(True) ].value = value(cdata_)
            for name_, index_, cdata_ in block.active_component_data(Var):
                soln.variable[ cdata_.parent_component().cname(True) ] = {'Value': cdata_.value}

        return soln

    def store(self, dp, components, namespace=None):
        for c in components:
            try:
                name = c.cname()
            except:
                name = c
            dp._data.get(namespace,{})[name] = c.data()

    def load(self, arg, namespaces=[None], symbol_map=None,
             allow_consistent_values_for_fixed_vars=False,
             comparison_tolerance_for_fixed_vars=1e-5,
             profile_memory=0, report_timing=False, 
             ignore_invalid_labels=False, id=0):
        """
        Load the model with data from a file or a Solution object
        """

        if arg is None or type(arg) is str:
            self._load_model_data(DataPortal(filename=arg,model=self), namespaces, profile_memory=profile_memory, report_timing=report_timing)
            return True
        elif type(arg) is DataPortal:
            self._load_model_data(arg, namespaces, profile_memory=profile_memory, report_timing=report_timing)
            return True
        elif type(arg) is dict:
            self._load_model_data(DataPortal(data_dict=arg,model=self), namespaces, profile_memory=profile_memory, report_timing=report_timing)
            return True
        elif type(arg) is pyomo.opt.SolverResults:
            # set the "stale" flag of each variable in the model prior to loading the
            # solution, so you known which variables have "real" values and which ones don't.
            self.flag_vars_as_stale()
            
            # if the solver status not one of either OK or Warning, then error.
            if (arg.solver.status != pyomo.opt.SolverStatus.ok) and \
               (arg.solver.status != pyomo.opt.SolverStatus.warning):

                if (arg.solver.status == pyomo.opt.SolverStatus.aborted) and (len(arg.solution) > 0):
                   print("WARNING - Loading a SolverResults object with an 'aborted' status, but containing a solution")
                else:
                   msg = 'Cannot load a SolverResults object with bad status: %s'
                   raise ValueError(msg % str( arg.solver.status ))

            # but if there is a warning, print out a warning, as someone should
            # probably take a look!
            if (arg.solver.status == pyomo.opt.SolverStatus.warning):
                print('WARNING - Loading a SolverResults object with a '       \
                      'warning status')

            if len(arg.solution) > 0:
                self._load_solution(
                    arg.solution(id),
                    symbol_map=arg.__dict__.get('_symbol_map', None),
                    allow_consistent_values_for_fixed_vars=allow_consistent_values_for_fixed_vars,
                    comparison_tolerance_for_fixed_vars=comparison_tolerance_for_fixed_vars,
                    ignore_invalid_labels=ignore_invalid_labels )
                return True
            else:
                return False
        elif type(arg) is pyomo.opt.Solution:
            # set the "stale" flag of each variable in the model prior to loading the
            # solution, so you known which variables have "real" values and which ones don't.
            self.flag_vars_as_stale()
            
            self._load_solution(
                arg,
                symbol_map=symbol_map,
                allow_consistent_values_for_fixed_vars=allow_consistent_values_for_fixed_vars,
                comparison_tolerance_for_fixed_vars=comparison_tolerance_for_fixed_vars,
                ignore_invalid_labels=ignore_invalid_labels )
            return True
        else:
            msg = "Cannot load model with object of type '%s'"
            raise ValueError(msg % str( type(arg) ))

    def Xstore_info(self, results):
        """
        Store model information into a SolverResults object
        """
        results.problem.name = self.name

        stat_keys = (
          'number_of_variables',
          'number_of_binary_variables',
          'number_of_integer_variables',
          'number_of_continuous_variables',
          'number_of_constraints',
          'number_of_objectives'
        )

        for key in stat_keys:
            results.problem.__dict__[key] = self.statistics.__dict__[key]

    def _tuplize(self, data, setobj):
        if data is None:            #pragma:nocover
            return None
        if setobj.dimen == 1:
            return data
        ans = {}
        for key in data:
            if type(data[key][0]) is tuple:
                return data
            ans[key] = tuplize(data[key], setobj.dimen, setobj.name)
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
        suspend_gc = PauseGC()

        #
        # Unlike the standard method in the pympler summary module, the tracker
        # doesn't print 0-byte entries to pad out the limit.
        #
        profile_memory = kwds.get('profile_memory', 0)

        #
        # It is often useful to report timing results for various activities during model construction.
        #
        report_timing = kwds.get('report_timing', False)

        if (pympler_available is True) and (profile_memory >= 2):
            mem_used = muppy.get_size(muppy.get_objects())
            print("")
            print("      Total memory = %d bytes prior to model construction" % mem_used)

        if (pympler_available is True) and (profile_memory >= 3):
            gc.collect()
            mem_used = muppy.get_size(muppy.get_objects())
            print("      Total memory = %d bytes prior to model construction (after garbage collection)" % mem_used)

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

        if report_timing is True:
            construction_start_time = time.time()

        for component_name, component in iteritems(self.components()):

            if component.type() is Model:
                continue

            if report_timing is True:
                start_time = time.time()
                clone_counters = (
                    pyomo.core.base.expr.generate_expression.clone_counter,
                    pyomo.core.base.expr.generate_relational_expression.clone_counter,
                    pyomo.core.base.expr.generate_intrinsic_function_expression.clone_counter,
                    )

            self._initialize_component(modeldata, namespaces, component_name, profile_memory)

            if report_timing is True:
                total_time = time.time() - start_time
                comp = self.find_component(component_name)
                if isinstance(comp, SparseIndexedComponent):
                    clen = len(comp)
                else:
                    assert isinstance(comp, Component)
                    clen = 1
                print("    %%6.%df seconds required to construct component=%s; %d indicies total" \
                          % (total_time>=0.005 and 2 or 0, component_name, clen) \
                          % total_time)
                tmp_clone_counters = (
                    pyomo.core.base.expr.generate_expression.clone_counter,
                    pyomo.core.base.expr.generate_relational_expression.clone_counter,
                    pyomo.core.base.expr.generate_intrinsic_function_expression.clone_counter,
                    )
                if clone_counters != tmp_clone_counters:
                    clone_counters = tmp_clone_counters
                    print("             Cloning detected! (clone counters: %d, %d, %d)" % clone_counters)

        # Note: As is, connectors are expanded when using command-line pyomo but not calling model.create(...) in a Python script.
        # John says this has to do with extension points which are called from commandline but not when writing scripts.
        # Uncommenting the next two lines switches this (command-line fails because it tries to expand connectors twice)
        #connector_expander = ConnectorExpander()
        #connector_expander.apply(instance=self)

        if report_timing is True:
            total_construction_time = time.time() - construction_start_time
            print("      %6.2f seconds required to construct instance=%s" % (total_construction_time, self.name))

        if (pympler_available is True) and (profile_memory >= 2):
            print("")
            print("      Summary of objects following instance construction")
            post_construction_summary = summary.summarize(muppy.get_objects())
            summary.print_(post_construction_summary, limit=100)

            print("")


    def _initialize_component(self, modeldata, namespaces, component_name, profile_memory):
        declaration = self.component(component_name)

        if component_name in modeldata._default.keys():
            if declaration.type() is not Set:
                declaration.set_default(modeldata._default[component_name])
        data = None
        
        for namespace in namespaces:
            if component_name in modeldata._data.get(namespace,{}).keys():
                if declaration.type() is Set:
                    data = self._tuplize(modeldata._data[namespace][component_name],
                                         declaration)
                else:
                    data = modeldata._data[namespace][component_name]
            if not data is None:
                break

        if __debug__ and logger.isEnabledFor(logging.DEBUG):
            _blockName = "Model" if self.parent_block() is None \
                else "Block '%s'" % self.cname(True)
            logger.debug( "Constructing %s '%s' on %s from data=%s",
                          declaration.__class__.__name__, 
                          declaration.cname(), _blockName, str(data) )
        try:
            declaration.construct(data)
        except:
            err = sys.exc_info()[1]
            logger.error(
                "Constructing component '%s' from data=%s failed:\n%s: %s",
                str(declaration.cname(True)), str(data).strip(),
                type(err).__name__, err )
            raise

        if __debug__ and logger.isEnabledFor(logging.DEBUG):
                _out = StringIO()
                declaration.pprint(ostream=_out)
                logger.debug("Constructed component '%s':\n%s" 
                             % ( declaration.cname(True), _out.getvalue()))
                
        if (pympler_available is True) and (profile_memory >= 2):
            mem_used = muppy.get_size(muppy.get_objects())
            print("      Total memory = %d bytes following construction of component=%s" % (mem_used, component_name))

        if (pympler_available is True) and (profile_memory >= 3):
            gc.collect()
            mem_used = muppy.get_size(muppy.get_objects())
            print("      Total memory = %d bytes following construction of component=%s (after garbage collection)" % (mem_used, component_name))


    def _load_solution( self, soln, symbol_map,
                        allow_consistent_values_for_fixed_vars=False,
                        comparison_tolerance_for_fixed_vars=1e-5,
                        ignore_invalid_labels=False ):
        """
        Load a solution. A solution can be either a tuple or list, or a pyomo.opt.Solution instance.
        - The allow_consistent_values_for_fixed_vars flag indicates whether a solution can specify
          consistent values for variables in the model that are fixed.
        - The ignore_invalid_labels flag indicates whether labels in the solution that don't
          appear in the model yield an error. This allows for loading a results object 
          generated from one model into another related, but not identical, model.
        """
        if symbol_map is None:
            same_instance = False
        else:
            same_instance = symbol_map.instance() is self

        # Generate the list of active import suffixes on this top level model
        valid_import_suffixes = dict(active_import_suffix_generator(self))
        # To ensure that import suffix data gets properly overwritten (e.g.,
        # the case where nonzero dual values exist on the suffix and but only
        # sparse dual values exist in the results object) we clear all active
        # import suffixes. 
        for suffix in itervalues(valid_import_suffixes):
            suffix.clearAllValues()

        # Load problem (model) level suffixes. These would only come from ampl
        # interfaced solution suffixes at this point in time.
        for _attr_key, attr_value in iteritems(soln.problem):
            attr_key = _attr_key[0].lower() + _attr_key[1:]
            if attr_key in valid_import_suffixes:
                # GAH: Unlike the var suffix information in the solution object,
                #      problem suffix values are ScalarData objects. I
                #      think it could be advantageous to make all suffix information
                #      ScalarData types. But for now I will take the simple route
                #      and maintain consistency with var suffixes, hence 
                #      attr_value.value rather than just attr_value
                valid_import_suffixes[attr_key].setValue(self,attr_value.value,expand=False)

        #
        # Load objective data (should simply be suffixes if they exist)
        #
        objective_skip_attrs = ['id','canonical_label','value']
        for label,entry in iteritems(soln.objective):

            if same_instance:
                obj_value = symbol_map.getObject(label)
            elif symbol_map is None:
                # We are going to assume the Solution was labeled with
                # the SolverResults pickler
                obj_value = self.find_component(label)
            else:
                obj_value = symbol_map.getEquivalentObject(label, self)

            if obj_value is SymbolMap.UnknownSymbol:
                if ignore_invalid_labels is True:
                    continue
                else:
                    raise KeyError("Objective with label '%s' is not in model '%s'."
                                   % ( label, self.name, ))

            if not isinstance(obj_value, _ObjectiveData):
                raise TypeError("Objective '%s' in model '%s' is type %s"
                                % ( label, self.name, str(type(obj_value)) ))

            for _attr_key, attr_value in iteritems(entry):
                attr_key = _attr_key[0].lower() + _attr_key[1:]
                if attr_key in valid_import_suffixes:
                    # GAH: Unlike the var suffix information in the solution object,
                    #      objective suffix values are ScalarData objects. I
                    #      think it could be advantageous to make all suffix information
                    #      ScalarData types. But for now I will take the simple route
                    #      and maintain consistency with var suffixes, hence 
                    #      attr_value.value rather than just attr_value
                    valid_import_suffixes[attr_key].setValue(obj_value,attr_value.value,expand=False)
        
        #
        # Load variable data
        #
        var_skip_attrs = ['id','canonical_label']
        for label, entry in iteritems(soln.variable):
            if same_instance:
                var_value = symbol_map.getObject(label)
            elif symbol_map is None:
                # We are going to assume the Solution was labeled with
                # the SolverResults pickler
                if 'canonical_label' in entry:
                    var_value = self.find_component(entry['canonical_label'])
                else:
                    # A last-ditch effort to resolve the object by the
                    # old lp-style labeling scheme
                    tmp = label
                    if tmp[-1] == ')': tmp = tmp[:-1]
                    tmp = tmp.replace('(',':').replace(')','.')
                    var_value = self.find_component(tmp)
            else:
                var_value = symbol_map.getEquivalentObject(label, self)

            if var_value is SymbolMap.UnknownSymbol:
                # NOTE: the following is a hack, to handle the ONE_VAR_CONSTANT
                #    variable that is necessary for the objective constant-offset
                #    terms.  probably should create a dummy variable in the model
                #    map at the same time the objective expression is being
                #    constructed.
                if label == "ONE_VAR_CONSTANT":
                    continue
                elif ignore_invalid_labels is True:
                    continue
                else:
                    raise KeyError("Variable label '%s' is not in model '%s'."
                                   % ( label, self.name, ))

            if not isinstance(var_value,_VarData):
                msg = "Variable '%s' in model '%s' is type %s"
                raise TypeError(msg % (
                    label, self.name, str(type(var_value)) ))

            if (allow_consistent_values_for_fixed_vars is False) and (var_value.fixed is True):
                msg = "Variable '%s' in model '%s' is currently fixed - new" \
                      ' value is not expected in solution'
                raise TypeError(msg % ( label, self.name ))

            for _attr_key, attr_value in iteritems(entry):
                attr_key = _attr_key[0].lower() + _attr_key[1:]
                if attr_key == 'value':
                    if (allow_consistent_values_for_fixed_vars is True) and (var_value.fixed is True) and (math.fabs(attr_value - var_value.value) > comparison_tolerance_for_fixed_vars):
                        msg = "Variable '%s' in model '%s' is currently fixed - a value of '%s' in solution is not within tolerance=%s of the current value of '%s'"
                        raise TypeError(msg % ( label, self.name, str(attr_value), str(comparison_tolerance_for_fixed_vars), str(var_value.value) ))
                    var_value.value = attr_value
                    var_value.stale = False
                elif attr_key in valid_import_suffixes:
                    valid_import_suffixes[attr_key].setValue(var_value,attr_value,expand=False)

        #
        # Load constraint data
        #
        con_skip_attrs = ['id', 'canonical_label']
        for label, entry in iteritems(soln.constraint):

            if same_instance:
                con_value = symbol_map.getObject(label)
            elif symbol_map is None:
                # We are going to assume the Solution was labeled with
                # the SolverResults pickler
                con_value = self.find_component(label)
            else:
                con_value = symbol_map.getEquivalentObject(label, self)

            if con_value is SymbolMap.UnknownSymbol:
                #
                # This is a hack - see above.
                #
                if label.endswith('ONE_VAR_CONSTANT'):
                    continue
                elif ignore_invalid_labels is True:
                    continue
                else:
                    raise KeyError("Constraint with label '%s' is not in model '%s'."
                                   % ( label, self.name, ))

            if not isinstance(con_value, _ConstraintData):
                raise TypeError("Constraint '%s' in model '%s' is type %s"
                                % ( label, self.name, str(type(con_value)) ))

            for _attr_key, attr_value in iteritems(entry):
                attr_key = _attr_key[0].lower() + _attr_key[1:]
                if attr_key in valid_import_suffixes:
                    # GAH: Unlike the var suffix information in the solution object,
                    #      constraint suffix values are ScalarData objects. I
                    #      think it could be advantageous to make all suffix information
                    #      ScalarData types. But for now I will take the simple route
                    #      and maintain consistency with var suffixes, hence 
                    #      attr_value.value rather than just attr_value
                    # JPW: The use of MapContainers was nixed for constraints (they have
                    #      long been gone for variables), due to their excessive memory
                    #      requirements. so at the moment, attr_value objects are not
                    #      ScalarData types. These container modifications do not, however,
                    #      contradict Gabe's desire above.
                    valid_import_suffixes[attr_key].setValue(con_value,attr_value,expand=False)
    
    def write(self, filename=None, format=ProblemFormat.cpxlp, solver_capability=None, io_options={}):
        """
        Write the model to a file, with a given format.

        TODO: verify that this method needs to return the filename and symbol_map.
        TODO: these should be returned in a Bunch() object.
        """
        
        if format is None and not filename is None:
            #
            # Guess the format if none is specified
            #
            format = guess_format(filename)
        if solver_capability is None:
            solver_capability = lambda x: True

        problem_writer = pyomo.opt.WriterFactory(format)
        if problem_writer is None:
            raise ValueError(\
                    "Cannot write model in format '%s': no model writer " \
                    "registered for that format" \
                    % str(format))

        (filename, symbol_map) = problem_writer(self, filename, solver_capability, io_options)

        if __debug__ and logger.isEnabledFor(logging.DEBUG):
                logger.debug("Writing model '%s' to file '%s' with format %s", self.name, str(filename), str(format))
        return filename, symbol_map


class ConcreteModel(Model):
    """
    A concrete optimization model that does not defer construction of
    components.
    """

    def __init__(self, *args, **kwds):
        kwds['_deprecate'] = False
        Model.__init__(self, *args, **kwds)
        self.config.create_functor = 'pyomo.model.default_constructor'
        self.construct()


class AbstractModel(Model):
    """
    An abstract optimization model that defers construction of
    components.
    """

    def __init__(self, *args, **kwds):
        kwds['_deprecate'] = False
        Model.__init__(self, *args, **kwds)
        self.config.create_functor = 'pyomo.model.default_constructor'



@pyomo.misc.pyomo_api(namespace='pyomo.model')
def default_constructor(data, model=None, filename=None, data_dict={}, name=None, namespace=None, namespaces=None, preprocess=True, profile_memory=0, report_timing=False, clone=None):
    """
    Create a concrete instance of this Model, possibly using data
    read in from a file.

    Required:
        model:              An AbstractModel object.

    Optional:
        filename:           The name of a Pyomo Data File that will be used to load
                                data into the model.
        data_dict:          A dictionary containing initialization data for the model
                                to be used if there is no filename
        name:               The name given to the model.
        namespace:          A namespace used to select data.
        namespaces:         A list of namespaces used to select data.
        preprocess:         If False, then preprocessing is suppressed.
        profile_memory:     A number that indicates the profiling level.
        report_timing:      Report timing statistics during construction.
        clone:              Force a clone of the model if this is True.

    Return:
        instance:           Return the model that is constructed.
    """
    if name is None:
        name = model.name
    #
    # Generate a warning if this is a concrete model but the filename is specified.
    # A concrete model is already constructed, so passing in a data file is a waste 
    # of time.
    #
    if model.is_constructed() and isinstance(filename,basestring):
        msg = "The filename=%s will not be loaded - supplied as an argument to the create() method of a ConcreteModel instance with name=%s." % (filename, name)
        logger.warning(msg)
    #
    # If construction is deferred, then clone the model and 
    #    
    if not model._constructed:
        instance = model.clone()

        if namespaces is None or len(namespaces) == 0:
            if filename is None:
                instance.load(data_dict, namespaces=[None], profile_memory=profile_memory, report_timing=report_timing)
            else:
                instance.load(filename, namespaces=[None], profile_memory=profile_memory, report_timing=report_timing)
        else:
            if filename is None:
                instance.load(data_dict, namespaces=namespaces+[None], profile_memory=profile_memory, report_timing=report_timing)
            else:
                instance.load(filename, namespaces=namespaces+[None], profile_memory=profile_memory, report_timing=report_timing)
    else:
        if clone:
            instance = model.clone()
        else:
            instance = model
    #
    # Preprocess the new model
    #    
    if preprocess is True:

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

    if not name is None:
        instance.name=name

    return Bunch(instance=instance)


register_component(Model, 'Model objects can be used as a component of other models.')
register_component(ConcreteModel, 'A concrete optimization model that does not defer construction of components.')
register_component(AbstractModel, 'An abstract optimization model that defers construction of components.')

