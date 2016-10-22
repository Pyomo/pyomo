#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

__all__ = ['pyomo_callback',
        'IPyomoExpression', 'ExpressionFactory', 'ExpressionRegistration',
        'IModelComponent',
        'ModelComponentFactory',
        'register_component',
        'IPyomoPresolver', 'IPyomoPresolveAction',
        'DataManagerFactory',
        'IParamRepresentation',
        'ParamRepresentationFactory',
        'IModelTransformation',
        'IPyomoScriptPreprocess',
        'IPyomoScriptCreateModel',
        'IPyomoScriptCreateDataPortal',
        'IPyomoScriptModifyInstance',
        'IPyomoScriptPrintModel',
        'IPyomoScriptPrintInstance',
        'IPyomoScriptSaveInstance',
        'IPyomoScriptPrintResults',
        'IPyomoScriptSaveResults',
        'IPyomoScriptPostprocess',
        'Transformation',
        'IModelTransformation',
        'TransformationFactory',
        'UnknownDataManager'
        ]

import pyutilib.misc
import logging

logger = logging.getLogger('pyomo.core')
registered_callback = {}

def pyomo_callback( name ):
    """This is a decorator that declares a function to be 
    a callback function.  The callback functions are 
    added to the solver when run from the pyomo script.

    Example:

    @pyomo_callback('cut-callback')
    def my_cut_generator(solver, model):
        ...
    """
    def fn(f):
        registered_callback[name] = f
        return f
    return fn


from pyomo.util.plugin import *


class IPyomoScriptPreprocess(Interface):

    def apply(self, **kwds):
        """Apply preprocessing step in the Pyomo script"""

class IPyomoScriptCreateModel(Interface):

    def apply(self, **kwds):
        """Apply model creation step in the Pyomo script"""

class IPyomoScriptModifyInstance(Interface):

    def apply(self, **kwds):
        """Modify and return the model instance"""

class IPyomoScriptCreateDataPortal(Interface):

    def apply(self, **kwds):
        """Apply model data creation step in the Pyomo script"""

class IPyomoScriptPrintModel(Interface):

    def apply(self, **kwds):
        """Apply model printing step in the Pyomo script"""

class IPyomoScriptPrintInstance(Interface):

    def apply(self, **kwds):
        """Apply instance printing step in the Pyomo script"""

class IPyomoScriptSaveInstance(Interface):

    def apply(self, **kwds):
        """Apply instance saving step in the Pyomo script"""

class IPyomoScriptPrintResults(Interface):

    def apply(self, **kwds):
        """Apply results printing step in the Pyomo script"""

class IPyomoScriptSaveResults(Interface):

    def apply(self, **kwds):
        """Apply results saving step in the Pyomo script"""

class IPyomoScriptPostprocess(Interface):

    def apply(self, **kwds):
        """Apply postprocessing step in the Pyomo script"""

class IPyomoPresolver(Interface):

    def get_actions(self):
        """Return a list of presolve actions, in the order in which
        they will be applied."""

    def activate_action(self, action):
        """Activate an action, but leave its default rank"""

    def deactivate_action(self, action):
        """Deactivate an action"""

    def set_actions(self, actions):
        """Set presolve action list"""

    def presolve(self, instance):
        """Apply the presolve actions to this instance, and return the
        revised instance"""


class IPyomoPresolveAction(Interface):

    def presolve(self, instance):
        """Apply the presolve action to this instance, and return the
        revised instance"""

    def rank(self):
        """Return an integer that is used to automatically order presolve actions,
        from low to high rank."""


class IPyomoExpression(Interface):

    def type(self):
        """Return the type of expression"""

    def create(self, args):
        """Create an instance of this expression type"""


class ExpressionRegistration(Plugin):

    implements(IPyomoExpression, service=False)

    def __init__(self, type, cls, swap=False):
        self._type = type
        self._cls = cls
        self._swap = swap

    def type(self):
        return self._type

    def create(self, args):
        if self._swap:
            args = list(args)
            args.reverse()
        return self._cls(args)

def ExpressionFactory(name=None, args=[]):
    ep = ExpressionFactory.ep
    if name is None:
        return map(lambda x: x.name, ep())
    return ep.service(name).create(args)
ExpressionFactory.ep = ExtensionPoint(IPyomoExpression)


class IModelComponent(Interface):
    pass

ModelComponentFactory = CreatePluginFactory(IModelComponent)

def register_component(cls, description):
    class TMP(Plugin):
        implements(IModelComponent, service=False)
        alias(cls.__name__, description)
        component = cls


class IDataManager(Interface):

    def available(self):
        """ Returns True if the data manager can be executed """
        pass

    def requirements(self):
        """ Return a string describing the packages that need to be installed for this plugin to be available """
        pass

    def initialize(self, filename, **kwds):
        """ Prepare to read a data file. """
        pass

    def add_options(self, **kwds):
        """ Add options """
        pass

    def open(self):
        """ Open the data file. """
        pass

    def close(self):
        """ Close the data file. """
        pass

    def read(self):
        """ Read the data file. """
        pass

    def process(self, model, data, default):
        """ Process the data. """
        pass

    def clear(self):
        """ Reset Plugin. """
        pass


class UnknownDataManager(Plugin):

    implements(IDataManager)

    def __init__(self, *args, **kwds):
        Plugin.__init__(self, **kwds)
        #
        # The 'type' is the class type of the solver instance
        #
        self.type = kwds["type"]

    def available(self):
        return False


#
# A DataManagerFactory is an instance of a plugin factory that is
# customized with a custom __call__ method
#
DataManagerFactory = CreatePluginFactory(IDataManager)
#
# This is the custom __call__ method
#
def __datamanager_call__(self, _name=None, args=[], **kwds):
    if _name is None:
        return self
    _name=str(_name)
    if _name in IDataManager._factory_active:
        dm = PluginFactory(IDataManager._factory_cls[_name], args, **kwds)
        if not dm.available():
            raise PluginError("Cannot process data in %s files.  The following python packages need to be installed: %s" % (_name, dm.requirements()))
    else:
        dm = UnknownDataManager(type=_name)
    return dm
#
# Adding the the custom __call__ method to DataManagerFactory
#
pyutilib.misc.add_method(DataManagerFactory, __datamanager_call__, name='__call__')



class IModelTransformation(Interface):

    def apply(self, model, **kwds):
        """Apply a model transformation and return a new model instance"""

    def __call__(self, model, **kwds):
        """Use this plugin instance as a functor to apply a transformation"""
        return self.apply(model, **kwds)


class IParamRepresentation(Interface):
    pass

ParamRepresentationFactory = CreatePluginFactory(IParamRepresentation)

class TransformationInfo(object): pass

class TransformationData(object):
    """
    This is a container class that supports named data objects.
    """

    def __init__(self):
        self._data = {}

    def __getitem__(self, name):
        if not name in self._data:
            self._data[name] = TransformationInfo()
        return self._data[name]


class Transformation(Plugin):
    """
    Base class for all model transformations.
    """

    implements(IModelTransformation, service=False)

    def __init__(self, **kwds):
        kwds["name"] = kwds.get("name", "transformation")
        super(Transformation, self).__init__(**kwds)

    def apply(self, model, **kwds):
        logger.warning(
"""DEPRECATION WARNING: Transformation.apply() has been deprecated.
Please use either Transformation.apply_to() for in-place transformations
or Transformation.create_using() for transformations that create a new,
independent transformed model instance.""")
        inplace = kwds.pop('inplace', True)
        if inplace:
            self.apply_to(model, **kwds)
        else:
            return self.create_using(model, **kwds)

    def apply_to(self, model, **kwds):
        """
        Apply the transformation to the given model.
        """
        if not hasattr(model, '_transformation_data'):
            model._transformation_data = TransformationData()
        self._apply_to(model, **kwds)

    def create_using(self, model, **kwds):
        """
        Create a new model with this transformation
        """
        if not hasattr(model, '_transformation_data'):
            model._transformation_data = TransformationData()
        return self._create_using(model, **kwds)

    def _apply_to(self, model, **kwds):
        raise RuntimeError("The Transformation.apply_to method is not implemented.")

    def _create_using(self, model, **kwds):
        instance = model.clone()
        self._apply_to(instance, **kwds)
        return instance


TransformationFactory = CreatePluginFactory(IModelTransformation)


def Xapply_transformation(*args, **kwds):
    """This function is deprecated"""
    if len(args) is 0:
        return TransformationFactory.services()
    xfrm = TransformationFactory(args[0])
    if len(args) == 1 or xfrm is None:
        return xfrm
    tmp=(args[1],)
    return xfrm.apply(*tmp, **kwds)


