#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

__all__ = ['pyomo_callback',
        'IPyomoExpression', 'ExpressionFactory', 'ExpressionRegistration',
        'IPyomoPresolver', 'IPyomoPresolveAction',
        'IParamRepresentation',
        'ParamRepresentationFactory',
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
        'ModelComponentFactory',
        'Transformation',
        'TransformationFactory',
        ]

import logging
import pyutilib.misc
from pyomo.common.deprecation import deprecated
from pyomo.common.modeling import unique_component_name
from pyomo.common import Factory
from pyomo.common.plugin import (
    alias, implements, Interface, Plugin, PluginFactory, CreatePluginFactory,
    PluginError, ExtensionPoint )
from pyomo.common.timing import TransformationTimer

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


class ModelComponentFactoryClass(Factory):

    def register(self, doc=None):
        def fn(cls):
            return super(ModelComponentFactoryClass, self).register(cls.__name__, doc)(cls)
        return fn

ModelComponentFactory = ModelComponentFactoryClass('model component')


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


class Transformation(object):
    """
    Base class for all model transformations.
    """
    def __init__(self, **kwds):
        kwds["name"] = kwds.get("name", "transformation")
        #super(Transformation, self).__init__(**kwds)

    #
    # Support "with" statements.
    #
    def __enter__(self):
        return self

    def __exit__(self, t, v, traceback):
        pass

    @deprecated(
        "Transformation.apply() has been deprecated.  Please use either "
        "Transformation.apply_to() for in-place transformations or "
        "Transformation.create_using() for transformations that create a "
        "new, independent transformed model instance.")
    def apply(self, model, **kwds):
        inplace = kwds.pop('inplace', True)
        if inplace:
            self.apply_to(model, **kwds)
        else:
            return self.create_using(model, **kwds)

    def apply_to(self, model, **kwds):
        """
        Apply the transformation to the given model.
        """
        timer = TransformationTimer(self, 'in-place')
        if not hasattr(model, '_transformation_data'):
            model._transformation_data = TransformationData()
        self._apply_to(model, **kwds)
        timer.report()

    def create_using(self, model, **kwds):
        """
        Create a new model with this transformation
        """
        timer = TransformationTimer(self, 'out-of-place')
        if not hasattr(model, '_transformation_data'):
            model._transformation_data = TransformationData()
        new_model = self._create_using(model, **kwds)
        timer.report()
        return new_model

    def _apply_to(self, model, **kwds):
        raise RuntimeError(
            "The Transformation.apply_to method is not implemented.")

    def _create_using(self, model, **kwds):
        # Put all the kwds onto the model so that when we clone the
        # model any references to things on the model are correctly
        # updated to point to the new instance.  Note that users &
        # transformation developers cannot rely on things happening by
        # argument side effect.
        name = unique_component_name(model, '_kwds')
        setattr(model, name, kwds)
        instance = model.clone()
        kwds = getattr(instance, name)
        delattr(model, name)
        delattr(instance, name)
        self._apply_to(instance, **kwds)
        return instance


TransformationFactory = Factory('transformation type')

@deprecated()
def apply_transformation(*args, **kwds):
    if len(args) is 0:
        return list(TransformationFactory)
    xfrm = TransformationFactory(args[0])
    if len(args) == 1 or xfrm is None:
        return xfrm
    tmp=(args[1],)
    return xfrm.apply(*tmp, **kwds)
