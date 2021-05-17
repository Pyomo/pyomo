#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import collections
import inspect
from weakref import ref as weakref_ref

from pyomo.common.errors import PyomoException
from pyomo.common.deprecation import deprecated, deprecation_warning


class PluginError(PyomoException):
    pass


@deprecated('The alias() function is deprecated, as Pyomo plugins do not '
            'support access by registered name', version='TBD')
def alias(name, doc=None, subclass=False):
    pass


def implements(interface, inherit=None, namespace=None, service=False):
    if namespace is not None:
        deprecation_warning(
            "The Pyomo plugin infrastructure is deprecated and no "
            "longer supports anything other than a single global namespace.",
            version='6.0')

    calling_frame = inspect.currentframe().f_back
    locals_ = calling_frame.f_locals

    #
    # Some sanity checks
    #
    assert locals_ is not calling_frame.f_globals and '__module__' in locals_, \
        'implements() can only be used in a class definition'
    assert issubclass(interface, Interface)

    locals_.setdefault('__implements__', []).append(
        (interface, inherit, service)
    )
    

class InterfaceMeta(type):
    def __new__(cls, name, bases, classdict, *args, **kwargs):
        classdict.setdefault('_plugins', {})
        return super().__new__(cls, name, bases, classdict, *args, **kwargs)


class Interface(metaclass=InterfaceMeta):
    pass


class _deprecated_plugin_dict(dict):
    def __init__(self, name, classdict):
        super().__init__()
        msg = classdict.pop('__deprecated_message__', None)
        if not msg:
            msg = 'The %s interface has been deprecated' % (name,)
        version = classdict.pop('__deprecated_version__', None)
        remove_in = classdict.pop('__deprecated_remove_in__', None)
        self._deprecation_info = {
            'msg': msg, 'version': version, 'remove_in': remove_in
        }

    def __setitem__(self, key, val):
        deprecation_warning(**self._deprecation_info)
        super().__setitem__(key, val)

    def items(self):
        deprecation_warning(**self._deprecation_info)
        return super().items()

class DeprecatedInterfaceMeta(InterfaceMeta):

    def __new__(cls, name, bases, classdict, *args, **kwargs):
        classdict.setdefault(
            '_plugins', _deprecated_plugin_dict(name, classdict)
        )
        return super().__new__(cls, name, bases, classdict, *args, **kwargs)


class DeprecatedInterface(Interface, metaclass=DeprecatedInterfaceMeta):
    pass


class PluginMeta(type):

    def __new__(cls, name, bases, classdict, *args, **kwargs):
        # This plugin is a singleton plugin based on the __singleton__
        # flag, OR if any base class is a singleton plugin
        _singleton = classdict.pop(
            '__singleton__',
            any(getattr(base, '__singleton__', None) is not None
                for base in bases)
        )
        classdict['__singleton__'] = None
        implements = classdict.setdefault('__implements__', [])
        for base in bases:
            implements.extend(getattr(base, '__implements__', []))
        for interface, inherit, service in implements:
            if not inherit:
                continue
            if not any(issubclass(base, interface) for base in bases):
                bases.append(interface)

        new_class = super().__new__(
            cls, name, bases, classdict, *args, **kwargs)

        for interface, inherit, service in implements:
            interface._plugins[new_class] = []

        if _singleton:
            new_class.__singleton__ = new_class()

        return new_class


class Plugin(object, metaclass=PluginMeta):
    def __new__(cls):
        if cls.__singleton__ is not None:
            raise RuntimeError(
                "Cannot create multiple singleton plugin instances of type %s"
                % (cls,))
        obj = super().__new__(cls)
        for interface, inherit, service in cls.__implements__:
            interface._plugins[cls].append((weakref_ref(obj), service))
        return obj

    def activate(self):
        cls = self.__class__
        for interface, inherit, service in cls.__implements__:
            for i, (obj, service) in enumerate(interface._plugins[cls]):
                if obj() is self and not service:
                    interface._plugins[cls][i] = (obj, True)

    def deactivate(self):
        cls = self.__class__
        for interface, inherit, service in cls.__implements__:
            for i, (obj, service) in enumerate(interface._plugins[cls]):
                if obj() is self and service:
                    interface._plugins[cls][i] = (obj, False)


class SingletonPlugin(Plugin):
    __singleton__ = True

class ExtensionPoint(object):
    def __init__(self, interface):
        assert issubclass(interface, Interface)
        self._interface = interface

    def __iter__(self):
        for cls, plugins in self._interface._plugins.items():
            remove = []
            for i, (obj, service) in enumerate(plugins):
                if not obj():
                    remove.append(i)
                elif service:
                    yield obj()
            if remove:
                for i in reversed(remove):
                    plugins.pop(i)

    def __len__(self):
        return len(list(self.__iter__()))

    def extensions(self, all=False, key=None):
        # We no longer support the all or key interfaces from PyUtilib
        assert all is False
        assert key is None
        return list(self)

    def __call__(self, key=None, all=False):
        return self.extensions(all=all, key=key)

    def service(self, key=None, all=False):
        """Return the unique service that matches the interface of this
        extension point.  An exception occurs if no service matches the
        specified key, or if multiple services match.
        """
        ans = self.extensions(all=all, key=key)
        if len(ans) == 1:
            #
            # There is a single service, so return it.
            #
            return ans.pop()
        elif len(ans) == 0:
            return None
        else:
            raise PluginError("The ExtensionPoint does not have a unique "
                              "service!  %d services are defined for interface"
                              " %s.  (key=%s)" %
                              (len(ans), self._interface.__name__, str(key)))


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
