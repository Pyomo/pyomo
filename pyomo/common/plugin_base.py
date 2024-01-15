#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
#
#  This module was originally developed as part of the PyUtilib project
#  Copyright (c) 2008 Sandia Corporation.
#  This software is distributed under the BSD License.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  ___________________________________________________________________________

import inspect
from weakref import ref as weakref_ref

from pyomo.common.errors import PyomoException
from pyomo.common.deprecation import deprecated, deprecation_warning


class PluginGlobals(object):
    @staticmethod
    @deprecated(
        "The PluginGlobals environment manager is deprecated: "
        "Pyomo only supports a single global environment",
        version='6.0',
    )
    def add_env(name):
        pass

    @staticmethod
    @deprecated(
        "The PluginGlobals environment manager is deprecated: "
        "Pyomo only supports a single global environment",
        version='6.0',
    )
    def pop_env():
        pass

    @staticmethod
    @deprecated(
        "The PluginGlobals environment manager is deprecated: "
        "Pyomo only supports a single global environment",
        version='6.0',
    )
    def clear():
        pass


class PluginError(PyomoException):
    pass


def alias(name, doc=None, subclass=None):
    if subclass is not None:
        deprecation_warning(
            "The Pyomo plugin infrastructure alias() function does "
            "not support the subclass flag.",
            version='6.0',
        )
    calling_frame = inspect.currentframe().f_back
    locals_ = calling_frame.f_locals
    #
    # Some sanity checks
    #
    assert (
        locals_ is not calling_frame.f_globals and '__module__' in locals_
    ), 'implements() can only be used in a class definition'
    #
    locals_.setdefault('__plugin_aliases__', []).append((name, doc))


def implements(interface, inherit=None, namespace=None, service=False):
    if namespace is not None:
        deprecation_warning(
            "The Pyomo plugin infrastructure only supports a "
            "single global namespace.",
            version='6.0',
        )
    calling_frame = inspect.currentframe().f_back
    locals_ = calling_frame.f_locals
    #
    # Some sanity checks
    #
    assert (
        locals_ is not calling_frame.f_globals and '__module__' in locals_
    ), 'implements() can only be used in a class definition'
    assert issubclass(interface, Interface)
    #
    locals_.setdefault('__implements__', []).append((interface, inherit, service))


class InterfaceMeta(type):
    def __new__(cls, name, bases, classdict, *args, **kwargs):
        # Ensure that all interfaces have their own _plugins & _aliases
        # dictionaries
        classdict.setdefault('_next_id', 0)
        classdict.setdefault('_plugins', {})
        classdict.setdefault('_aliases', {})
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
            'msg': msg,
            'version': version,
            'remove_in': remove_in,
        }

    def __setitem__(self, key, val):
        deprecation_warning(**self._deprecation_info)
        super().__setitem__(key, val)

    def items(self):
        deprecation_warning(**self._deprecation_info)
        return super().items()


class DeprecatedInterfaceMeta(InterfaceMeta):
    def __new__(cls, name, bases, classdict, *args, **kwargs):
        classdict.setdefault('_plugins', _deprecated_plugin_dict(name, classdict))
        return super().__new__(cls, name, bases, classdict, *args, **kwargs)


class DeprecatedInterface(Interface, metaclass=DeprecatedInterfaceMeta):
    pass


class PluginMeta(type):
    def __new__(cls, name, bases, classdict, *args, **kwargs):
        # This plugin is a singleton plugin based on the __singleton__
        # class attribute, OR if not specified, if any base class is a
        # singleton plugin
        _singleton = classdict.pop(
            '__singleton__',
            any(getattr(base, '__singleton__', None) is not None for base in bases),
        )
        # This prevents base class __singleton__, __plugin_aliases__,
        # and __implements__ from implicitly bleeding through and being
        # accidentally shared across subclasses.
        classdict['__singleton__'] = None
        aliases = classdict.setdefault('__plugin_aliases__', [])
        implements = classdict.setdefault('__implements__', [])
        # If multiple classes (classdict, and/or any base) implement()
        # the same interface, use standard Python rules to determine
        # which implements() should govern (i.e. classdict supersedes
        # bases, bases resolved in order)
        interfaces = set(impl[0] for impl in implements)
        for base in bases:
            implements.extend(
                ep
                for ep in getattr(base, '__implements__', [])
                if ep[0] not in interfaces
            )
            interfaces.update(impl[0] for impl in implements)
        for interface, inherit, service in implements:
            if not inherit:
                continue
            if not any(issubclass(base, interface) for base in bases):
                bases = bases + (interface,)
                # Python requires that a class' metaclass be a
                # (nonstrict) subclass of the metaclasses of all its
                # base classes.  Check, and declare a new metaclass if
                # necessary.
                if not issubclass(cls, type(interface)):

                    class tmp_meta(cls, type(interface)):
                        def __new__(cls, name, bases, classdict, *args, **kwargs):
                            # This is a plugin and not an Interface.  Do
                            # not set up dicts for the interface
                            # definition.
                            classdict.setdefault('_plugins', None)
                            classdict.setdefault('_aliases', None)
                            return super().__new__(
                                cls, name, bases, classdict, *args, **kwargs
                            )

                    cls = tmp_meta

        new_class = super().__new__(cls, name, bases, classdict, *args, **kwargs)

        # Register the new class with the interfaces
        for interface, inherit, service in implements:
            interface._plugins[new_class] = {}
            interface._aliases.update({name: (new_class, doc) for name, doc in aliases})

        if _singleton:
            new_class.__singleton__ = new_class()

        return new_class


class Plugin(object, metaclass=PluginMeta):
    def __new__(cls):
        if cls.__singleton__ is not None:
            raise RuntimeError(
                "Cannot create multiple singleton plugin instances of type %s" % (cls,)
            )
        obj = super().__new__(cls)
        obj._plugin_ids = {}
        # Record this instance (service) with all Interfaces
        for interface, inherit, service in cls.__implements__:
            _id = interface._next_id
            interface._next_id += 1
            obj._plugin_ids[interface] = _id
            interface._plugins[cls][_id] = (weakref_ref(obj), service)
        return obj

    def activate(self):
        cls = self.__class__
        for interface, inherit, service in cls.__implements__:
            _id = self._plugin_ids[interface]
            obj, service = interface._plugins[cls][_id]
            if not service:
                interface._plugins[cls][_id] = obj, True

    enable = activate

    def deactivate(self):
        cls = self.__class__
        for interface, inherit, service in cls.__implements__:
            _id = self._plugin_ids[interface]
            obj, service = interface._plugins[cls][_id]
            if service:
                interface._plugins[cls][_id] = obj, False

    disable = deactivate

    def enabled(self):
        cls = self.__class__
        return any(
            interface._plugins[cls][self._plugin_ids[interface]][1]
            for interface, inherit, service in cls.__implements__
        )


class SingletonPlugin(Plugin):
    __singleton__ = True


class ExtensionPoint(object):
    def __init__(self, interface):
        assert issubclass(interface, Interface)
        self._interface = interface

    def __iter__(self, key=None, all=False):
        for cls, plugins in self._interface._plugins.items():
            remove = []
            for i, (obj, service) in plugins.items():
                if not obj():
                    remove.append(i)
                elif (all or service) and (
                    key is None or key is cls or key == cls.__name__
                ):
                    yield obj()
            for i in remove:
                del plugins[i]

    def __len__(self):
        return len(list(self.__iter__()))

    def extensions(self, all=False, key=None):
        return list(self.__iter__(key=key, all=all))

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
            return ans[0]
        elif not ans:
            return None
        else:
            raise PluginError(
                "The ExtensionPoint does not have a unique "
                "service!  %d services are defined for interface"
                " '%s' (key=%s)." % (len(ans), self._interface.__name__, str(key))
            )


class PluginFactory(object):
    def __init__(self, interface):
        self.interface = interface

    def __call__(self, name, *args, **kwds):
        name = str(name)
        if name not in self.interface._aliases:
            return None
        else:
            return self.interface._aliases[name][0](*args, **kwds)

    def services(self):
        return list(self.interface._aliases)

    def get_class(self, name):
        return self.interface._aliases.get(name, [None])[0]

    def doc(self, name):
        name = str(name)
        if name not in self.interface._aliases:
            return ""
        else:
            return self.interface._aliases[name][1]

    def deactivate(self, name):
        if isinstance(name, str):
            cls = self.get_class(name)
        if cls is None:
            return
        for service in ExtensionPoint(self.interface)(key=cls):
            service.deactivate()

    def activate(self, name):
        if isinstance(name, str):
            cls = self.get_class(name)
        if cls is None:
            return
        for service in ExtensionPoint(self.interface)(all=True, key=cls):
            service.activate()


# Old name for creating plugin factories
CreatePluginFactory = PluginFactory
