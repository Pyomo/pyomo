"""
Convert to Python3 with the following command:

cp plugin2.py plugin3.py
2to3 -w plugin3.py
"""

__all__ = ['Plugin', 'implements', 'Interface', 'CreatePluginFactory', 'PluginMeta', 'alias', 'ExtensionPoint', 'SingletonPlugin', 'PluginFactory', 'PluginError', 'push', 'pop', 'clear', 'display', 'with_metaclass']

import sys
import weakref
from six import itervalues
import logging
logger = logging.getLogger('pyomo.misc.plugin')

# This is a copy of the with_metaclass function from 'six' from the 
# development branch.  This fixes a bug in six 1.6.1.
# 
# Copyright (c) 2010-2014 Benjamin Peterson
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
def with_metaclass(meta, *bases):
    """Create a base class with a metaclass."""
    # This requires a bit of explanation: the basic idea is to make a
    # dummy metaclass for one level of class instantiation that replaces
    # itself with the actual metaclass.  Because of internal type checks
    # we also need to make sure that we downgrade the custom metaclass
    # for one level to something closer to type (that's why __call__ and
    # __init__ comes back from type etc.).
    class metaclass(meta):
        __call__ = type.__call__
        __init__ = type.__init__
        def __new__(cls, name, this_bases, d):
            if this_bases is None:
                return type.__new__(cls, name, (), d)
            return meta(name, bases, d)
    return metaclass('temporary_class', None, {})

#
# Plugins define within Pyomo
#

# A dictionary of interface classes mapped to sets of plugin class instance ids
#   interface cls -> set(ids)
interface_services = {}

# A dictionary of plugin instances
#   id -> weakref(instance)
plugin_instances = {}


class Environment(object):

    def __init__(self, name):
        self.name = name
        # A dictionary of plugin classes
        #   name -> plugin cls
        self.plugin_registry = {}
        # The set of interfaces that have been defined
        self.interfaces = set()
        # A dictionary of singleton plugin class instance ids
        #   plugin cls -> id
        self.singleton_services = {}

    def cleanup(self):
        global plugin_instances
        if plugin_instances is None:
            return
        for id_ in itervalues(self.singleton_services):
            if id_ in plugin_instances and not plugin_instances[id_] is None:
                del plugin_instances[id_]

# Environment stack
env = [Environment('pyomo')]


def push(name):
    global env
    env.append(Environment(name))

def pop():
    global env
    tmp = env.pop()
    tmp.cleanup()

def clear():
    global interface_services
    global plugin_instances
    global env
    while len(env) > 0:
        tmp = env.pop()
        tmp.cleanup()
    env = []
    interface_services = {}
    plugin_instances = {}

def display(interface=None, verbose=False):
    print("Plugin Instances:", len(plugin_instances))
    if not interface is None:
        print("Interface:",interface.name)
        print("Count:",len(interface_services.get(interface,[])))
        if verbose:
            for service in interface.services.get(interface,[]):
                print(service)
    else:
        print("Interfaces", len(interface_services))
        for interface in interface_services:
            print("  Interface:", interface)
            print("  Count:",len(interface_services.get(interface,[])))
            if verbose:
                for service in interface_services.get(interface,[]):
                    print("     ",plugin_instances[service])

    print("")
    for env_ in env:
        print("Plugin Declarations:",env_.name) 
        #print env_.interfaces
        for interface in sorted(env_.interfaces, key=lambda v: v.upper()):
            print("Interface:", interface)
            #print "Aliases:"
            #for alias in sorted(interface._factory_cls.keys(), key=lambda v: v.upper()):
                #print "   ",alias,interface._factory_cls[alias]


class PluginError(Exception):
    """Exception base class for plugin errors."""

    def __init__(self, value):
        """Constructor, whose argument is the error message"""
        self.value = value

    def __str__(self):
        """Return a string value for this message"""
        return str(self.value)


class InterfaceMeta(type):
    """Meta class that registered the declaration of an interface"""

    def __new__(cls, name, bases, d):
        """Register this interface"""
        new_class = type.__new__(cls, name, bases, d)
        if name != "Interface":
            if name in env[-1].interfaces:
                raise PluginError("Interface %s has already been defined" % name)
            env[-1].interfaces.add(name)
        return new_class


class Interface(with_metaclass(InterfaceMeta,object)):
    """
    Marker base class for extension point interfaces.  This class
    is not intended to be instantiated.  Instead, the declaration
    of subclasses of Interface are recorded, and these
    classes are used to define extension points.
    """
    pass


class ExtensionPoint(object):
    """Marker class for extension points in services."""

    def __init__(self, *args):
        """Create the extension point.

        @param interface: the `Interface` subclass that defines the protocol
            for the extension point
        @param env: the `PluginEnvironment` instance that this extension point
            references
        """
        #
        # Construct the interface, passing in this extension
        #
        nargs=len(args)
        if nargs == 0:
            raise PluginError("Must specify interface class used in the ExtensionPoint")
        self.interface = args[0]
        self.__doc__ = 'List of services that implement `%s`' % self.interface.__name__

    def __iter__(self):
        """
        Return an iterator to a set of services that match the interface of this
        extension point.
        """
        return self.extensions().__iter__()

    def __call__(self, key=None, all=False):
        """
        Return a set of services that match the interface of this
        extension point.
        """
        if type(key) in (int, int):
            raise PluginError("Access of the n-th extension point is disallowed.  This is not well-defined, since ExtensionPoints are stored as unordered sets.")
        return self.extensions(all=all, key=key)

    def service(self, key=None, all=False):
        """
        Return the unique service that matches the interface of this
        extension point.  An exception occurs if no service matches the
        specified key, or if multiple services match.
        """
        ans = ExtensionPoint.__call__(self, key=key, all=all)
        if len(ans)== 1:
            #
            # There is a single service, so return it.
            #
            return ans.pop()
        elif len(ans) == 0:
            return None
        else:
            raise PluginError("The ExtensionPoint does not have a unique service!  %d services are defined for interface %s.  (key=%s)" % (len(ans), self.interface.__name__, str(key)))

    def __len__(self):
        """
        Return the number of services that match the interface of this
        extension point.
        """
        return len(self.extensions())

    def extensions(self, all=False, key=None):
        """
        Return a set of services that match the interface of this
        extension point.  This tacitly filters out disabled extension points.
        """
        strkey = str(key)
        ans = set()
        remove = set()
        if self.interface in interface_services:
            for id in interface_services[self.interface]:
                try:
                    if id < 0:
                        plugin = plugin_instances[id]
                    else:
                        plugin = plugin_instances[id]()
                except KeyError:
                    remove.add(id)
                    continue
                if plugin is None:
                    remove.add(id)
                elif (all or plugin._enable) and (key is None or strkey == plugin.name):
                    ans.add(plugin)
            # Remove weakrefs that were empty
            for id in remove:
                interface_services[self.interface].remove(id)
        return sorted( ans, key=lambda x:x.id )


class PluginMeta(type):
    """Meta class for the Plugin class.  This meta class
    takes care of service and extension point registration.  This class
    also instantiates singleton plugins.
    """

    def __new__(cls, name, bases, d):
        """Find all interfaces that need to be registered."""
        #
        # Avoid cycling in the Python logic by hard-coding the behavior
        # for the Plugin and SingletonPlugin classes.
        #
        if name == "Plugin":
            d['__singleton__'] = False
            return type.__new__(cls, name, bases, d)
        if name == "SingletonPlugin":
            d['__singleton__'] = True
            return type.__new__(cls, name, bases, d)
        #
        # Check if plugin has already been registered
        #
        if len(d.get('_implements', [])) == 0 and name in env[-1].plugin_registry:
            return env[-1].plugin_registry[name]
            #raise PluginError("Plugin class %r does not implement an interface, and it has already been defined" % name)
        #
        # Find all interfaces that this plugin will support
        #
        __interfaces__ = {}
        for interface in d.get('_implements', {}):
            __interfaces__.setdefault(interface,[]).extend( d['_implements'][interface] )
        for base in [base for base in bases if hasattr(base, '__interfaces__')]:
            for interface in base.__interfaces__:
                __interfaces__.setdefault(interface,[]).extend( base.__interfaces__[interface] )
        d['__interfaces__'] = __interfaces__
        #
        # Create a boolean, which indicates whether this is
        # a singleton class.
        #
        if True in [issubclass(x, SingletonPlugin) for x in bases]:
            d['__singleton__'] = True
        else:
            d['__singleton__'] = False
        #
        # Add interfaces to the list of base classes if they are
        # declared inherited.
        #
        flag = False
        bases = list(bases)
        for interface in d.get('_inherited_interfaces', set()):
            if not interface in bases:
                bases.append(interface)
                flag = True
        if flag:
            cls=MergedPluginMeta
        #
        # Create new class
        #
        try:
            new_class = type.__new__(cls, name, tuple(bases), d)
        except:
            #print "HERE", cls, name, bases, d
            raise
        setattr(new_class,'__name__',name)
        #
        for _interface in __interfaces__:
            if getattr(_interface, '_factory_active', None) is None:
                continue
            for _name,_doc,_subclass in getattr(new_class,"_factory_aliases",[]):
                if _name in _interface._factory_active:
                    if _subclass:
                        continue
                    else:
                        raise PluginError("Alias '%s' has already been defined for interface '%s'" % (_name,str(_interface)))
                _interface._factory_active[_name] = name
                _interface._factory_doc[_name] = _doc
                _interface._factory_cls[_name] = new_class
        #
        if d['__singleton__']:
            #
            # Here, we create an instance of a singleton class, which
            # registers itself in singleton_services
            #
            env[-1].singleton_services[new_class] = True
            __instance__ = new_class()
            plugin_instances[__instance__.id] = __instance__
            env[-1].singleton_services[new_class] = __instance__.id
        else:
            __instance__ = None
        #
        # Register this plugin
        #
        env[-1].plugin_registry[name] = new_class
        return new_class


class MergedPluginMeta(PluginMeta,InterfaceMeta):

    def __new__(cls, name, bases, d):
        return PluginMeta.__new__(cls, name, bases, d)


class Plugin(with_metaclass(PluginMeta, object)):
    """Base class for plugins.  A 'service' is an instance of a Plugin.

    Every Plugin class can declare what extension points it provides, as
    well as what extension points of other Plugins it extends.
    """

    # A counter used to generate unique IDs for plugin instances
    plugin_counter = 0

    def __del__(self):
        self.deactivate()
        if not plugin_instances is None and self.id in plugin_instances and not plugin_instances[self.id] is None:
            del plugin_instances[self.id]

    def __init__(self, **kwargs):
        if "name" in kwargs:
            self.name=kwargs["name"]

    def __new__(cls, *args, **kwargs):
        """Plugin constructor"""
        #
        # If this service is a singleton, then allocate and configure
        # it differently.
        #
        if cls in env[-1].singleton_services:       #pragma:nocover
            id = env[-1].singleton_services[cls]
            if id is True:
                self = super(Plugin, cls).__new__(cls)
                Plugin.plugin_counter += 1
                self.id = - Plugin.plugin_counter
                self.name = self.__class__.__name__
                self._enable = True
                self.activate()
            else:
                self = plugin_instances[id]
            return self
        #
        # Else we generate a normal plugin
        #
        self = super(Plugin, cls).__new__(cls)
        Plugin.plugin_counter += 1
        self.id = Plugin.plugin_counter
        self.name = "Plugin."+str(self.id)
        self._enable = True
        plugin_instances[self.id] = weakref.ref(self)
        if getattr(cls, '_service', True):
            self.activate()
        return self

    def activate(self):
        """
        Register this plugin with all interfaces that it implements.
        """
        for interface in self.__interfaces__:
            interface_services.setdefault(interface,set()).add(self.id)

    def deactivate(self):
        """
        Unregister this plugin with all interfaces that it implements.
        """
        global interface_services
        if interface_services is None:
            # This could happen when python quits
            return
        for interface in self.__interfaces__:
            if interface in interface_services:
                # Remove an element if it exists
                interface_services[interface].discard(self.id)

    @staticmethod
    def alias(name, doc=None, subclass=False):
        """
        This function is used to declare aliases that can be used by a factory for constructing
        plugin instances.

        When the subclass option is True, then subsequent calls to alias() with this class name
        are ignored, because they are assumed to be due to subclasses of the original class
        declaration.
        """
        frame = sys._getframe(1)
        locals_ = frame.f_locals
        assert locals_ is not frame.f_globals and '__module__' in locals_, \
               'alias() can only be used in a class definition'
        #print "HERE", name, doc, subclass
        locals_.setdefault('_factory_aliases', set()).add((name,doc,subclass))

    @staticmethod
    def implements(interface, inherit=None, namespace=None, service=False):
        """
        Can be used in the class definition of `Plugin` subclasses to
        declare the extension points that are implemented by this
        interface class.
        """
        frame = sys._getframe(1)
        locals_ = frame.f_locals
        #
        # Some sanity checks
        #
        assert namespace is None or isinstance(namespace,str), \
               'second implements() argument must be a string'
        assert locals_ is not frame.f_globals and '__module__' in locals_, \
               'implements() can only be used in a class definition'
        #
        locals_.setdefault('_implements', {}).setdefault(interface,[]).append(namespace)
        if inherit:
            locals_.setdefault('_inherited_interfaces', set()).add(interface)
        locals_['_service'] = service

    def disable(self):
        """Disable this plugin"""
        self._enable = False

    def enable(self):
        """Enable this plugin"""
        self._enable = True

    def enabled(self):
        """Return value indicating if this plugin is enabled"""
        return self._enable

alias = Plugin.alias
implements = Plugin.implements


class SingletonPlugin(Plugin):
    """The base class for singleton plugins.  The PluginMeta class
    instantiates a SingletonPlugin class when it is declared.  Note that
    only one instance of a SingletonPlugin class is created in
    any environment.
    """
    pass


def CreatePluginFactory(_interface):
    if getattr(_interface, '_factory_active', None) is None:
        setattr(_interface, '_factory_active', {})
        setattr(_interface, '_factory_doc', {})
        setattr(_interface, '_factory_cls', {})
        setattr(_interface, '_factory_deactivated', {})

    class PluginFactoryFunctor(object):
        def __call__(self, _name=None, args=[], **kwds):
            if _name is None:
                return self
            _name=str(_name)
            if not _name in _interface._factory_active:
                return None
            return PluginFactory(_interface._factory_cls[_name], args, **kwds)
        def services(self):
            return list(_interface._factory_active.keys())
        def get_class(self, name):
            return _interface._factory_cls[name]
        def doc(self, name):
            tmp = _interface._factory_doc[name]
            if tmp is None:
                return ""
            return tmp
        def deactivate(self, name):
            if name in _interface._factory_active:
                _interface._factory_deactivated[name] = _interface._factory_active[name]
                del _interface._factory_active[name]
        def activate(self, name):
            if name in _interface._factory_deactivated:
                _interface._factory_active[name] = _interface._factory_deactivated[name]
                del _interface._factory_deactivated[name]
    return PluginFactoryFunctor()


def PluginFactory(classname, args=[], **kwds):
    """Construct a Plugin instance, and optionally assign it a name"""
    if isinstance(classname, str):
        try:
            cls = env[-1].plugin_registry[classname]
        except KeyError:
            raise PluginError("Unknown class %r" % str(classname))
    else:
        cls = classname
    obj = cls(*args, **kwds)
    if 'name' in kwds:
        obj.name = kwds['name']
    if __debug__ and logger.isEnabledFor(logging.DEBUG):
        if obj is None:
            logger.debug("Failed to create plugin %s" % (classname))
        else:
            logger.debug("Creating plugin %s with name %s" % (classname, obj.name))
    return obj

