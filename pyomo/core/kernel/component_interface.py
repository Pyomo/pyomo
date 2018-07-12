#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import sys
import abc
import copy
import weakref

import six
from six.moves import xrange

def _not_implemented(*args, **kwds):
    raise NotImplementedError     #pragma:nocover

def _abstract_readwrite_property(**kwds):
    p = abc.abstractproperty(fget=_not_implemented,
                             fset=_not_implemented,
                             **kwds)
    if 'doc' in kwds:
        p.__doc__ = kwds['doc']
    return p

def _abstract_readonly_property(**kwds):
    p = abc.abstractproperty(fget=_not_implemented,
                             **kwds)
    if 'doc' in kwds:
        p.__doc__ = kwds['doc']
    return p

class _ICategorizedObjectMeta(abc.ABCMeta):
    # This allows the _ctype property on the
    # ICategorizedObject class to "officially" remain
    # private (starts with _, so if anyone attempts to
    # change it, they get what they deserve), while still
    # allowing users to access it via the class or the
    # instance level. If the property below were removed,
    # then ICategorizedObject.ctype would return the
    # property method itself and not the value of the _ctype
    # attribute.
    @property
    def ctype(cls):
        return cls._ctype
@six.add_metaclass(_ICategorizedObjectMeta)
class ICategorizedObject(object):
    """
    Interface for objects that maintain a weak reference to
    a parent storage object and have a category type.

    This class is abstract. It assumes any derived class
    declares the attributes below at the class or instance
    level (with or without __slots__):

    Attributes:
        _ctype: The objects category type.
        _parent: A weak reference to the object's parent or
            :const:`None`.
        _storage_key: The key this object is stored under
            within the parent container.
    """
    __slots__ = ()

    # These flags can be used by implementations to speed up
    # code. The use of ABCMeta as a metaclass slows down
    # isinstance calls by an order of magnitude! So for
    # instance, use hasattr(obj, '_is_categorized_object') as
    # opposed to isinstance(obj, ICategorizedObject)
    _is_categorized_object = True
    """A flag used to indicate the class is an instance of
    ICategorized object. This is a workaround for the slow
    behavior of isinstance on classes that use abc.ABCMeta
    as a metaclass."""

    _is_component = False
    """A flag used to indicate that the class is an instance
    of IComponent. This is a workaround for the slow
    behavior of isinstance on classes that use abc.ABCMeta
    as a metaclass."""

    _is_container = False
    """A flag used to indicate that the class is an instance
    IComponentContainer.  This is a workaround for the slow
    behavior of isinstance on classes that use abc.ABCMeta
    as a metaclass."""

    #
    # Interface
    #

    @property
    def ctype(self):
        """The object's category type."""
        return self._ctype

    @property
    def parent(self):
        """The object's parent"""
        if isinstance(self._parent, weakref.ReferenceType):
            return self._parent()
        else:
            return self._parent

    @property
    def storage_key(self):
        """The object's storage key within its parent"""
        return self._storage_key

    @property
    def parent_block(self):
        """The first block above this object in the storage tree"""
        parent = self.parent
        while (parent is not None) and \
              (not parent._is_component):
            parent = parent.parent
        return parent

    @property
    def root_block(self):
        """The root storage block above this object"""
        root_block = None
        if self._is_component and self._is_container:
            root_block = self
        parent_block = self.parent_block
        while parent_block is not None:
            root_block = parent_block
            parent_block = parent_block.parent_block
        return root_block

    def getname(self,
                fully_qualified=False,
                name_buffer={}, # HACK: ignored (required to work with some solver interfaces, but that code should change soon)
                convert=str, relative_to=None):
        """
        Dynamically generates a name for this object.

        Args:
            fully_qualified (bool): Generate a full name by
                iterating through all anscestor containers.
                Default is :const:`False`.
            convert (function): A function that converts a
                storage key into a string
                representation. Default is the built-in
                function str.
            relative_to (object): When generating a fully
                qualified name, generate the name relative
                to this block.

        Returns:
            If a parent exists, this method returns a string
            representing the name of the object in the
            context of its parent; otherwise (if no parent
            exists), this method returns :const:`None`.

        .. warning::
            Name generation can be slow. See the
            generate_names method, found on most containers,
            for a way to generate a static set of component
            names.
        """
        parent = self.parent
        if parent is None:
            return None

        key = self.storage_key
        name = parent._child_storage_entry_string % convert(key)
        if fully_qualified:
            parent_name = parent.getname(fully_qualified=True)
            if (parent_name is not None) and \
               ((relative_to is None) or \
                (parent is not relative_to)):
                return (parent_name +
                        parent._child_storage_delimiter_string +
                        name)
            else:
                return name
        else:
            return name

    @property
    def name(self):
        """The object's fully qualified name. Alias for
        `obj.getname(fully_qualified=True)`.

        .. warning::
            Name generation can be slow. See the
            generate_names method, found on most containers,
            for a way to generate a static set of component
            names.
        """
        return self.getname(fully_qualified=True)

    @property
    def local_name(self):
        """The object's local name within the context of its
        parent. Alias for
        `obj.getname(fully_qualified=False)`.

        .. warning::
            Name generation can be slow. See the
            generate_names method, found on most containers,
            for a way to generate a static set of component
            names.
        """
        return self.getname(fully_qualified=False)

    def __str__(self):
        """Convert this object to a string by first
        attempting to generate its fully qualified name. If
        the object does not have a name (because it does not
        have a parent, then a string containing the class
        name is returned.

        .. warning::
            Name generation can be slow. See the
            generate_names method, found on most containers,
            for a way to generate a static set of component
            names.
        """
        name = self.name
        if name is None:
            return "<"+self.__class__.__name__+">"
        else:
            return name

    #
    # The following method must be defined to allow
    # expressions and blocks to be correctly cloned.
    # (copied from JDS code in original component.py,
    # comments removed for now)
    #

    def __deepcopy__(self, memo):
        if '__block_scope__' in memo and \
                id(self) not in memo['__block_scope__']:
            _known = memo['__block_scope__']
            _new = []
            tmp = self.parent_block
            tmpId = id(tmp)
            while tmpId not in _known:
                _new.append(tmpId)
                tmp = tmp.parent_block
                tmpId = id(tmp)

            for _id in _new:
                _known[_id] = _known[tmpId]

            if not _known[tmpId]:
                # component is out-of-scope. shallow copy only
                ans = memo[id(self)] = self
                return ans

        ans = memo[id(self)] = self.__class__.__new__(self.__class__)
        ans.__setstate__(copy.deepcopy(self.__getstate__(), memo))
        return ans

    #
    # The following two methods allow implementations to be
    # pickled. These should work whether or not the
    # implementation makes use of __slots__, and whether or
    # not non-empty __slots__ declarations appear on
    # multiple classes in the inheritance chain.
    #

    def __getstate__(self):
        state = getattr(self, "__dict__", {}).copy()
        # Get all slots in the inheritance chain
        for cls in self.__class__.__mro__:
            for key in cls.__dict__.get("__slots__",()):
                state[key] = getattr(self, key)
        # make sure we don't store the __dict__ in
        # duplicate (it can be declared as a slot)
        state.pop('__dict__', None)
        # make sure not to pickle the __weakref__
        # slot if it was declared
        state.pop('__weakref__', None)
        # make sure to dereference the parent weakref
        state['_parent'] = self.parent
        return state

    def __setstate__(self, state):
        for key, value in state.items():
            # bypass a possibly overridden __setattr__
            object.__setattr__(self, key, value)
        # make sure _parent is a weakref
        # if it is not None
        if self._parent is not None:
            self._parent = weakref.ref(self._parent)

class IComponent(ICategorizedObject):
    """
    Interface for components that can be stored inside
    objects of type IComponentContainer.

    This class is abstract, but it partially implements the
    ICategorizedObject interface by defining the following
    attributes:

    Attributes:
        _is_component: :const:`True`
        _is_container: :const:`False`
    """
    _is_component = True
    _is_container = False
    __slots__ = ()

class IComponentContainer(ICategorizedObject):
    """
    Interface for containers of components or other
    containers.

    This class is abstract, but it partially implements the
    ICategorizedObject interface by defining the following
    attributes:

    Attributes:
        _is_component: :const:`False`
        _is_container: :const:`True`
    """
    _is_component = False
    _is_container = True
    _child_storage_delimiter_string = ""
    _child_storage_entry_string = "[%s]"
    __slots__ = ()

    #
    # Interface
    #

    @abc.abstractmethod
    def components(self, *args, **kwds):
        """A generator over the set of components stored
        under this container."""
        raise NotImplementedError     #pragma:nocover

    @abc.abstractmethod
    def child(self, *args, **kwds):
        """Returns a child of this container given a storage
        key."""
        raise NotImplementedError     #pragma:nocover

    @abc.abstractmethod
    def children(self, *args, **kwds):
        """A generator over the children of this container."""
        raise NotImplementedError     #pragma:nocover

    @abc.abstractmethod
    def preorder_traversal(self, *args, **kwds):
        """A generator over all descendents in prefix order."""
        raise NotImplementedError     #pragma:nocover

    @abc.abstractmethod
    def preorder_visit(self, *args, **kwds):
        """Visit all descendents in prefix order."""
        raise NotImplementedError     #pragma:nocover

    @abc.abstractmethod
    def postorder_traversal(self, *args, **kwds):
        """A generator over all descendents in postfix order."""
        raise NotImplementedError     #pragma:nocover

class _ActiveObjectMixin(object):
    """
    To be used as an additional base class in IComponent or
    IComponentContainer implementations to add fuctionality
    for activating and deactivating the component.

    This class is abstract. It assumes any derived class
    declares the attributes below at the class or instance
    level (with or without __slots__):

    Attributes:
        _active (bool): A boolean indicating whethor or not
            this component is active.
    """
    __slots__ = ()

    #
    # Interface
    #

    @property
    def active(self):
        """The active status of this container."""
        return self._active
    @active.setter
    def active(self, value):
        raise AttributeError(
            "Assignment not allowed. Use the "
            "(de)activate method")

    def activate(self, shallow=True):
        """Activate this component."""
        self._active = True
        if (not shallow) and \
           (self._is_container):
            for child in self.children():
                if isinstance(child, _ActiveObjectMixin):
                    child.activate(shallow=False)

    def deactivate(self, shallow=True):
        """Deactivate this component."""
        self._active = False
        if (not shallow) and \
           (self._is_container):
            for child in self.children():
                if isinstance(child, _ActiveObjectMixin):
                    child.deactivate(shallow=False)

# used frequently below, so I'm caching it here
_active_flag_name = "active"
class _SimpleContainerMixin(object):
    """
    A partial implementation of the IComponentContainer
    interface for implementations that store a single
    component category.

    Complete implementations need to set the _ctype property
    at the class level and declare the remaining required
    abstract properties of the IComponentContainer base
    class.

    Note that this implementation allows nested storage of
    other :class:`IComponentContainer` implementations that
    are defined with the same ctype.
    """
    __slots__ = ()

    def _prepare_for_add(self, key, obj):
        """This method must be called any time a new child
        is inserted into this container."""
        obj._parent = weakref.ref(self)
        obj._storage_key = key

    def _prepare_for_delete(self, obj):
        """This method must be called any time a new child
        is removed from this container."""
        obj._parent = None
        obj._storage_key = None

    def components(self,
                   active=None):
        """
        Generates an efficient traversal of all components
        stored under this container. Components are leaf
        nodes in a storage tree (not containers themselves,
        except for blocks).

        Args:
            active (:const:`True`/:const:`None`): Set to
                :const:`True` to indicate that only active
                objects should be included. The default
                value of :const:`None` indicates that all
                components (including those that have been
                deactivated) should be included. *Note*:
                This flag is ignored for any objects that do
                not have an active flag.

        Returns:
            iterator of objects in the storage tree
        """
        assert active in (None, True)

        # if not active, then no children can be active
        if (active is not None) and \
           not getattr(self, _active_flag_name, True):
            return

        for child in self.children():

            # check active status (if appropriate)
            if (active is not None) and \
               not getattr(child, _active_flag_name, True):
                continue

            if child._is_component:
                yield child
            else:
                for item in child.components(active=active):
                    yield item

    def preorder_traversal(self,
                           active=None):
        """
        Generates a preorder traversal of the storage tree.

        Args:
            active (:const:`True`/:const:`None`): Set to
                :const:`True` to indicate that only active
                objects should be included. The default
                value of :const:`None` indicates that all
                components (including those that have been
                deactivated) should be included. *Note*:
                This flag is ignored for any objects that do
                not have an active flag.

        Returns:
            iterator of objects in the storage tree
        """
        assert active in (None, True)

        # if not active, then no children can be active
        if (active is not None) and \
           not getattr(self, _active_flag_name, True):
            return

        yield self
        for child in self.children():

            # check active status (if appropriate)
            if (active is not None) and \
               not getattr(child, _active_flag_name, True):
                continue

            if child._is_component:
                # this is a leaf node
                yield child
            else:
                assert child._is_container
                for item in child.preorder_traversal(
                        active=active):
                    yield item

    def preorder_visit(self,
                       visit,
                       active=None):
        """
        Visits each node in the storage tree using a
        preorder traversal.

        Args:
            visit: A function that is called on each node in
                the storage tree. When the return value of
                the function evaluates to to :const:`True`,
                this indicates that the traversal should
                continue with the children of the current
                node; otherwise, the traversal does not go
                below the current node.
            active (:const:`True`/:const:`None`): Set to
                :const:`True` to indicate that only active
                objects should be included. The default
                value of :const:`None` indicates that all
                components (including those that have been
                deactivated) should be included. *Note*:
                This flag is ignored for any objects that do
                not have an active flag.
        """
        assert active in (None, True)

        # if not active, then no children can be active
        if (active is not None) and \
           not getattr(self, _active_flag_name, True):
            return

        if not visit(self):
            return
        for child in self.children():

            # check active status (if appropriate)
            if (active is not None) and \
               not getattr(child, _active_flag_name, True):
                continue

            if child._is_component:
                visit(child)
            else:
                assert child._is_container
                child.preorder_visit(visit,
                                     active=active)

    def postorder_traversal(self, active=None):
        """
        Generates a postorder traversal of the storage tree.

        Args:
            active (:const:`True`/:const:`None`): Set to
                :const:`True` to indicate that only active
                objects should be included. The default
                value of :const:`None` indicates that all
                components (including those that have been
                deactivated) should be included. *Note*:
                This flag is ignored for any objects that do
                not have an active flag.

        Returns:
            iterator of objects in storage tree
        """
        assert active in (None, True)

        # if not active, then no children can be active
        if (active is not None) and \
           not getattr(self, _active_flag_name, True):
            return

        for child in self.children():

            # check active status (if appropriate)
            if (active is not None) and \
               not getattr(child, _active_flag_name, True):
                continue

            if child._is_component:
                # this is a leaf node
                yield child
            else:
                assert child._is_container
                for item in child.postorder_traversal(
                        active=active):
                    yield item

        yield self

    def generate_names(self,
                       active=None,
                       descend_into=True,
                       convert=str,
                       prefix=""):
        """
        Generate a container of fully qualified names (up to
        this container) for objects stored under this
        container.

        Args:
            active (:const:`True`/:const:`None`): Set to
                :const:`True` to indicate that only active
                components should be included. The default
                value of :const:`None` indicates that all
                components (including those that have been
                deactivated) should be included. *Note*:
                This flag is ignored for any objects that do
                not have an active flag.
            descend_into (bool): Indicates whether or not to
                include subcomponents of any container
                objects that are not components. Default is
                :const:`True`.
            convert (function): A function that converts a
                storage key into a string
                representation. Default is str.
            prefix (str): A string to prefix names with.

        Returns:
            A component map that behaves as a dictionary
            mapping component objects to names.
        """
        assert active in (None, True)
        from pyomo.core.kernel.component_map import ComponentMap
        names = ComponentMap()

        # if not active, then no children can be active
        if (active is not None) and \
           not getattr(self, _active_flag_name, True):
            return names

        name_template = (prefix +
                         self._child_storage_delimiter_string +
                         self._child_storage_entry_string)
        for child in self.children():
            if (active is None) or \
               getattr(child, _active_flag_name, True):
                names[child] = (name_template
                                % convert(child.storage_key))
                if descend_into and child._is_container and \
                   (not child._is_component):
                    names.update(child.generate_names(
                        active=active,
                        descend_into=True,
                        convert=convert,
                        prefix=names[child]))
        return names
