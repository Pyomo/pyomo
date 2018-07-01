#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import abc
import copy
import weakref

def _not_implemented_property(*args, **kwds):
    raise NotImplementedError("This property is abstract")     #pragma:nocover

def _abstract_readwrite_property(**kwds):
    p = property(fget=_not_implemented_property,
                 fset=_not_implemented_property,
                 **kwds)
    return p

def _abstract_readonly_property(**kwds):
    p = property(fget=_not_implemented_property,
                 **kwds)
    return p

class _no_ctype(object):
    """The default argument for methods that accept a ctype."""
    pass

# This will be populated outside of core.kernel. It will map
# AML classes (which are the ctypes used by all of the
# solver interfaces) to Kernel classes
_convert_ctype = {}

class ICategorizedObject(object):
    """
    Interface for objects that maintain a weak reference to
    a parent storage object and have a category type.

    This class is abstract. It assumes any derived class
    declares the attributes below at the class or instance
    level (with or without __slots__):

    Attributes:
        _ctype: Stores the object's category type, which
            should be some class derived from ICategorizedObject.
        _parent: Stores a weak reference to the object's
            parent container or :const:`None`.
        _storage_key: Stores key this object can be accessed
            with through its parent container.
        _active (bool): Stores the active status of this
            object.
    """
    __slots__ = ()

    # These flags can be used by implementations
    # to avoid isinstance calls.
    _is_container = False
    """A flag used to indicate that the class is an instance
    of ICategorizedObjectContainer."""

    _is_heterogeneous_container = False
    """A flag used to indicate that the class is an instance
    of ICategorizedObjectContainer that stores objects with
    different category types than its own."""

    #
    # Interface
    #

    @property
    def ctype(self):
        """The object's category type."""
        return self._ctype

    @property
    def parent(self):
        """The object's parent (possibly None)."""
        if isinstance(self._parent, weakref.ReferenceType):
            return self._parent()
        else:
            return self._parent

    @property
    def storage_key(self):
        """The object's storage key within its parent"""
        return self._storage_key

    @property
    def active(self):
        """The active status of this object."""
        return self._active
    @active.setter
    def active(self, value):
        raise AttributeError(
            "Assignment not allowed. Use the "
            "(de)activate method")

    def activate(self):
        """Activate this object."""
        self._active = True

    def deactivate(self):
        """Deactivate this object."""
        self._active = False

    def getname(self,
                fully_qualified=False,
                name_buffer={}, # HACK: ignored (required to work with some solver interfaces, but that code should change soon)
                convert=str,
                relative_to=None):
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
        assert fully_qualified or \
            (relative_to is None)
        parent = self.parent
        if parent is None:
            return None

        key = self.storage_key
        name = parent._child_storage_entry_string % convert(key)
        if fully_qualified:
            parent_name = parent.getname(fully_qualified=True,
                                         relative_to=relative_to)
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

    def clone(self):
        """
        Returns a copy of this object with the parent
        pointer set to :const:`None`.

        A clone is almost equivalent to deepcopy except that
        any categorized objects encountered that are not
        descendents of this object will reference the same
        object on the clone.
        """
        save_parent, self._parent = self._parent, None
        try:
            new_block = copy.deepcopy(self,
                                      {'__categorized_object_scope__':
                                       {id(self): True, id(None): False}})
        finally:
            self._parent = save_parent
        return new_block

    #
    # The following method must be defined to allow
    # expressions and blocks to be correctly cloned.
    # (copied from JDS code in original component.py,
    # comments removed for now)
    #

    def __deepcopy__(self, memo):
        if '__categorized_object_scope__' in memo and \
                id(self) not in memo['__categorized_object_scope__']:
            _known = memo['__categorized_object_scope__']
            _new = []
            tmp = self.parent
            tmpId = id(tmp)
            while tmpId not in _known:
                _new.append(tmpId)
                tmp = tmp.parent
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

class ICategorizedObjectContainer(ICategorizedObject):
    """
    Interface for categorized containers of categorized
    objects.
    """
    _is_container = True
    _child_storage_delimiter_string = None
    _child_storage_entry_string = None
    __slots__ = ()

    #
    # Interface
    #

    def activate(self, shallow=True):
        """Activate this container."""
        super(ICategorizedObjectContainer, self).activate()
        if not shallow:
            for child in self.children():
                if not child._is_container:
                    child.activate()
                else:
                    child.activate(shallow=False)

    def deactivate(self, shallow=True):
        """Deactivate this container."""
        super(ICategorizedObjectContainer, self).deactivate()
        if not shallow:
            for child in self.children():
                if not child._is_container:
                    child.deactivate()
                else:
                    child.deactivate(shallow=False)

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
    def components(self, *args, **kwds):
        """A generator over the set of components stored
        under this container."""
        raise NotImplementedError     #pragma:nocover

    @abc.abstractmethod
    def preorder_traversal(self, *args, **kwds):
        """A generator over all descendents in prefix order."""
        raise NotImplementedError     #pragma:nocover

class IHomogeneousContainer(ICategorizedObjectContainer):
    """
    A partial implementation of the ICategorizedObjectContainer
    interface for implementations that store a single
    category of objects and that uses the same category
    as the objects it stores.

    Complete implementations need to set the _ctype property
    at the class level and declare the remaining required
    abstract properties of the ICategorizedObjectContainer base
    class.

    Note that this implementation allows nested storage of
    other :class:`ICategorizedObjectContainer` implementations that
    are defined with the same ctype.
    """
    __slots__ = ()

    #
    # Define the ICategorizedObjectContainer abstract methods
    #

    #def child(self, *args, **kwds):
    # ... not defined here

    #def children(self, *args, **kwds):
    # ... not defined here

    def components(self,
                   active=None):
        """
        Generates an efficient traversal of all components
        stored under this container. Components are ...TODO

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
           (not self.active):
            return

        for child in self.children():

            if (not child._is_container) or \
               child._is_heterogeneous_container:
                # it is either not a container or
                # it is a hetergeneous container, so
                # we treat it is a leaf node
                if (active is None) or \
                   child.active:
                    yield child
            else:
                assert child._is_container
                for obj in child.components(active=active):
                    yield obj

    def preorder_traversal(self,
                           active=None,
                           descend=None):
        """
        A generator that visits each node in the storage
        tree using a preorder traversal.

        Args:
            active (:const:`True`/:const:`None`): Set to
                :const:`True` to indicate that only active
                objects should be included. The default
                value of :const:`None` indicates that all
                components (including those that have been
                deactivated) should be included. *Note*:
                This flag is ignored for any objects that do
                not have an active flag.
            descend: A function that can be used to control
                if a container object should be descended
                into. When the return value is False, the
                traversal will not continue into children
                of the container.
        """
        assert active in (None, True)

        # if not active, then nothing below is active
        if (active is not None) and \
           (not self.active):
            return

        if (descend is not None) and \
           (not descend(self)):
            yield self
            return
        else:
            yield self

        for child in self.children():

            if not child._is_container:
                if (active is None) or \
                   child.active:
                    yield child
            elif child._is_heterogeneous_container:
                if (active is None) or \
                   child.active:
                    if descend is not None:
                        descend(child)
                    yield child
            else:
                for obj in child.preorder_traversal(
                        active=active,
                        descend=descend):
                    yield obj

class IHeterogeneousContainer(ICategorizedObjectContainer):
    """
    A partial implementation of the ICategorizedObjectContainer
    interface for implementations that store multiple
    categories of objects.

    Complete implementations need to set the _ctype property
    at the class level and declare the remaining required
    abstract properties of the ICategorizedObjectContainer base
    class.
    """
    __slots__ = ()
    _is_heterogeneous_container = True

    #
    # Interface
    #

    def collect_ctypes(self,
                       active=None,
                       descend_into=True):
        """Returns the set of child object category types
        under this container.

        Args:
            active (:const:`True`/:const:`None`): Set to
                :const:`True` to indicate that only active
                objects should be included. The default
                value of :const:`None` indicates that all
                components (including those that have been
                deactivated) should be included. *Note*:
                This flag is ignored for any objects that do
                not have an active flag.
            descend_into (bool): Indicates whether or not to
                descend into heterogeneous containers.
                Default is True.

        Returns:
            A set object category types
        """
        assert active in (None, True)
        ctypes = set()
        # if not active, then nothing below is active
        if (active is not None) and \
           (not self.active):
            return ctypes

        for child_ctype in self.child_ctypes():
            for obj in self.components(
                    ctype=child_ctype,
                    active=active,
                    descend_into=False):
                ctypes.add(child_ctype)
                # just need 1 to appear in order to
                # count the child_ctype
                break

        if descend_into:
            for child_ctype in tuple(ctypes):
                if child_ctype._is_heterogeneous_container:
                    for obj in self.components(
                            ctype=child_ctype,
                            active=active,
                            descend_into=False):
                        assert obj._is_container
                        ctypes.update(obj.collect_ctypes(
                            active=active,
                            descend_into=True))

        return ctypes

    def child_ctypes(self, *args, **kwds):
        """Returns the set of child object category types
        for this container."""
        raise NotImplementedError     #pragma:nocover

    #
    # Define the ICategorizedObjectContainer abstract methods
    #

    #def child(self, *args, **kwds):
    # ... not defined here

    #def children(self, *args, **kwds):
    # ... not defined here

    def components(self,
                   ctype=_no_ctype,
                   active=None,
                   descend_into=True):
        """
        Generates an efficient traversal of all components
        stored under this container. Components are ...TODO

        Args:
            ctype: Indicates the category of components to
                include. The default value indicates that
                all categories should be included.
            active (:const:`True`/:const:`None`): Set to
                :const:`True` to indicate that only active
                objects should be included. The default
                value of :const:`None` indicates that all
                components (including those that have been
                deactivated) should be included. *Note*:
                This flag is ignored for any objects that do
                not have an active flag.
            descend_into (bool): Indicates whether or not to
                descend into heterogeneous containers.
                Default is True.

        Returns:
            iterator of objects in the storage tree
        """
        assert active in (None, True)
        # if not active, then nothing below is active
        if (active is not None) and \
           (not self.active):
            return

        # convert AML types into Kernel types (hack for the
        # solver interfaces)
        ctype = _convert_ctype.get(ctype, ctype)

        if ctype is _no_ctype:

            for child in self.children():

                if not child._is_container:
                    if (active is None) or \
                       child.active:
                        yield child
                elif child._is_heterogeneous_container:
                    if (active is None) or \
                       child.active:
                        yield child
                        if descend_into:
                            for obj in child.components(
                                    active=active,
                                    descend_into=True):
                                yield obj
                elif descend_into and \
                     child.ctype._is_heterogeneous_container:
                    assert child._is_container
                    for obj in child.components(
                            active=active):
                        assert obj._is_container and \
                            obj._is_heterogeneous_container
                        yield obj
                        for item in obj.components(
                                active=active,
                                descend_into=True):
                            yield item
                else:
                    assert child._is_container
                    for obj in child.components(active=active):
                        yield obj

        else:

            # Generate components from immediate children first
            for child in self.children(ctype=ctype):

                if (not child._is_container) or \
                   child._is_heterogeneous_container:
                    if (active is None) or \
                       child.active:
                        yield child
                else:
                    assert child._is_container
                    for obj in child.components(active=active):
                        yield obj

            if descend_into:
                for child_ctype in self.child_ctypes():
                    if child_ctype._is_heterogeneous_container:
                        # yield components on all
                        # heterogeneous containers
                        for child in self.children(ctype=child_ctype):
                            assert child._is_container
                            if not child._is_heterogeneous_container:
                                for obj in child.components(
                                        active=active):
                                    assert obj.ctype is child_ctype
                                    assert obj._is_container and \
                                        obj._is_heterogeneous_container
                                    for item in obj.components(
                                            ctype=ctype,
                                            active=active,
                                            descend_into=True):
                                        yield item
                            else:
                                for obj in child.components(
                                        ctype=ctype,
                                        active=active,
                                        descend_into=True):
                                    yield obj

    def preorder_traversal(self,
                           ctype=_no_ctype,
                           active=None,
                           descend=None):
        """
        A generator that visits each node in the storage
        tree using a preorder traversal. This includes all
        components and all component containers (optionally)
        matching the requested type.

        Args:
            ctype: Indicates the category of components to
                include. The default value indicates that
                all categories should be included.
            active (:const:`True`/:const:`None`): Set to
                :const:`True` to indicate that only active
                objects should be included. The default
                value of :const:`None` indicates that all
                components (including those that have been
                deactivated) should be included. *Note*:
                This flag is ignored for any objects that do
                not have an active flag.
            descend: A function that can be used to control
                if a container object should be descended
                into. When the return value is False, the
                traversal will not continue into children
                of the container.

        Returns:
            iterator of objects in the storage tree
        """
        assert active in (None, True)
        # if not active, then nothing below is active
        if (active is not None) and \
           (not self.active):
            return

        # convert AML types into Kernel types (hack for the
        # solver interfaces)
        ctype = _convert_ctype.get(ctype, ctype)

        if (descend is not None) and \
           (not descend(self)):
            yield self
            return
        else:
            yield self

        for child in self.children():
            child_ctype = child.ctype
            if not child._is_container:
                # not a container
                if (active is None) or \
                   child.active:
                    if (ctype is _no_ctype) or \
                       (child_ctype is ctype):
                        yield child
            elif child._is_heterogeneous_container:
                # a non-homegenous container, so use
                # its traversal method
                for obj in child.preorder_traversal(
                        ctype=ctype,
                        active=active,
                        descend=descend):
                    yield obj
            else:
                # a homogeneous container
                if child_ctype._is_heterogeneous_container:
                    def descend_(obj):
                        if obj._is_heterogeneous_container or \
                           (descend is None):
                            return True
                        else:
                            return descend(obj)
                    for obj in child.preorder_traversal(
                            active=active,
                            descend=descend_):
                        if not obj._is_heterogeneous_container:
                            yield obj
                        else:
                            # a non-homegenous container, so use
                            # its traversal method
                            for item in obj.preorder_traversal(
                                    ctype=ctype,
                                    active=active,
                                    descend=descend):
                                yield item
                elif (ctype is _no_ctype) or \
                     (child_ctype is ctype):
                    for obj in child.preorder_traversal(
                            active=active,
                            descend=descend):
                        yield obj
