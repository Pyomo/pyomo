#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.core.kernel.base import \
    (_no_ctype,
     _convert_ctype,
     ICategorizedObjectContainer)

class IHeterogeneousContainer(ICategorizedObjectContainer):
    """
    A partial implementation of the ICategorizedObjectContainer
    interface for implementations that store multiple
    categories of objects.

    Complete implementations need to set the _ctype
    attribute and declare the remaining required abstract
    properties of the ICategorizedObjectContainer base
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
                deactivated) should be included.
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
        stored under this container. Components are
        categorized objects that are either not containers,
        or are heterogeneous containers having a category
        type different from their children.

        Args:
            ctype: Indicates the category of components to
                include. The default value indicates that
                all categories should be included.
            active (:const:`True`/:const:`None`): Set to
                :const:`True` to indicate that only active
                objects should be included. The default
                value of :const:`None` indicates that all
                components (including those that have been
                deactivated) should be included.
            descend_into (bool): Indicates whether or not to
                descend into heterogeneous containers.
                Default is True.

        Returns:
            iterator of objects in the storage tree
        """
        assert active in (None, True)
        # convert AML types into Kernel types (hack for the
        # solver interfaces)
        ctype = _convert_ctype.get(ctype, ctype)

        if ctype is _no_ctype:

            # if not active, then nothing below is active
            if (active is not None) and \
               (not self.active):
                return

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

            if not descend_into:
                # if not active, then nothing below is active
                if (active is not None) and \
                   (not self.active):
                    return
                items = (self,)
            else:
                items = self.heterogeneous_containers(active=active,
                                                      descend_into=True)

            for item in items:
                for child in item.children(ctype=ctype):
                    if (not child._is_container) or \
                       child._is_heterogeneous_container:
                        if (active is None) or \
                           child.active:
                            yield child
                    else:
                        assert child._is_container
                        for obj in child.components(active=active):
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
                deactivated) should be included.
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
                # a heterogeneous container, so use
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
                            # a heterogeneous container, so use
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

    def heterogeneous_containers(self,
                                 ctype=_no_ctype,
                                 active=None,
                                 descend_into=True):
        """
        A generator that visits each heterogeneous container
        in the storage tree. Heterogeneous containers are
        categorized objects, such as blocks, with a category
        type different from their children.

        Args:
            ctype: Indicates the category of objects to
                include. The default value indicates that
                all categories should be included.
            active (:const:`True`/:const:`None`): Set to
                :const:`True` to indicate that only active
                objects should be included. The default
                value of :const:`None` indicates that all
                components (including those that have been
                deactivated) should be included.
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

        if (ctype is _no_ctype) or \
           (self.ctype is ctype):
            yield self

        for child_ctype in self.child_ctypes():

            if not child_ctype._is_heterogeneous_container:
                continue

            for child in self.children(ctype=child_ctype):
                assert child._is_container
                if child._is_heterogeneous_container:
                    if not descend_into:
                        if (active is None) or \
                           child.active:
                            if (ctype is _no_ctype) or \
                               (child_ctype is ctype):
                                yield child
                    else:
                        for obj in child.heterogeneous_containers(
                                ctype=ctype,
                                active=active,
                                descend_into=True):
                            yield obj
                else:
                    # a homogeneous container
                    for obj in child.components(active=active):
                        assert obj._is_heterogeneous_container
                        if not descend_into:
                            if (ctype is _no_ctype) or \
                               (child_ctype is ctype):
                                yield obj
                        else:
                            for item in obj.heterogeneous_containers(
                                    ctype=ctype,
                                    active=active,
                                    descend_into=True):
                                yield item
