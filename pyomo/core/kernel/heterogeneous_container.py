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
     _convert_descend_into,
     ICategorizedObjectContainer)

def heterogeneous_containers(node,
                             ctype=_no_ctype,
                             active=True,
                             descend_into=True):
    """
    A generator that yields all heterogeneous containers
    included in an object storage tree, including the root
    object. Heterogeneous containers are categorized objects
    with a category type different from their children.

    Args:
        node: The root object.
        ctype: Indicates the category of objects to
            include. The default value indicates that
            all categories should be included.
        active (:const:`True`/:const:`None`): Controls
            whether or not to filter the iteration to
            include only the active part of the storage
            tree. The default is :const:`True`. Setting this
            keyword to :const:`None` causes the active
            status of objects to be ignored.
        descend_into (bool, function): Indicates whether or
            not to descend into a heterogeneous
            container. Default is True, which is equivalent
            to `lambda x: True`, meaning all heterogeneous
            containers will be descended into.

    Returns:
        iterator of heterogeneous containers in the storage
        tree, include the root object.
    """
    assert active in (None, True)

    # if not active, then nothing below is active
    if (active is not None) and \
       (not node.active):
        return

    if not node.ctype._is_heterogeneous_container:
        return

    if not node._is_heterogeneous_container:
        assert node._is_container
        # a homogeneous container that stores
        # heterogeneous objects
        for obj in node.components(active=active):
            assert obj._is_heterogeneous_container
            for item in heterogeneous_containers(
                    obj,
                    ctype=ctype,
                    active=active,
                    descend_into=descend_into):
                yield item
        return

    # convert AML types into Kernel types (hack for the
    # solver interfaces)
    ctype = _convert_ctype.get(ctype, ctype)
    assert (ctype is _no_ctype) or \
        ctype._is_heterogeneous_container

    # convert descend_into to a function if
    # it is not already one
    descend_into = _convert_descend_into(descend_into)

    # a heterogeneous container
    if (ctype is _no_ctype) or \
       (node.ctype is ctype):
        yield node

    if not descend_into(node):
        return

    for child_ctype in node.child_ctypes():

        if not child_ctype._is_heterogeneous_container:
            continue

        for child in node.children(ctype=child_ctype):
            assert child._is_container

            if (active is not None) and \
               (not child.active):
                continue

            for obj in heterogeneous_containers(
                    child,
                    ctype=ctype,
                    active=active,
                    descend_into=descend_into):
                yield obj

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
                       active=True,
                       descend_into=True):
        """Returns the set of object category types that can
        be found under this container.

        Args:
            active (:const:`True`/:const:`None`): Controls
                whether or not to filter the iteration to
                include only the active part of the storage
                tree. The default is :const:`True`. Setting
                this keyword to :const:`None` causes the
                active status of objects to be ignored.
            descend_into (bool, function): Indicates whether
                or not to descend into a heterogeneous
                container. Default is True, which is
                equivalent to `lambda x: True`, meaning all
                heterogeneous containers will be descended
                into.

        Returns:
            A set of object category types
        """
        assert active in (None, True)

        ctypes = set()
        # if not active, then nothing below is active
        if (active is not None) and \
           (not self.active):
            return ctypes

        # convert descend_into to a function if
        # it is not already one
        descend_into = _convert_descend_into(descend_into)

        for child_ctype in self.child_ctypes():
            for obj in self.components(
                    ctype=child_ctype,
                    active=active,
                    descend_into=_convert_descend_into._false):
                ctypes.add(child_ctype)
                # just need 1 to appear in order to
                # count the child_ctype
                break

        if descend_into is _convert_descend_into._false:
            return ctypes

        # iterate over a copy of the set as we will
        # modify it in this loop
        for child_ctype in tuple(ctypes):
            if child_ctype._is_heterogeneous_container:
                for obj in self.components(
                        ctype=child_ctype,
                        active=active,
                        descend_into=_convert_descend_into._false):
                    assert obj._is_heterogeneous_container
                    if descend_into(obj):
                        ctypes.update(obj.collect_ctypes(
                            active=active,
                            descend_into=descend_into))

        return ctypes

    def child_ctypes(self, *args, **kwds):
        """Returns the set of child object category types
        stored in this container."""
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
                   active=True,
                   descend_into=True):
        """
        Generates an efficient traversal of all components
        stored under this container. Components are
        categorized objects that are either (1) not
        containers, or (2) are heterogeneous containers.

        Args:
            ctype: Indicates the category of components to
                include. The default value indicates that
                all categories should be included.
            active (:const:`True`/:const:`None`): Controls
                whether or not to filter the iteration to
                include only the active part of the storage
                tree. The default is :const:`True`. Setting
                this keyword to :const:`None` causes the
                active status of objects to be ignored.
            descend_into (bool, function): Indicates whether
                or not to descend into a heterogeneous
                container. Default is True, which is
                equivalent to `lambda x: True`, meaning all
                heterogeneous containers will be descended
                into.

        Returns:
            iterator of components in the storage tree
        """
        assert active in (None, True)

        # if not active, then nothing below is active
        if (active is not None) and \
           (not self.active):
            return

        # convert AML types into Kernel types (hack for the
        # solver interfaces)
        ctype = _convert_ctype.get(ctype, ctype)

        # convert descend_into to a function if
        # it is not already one
        descend_into = _convert_descend_into(descend_into)

        if ctype is _no_ctype:

            for child in self.children():

                if (active is not None) and \
                   (not child.active):
                    continue

                if not child._is_container:
                    yield child
                elif child._is_heterogeneous_container:
                    yield child
                    if descend_into(child):
                        for obj in child.components(
                                active=active,
                                descend_into=descend_into):
                            yield obj
                elif (descend_into is _convert_descend_into._false) or \
                     (not child.ctype._is_heterogeneous_container):
                    assert child._is_container
                    for obj in child.components(active=active):
                        yield obj
                else:
                    assert child._is_container
                    for obj in child.components(
                            active=active):
                        assert obj._is_heterogeneous_container
                        yield obj
                        if descend_into(obj):
                            for item in obj.components(
                                    active=active,
                                    descend_into=descend_into):
                                yield item

        else:

            for item in heterogeneous_containers(self,
                                                 active=active,
                                                 descend_into=descend_into):
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
