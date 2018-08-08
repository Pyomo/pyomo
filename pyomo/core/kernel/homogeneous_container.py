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
     ICategorizedObjectContainer)

class IHomogeneousContainer(ICategorizedObjectContainer):
    """
    A partial implementation of the ICategorizedObjectContainer
    interface for implementations that store a single
    category of objects and that uses the same category
    as the objects it stores.

    Complete implementations need to set the _ctype
    attribute and declare the remaining required abstract
    properties of the ICategorizedObjectContainer base
    class.

    Note that this implementation allows nested storage of
    other :class:`ICategorizedObjectContainer`
    implementations that are defined with the same ctype.
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
        stored under this container. Components are
        categorized objects that are either not containers,
        or are heterogeneous containers having a category
        type different from their children.

        Args:
            active (:const:`True`/:const:`None`): Set to
                :const:`True` to indicate that only active
                objects should be included. The default
                value of :const:`None` indicates that all
                components (including those that have been
                deactivated) should be included.

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
                deactivated) should be included.
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
            if (not child._is_container) or \
               child._is_heterogeneous_container:
                if (active is None) or \
                   child.active:
                    yield child
            else:
                assert child._is_container
                for obj in child.preorder_traversal(
                        active=active,
                        descend=descend):
                    yield obj
