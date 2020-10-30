#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.core.kernel.base import ICategorizedObjectContainer

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

    def components(self, active=True):
        """
        Generates an efficient traversal of all components
        stored under this container. Components are
        categorized objects that are either (1) not
        containers, or (2) are heterogeneous containers.

        Args:
            active (:const:`True`/:const:`None`): Controls
                whether or not to filter the iteration to
                include only the active part of the storage
                tree. The default is :const:`True`. Setting
                this keyword to :const:`None` causes the
                active status of objects to be ignored.

        Returns:
            iterator of components in the storage tree
        """
        assert active in (None, True)

        # if not active, then no children can be active
        if (active is not None) and \
           (not self.active):
            return

        for child in self.children():
            if (active is None) or \
               child.active:
                if (not child._is_container) or \
                   child._is_heterogeneous_container:
                        yield child
                else:
                    assert child._is_container
                    for obj in child.components(active=active):
                        yield obj
