#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import copy
import abc
import logging
import weakref
import math
from collections import defaultdict
try:
    from collections import OrderedDict
except ImportError:                         #pragma:nocover
    from ordereddict import OrderedDict

from pyomo.core.kernel.component_interface import \
    (IActiveObject,
     ICategorizedObject,
     IComponent,
     IComponentContainer,
     _ActiveComponentContainerMixin)
from pyomo.core.kernel.component_objective import IObjective
from pyomo.core.kernel.component_variable import IVariable, variable
from pyomo.core.kernel.component_constraint import IConstraint
from pyomo.core.kernel.component_dict import ComponentDict
from pyomo.core.kernel.component_tuple import ComponentTuple
from pyomo.core.kernel.component_list import ComponentList
from pyomo.core.kernel.component_map import ComponentMap
from pyomo.core.kernel.component_suffix import import_suffix_generator
from pyomo.core.kernel.symbol_map import SymbolMap
import pyomo.opt

import six
from six import itervalues, iteritems

logger = logging.getLogger('pyomo.core')

_no_ctype = object()

# used frequently in this file,
# so I'm caching it here
_active_flag_name = "active"

class IBlockStorage(IComponent,
                    IComponentContainer,
                    _ActiveComponentContainerMixin):
    """A container that stores multiple types.

    This class is abstract, but it partially implements the
    :class:`ICategorizedObject` interface by defining the
    following attributes:

    Attributes:
        _is_component: :const:`True`
        _is_container: :const:`True`
    """
    _is_component = True
    _is_container = True
    _child_storage_delimiter_string = "."
    _child_storage_entry_string = "%s"
    __slots__ = ()

    #
    # These methods are already declared abstract on
    # IComponentContainer, but we redeclare them here to
    # point out that they can accept a ctype
    #

    @abc.abstractmethod
    def children(self, *args, **kwds):
        raise NotImplementedError     #pragma:nocover

    @abc.abstractmethod
    def components(self,  *args, **kwds):
        raise NotImplementedError     #pragma:nocover

    #
    # Interface
    #

    def clone(self):
        """
        Clones this block. Returns a new block with whose
        parent pointer is set to :const:`None`. Any
        components encountered that are descendents of this
        block will be deepcopied, otherwise a reference to
        the original component is retained.
        """
        save_parent, self._parent = self._parent, None
        try:
            new_block = copy.deepcopy(
                self, {
                    '__block_scope__': {id(self): True, id(None): False},
                    '__paranoid__': False,
                    })
        except:
            new_block = copy.deepcopy(
                self, {
                    '__block_scope__': {id(self): True, id(None): False},
                    '__paranoid__': True,
                    })
        finally:
            self._parent = save_parent

        return new_block

    @abc.abstractmethod
    def blocks(self, *args, **kwds):
        raise NotImplementedError     #pragma:nocover

    @abc.abstractmethod
    def collect_ctypes(self, *args, **kwds):
        raise NotImplementedError     #pragma:nocover

class _block_base(object):
    """
    A base class shared by :class:`block` and
    :class:`tiny_block` that implements a few
    :class:`IBlockStorage` abstract methods.
    """
    __slots__ = ()

    # Blocks do not change their active status
    # based on changes in status of their children
    def _increment_active(self):
        pass
    def _decrement_active(self):
        pass

    def activate(self,
                 shallow=True,
                 descend_into=False,
                 _from_parent_=False):
        """Activates this block.

        Args:
            shallow (bool): If :const:`False`, all children
                of the block will be activated. By default,
                the active status of children are not
                changed.
            descend_into (bool): Indicates whether or not to
                perform the same action on sub-blocks. The
                default is :const:`False`, as a shallow
                operation on the top-level block is
                sufficient.
        """
        block_ctype = self.ctype
        if (not self.active) and \
           (not _from_parent_):
            # inform the parent
            parent = self.parent
            if parent is not None:
                parent._increment_active()
        self._active = True
        if not shallow:
            for child in self.children():
                if isinstance(child, IActiveObject):
                    child.activate(_from_parent_=True)
        if descend_into:
            for obj in self.components(ctype=block_ctype):
                obj.activate(shallow=shallow,
                             descend_into=False,
                             _from_parent_=True)

    def deactivate(self,
                   shallow=True,
                   descend_into=False,
                   _from_parent_=False):
        """Deactivates this block.

        Args:
            shallow (bool): If :const:`False`, all children
                of the block will be deactivated. By
                default, the active status of children are
                not changed, but they become effectively
                inactive for anything above this block.
            descend_into (bool): Indicates whether or not to
                perform the same action on sub-blocks. The
                default is :const:`False`, as a shallow
                operation on the top-level block is
                sufficient.
        """
        block_ctype = self.ctype
        if self.active and \
           (not _from_parent_):
            # inform the parent
            parent = self.parent
            if parent is not None:
                parent._decrement_active()
        self._active = False
        if not shallow:
            for child in self.children():
                if isinstance(child, IActiveObject):
                    child.deactivate(_from_parent_=True)
        if descend_into:
            for obj in self.components(ctype=block_ctype):
                obj.deactivate(shallow=shallow,
                               descend_into=False,
                               _from_parent_=True)

    def child(self, key):
        """Get the child object associated with a given
        storage key for this container.

        Raises:
            KeyError: if the argument is not a storage key
                for any children of this container
        """
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(str(key))

    def preorder_traversal(self,
                           ctype=_no_ctype,
                           active=None,
                           include_all_parents=True,
                           return_key=False,
                           root_key=None):
        """
        Generates a preorder traversal of the storage
        tree. This includes all components and all component
        containers (optionally) matching the requested type.

        Args:
            ctype: Indicate the type of components to
                include. The default value indicates that
                all types should be included.
            active (:const:`True`/:const:`None`): Set to
                :const:`True` to indicate that only active
                objects should be included. The default
                value of :const:`None` indicates that all
                components (including those that have been
                deactivated) should be included. *Note*:
                This flag is ignored for any objects that do
                not have an active flag.
            include_all_parents (bool): Indicates if all
                parent containers (such as blocks and simple
                block containers) should be included in the
                traversal even when the :attr:`ctype`
                keyword is set to something that is not
                Block. Default is :const:`True`.
            return_key (bool): Set to :const:`True` to
                indicate that the return type should be a
                2-tuple consisting of the local storage key
                of the object within its parent and the
                object itself. By default, only the objects
                are returned.
            root_key: The key to return with this object.
                Ignored when :attr:`return_key` is
                :const:`False`.

        Returns:
            iterator of objects or (key,object) tuples
        """
        assert active in (None, True)
        block_ctype = self.ctype

        # if this block is not active, then nothing below it
        # can be active
        if active and (not self.active):
            return

        if include_all_parents or \
           (ctype is _no_ctype) or \
           (ctype is block_ctype):
            if return_key:
                yield root_key, self
            else:
                yield self
        for key, child in self.children(return_key=True):

            # check for appropriate ctype
            if (ctype is not _no_ctype) and \
               (child.ctype is not ctype) and \
               (child.ctype is not block_ctype):
                continue

            # check active status (if appropriate)
            if (active is not None) and \
               not getattr(child, _active_flag_name, True):
                continue

            if not child._is_container:
                # not a container (thus, also not a block),
                # so it is a leaf node
                if return_key:
                    yield key, child
                else:
                    yield child
            elif not child._is_component:
                # a container and not a component (thus, not a block)
                if child.ctype is block_ctype:
                    # this is a simple container of blocks
                    # Note: we treat the simple block
                    #   containers differently because we
                    #   want to propagate the ctype filter
                    #   beyond the simple container methods
                    #   (which don't have a ctype keyword)
                    for obj_key, obj in child.preorder_traversal(
                            active=active,
                            return_key=True,
                            root_key=key):
                        if not obj._is_component:
                            # a container of blocks
                            if (ctype is _no_ctype) or \
                               (ctype is block_ctype) or \
                               include_all_parents:
                                if return_key:
                                    yield obj_key, obj
                                else:
                                    yield obj
                        else:
                            # a block
                            for item in obj.preorder_traversal(
                                    ctype=ctype,
                                    active=active,
                                    include_all_parents=include_all_parents,
                                    return_key=return_key,
                                    root_key=obj_key):
                                yield item

                else:
                    # a simple container, call its traversal method
                    for item in child.preorder_traversal(
                            active=active,
                            return_key=return_key,
                            root_key=key):
                        yield item
            else:
                # a block, call its traversal method
                for item in child.preorder_traversal(
                        ctype=ctype,
                        active=active,
                        include_all_parents=include_all_parents,
                        return_key=return_key,
                        root_key=key):
                    yield item

    def preorder_visit(self,
                       visit,
                       ctype=_no_ctype,
                       active=None,
                       include_all_parents=True,
                       include_key=False,
                       root_key=None):
        """
        Visits each node in the storage tree using a
        preorder traversal. This includes all components and
        all component containers (optionally) matching the
        requested type.

        Args:
            visit: A function that is called on each node in
                the storage tree. When the
                :attr:`include_key` keyword is
                :const:`False`, the function signature
                should be `visit(node) -> [True|False]`.
                When the :attr:`include_key` keyword is
                :const:`True`, the function signature should
                be `visit(key,node) -> [True|False]`. When
                the return value of the function evaluates
                to to :const:`True`, this indicates that the
                traversal should continue with the children
                of the current node; otherwise, the
                traversal does not go below the current
                node.
            ctype: Indicate the type of components to
                include. The default value indicates that
                all types should be included.
            active (:const:`True`/:const:`None`): Set to
                :const:`True` to indicate that only active
                objects should be included. The default
                value of :const:`None` indicates that all
                components (including those that have been
                deactivated) should be included. *Note*:
                This flag is ignored for any objects that do
                not have an active flag.
            include_all_parents (bool): Indicates if all
                parent containers (such as blocks and simple
                block containers) should be included in the
                traversal even when the :attr:`ctype`
                keyword is set to something that is not
                Block. Default is :const:`True`.
            include_key (bool): Set to :const:`True` to
                indicate that 2 arguments should be passed
                to the visit function, with the first being
                the local storage key of the object within
                its parent and the second being the object
                itself. By default, only the objects are
                passed to the function.
            root_key: The key to pass with this object.
                Ignored when :attr:`include_key` is
                :const:`False`.
        """
        assert active in (None, True)
        block_ctype = self.ctype

        # if this block is not active, then nothing below it
        # can be active
        if active and (not self.active):
            return

        go = True
        if include_all_parents or \
           (ctype is _no_ctype) or \
           (ctype is block_ctype):
            if include_key:
                go = visit(root_key, self)
            else:
                go = visit(self)
        if not go:
            return
        for key, child in self.children(return_key=True):

            # check for appropriate ctype
            if (ctype is not _no_ctype) and \
               (child.ctype is not ctype) and \
               (child.ctype is not block_ctype):
                continue

            # check active status (if appropriate)
            if (active is not None) and \
               not getattr(child, _active_flag_name, True):
                continue

            if not child._is_container:
                # not a container (thus, also not a block),
                # so it is a leaf node
                if include_key:
                    visit(key, child)
                else:
                    visit(child)
            elif not child._is_component:
                # a container and not a component (thus, not a block)
                if child.ctype is block_ctype:
                    # this is a simple container of blocks
                    # Note: we treat the simple block
                    #   containers differently because we
                    #   want to propagate the ctype filter
                    #   beyond the simple container methods
                    #   (which don't have a ctype keyword)
                    stack = [(key,child)]
                    while len(stack):
                        obj_key, obj = stack.pop()

                        # check active status (if appropriate)
                        if (active is not None) and \
                           not getattr(obj, _active_flag_name, True):
                            continue

                        if not obj._is_component:
                            # a simple container of blocks
                            go = True
                            if (ctype is _no_ctype) or \
                               (ctype is block_ctype) or \
                               include_all_parents:
                                if include_key:
                                    go = visit(obj_key, obj)
                                else:
                                    go = visit(obj)
                            if go:
                                stack.extend(
                                    obj.children(return_key=True))
                        else:
                            # a block
                            obj.preorder_visit(
                                visit,
                                ctype=ctype,
                                active=active,
                                include_all_parents=include_all_parents,
                                include_key=include_key,
                                root_key=obj_key)

                else:
                    # a simple container, call its visit method
                    child.preorder_visit(
                        visit,
                        active=active,
                        include_key=include_key,
                        root_key=key)
            else:
                # a block, call its visit method
                child.preorder_visit(
                    visit,
                    ctype=ctype,
                    active=active,
                    include_all_parents=include_all_parents,
                    include_key=include_key,
                    root_key=key)

    def postorder_traversal(self,
                            ctype=_no_ctype,
                            active=None,
                            include_all_parents=True,
                            return_key=False,
                            root_key=None):
        """
        Generates a postorder traversal of the storage
        tree. This includes all components and all component
        containers (optionally) matching the requested type.

        Args:
            ctype: Indicate the type of components to
                include. The default value indicates that
                all types should be included.
            active (:const:`True`/:const:`None`): Set to
                :const:`True` to indicate that only active
                objects should be included. The default
                value of :const:`None` indicates that all
                components (including those that have been
                deactivated) should be included. *Note*:
                This flag is ignored for any objects that do
                not have an active flag.
            include_all_parents (bool): Indicates if all
                parent containers (such as blocks and simple
                block containers) should be included in the
                traversal even when the :attr:`ctype`
                keyword is set to something that is not
                Block. Default is :const:`True`.
            return_key (bool): Set to :const:`True` to
                indicate that the return type should be a
                2-tuple consisting of the local storage key
                of the object within its parent and the
                object itself. By default, only the objects
                are returned.
            root_key: The key to return with this object.
                Ignored when :attr:`return_key` is
                :const:`False`.

        Returns:
            iterator of objects or (key,object) tuples
        """
        assert active in (None, True)
        block_ctype = self.ctype

        # if this block is not active, then nothing below it
        # can be active
        if active and (not self.active):
            return

        for key, child in self.children(return_key=True):

            # check for appropriate ctype
            if (ctype is not _no_ctype) and \
               (child.ctype is not ctype) and \
               (child.ctype is not block_ctype):
                continue

            # check active status (if appropriate)
            if (active is not None) and \
               not getattr(child, _active_flag_name, True):
                continue

            if not child._is_container:
                # not a container (thus, also not a block),
                # so it is a leaf node
                if return_key:
                    yield key, child
                else:
                    yield child
            elif not child._is_component:
                # a container and not a component (thus, not a block)
                if child.ctype is block_ctype:
                    # this is a simple container of blocks
                    # Note: we treat the simple block
                    #   containers differently because we
                    #   want to propagate the ctype filter
                    #   beyond the simple container methods
                    #   (which don't have a ctype keyword)
                    for obj_key, obj in child.postorder_traversal(
                            active=active,
                            return_key=True,
                            root_key=key):
                        if not obj._is_component:
                            # a container of blocks
                            if (ctype is _no_ctype) or \
                               (ctype is block_ctype) or \
                               include_all_parents:
                                if return_key:
                                    yield obj_key, obj
                                else:
                                    yield obj
                        else:
                            # a block
                            for item in obj.postorder_traversal(
                                    ctype=ctype,
                                    active=active,
                                    include_all_parents=include_all_parents,
                                    return_key=return_key,
                                    root_key=obj_key):
                                yield item

                else:
                    # a simple container, call its traversal method
                    for item in child.postorder_traversal(
                            active=active,
                            return_key=return_key,
                            root_key=key):
                        yield item
            else:
                # a block, call its traversal method
                for item in child.postorder_traversal(
                        ctype=ctype,
                        active=active,
                        include_all_parents=include_all_parents,
                        return_key=return_key,
                        root_key=key):
                    yield item

        if include_all_parents or \
           (ctype is _no_ctype) or \
           (ctype is block_ctype):
            if return_key:
                yield root_key, self
            else:
                yield self

    def components(self,
                   ctype=_no_ctype,
                   active=None,
                   return_key=False,
                   descend_into=True):
        """
        Generates an efficient traversal of all components
        stored under this block. Components are leaf nodes
        in a storage tree (not containers themselves, except
        for blocks).

        Args:
            ctype: Indicate the type of components to
                include. The default value indicates that
                all types should be included.
            active (:const:`True`/:const:`None`): Set to
                :const:`True` to indicate that only active
                objects should be included. The default
                value of :const:`None` indicates that all
                components (including those that have been
                deactivated) should be included. *Note*:
                This flag is ignored for any objects that do
                not have an active flag.
            return_key (bool): Set to :const:`True` to
                indicate that the return type should be a
                2-tuple consisting of the local storage key
                of the object within its parent and the
                object itself. By default, only the objects
                are returned.
            descend_into (bool): Indicates whether or not to
                include components on sub-blocks. Default is
                :const:`True`.

        Returns:
            iterator of objects or (key,object) tuples
        """

        assert active in (None, True)
        block_ctype = self.ctype

        # if this block is not active, then nothing below it
        # can be active
        if active and (not self.active):
            return

        # Generate components from immediate children first
        for child_key, child in self.children(ctype=ctype, return_key=True):

            # check active status (if appropriate)
            if (active is not None) and \
               not getattr(child, _active_flag_name, True):
                continue

            if child._is_component:
                # child is a component (includes blocks), so yield it
                if return_key:
                    yield child_key, child
                else:
                    yield child
            else:
                assert child._is_container
                # child is a container (but not a block)
                if (active is not None) and \
                   isinstance(child, _ActiveComponentContainerMixin):
                    for component_key, component in child.components(return_key=True):
                        if getattr(component,
                                   _active_flag_name,
                                   True):
                            if return_key:
                                yield component_key, component
                            else:
                                yield component
                else:
                    for item in child.components(return_key=return_key):
                        yield item

        if descend_into:
            # now recurse into subblocks
            for child in self.children(ctype=block_ctype):

                # check active status (if appropriate)
                if (active is not None) and \
                   not getattr(child, _active_flag_name, True):
                    continue

                if child._is_component:
                    # child is a block
                    for item in child.components(
                            ctype=ctype,
                            active=active,
                            return_key=return_key,
                            descend_into=descend_into):
                        yield item
                else:
                    # child is a container of blocks,
                    # but not a block itself
                    for _comp in child.components():
                        if (active is None) or \
                           getattr(_comp,
                                   _active_flag_name,
                                   True):
                            for item in _comp.components(
                                    ctype=ctype,
                                    active=active,
                                    return_key=return_key,
                                    descend_into=descend_into):
                                yield item

    def blocks(self,
               active=None,
               descend_into=True):
        """
        Generates a traversal of all blocks associated with
        this one (including itself). This method yields
        identical behavior to calling the components()
        method with ctype=Block, except that this block is
        included (as the first item in the generator).
        """
        assert active in (None, True)
        block_ctype = self.ctype

        # if this block is not active, then nothing below it
        # can be active
        if active and (not self.active):
            return

        yield self
        for component in self.components(ctype=block_ctype,
                                         active=active,
                                         descend_into=descend_into):
            yield component

    def generate_names(self,
                       ctype=_no_ctype,
                       active=None,
                       descend_into=True,
                       convert=str,
                       prefix=""):
        """
        Generate a container of fully qualified names (up to
        this block) for objects stored under this block.

        This function is useful in situations where names
        are used often, but they do not need to be
        dynamically regenerated each time.

        Args:
            ctype: Indicate the type of components to
                include. The default value indicates that
                all types should be included.
            active (:const:`True`/:const:`None`): Set to
                :const:`True` to indicate that only active
                components should be included. The default
                value of :const:`None` indicates that all
                components (including those that have been
                deactivated) should be included. *Note*:
                This flag is ignored for any objects that do
                not have an active flag.
            descend_into (bool): Indicates whether or not to
                include components on sub-blocks. Default is
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

        names = ComponentMap()

        # if this block is not active, then nothing below it
        # can be active
        if active and (not self.active):
            return names

        if descend_into:
            traversal = self.preorder_traversal(ctype=ctype,
                                                active=active,
                                                include_all_parents=True,
                                                return_key=True)
            # skip the root (this block)
            six.next(traversal)
        else:
            traversal = self.children(ctype=ctype,
                                      return_key=True)
        for key, obj in traversal:
            parent = obj.parent
            name = parent._child_storage_entry_string % convert(key)
            if parent is not self:
                names[obj] = (names[parent] +
                              parent._child_storage_delimiter_string +
                              name)
            else:
                names[obj] = prefix + name

        return names

    def write(self,
              filename,
              format=None,
              _solver_capability=None,
              _called_by_solver=False,
              **kwds):
        """
        Write the model to a file, with a given format.

        Args:
            filename (str): The name of the file to write.
            format: The file format to use. If this is not
                specified, the file format will be inferred
                from the filename suffix.
            **kwds: Additional keyword options passed to the
                model writer.

        Returns:
            a :class:`SymbolMap`
        """
        #
        # Guess the format if none is specified
        #
        if format is None:
            format = pyomo.opt.base.guess_format(filename)
        problem_writer = pyomo.opt.WriterFactory(format)

        # TODO: I have no idea how to properly check if the
        #       WriterFactory lookup failed. When it does
        #       fail, it seems to return something of type:
        # 'pyutilib.component.core.core.PluginFactoryFunctor'
        #       which is not a class in the global namespace
        #       of that module. So for now, I am simply
        #       checking for a few methods that exist on this
        #       strange class.
        if (problem_writer is None) or \
           (hasattr(problem_writer, 'get_class') and \
            hasattr(problem_writer, 'services')):
            raise ValueError(
                "Cannot write model in format '%s': no model "
                "writer registered for that format"
                % str(format))

        if _solver_capability is None:
            _solver_capability = lambda x: True
        (filename_, smap) = problem_writer(self,
                                           filename,
                                           _solver_capability,
                                           kwds)
        assert filename_ == filename

        if _called_by_solver:
            # BIG HACK
            smap_id = id(smap)
            if not hasattr(self, "._symbol_maps"):
                setattr(self, "._symbol_maps", {})
            getattr(self, "._symbol_maps")[smap_id] = smap
            return smap_id
        else:
            return smap

    def _flag_vars_as_stale(self):
        from pyomo.core.kernel.component_variable import variable
        for var in self.components(variable.ctype,
                                   active=True):
            var.stale = True

    def load_solution(self,
                      solution,
                      allow_consistent_values_for_fixed_vars=False,
                      comparison_tolerance_for_fixed_vars=1e-5):
        """
        Load a solution.

        Args:
            solution: A :class:`pyomo.opt.Solution` object with a
                symbol map. Optionally, the solution can be tagged
                with a default variable value (e.g., 0) that will be
                applied to those variables in the symbol map that do
                not have a value in the solution.
            allow_consistent_values_for_fixed_vars:
                Indicates whether a solution can specify
                consistent values for variables that are
                fixed.
            comparison_tolerance_for_fixed_vars: The
                tolerance used to define whether or not a
                value in the solution is consistent with the
                value of a fixed variable.
        """
        symbol_map = solution.symbol_map
        default_variable_value = getattr(solution,
                                         "default_variable_value",
                                         None)

        # Generate the list of active import suffixes on
        # this top level model
        valid_import_suffixes = \
            dict(import_suffix_generator(self,
                                         active=True,
                                         return_key=True))
        # To ensure that import suffix data gets properly
        # overwritten (e.g., the case where nonzero dual
        # values exist on the suffix and but only sparse
        # dual values exist in the results object) we clear
        # all active import suffixes.
        for suffix in itervalues(valid_import_suffixes):
            suffix.clear()

        # Load problem (model) level suffixes. These would
        # only come from ampl interfaced solution suffixes
        # at this point in time.
        for _attr_key, attr_value in iteritems(solution.problem):
            attr_key = _attr_key[0].lower() + _attr_key[1:]
            if attr_key in valid_import_suffixes:
                valid_import_suffixes[attr_key][self] = attr_value

        #
        # Load variable data
        #
        self._flag_vars_as_stale()
        var_skip_attrs = ['id','canonical_label']
        seen_var_ids = set()
        for label, entry in iteritems(solution.variable):
            var = symbol_map.getObject(label)
            if (var is None) or \
               (var is SymbolMap.UnknownSymbol):
                # NOTE: the following is a hack, to handle
                #    the ONE_VAR_CONSTANT variable that is
                #    necessary for the objective
                #    constant-offset terms.  probably should
                #    create a dummy variable in the model
                #    map at the same time the objective
                #    expression is being constructed.
                if "ONE_VAR_CONST" in label:
                    continue
                else:
                    raise KeyError("Variable associated with symbol '%s' "
                                   "is not found on this block"
                                   % (label))

            seen_var_ids.add(id(var))

            if (not allow_consistent_values_for_fixed_vars) and \
               var.fixed:
                raise ValueError("Variable '%s' is currently fixed. "
                                 "A new value is not expected "
                                 "in solution" % (var.name))

            for _attr_key, attr_value in iteritems(entry):
                attr_key = _attr_key[0].lower() + _attr_key[1:]
                if attr_key == 'value':
                    if allow_consistent_values_for_fixed_vars and \
                       var.fixed and \
                       (math.fabs(attr_value - var.value) > \
                        comparison_tolerance_for_fixed_vars):
                        raise ValueError(
                            "Variable %s is currently fixed. "
                            "A value of '%s' in solution is "
                            "not within tolerance=%s of the current "
                            "value of '%s'"
                            % (var.name, attr_value,
                               comparison_tolerance_for_fixed_vars,
                               var.value))
                    var.value = attr_value
                    var.stale = False
                elif attr_key in valid_import_suffixes:
                    valid_import_suffixes[attr_key][var] = attr_value

        # start to build up the set of unseen variable ids
        unseen_var_ids = set(symbol_map.byObject.keys())
        # at this point it contains ids for non-variable types
        unseen_var_ids.difference_update(seen_var_ids)

        #
        # Load objective solution (should simply be suffixes if
        # they exist)
        #
        objective_skip_attrs = ['id','canonical_label','value']
        for label,entry in iteritems(solution.objective):
            obj = symbol_map.getObject(label)
            if (obj is None) or \
               (obj is SymbolMap.UnknownSymbol):
                raise KeyError("Objective associated with symbol '%s' "
                                "is not found on this block"
                                % (label))
            unseen_var_ids.remove(id(obj))
            for _attr_key, attr_value in iteritems(entry):
                attr_key = _attr_key[0].lower() + _attr_key[1:]
                if attr_key in valid_import_suffixes:
                    valid_import_suffixes[attr_key][obj] = \
                        attr_value

        #
        # Load constraint solution
        #
        con_skip_attrs = ['id', 'canonical_label']
        for label, entry in iteritems(solution.constraint):
            con = symbol_map.getObject(label)
            if con is SymbolMap.UnknownSymbol:
                #
                # This is a hack - see above.
                #
                if "ONE_VAR_CONST" in label:
                    continue
                else:
                    raise KeyError("Constraint associated with symbol '%s' "
                                   "is not found on this block"
                                   % (label))
            unseen_var_ids.remove(id(con))
            for _attr_key, attr_value in iteritems(entry):
                attr_key = _attr_key[0].lower() + _attr_key[1:]
                if attr_key in valid_import_suffixes:
                    valid_import_suffixes[attr_key][con] = \
                        attr_value


        #
        # Load sparse variable solution
        #
        if default_variable_value is not None:
            for var_id in unseen_var_ids:
                var = symbol_map.getObject(symbol_map.byObject[var_id])
                if var.ctype is not variable.ctype:
                    continue
                if (not allow_consistent_values_for_fixed_vars) and \
                   var.fixed:
                    raise ValueError("Variable '%s' is currently fixed. "
                                     "A new value is not expected "
                                     "in solution" % (var.name))

                if allow_consistent_values_for_fixed_vars and \
                   var.fixed and \
                   (math.fabs(default_variable_value - var.value) > \
                    comparison_tolerance_for_fixed_vars):
                    raise ValueError(
                        "Variable %s is currently fixed. "
                        "A value of '%s' in solution is "
                        "not within tolerance=%s of the current "
                        "value of '%s'"
                        % (var.name, default_variable_value,
                           comparison_tolerance_for_fixed_vars,
                           var.value))
                var.value = default_variable_value
                var.stale = False

class block(_block_base, IBlockStorage):
    """An implementation of the :class:`IBlockStorage` interface."""
    # To avoid a circular import, for the time being, this
    # property will be set externally
    _ctype = None
    def __init__(self):
        self._parent = None
        self._active = True
        # This implementation is quite piggish at the
        # moment. It can probably be streamlined by doing
        # something similar to what _BlockData does in
        # block.py (e.g., using _ctypes, _decl, and
        # _decl_order). However, considering that we now
        # have other means of producing lightweight blocks
        # (tiny_block) as well as the more lightweight
        # implementation of singleton types, it is hard to
        # justify making this implementation harder to
        # follow until we do some more concrete profiling.
        self._byctype = defaultdict(OrderedDict)
        self._order = OrderedDict()

    #
    # Define the IComponentContainer abstract methods
    #

    # overridden by the IBlockStorage interface
    #def components(self):
    #    pass

    def child_key(self, child):
        """Get the lookup key associated with a child of
        this container.

        Raises:
            ValueError: if the argument is not a child of
                this container
        """
        if getattr(child, "parent", None) is self:
            if child.ctype in self._byctype:
                for key, val in iteritems(self._byctype[child.ctype]):
                    if val is child:
                        return key
        raise ValueError

    #
    # Define the IBlockStorage abstract methods
    #

    def children(self,
                 ctype=_no_ctype,
                 return_key=False):
        """Iterate over the children of this block.

        Args:
            ctype: Indicate the type of children to iterate
                over. The default value indicates that all
                types should be included.
            return_key (bool): Set to :const:`True` to
                indicate that the return type should be a
                2-tuple consisting of the child storage key
                and the child object. By default, only the
                child objects are returned.

        Returns:
            iterator of objects or (key,object) tuples
        """
        if return_key:
            itermethod = iteritems
        else:
            itermethod = itervalues
        if ctype is _no_ctype:
            return itermethod(self._order)
        else:
            return itermethod(self._byctype.get(ctype,{}))

    #
    # Interface
    #

    def __setattr__(self, name, obj):
        if hasattr(obj, '_is_categorized_object'):
            if obj._parent is None:
                if hasattr(self, name) and \
                   hasattr(getattr(self, name),
                           '_is_categorized_object'):
                    logger.warning(
                        "Implicitly replacing the categorized attribute "
                        "%s (type=%s) on block with a new object "
                        "(type=%s).\nThis is usually indicative of a "
                        "modeling error.\nTo avoid this warning, delete "
                        "the original object from the block before "
                        "assigning a new object with the same name."
                        % (name,
                           type(getattr(self, name)),
                           type(obj)))
                    delattr(self, name)
                self._byctype[obj.ctype][name] = obj
                self._order[name] = obj
                obj._parent = weakref.ref(self)
                # children that are not of type
                # _ActiveComponentMixin retain the active status
                # of their parent, which is why the default
                # return value from getattr is False
                if getattr(obj, _active_flag_name, False):
                    self._increment_active()
            elif hasattr(self, name) and \
                 (getattr(self, name) is obj):
                # a very special case that makes sense to handle
                # because the implied order should be: (1) delete
                # the object at the current index, (2) insert the
                # the new object. This performs both without any
                # actions, but it is an extremely rare case, so
                # it should go last.
                pass
            else:
                raise ValueError(
                    "Invalid assignment to %s type with name '%s' "
                    "at entry %s. A parent container has already "
                    "been assigned to the object being inserted: %s"
                    % (self.__class__.__name__,
                       self.name,
                       name,
                       obj.parent.name))
        super(block, self).__setattr__(name, obj)

    def __delattr__(self, name):
        obj = getattr(self, name)
        if hasattr(obj, '_is_categorized_object'):
            del self._order[name]
            del self._byctype[obj.ctype][name]
            if len(self._byctype[obj.ctype]) == 0:
                del self._byctype[obj.ctype]
            obj._parent = None
            # children that are not of type
            # IActiveObject retain the active status
            # of their parent, which is why the default
            # return value from getattr is False
            if getattr(obj, _active_flag_name, False):
                self._decrement_active()
        super(block, self).__delattr__(name)

    def collect_ctypes(self,
                       active=None,
                       descend_into=True):
        """
        Count all object category types stored on or under
        this block.

        Args:
            active (:const:`True`/:const:`None`): Set to
                :const:`True` to indicate that only active
                categorized objects should be counted. The
                default value of :const:`None` indicates
                that all categorized objects (including
                those that have been deactivated) should be
                counted. *Note*: This flag is ignored for
                any objects that do not have an active flag.
            descend_into (bool): Indicates whether or not
                category types should be counted on
                sub-blocks. Default is :const:`True`.

        Returns:
            set of category types
        """
        assert active in (None, True)
        ctypes = set()
        if not descend_into:
            if active is None:
                ctypes.update(ctype for ctype in self._byctype)
            else:
                assert active is True
                for ctype in self._byctype:
                    for component in self.components(
                            ctype=ctype,
                            active=True,
                            descend_into=False):
                        ctypes.add(ctype)
                        # just need 1 to appear in order to
                        # count the ctype
                        break
        else:
            for blk in self.blocks(active=active,
                                   descend_into=True):
                ctypes.update(blk.collect_ctypes(
                    active=active,
                    descend_into=False))
        return ctypes

class tiny_block(_block_base, IBlockStorage):
    """
    A memory efficient block for storing a small number
    of child components.
    """
    # To avoid a circular import, for the time being, this
    # property will be set externally
    _ctype = None
    def __init__(self):
        self._parent = None
        self._active = True
        self._order = []

    def __setattr__(self, name, obj):
        if hasattr(obj, '_is_categorized_object'):
            if obj._parent is None:
                if hasattr(self, name) and \
                   hasattr(getattr(self,name),
                           '_is_categorized_object'):
                    logger.warning(
                        "Implicitly replacing the categorized attribute "
                        "%s (type=%s) on block with a new object "
                        "(type=%s).\nThis is usually indicative of a "
                        "modeling error.\nTo avoid this warning, delete "
                        "the original object from the block before "
                        "assigning a new object with the same name."
                        % (name,
                           type(getattr(self, name)),
                           type(obj)))
                    delattr(self, name)
                obj._parent = weakref.ref(self)
                self._order.append(name)
                # children that are not of type
                # IActiveObject retain the active status
                # of their parent, which is why the default
                # return value from getattr is False
                if getattr(obj, _active_flag_name, False):
                    self._increment_active()
            elif hasattr(self, name) and \
                 (getattr(self, name) is obj):
                # a very special case that makes sense to handle
                # because the implied order should be: (1) delete
                # the object at the current index, (2) insert the
                # the new object. This performs both without any
                # actions, but it is an extremely rare case, so
                # it should go last.
                pass
            else:
                raise ValueError(
                    "Invalid assignment to %s type with name '%s' "
                    "at entry %s. A parent container has already "
                    "been assigned to the object being inserted: %s"
                    % (self.__class__.__name__,
                       self.name,
                       name,
                       obj.parent.name))
        super(tiny_block, self).__setattr__(name, obj)

    def __delattr__(self, name):
        obj = getattr(self, name)
        if hasattr(obj, '_is_categorized_object'):
            obj._parent = None
            for ndx, key in enumerate(self._order):
                if getattr(self, key) is obj:
                    break
            else:        #pragma:nocover
                # shouldn't happen
                assert False
            del self._order[ndx]
            # children that are not of type
            # IActiveObject retain the active status
            # of their parent, which is why the default
            # return value from getattr is False
            if getattr(obj, _active_flag_name, False):
                self._decrement_active()
        super(tiny_block, self).__delattr__(name)

    #
    # Define the IComponentContainer abstract methods
    #

    # overridden by the IBlockStorage interface
    #def components(...)

    def child_key(self, child):
        """Get the lookup key associated with a child of
        this container.

        Raises:
            ValueError: if the argument is not a child of
                this container
        """
        if getattr(child, "parent", None) is self:
            for key in self._order:
                if getattr(self, key) is child:
                    return key
        raise ValueError

    # overridden by the IBlockStorage interface
    #def children(...)

    #
    # Define the IBlockStorage abstract methods
    #

    def children(self,
                 ctype=_no_ctype,
                 return_key=False):
        """Iterate over the children of this block.

        Args:
            ctype: Indicate the type of children to iterate
                over. The default value indicates that all
                types should be included.
            return_key (bool): Set to :const:`True` to
                indicate that the return type should be a
                2-tuple consisting of the child storage key
                and the child object. By default, only the
                child objects are returned.

        Returns:
            iterator of objects or (key,object) tuples
        """
        for key in self._order:
            child = getattr(self, key)
            if (ctype is _no_ctype) or (child.ctype == ctype):
                if return_key:
                    yield key, child
                else:
                    yield child

    # implemented by _block_base
    # def components(...)

    # implemented by _block_base
    # def blocks(...)

    def collect_ctypes(self,
                       active=None,
                       descend_into=True):
        """
        Count all object category types stored on or under
        this block.

        Args:
            active (:const:`True`/:const:`None`): Set to
                :const:`True` to indicate that only active
                categorized objects should be counted. The
                default value of :const:`None` indicates
                that all categorized objects (including
                those that have been deactivated) should be
                counted. *Note*: This flag is ignored for
                any objects that do not have an active flag.
            descend_into (bool): Indicates whether or not
                category types should be counted on
                sub-blocks. Default is :const:`True`.

        Returns:
            set of category types
        """
        assert active in (None, True)
        ctypes = set()
        if not descend_into:
            for component in self.components(active=active,
                                             descend_into=False):
                ctypes.add(component.ctype)
        else:
            for blk in self.blocks(active=active,
                                   descend_into=True):
                ctypes.update(blk.collect_ctypes(
                    active=active,
                    descend_into=False))
        return ctypes

class block_tuple(ComponentTuple,
                  _ActiveComponentContainerMixin):
    """A tuple-style container for blocks."""
    # To avoid a circular import, for the time being, this
    # property will be set externally
    _ctype = None
    __slots__ = ("_parent",
                 "_active",
                 "_data")
    if six.PY3:
        # This has to do with a bug in the abc module
        # prior to python3. They forgot to define the base
        # class using empty __slots__, so we shouldn't add a slot
        # for __weakref__ because the base class has a __dict__.
        __slots__ = list(__slots__) + ["__weakref__"]

    def __init__(self, *args, **kwds):
        self._parent = None
        self._active = True
        super(block_tuple, self).__init__(*args, **kwds)

class block_list(ComponentList,
                 _ActiveComponentContainerMixin):
    """A list-style container for blocks."""
    # To avoid a circular import, for the time being, this
    # property will be set externally
    _ctype = None
    __slots__ = ("_parent",
                 "_active",
                 "_data")
    if six.PY3:
        # This has to do with a bug in the abc module
        # prior to python3. They forgot to define the base
        # class using empty __slots__, so we shouldn't add a slot
        # for __weakref__ because the base class has a __dict__.
        __slots__ = list(__slots__) + ["__weakref__"]

    def __init__(self, *args, **kwds):
        self._parent = None
        self._active = True
        super(block_list, self).__init__(*args, **kwds)

class block_dict(ComponentDict,
                 _ActiveComponentContainerMixin):
    """A dict-style container for blocks."""
    # To avoid a circular import, for the time being, this
    # property will be set externally
    _ctype = None
    __slots__ = ("_parent",
                 "_active",
                 "_data")
    if six.PY3:
        # This has to do with a bug in the abc module
        # prior to python3. They forgot to define the base
        # class using empty __slots__, so we shouldn't add a slot
        # for __weakref__ because the base class has a __dict__.
        __slots__ = list(__slots__) + ["__weakref__"]

    def __init__(self, *args, **kwds):
        self._parent = None
        self._active = True
        super(block_dict, self).__init__(*args, **kwds)
