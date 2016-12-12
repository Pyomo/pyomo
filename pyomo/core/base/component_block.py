#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

__all__ = ("block",
           "block_list",
           "block_dict",
           "StaticBlock")

import copy
import abc
import logging
import weakref
from collections import defaultdict
try:
    from collections import OrderedDict
except ImportError:                         #pragma:nocover
    from ordereddict import OrderedDict

from pyomo.core.base.component_interface import (ICategorizedObject,
                                                 IComponent,
                                                 IComponentContainer,
                                                 _IActiveComponentContainer)
from pyomo.core.base.component_objective import IObjective
from pyomo.core.base.component_variable import IVariable
from pyomo.core.base.component_constraint import IConstraint
from pyomo.core.base.component_dict import ComponentDict
from pyomo.core.base.component_list import ComponentList
from pyomo.core.base.component_map import ComponentMap
from pyomo.core.base.suffix import active_import_suffix_generator
from pyomo.core.base.symbol_map import SymbolMap
import pyomo.opt

import six
from six import itervalues, iteritems

logger = logging.getLogger('pyomo.core')

_no_ctype = object()

# TODO: Possibly take some additional steps in the basic block
#       implementation to ensure that ordering is retained in
#       a cloned / deepcopied / pickled block. I'm seeing some
#       sporadic test failures that indicate that there might be
#       a problem with this.

class IBlockStorage(IComponent,
                    IComponentContainer,
                    _IActiveComponentContainer):
    """A container that stores multiple types."""
    _is_component = True
    _is_container = True
    _child_storage_delimiter_string = "."
    _child_storage_entry_string = "%s"
    __slots__ = ()

    #
    # These methods are already declared abstract on
    # IComponentContainer, but we redeclare them here to
    # point out that the can accept a ctype
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
        parent pointer is set to None. Any components
        encountered that are descendents of this
        block will be deepcopied, otherwise a reference
        to the original component is retained.
        """
        new_block = copy.deepcopy(
            self, {'__block_scope__': {id(self):True, id(None):False}} )
        new_block._parent = None
        return new_block

    @abc.abstractmethod
    def blocks(self, *args, **kwds):
        raise NotImplementedError     #pragma:nocover

    @abc.abstractmethod
    def collect_ctypes(self, *args, **kwds):
        raise NotImplementedError     #pragma:nocover

class _block_base(object):
    """
    A base class shared by 'block' and StaticBlock that
    implements a few IBlockStorage abstract methods.
    """
    __slots__ = ()

    if False:                           #pragma:nocover
        # This original implementation for preorder traversal
        # technically works, but does not respect declaration
        # order, which makes it difficult to test.  I'm keeping
        # it around to do some performance profiling later on.
        def _Xpreorder_traversal(self,
                               ctype=_no_ctype,
                               active=None,
                               include_parent_blocks=True,
                               return_key=False):
            assert active in (None, True)
            # TODO
            from pyomo.core.base.block import Block
            _active_flag_name = "active"
            # rule: only containers get placed in the stack
            stack = []
            if (active is None) or \
               getattr(self, _active_flag_name, True):
                stack.append((None, self))
            while len(stack) > 0:
                node_key, node = stack.pop()
                if (include_parent_blocks) or \
                   (ctype is _no_ctype) or \
                   (node.ctype == ctype):
                    if return_key:
                        yield node_key, node
                    else:
                        yield node
                assert node._is_container
                if isinstance(node, IBlockStorage):
                    children = node.children(ctype=ctype,
                                             return_key=True)
                    isblock = True
                else:
                    children = node.children(return_key=True)
                    isblock = False
                for key, child in children:
                    if (active is None) or \
                       getattr(child, _active_flag_name, True):
                        if not child._is_container:
                            if return_key:
                               yield key, child
                            else:
                                yield child
                        else:
                            stack.append((key, child))
                if isblock and (ctype != Block) and (ctype != _no_ctype):
                    for key, child in node.children(ctype=Block,
                                                    return_key=True):
                        if (active is None) or \
                           getattr(child, _active_flag_name, True):
                            stack.append((key, child))

    if False:                           #pragma:nocover
        # This original implementation for postorder traversal
        # technically works, but does not respect declaration
        # order, which makes it difficult to test.  I'm keeping
        # it around to do some performance profiling later on.
        def _Xpostorder_traversal(self,
                                  ctype=_no_ctype,
                                  active=None,
                                  include_parent_blocks=True,
                                  return_key=False):
            assert active in (None, True)
            # TODO
            from pyomo.core.base.block import Block
            _active_flag_name = "active"
            # rule: only containers get placed in the stack
            stack = []
            if (active is None) or \
               getattr(self, _active_flag_name, True):
                stack.append((None, self))
            used = set()
            while len(stack) > 0:
                node_key, node = stack.pop()
                assert node._is_container
                if id(node) in used:
                    if (include_parent_blocks) or \
                       (ctype is _no_ctype) or \
                       (node.ctype == ctype):
                        if return_key:
                            yield node_key, node
                        else:
                            yield node
                else:
                    used.add(id(node))
                    stack.append((node_key, node))
                    if isinstance(node, IBlockStorage):
                        children = node.children(ctype=ctype,
                                                 return_key=True)
                        isblock = True
                    else:
                        children = node.children(return_key=True)
                        isblock = False
                    for key, child in children:
                        if (active is None) or \
                           getattr(child, _active_flag_name, True):
                            if not child._is_container:
                                if return_key:
                                   yield key, child
                                else:
                                    yield child
                            else:
                                stack.append((key, child))
                    if isblock and (ctype != Block) and (ctype != _no_ctype):
                        for key, child in node.children(ctype=Block,
                                                        return_key=True):
                            if (active is None) or \
                               getattr(self, _active_flag_name, True):
                                stack.append((key, child))

    def preorder_traversal(self,
                           ctype=_no_ctype,
                           active=None,
                           include_parent_blocks=True,
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
            active (True/None): Set to True to indicate that
                only active objects should be included. The
                default value of None indicates that all
                components (including those that have been
                deactivated) should be included. *Note*: This
                flag is ignored for any objects that do not
                have an active flag.
            include_parent_blocks (bool): Indicates if
              parent block containers should be included
              even when the ctype is set to something that
              is not Block. Default is True.
            return_key (bool): Set to True to indicate that
                the return type should be a 2-tuple
                consisting of the local storage key of the
                object within its parent and the object
                itself. By default, only the objects are
                returned.

        Returns: an iterator of objects or (key,object) tuples
        """
        assert active in (None, True)
        # TODO
        from pyomo.core.base.block import Block
        _active_flag_name = "active"

        # if this block is not active, then nothing below it
        # can be active
        if active and (not self.active):
            return

        if include_parent_blocks or \
           (ctype is _no_ctype) or \
           (ctype is Block):
            if return_key:
                yield root_key, self
            else:
                yield self
        for key, child in self.children(return_key=True):

            # check for appropriate ctype
            if (ctype is not _no_ctype) and \
               (child.ctype is not ctype) and \
               (child.ctype is not Block):
                continue

            # check active status (if appropriate)
            if (active is not None) and \
               not getattr(child, _active_flag_name, True):
                continue

            if not child._is_container:
                # not a block and not a container,
                # so it is a leaf node
                if return_key:
                    yield key, child
                else:
                    yield child
            elif not child._is_component:
                # not a block
                if child.ctype is Block:
                    # but a container of blocks
                    for obj_key, obj in child.preorder_traversal(
                            active=active,
                            return_key=True,
                            root_key=key):
                        if not obj._is_component:
                            # a container of blocks
                            if (ctype is _no_ctype) or \
                               (ctype is Block) or \
                               include_parent_blocks:
                                if return_key:
                                    yield obj_key, obj
                                else:
                                    yield obj
                        else:
                            # a block
                            for item in obj.preorder_traversal(
                                    ctype=ctype,
                                    active=active,
                                    include_parent_blocks=include_parent_blocks,
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
                # a block
                for item in child.preorder_traversal(
                        ctype=ctype,
                        active=active,
                        include_parent_blocks=include_parent_blocks,
                        return_key=return_key,
                        root_key=key):
                    yield item

    def postorder_traversal(self,
                            ctype=_no_ctype,
                            active=None,
                            include_parent_blocks=True,
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
            active (True/None): Set to True to indicate that
                only active objects should be included. The
                default value of None indicates that all
                components (including those that have been
                deactivated) should be included. *Note*: This
                flag is ignored for any objects that do not
                have an active flag.
            include_parent_blocks (bool): Indicates if
              parent block containers should be included
              even when the ctype is set to something that
              is not Block. Default is True.
            return_key (bool): Set to True to indicate that
                the return type should be a 2-tuple
                consisting of the local storage key of the
                object within its parent and the object
                itself. By default, only the objects are
                returned.

        Returns: an iterator of objects or (key,object) tuples
        """
        assert active in (None, True)
        # TODO
        from pyomo.core.base.block import Block
        _active_flag_name = "active"

        # if this block is not active, then nothing below it
        # can be active
        if active and (not self.active):
            return

        for key, child in self.children(return_key=True):

            # check for appropriate ctype
            if (ctype is not _no_ctype) and \
               (child.ctype is not ctype) and \
               (child.ctype is not Block):
                continue

            # check active status (if appropriate)
            if (active is not None) and \
               not getattr(child, _active_flag_name, True):
                continue

            if not child._is_container:
                # not a block and not a container,
                # so it is a leaf node
                if return_key:
                    yield key, child
                else:
                    yield child
            elif not child._is_component:
                # not a block
                if child.ctype is Block:
                    # but a container of blocks
                    for obj_key, obj in child.postorder_traversal(
                            active=active,
                            return_key=True,
                            root_key=key):
                        if not obj._is_component:
                            # a container of blocks
                            if (ctype is _no_ctype) or \
                               (ctype is Block) or \
                               include_parent_blocks:
                                if return_key:
                                    yield obj_key, obj
                                else:
                                    yield obj
                        else:
                            # a block
                            for item in obj.postorder_traversal(
                                    ctype=ctype,
                                    active=active,
                                    include_parent_blocks=include_parent_blocks,
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
                # a block
                for item in child.postorder_traversal(
                        ctype=ctype,
                        active=active,
                        include_parent_blocks=include_parent_blocks,
                        return_key=return_key,
                        root_key=key):
                    yield item

        if include_parent_blocks or \
           (ctype is _no_ctype) or \
           (ctype is Block):
            if return_key:
                yield root_key, self
            else:
                yield self

    def components(self,
                   ctype=_no_ctype,
                   active=None,
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
            active (True/None): Set to True to indicate that
                only active objects should be included. The
                default value of None indicates that all
                components (including those that have been
                deactivated) should be included. *Note*: This
                flag is ignored for any objects that do not
                have an active flag.
            descend_into (bool): Indicates whether or not to
                include components on sub-blocks. Default is
                True.

        Returns: an iterator of objects or (key,object) tuples
        """

        assert active in (None, True)
        # TODO
        from pyomo.core.base.block import Block
        _active_flag_name = "active"

        # if this block is not active, then nothing below it
        # can be active
        if active and (not self.active):
            return

        # Generate components from immediate children first
        for child in self.children(ctype=ctype):

            # check active status (if appropriate)
            if (active is not None) and \
               not getattr(child, _active_flag_name, True):
                continue

            if child._is_component:
                # child is a component (includes blocks), so yield it
                yield child
            else:
                assert child._is_container
                # child is a container (but not a block)
                if (active is not None) and \
                   isinstance(child, _IActiveComponentContainer):
                    for component in child.components():
                        if getattr(component,
                                   _active_flag_name,
                                   True):
                            yield component
                else:
                    for component in child.components():
                        yield component

        if descend_into:
            # now recurse into subblocks
            for child in self.children(ctype=Block):

                # check active status (if appropriate)
                if (active is not None) and \
                   not getattr(child, _active_flag_name, True):
                    continue

                if child._is_component:
                    # child is a block
                    for component in child.components(
                            ctype=ctype,
                            active=active,
                            descend_into=descend_into):
                        yield component
                else:
                    # child is a container of blocks,
                    # but not a block itself
                    for _comp in child.components():
                        if (active is None) or \
                           getattr(_comp,
                                   _active_flag_name,
                                   True):
                            for component in _comp.components(
                                    ctype=ctype,
                                    active=active,
                                    descend_into=descend_into):
                                yield component

    def blocks(self,
               active=None,
               descend_into=True):
        """
        Generates a traversal of all blocks associated with
        this one (include itself). This method yields
        identical behavior to calling the components()
        method with ctype=Block, except that this block is
        included (as the first item in the generator).
        """
        assert active in (None, True)
        # TODO
        from pyomo.core.base.block import Block

        # if this block is not active, then nothing below it
        # can be active
        if active and (not self.active):
            return

        yield self
        for component in self.components(ctype=Block,
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
                include.  The default value indicates that
                all types should be included.
            active (True/None): Set to True to indicate that
                only active components should be
                included. The default value of None
                indicates that all components (including
                those that have been deactivated) should be
                included. *Note*: This flag is ignored for
                any objects that do not have an active flag.
            descend_into (bool): Indicates whether or not to
                include components on sub-blocks. Default is
                True.
            convert (function): A function that converts a
                storage key into a string
                representation. Default is str.
            prefix (str): A string to prefix names with.

        Returns:
            A component map that behaves as a dictionary
            mapping component objects to names.
        """
        assert active in (None, True)
        # TODO
        from pyomo.core.base.block import Block

        names = ComponentMap()

        # if this block is not active, then nothing below it
        # can be active
        if active and (not self.active):
            return names

        if descend_into:
            traversal = self.preorder_traversal(ctype=ctype,
                                                active=active,
                                                include_parent_blocks=True,
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

    def load_solution(self,
                      solution,
                      allow_consistent_values_for_fixed_vars=False,
                      comparison_tolerance_for_fixed_vars=1e-5):
        """
        Load a solution.

        Args:
            solution: A pyomo.opt.Solution object with
                a symbol map.
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

        # Generate the list of active import suffixes on
        # this top level model
        valid_import_suffixes = \
            dict(active_import_suffix_generator(self))
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
                valid_import_suffixes[attr_key][self] = \
                    attr_value.value

        #
        # Load objective data (should simply be suffixes if
        # they exist)
        #
        objective_skip_attrs = ['id','canonical_label','value']
        for label,entry in iteritems(solution.objective):
            obj_value = symbol_map.getObject(label)
            if (obj_value is None) or \
               (obj_value is SymbolMap.UnknownSymbol):
                raise KeyError("Objective associated with symbol '%s' "
                                "is not found on this block"
                                % (label))
            for _attr_key, attr_value in iteritems(entry):
                attr_key = _attr_key[0].lower() + _attr_key[1:]
                if attr_key in valid_import_suffixes:
                    valid_import_suffixes[attr_key][obj_value] = \
                        attr_value.value

        #
        # Load variable data
        #
        var_skip_attrs = ['id','canonical_label']
        for label, entry in iteritems(solution.variable):
            var_value = symbol_map.getObject(label)
            if (var_value is None) or \
               (var_value is SymbolMap.UnknownSymbol):
                # NOTE: the following is a hack, to handle
                #    the ONE_VAR_CONSTANT variable that is
                #    necessary for the objective
                #    constant-offset terms.  probably should
                #    create a dummy variable in the model
                #    map at the same time the objective
                #    expression is being constructed.
                if label == "ONE_VAR_CONSTANT":
                    continue
                else:
                    raise KeyError("Variable associated with symbol '%s' "
                                   "is not found on this block"
                                   % (label))

            if (not allow_consistent_values_for_fixed_vars) and \
               var_value.fixed:
                raise ValueError("Variable '%s' on this block is "
                                "currently fixed - new value is "
                                "not expected in solution"
                                % (label))

            for _attr_key, attr_value in iteritems(entry):
                attr_key = _attr_key[0].lower() + _attr_key[1:]
                if attr_key == 'value':
                    if allow_consistent_values_for_fixed_vars and \
                       var_value.fixed and \
                       (math.fabs(attr_value - var_value.value) > \
                        comparison_tolerance_for_fixed_vars):
                        raise ValueError(
                            "Variable %s on this block is currently "
                            "fixed - a value of '%s' in solution is "
                            "not within tolerance=%s of the current "
                            "value of '%s'"
                            % (label, str(attr_value),
                               str(comparison_tolerance_for_fixed_vars),
                               str(var_value.value)))
                    var_value.value = attr_value
                    var_value.stale = False
                elif attr_key in valid_import_suffixes:
                    valid_import_suffixes[attr_key][var_value] = attr_value

        #
        # Load constraint data
        #
        con_skip_attrs = ['id', 'canonical_label']
        for label, entry in iteritems(solution.constraint):
            con_value = symbol_map.getObject(label)
            if con_value is SymbolMap.UnknownSymbol:
                #
                # This is a hack - see above.
                #
                if label.endswith('ONE_VAR_CONSTANT'):
                    continue
                else:
                    raise KeyError("Constraint associated with symbol '%s' "
                                   "is not found on this block"
                                   % (label))

            for _attr_key, attr_value in iteritems(entry):
                attr_key = _attr_key[0].lower() + _attr_key[1:]
                if attr_key in valid_import_suffixes:
                    valid_import_suffixes[attr_key][con_value] = \
                        attr_value

class block(_block_base, IBlockStorage):
    """An implementation of the IBlockStorage interface."""
    # To avoid a circular import, for the time being, this
    # property will be set in block.py
    _ctype = None
    def __init__(self):
        self.__parent = None
        self.__active = True
        self.__byctype = defaultdict(OrderedDict)
        self.__order = OrderedDict()

    # The base class implementation of __getstate__ handles
    # the parent weakref assumed to be stored with the
    # attribute name: _parent.  We need to remove the
    # duplicate reference to the parent stored at
    # _block__parent to avoid letting a weakref object
    # remain in the final state dictionary.
    def __getstate__(self):
        state = super(block, self).__getstate__()
        state['_block__parent'] = state.pop('_parent')
        return state

    @property
    def _parent(self):
        return self.__parent
    @_parent.setter
    def _parent(self, parent):
        self.__parent = parent

    @property
    def _active(self):
        return self.__active
    @_active.setter
    def _active(self, active):
        self.__active = active

    #
    # Define the IComponentContainer abstract methods
    #

    # overridden by the IBlockStorage interface
    #def components(self):
    #    pass

    def child_key(self, child):
        if child.ctype in self.__byctype:
            for key, val in iteritems(self.__byctype[child.ctype]):
                if val is child:
                    return key
        raise ValueError("No child entry: %s"
                         % (child))

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
            return_key (bool): Set to True to indicate that
                the return type should be a 2-tuple
                consisting the child storage key and the
                child object. By default, only the child
                objects are returned.

        Returns: an iterator of objects or (key,object) tuples
        """
        if return_key:
            itermethod = iteritems
        else:
            itermethod = itervalues
        if ctype is _no_ctype:
            return itermethod(self.__order)
        else:
            return itermethod(self.__byctype.get(ctype,{}))

    #
    # Interface
    #

    def __setattr__(self, name, component):
        if isinstance(component, ICategorizedObject):
            if component._parent is None:
                if name in self.__dict__:
                    logger.warning(
                        "Implicitly replacing the component attribute "
                        "%s (type=%s) on block with a new Component "
                        "(type=%s).\nThis is usually indicative of a "
                        "modeling error.\nTo avoid this warning, delete "
                        "the original component from the block before "
                        "assigning a new component with the same name."
                        % (name,
                           type(getattr(self, name)),
                           type(component)))
                    delattr(self, name)
                self.__byctype[component.ctype][name] = component
                self.__order[name] = component
                component._parent = weakref.ref(self)
            elif hasattr(self, name) and \
                 (getattr(self, name) is component):
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
                    "been assigned to the component being "
                    "inserted: %s"
                    % (self.__class__.__name__,
                       self.name,
                       name,
                       component.parent.name))
        super(block, self).__setattr__(name, component)

    def __delattr__(self, name):
        component = getattr(self, name)
        if isinstance(component, ICategorizedObject):
            del self.__order[name]
            del self.__byctype[component.ctype][name]
            if len(self.__byctype[component.ctype]) == 0:
                del self.__byctype[component.ctype]
            component._parent = None
        super(block, self).__delattr__(name)

    def collect_ctypes(self,
                       active=None,
                       descend_into=True):
        """
        Count all component types stored on or under this
        block.

        Args:
            active (True/None): Set to True to indicate that
                only active components should be
                counted. The default value of None indicates
                that all components (including those that
                have been deactivated) should be
                counted. *Note*: This flag is ignored for
                any objects that do not have an active flag.
            descend_into (bool): Indicates whether or not
                component types should be counted on
                sub-blocks. Default is True.

        Returns: a set of component types.
        """
        assert active in (None, True)
        ctypes = set()
        if not descend_into:
            if active is None:
                ctypes.update(ctype for ctype in self.__byctype)
            else:
                assert active is True
                for ctype in self.__byctype:
                    for component in self.components(
                            ctype=ctype,
                            active=True,
                            descend_into=False):
                        ctypes.add(ctype)
                        break # just need 1 or more
        else:
            for blk in self.blocks(active=active,
                                   descend_into=True):
                ctypes.update(blk.collect_ctypes(
                    active=active,
                    descend_into=False))
        return ctypes

    # TODO
    #def write(self, ...):

class block_list(ComponentList,
                 _IActiveComponentContainer):
    """A list-style container for blocks."""
    # To avoid a circular import, for the time being, this
    # property will be set in block.py
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
                 _IActiveComponentContainer):
    """A dict-style container for blocks."""
    # To avoid a circular import, for the time being, this
    # property will be set in block.py
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

class StaticBlock(_block_base, IBlockStorage):
    """
    A helper class for implementing blocks with a static
    set of components using __slots__. Derived classes
    should assign a static set of components to the instance
    in the __init__ method before calling this base class's
    __init__ method. The set of components should be
    identified using __slots__.

    This implementation can be used in class hierarchies
    that extend more than one level beyond this base class.
    """
    # To avoid a circular import, for the time being, this
    # property will be set in block.py
    _ctype = None
    __slots__ = ("_parent",
                 "_active",
                 "__weakref__")
    def __init__(self):
        self._parent = None
        self._active = True

    def _getattrs(self):
        """Returns all attributes that may appear on this
        class in the inheritance hierarchy."""
        # Get all slots in the inheritance chain
        # (base classes first)
        for cls in reversed(self.__class__.__mro__):
            for key in cls.__dict__.get("__slots__",()):
                yield key, getattr(self, key)
        for item in getattr(self, "__dict__",{}).items():
            yield item

    def __setattr__(self, name, component):
        if isinstance(component, ICategorizedObject):
            if component._parent is None:
                if hasattr(self, name):
                    logger.warning(
                        "Implicitly replacing the component attribute "
                        "%s (type=%s) on block with a new Component "
                        "(type=%s).\nThis is usually indicative of a "
                        "modeling error.\nTo avoid this warning, delete "
                        "the original component from the block before "
                        "assigning a new component with the same name."
                        % (name,
                           type(getattr(self, name)),
                           type(component)))
                    delattr(self, name)
                component._parent = weakref.ref(self)
            elif hasattr(self, name) and \
                 (getattr(self, name) is component):
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
                    "been assigned to the component being "
                    "inserted: %s"
                    % (self.__class__.__name__,
                       self.name,
                       name,
                       component.parent.name))
        super(StaticBlock, self).__setattr__(name, component)

    def __delattr__(self, name):
        component = getattr(self, name)
        if isinstance(component, ICategorizedObject):
            component._parent = None
        super(StaticBlock, self).__delattr__(name)

    #
    # Define the IComponentContainer abstract methods
    #

    # overridden by the IBlockStorage interface
    #def components(...)

    def child_key(self, child):
        for key, obj in self._getattrs():
            if obj is child:
                return key
        raise ValueError("No child entry: %s"
                         % (child))

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
            return_key (bool): Set to True to indicate that
                the return type should be a 2-tuple
                consisting the child storage key and the
                child object. By default, only the child
                objects are returned.

        Returns: an iterator objects or (key,object) tuples
        """
        for key, child in self._getattrs():
            if isinstance(child, ICategorizedObject) and \
               ((ctype is _no_ctype) or (child.ctype == ctype)):
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
        Count all component types stored on or under this
        block.

        Args:
            active (True/None): Set to True to indicate that
                only active components should be
                counted. The default value of None indicates
                that all components (including those that
                have been deactivated) should be
                counted. *Note*: This flag is ignored for
                any objects that do not have an active flag.
            descend_into (bool): Indicates whether or not
                component types should be counted on
                sub-blocks. Default is True.

        Returns: a set object of component types.
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
