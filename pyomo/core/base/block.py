#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

__all__ = ['Block', 'TraversalStrategy', 'SortComponents',
           'active_components', 'components', 'active_components_data',
           'components_data', 'SimpleBlock']

import collections
import copy
import logging
import sys
import weakref
import textwrap

from inspect import isclass
from operator import itemgetter
from six import iteritems, iterkeys, itervalues, StringIO, string_types, \
    advance_iterator, PY3

from pyutilib.misc.indent_io import StreamIndenter

from pyomo.common.collections import ComponentMap, Mapping
from pyomo.common.deprecation import deprecated, deprecation_warning
from pyomo.common.log import is_debug_set
from pyomo.common.timing import ConstructionTimer
from pyomo.core.base.plugin import ModelComponentFactory
from pyomo.core.base.component import (
    Component, ActiveComponentData,
)
from pyomo.core.base.componentuid import ComponentUID
from pyomo.core.base.set import GlobalSetBase, _SetDataBase
from pyomo.core.base.var import Var
from pyomo.core.base.util import Initializer
from pyomo.core.base.indexed_component import (
    ActiveIndexedComponent, UnindexedComponent_set,
)

from pyomo.opt.base import ProblemFormat, guess_format
from pyomo.opt import WriterFactory

logger = logging.getLogger('pyomo.core')


class _generic_component_decorator(object):
    """A generic decorator that wraps Block.__setattr__()

    Arguments
    ---------
        component: the Pyomo Component class to construct
        block: the block onto which to add the new component
        *args: positional arguments to the Component constructor
               (*excluding* the block argument)
        **kwds: keyword arguments to the Component constructor
    """
    def __init__(self, component, block, *args, **kwds):
        self._component = component
        self._block = block
        self._args = args
        self._kwds = kwds

    def __call__(self, rule):
        setattr(
            self._block,
            rule.__name__,
            self._component(*self._args, rule=rule, **(self._kwds))
        )
        return rule


class _component_decorator(object):
    """A class that wraps the _generic_component_decorator, which remembers
    and provides the Block and component type to the decorator.

    Arguments
    ---------
        component: the Pyomo Component class to construct
        block: the block onto which to add the new component

    """
    def __init__(self, block, component):
        self._block = block
        self._component = component

    def __call__(self, *args, **kwds):
        return _generic_component_decorator(
            self._component, self._block, *args, **kwds)


class SubclassOf(object):
    """This mocks up a tuple-like interface based on subclass relationship.

    Instances of this class present a somewhat tuple-like interface for
    use in PseudoMap ctype / descend_into.  The constructor takes a
    single ctype argument.  When used with PseudoMap (through Block APIs
    like component_objects()), it will match any ctype that is a
    subclass of the reference ctype.

    This allows, for example:

        model.component_data_objects(Var, descend_into=SubclassOf(Block))
    """
    def __init__(self, *ctype):
        self.ctype = ctype
        self.__name__ = 'SubclassOf(%s)' % (
            ','.join(x.__name__ for x in ctype),)

    def __contains__(self, item):
        return issubclass(item, self.ctype)

    def __len__(self):
        return 1

    def __getitem__(self, item):
        return self

class SortComponents(object):

    """
    This class is a convenient wrapper for specifying various sort
    ordering.  We pass these objects to the "sort" argument to various
    accessors / iterators to control how much work we perform sorting
    the resultant list.  The idea is that
    "sort=SortComponents.deterministic" is more descriptive than
    "sort=True".
    """
    unsorted = set()
    indices = set([1])
    declOrder = set([2])
    declarationOrder = declOrder
    alphaOrder = set([3])
    alphabeticalOrder = alphaOrder
    alphabetical = alphaOrder
    # both alpha and decl orders are deterministic, so only must sort indices
    deterministic = indices
    sortBoth = indices | alphabeticalOrder         # Same as True
    alphabetizeComponentAndIndex = sortBoth

    @staticmethod
    def default():
        return set()

    @staticmethod
    def sorter(sort_by_names=False, sort_by_keys=False):
        sort = SortComponents.default()
        if sort_by_names:
            sort |= SortComponents.alphabeticalOrder
        if sort_by_keys:
            sort |= SortComponents.indices
        return sort

    @staticmethod
    def sort_names(flag):
        if type(flag) is bool:
            return flag
        else:
            try:
                return SortComponents.alphaOrder.issubset(flag)
            except:
                return False

    @staticmethod
    def sort_indices(flag):
        if type(flag) is bool:
            return flag
        else:
            try:
                return SortComponents.indices.issubset(flag)
            except:
                return False


class TraversalStrategy(object):
    BreadthFirstSearch = (1,)
    PrefixDepthFirstSearch = (2,)
    PostfixDepthFirstSearch = (3,)
    # aliases
    BFS = BreadthFirstSearch
    ParentLastDepthFirstSearch = PostfixDepthFirstSearch
    PostfixDFS = PostfixDepthFirstSearch
    ParentFirstDepthFirstSearch = PrefixDepthFirstSearch
    PrefixDFS = PrefixDepthFirstSearch
    DepthFirstSearch = PrefixDepthFirstSearch
    DFS = DepthFirstSearch


def _sortingLevelWalker(list_of_generators):
    """Utility function for iterating over all members of a list of
    generators that prefixes each item with the index of the original
    generator that produced it.  This is useful for creating lists where
    we want to preserve the original generator order but want to sort
    the sub-lists.

    Note that the generators must produce tuples.
    """
    lastName = ''
    nameCounter = 0
    for gen in list_of_generators:
        nameCounter += 1  # Each generator starts a new component name
        for item in gen:
            if item[0] != lastName:
                nameCounter += 1
                lastName = item[0]
            yield (nameCounter,) + item


def _levelWalker(list_of_generators):
    """Simple utility function for iterating over all members of a list of
    generators.
    """
    for gen in list_of_generators:
        for item in gen:
            yield item


class _BlockConstruction(object):
    """
    This class holds a "global" dict used when constructing
    (hierarchical) models.
    """
    data = {}


class PseudoMap(object):
    """
    This class presents a "mock" dict interface to the internal
    _BlockData data structures.  We return this object to the
    user to preserve the historical "{ctype : {name : obj}}"
    interface without actually regenerating that dict-of-dicts data
    structure.

    We now support {ctype : PseudoMap()}
    """

    __slots__ = ('_block', '_ctypes', '_active', '_sorted')

    def __init__(self, block, ctype, active=None, sort=False):
        """
        TODO
        """
        self._block = block
        if isclass(ctype):
            self._ctypes = (ctype,)
        else:
            self._ctypes = ctype
        self._active = active
        self._sorted = SortComponents.sort_names(sort)

    def __iter__(self):
        """
        TODO
        """
        return self.iterkeys()

    def __getitem__(self, key):
        """
        TODO
        """
        if key in self._block._decl:
            x = self._block._decl_order[self._block._decl[key]]
            if self._ctypes is None or x[0].ctype in self._ctypes:
                if self._active is None or x[0].active == self._active:
                    return x[0]
        msg = ""
        if self._active is not None:
            msg += self._active and "active " or "inactive "
        if self._ctypes is not None:
            if len(self._ctypes) == 1:
                msg += self._ctypes[0].__name__ + " "
            else:
                types = sorted(x.__name__ for x in self._ctypes)
                msg += '%s or %s ' % (', '.join(types[:-1]), types[-1])
        raise KeyError("%scomponent '%s' not found in block %s"
                       % (msg, key, self._block.name))

    def __nonzero__(self):
        """
        TODO
        """
        # Shortcut: this will bail after finding the first
        # (non-None) item.  Note that we temporarily disable sorting
        # -- otherwise, if this is a sorted PseudoMap the entire
        # list will be walked and sorted before returning the first
        # element.
        sort_order = self._sorted
        try:
            self._sorted = False
            for x in itervalues(self):
                return True
            return False
        finally:
            self._sorted = sort_order

    __bool__ = __nonzero__

    def __len__(self):
        """
        TODO
        """
        #
        # If _active is None, then the number of components is
        # simply the total of the counts of the ctypes that have
        # been added.
        #
        if self._active is None:
            if self._ctypes is None:
                return sum(x[2] for x in itervalues(self._block._ctypes))
            else:
                return sum(self._block._ctypes.get(x, (0, 0, 0))[2]
                           for x in self._block._ctypes
                           if x in self._ctypes)
        #
        # If _active is True or False, then we have to count by brute force.
        #
        ans = 0
        for x in itervalues(self):
            ans += 1
        return ans

    def __contains__(self, key):
        """
        TODO
        """
        # Return True is the underlying Block contains the component
        # name.  Note, if this Pseudomap soecifies a ctype or the
        # active flag, we need to check that the underlying
        # component matches those flags
        if key in self._block._decl:
            x = self._block._decl_order[self._block._decl[key]]
            if self._ctypes is None or x[0].ctype in self._ctypes:
                return self._active is None or x[0].active == self._active
        return False

    def _ctypewalker(self):
        """
        TODO
        """
        # Note: since push/pop from the end of lists is slightly more
        # efficient, we will reverse-sort so the next ctype index is
        # at the end of the list.
        _decl_order = self._block._decl_order
        _idx_list = sorted((self._block._ctypes[x][0]
                            for x in self._block._ctypes
                            if x in self._ctypes),
                           reverse=True)
        while _idx_list:
            _idx = _idx_list.pop()
            while _idx is not None:
                _obj, _next = _decl_order[_idx]
                if _obj is not None:
                    yield _obj
                _idx = _next
                if _idx is not None and _idx_list and _idx > _idx_list[-1]:
                    _idx_list.append(_idx)
                    _idx_list.sort(reverse=True)
                    break

    def iterkeys(self):
        """
        TODO
        """
        # Iterate over the PseudoMap keys (the component names) in
        # declaration order
        #
        # Ironically, the values are the fundamental thing that we
        # can (efficiently) iterate over in decl_order.  iterkeys
        # just wraps itervalues.
        for obj in self.itervalues():
            yield obj._name

    def itervalues(self):
        """
        TODO
        """
        # Iterate over the PseudoMap values (the component objects) in
        # declaration order
        _active = self._active
        if self._ctypes is None:
            # If there is no ctype, then we will just iterate over
            # all components and return them all
            if _active is None:
                walker = (obj for obj, idx in self._block._decl_order
                          if obj is not None)
            else:
                walker = (obj for obj, idx in self._block._decl_order
                          if obj is not None and obj.active == _active)
        else:
            # The user specified a desired ctype; we will leverage
            # the _ctypewalker generator to walk the underlying linked
            # list and just return the desired objects (again, in
            # decl order)
            if _active is None:
                walker = (obj for obj in self._ctypewalker())
            else:
                walker = (obj for obj in self._ctypewalker()
                          if obj.active == _active)
        # If the user wants this sorted by name, then there is
        # nothing we can do to save memory: we must create the whole
        # list (so we can sort it) and then iterate over the sorted
        # temporary list
        if self._sorted:
            return (obj for obj in sorted(walker, key=lambda _x: _x.local_name))
        else:
            return walker

    def iteritems(self):
        """
        TODO
        """
        # Ironically, the values are the fundamental thing that we
        # can (efficiently) iterate over in decl_order.  iteritems
        # just wraps itervalues.
        for obj in self.itervalues():
            yield (obj._name, obj)

    def keys(self):
        """
        Return a list of dictionary keys
        """
        return list(self.iterkeys())

    def values(self):
        """
        Return a list of dictionary values
        """
        return list(self.itervalues())

    def items(self):
        """
        Return a list of (key, value) tuples
        """
        return list(self.iteritems())

# In Python3, the items(), etc methods of dict-like things return
# generator-like objects.
if PY3:
    PseudoMap.keys = PseudoMap.iterkeys
    PseudoMap.values = PseudoMap.itervalues
    PseudoMap.items = PseudoMap.iteritems


class _BlockData(ActiveComponentData):
    """
    This class holds the fundamental block data.
    """
    _Block_reserved_words = set()

    def __init__(self, component):
        #
        # BLOCK DATA ELEMENTS
        #
        #   _decl_order:  [ (component, id_of_next_ctype_in_decl_order), ...]
        #   _decl:        { name : index_in_decl_order }
        #   _ctypes:      { ctype : [ id_first_ctype, id_last_ctype, count ] }
        #
        #
        # We used to define an internal structure that looked like:
        #
        #    _component    = { ctype : OrderedDict( name : obj ) }
        #    _declarations = OrderedDict( name : obj )
        #
        # This structure is convenient, but the overhead of carrying
        # around roughly 20 dictionaries for every block consumed a
        # nontrivial amount of memory.  Plus, the generation and
        # maintenance of OrderedDicts appeared to be disturbingly slow.
        #
        # We now "mock up" this data structure using 2 dicts and a list:
        #
        #    _ctypes     = { ctype : [ first idx, last idx, count ] }
        #    _decl       = { name  : idx }
        #    _decl_order = [ (obj, next_ctype_idx) ]
        #
        # Some notes: As Pyomo models rarely *delete* objects, we
        # currently never remove items from the _decl_order list.  If
        # the component is ever removed / cleared, we simply mark the
        # object as None.  If models crop up where we start seeing a
        # significant amount of adding / removing components, then we
        # can revisit this decision (although we will probably continue
        # marking entries as None and just periodically rebuild the list
        # as opposed to maintaining the list without any holes).
        #
        ActiveComponentData.__init__(self, component)
        # Note: call super() here to bypass the Block __setattr__
        #   _ctypes:      { ctype -> [1st idx, last idx, count] }
        #   _decl:        { name -> idx }
        #   _decl_order:  list( tuples( obj, next_type_idx ) )
        super(_BlockData, self).__setattr__('_ctypes', {})
        super(_BlockData, self).__setattr__('_decl', {})
        super(_BlockData, self).__setattr__('_decl_order', [])

    def __getstate__(self):
        # Note: _BlockData is NOT slot-ized, so we must pickle the
        # entire __dict__.  However, we want the base class's
        # __getstate__ to override our blanket approach here (i.e., it
        # will handle the _component weakref), so we will call the base
        # class's __getstate__ and allow it to overwrite the catch-all
        # approach we use here.
        ans = dict(self.__dict__)
        ans.update(super(_BlockData, self).__getstate__())
        # Note sure why we are deleting these...
        if '_repn' in ans:
            del ans['_repn']
        return ans

    #
    # The base class __setstate__ is sufficient (assigning all the
    # pickled attributes to the object is appropriate
    #
    # def __setstate__(self, state):
    #    pass

    def __getattr__(self, val):
        if val in ModelComponentFactory:
            return _component_decorator(
                self, ModelComponentFactory.get_class(val))
        # Since the base classes don't support getattr, we can just
        # throw the "normal" AttributeError
        raise AttributeError("'%s' object has no attribute '%s'"
                             % (self.__class__.__name__, val))

    def __setattr__(self, name, val):
        """
        Set an attribute of a block data object.
        """
        #
        # In general, the most common case for this is setting a *new*
        # attribute.  After that, there is updating an existing
        # Component value, with the least common case being resetting an
        # existing general attribute.
        #
        # Case 1.  Add an attribute that is not currently in the class.
        #
        if name not in self.__dict__:
            if isinstance(val, Component):
                #
                # Pyomo components are added with the add_component method.
                #
                self.add_component(name, val)
            else:
                #
                # Other Python objects are added with the standard __setattr__
                # method.
                #
                super(_BlockData, self).__setattr__(name, val)
        #
        # Case 2.  The attribute exists and it is a component in the
        #          list of declarations in this block.  We will use the
        #          val to update the value of that [scalar] component
        #          through its set_value() method.
        #
        elif name in self._decl:
            if isinstance(val, Component):
                #
                # The value is a component, so we replace the component in the
                # block.
                #
                if self._decl_order[self._decl[name]][0] is val:
                    return
                logger.warning(
                    "Implicitly replacing the Component attribute "
                    "%s (type=%s) on block %s with a new Component (type=%s)."
                    "\nThis is usually indicative of a modelling error.\n"
                    "To avoid this warning, use block.del_component() and "
                    "block.add_component()."
                    % (name, type(self.component(name)), self.name,
                       type(val)))
                self.del_component(name)
                self.add_component(name, val)
            else:
                #
                # The incoming value is not a component, so we set the
                # value in the existing component.
                #
                # Because we want to log a special error only if the
                # set_value attribute is missing, we will fetch the
                # attribute first and then call the method outside of
                # the try-except so as to not suppress any exceptions
                # generated while setting the value.
                #
                try:
                    _set_value = self._decl_order[self._decl[name]][0].set_value
                except AttributeError:
                    logger.error(
                        "Expected component %s (type=%s) on block %s to have a "
                        "'set_value' method, but none was found." %
                        (name, type(self.component(name)),
                         self.name))
                    raise
                #
                # Call the set_value method.
                #
                _set_value(val)
        #
        # Case 3. Handle setting non-Component attributes
        #
        else:
            #
            # NB: This is important: the _BlockData is either a scalar
            # Block (where _parent and _component are defined) or a
            # single block within an Indexed Block (where only
            # _component is defined).  Regardless, the
            # _BlockData.__init__() method declares these methods and
            # sets them either to None or a weakref.  Thus, we will
            # never have a problem converting these objects from
            # weakrefs into Blocks and back (when pickling); the
            # attribute is already in __dict__, we will not hit the
            # add_component / del_component branches above.  It also
            # means that any error checking we want to do when assigning
            # these attributes should be done here.
            #
            # NB: isintance() can be slow and we generally avoid it in
            # core methods.  However, it is only slow when it returns
            # False.  Since the common paths on this branch should
            # return True, this shouldn't be too inefficient.
            #
            if name == '_parent':
                if val is not None and not isinstance(val(), _BlockData):
                    raise ValueError(
                        "Cannot set the '_parent' attribute of Block '%s' "
                        "to a non-Block object (with type=%s); Did you "
                        "try to create a model component named '_parent'?"
                        % (self.name, type(val)))
                super(_BlockData, self).__setattr__(name, val)
            elif name == '_component':
                if val is not None and not isinstance(val(), _BlockData):
                    raise ValueError(
                        "Cannot set the '_component' attribute of Block '%s' "
                        "to a non-Block object (with type=%s); Did you "
                        "try to create a model component named '_component'?"
                        % (self.name, type(val)))
                super(_BlockData, self).__setattr__(name, val)
            #
            # At this point, we should only be seeing non-component data
            # the user is hanging on the blocks (uncommon) or the
            # initial setup of the object data (in __init__).
            #
            elif isinstance(val, Component):
                logger.warning(
                    "Reassigning the non-component attribute %s\n"
                    "on block (model).%s with a new Component\nwith type %s.\n"
                    "This is usually indicative of a modelling error.\n"
                    "To avoid this warning, explicitly delete the attribute:\n"
                    "    del %s.%s" % (
                        name, self.name, type(val), self.name, name))
                delattr(self, name)
                self.add_component(name, val)
            else:
                super(_BlockData, self).__setattr__(name, val)

    def __delattr__(self, name):
        """
        Delete an attribute on this Block.
        """
        #
        # It is important that we call del_component() whenever a
        # component is removed from a block.  Defining __delattr__
        # catches the case when a user attempts to remove components
        # using, e.g. "del model.myParam"
        #
        if name in self._decl:
            #
            # The attribute exists and it is a component in the
            # list of declarations in this block.
            #
            self.del_component(name)
        else:
            #
            # Other Python objects are removed with the standard __detattr__
            # method.
            #
            super(_BlockData, self).__delattr__(name)

    def _compact_decl_storage(self):
        idxMap = {}
        _new_decl_order = []
        j = 0
        # Squeeze out the None entries
        for i, entry in enumerate(self._decl_order):
            if entry[0] is not None:
                idxMap[i] = j
                j += 1
                _new_decl_order.append(entry)
        # Update the _decl map
        self._decl = {k:idxMap[idx] for k,idx in iteritems(self._decl)}
        # Update the ctypes, _decl_order linked lists
        for ctype, info in iteritems(self._ctypes):
            idx = info[0]
            entry = self._decl_order[idx]
            while entry[0] is None:
                idx = entry[1]
                entry = self._decl_order[idx]
            info[0] = last = idxMap[idx]
            while entry[1] is not None:
                idx = entry[1]
                entry = self._decl_order[idx]
                if entry[0] is not None:
                    this = idxMap[idx]
                    _new_decl_order[last] = (_new_decl_order[last][0], this)
                    last = this
            info[1] = last
            _new_decl_order[last] = (_new_decl_order[last][0], None)
        self._decl_order = _new_decl_order

    def set_value(self, val):
        raise RuntimeError(textwrap.dedent(
            """\
            Block components do not support assignment or set_value().
            Use the transfer_attributes_from() method to transfer the
            components and public attributes from one block to another:
                model.b[1].transfer_attributes_from(other_block)
            """))

    def clear(self):
        for name in iterkeys(self.component_map()):
            if name not in self._Block_reserved_words:
                self.del_component(name)
        for attr in tuple(self.__dict__):
            if attr not in self._Block_reserved_words:
                delattr(self, attr)
        self._compact_decl_storage()

    def transfer_attributes_from(self, src):
        """Transfer user-defined attributes from src to this block

        This transfers all components and user-defined attributes from
        the block or dictionary `src` and places them on this Block.
        Components are transferred in declaration order.

        If a Component on `src` is also declared on this block as either
        a Component or attribute, the local Component or attribute is
        replaced by the incoming component.  If an attribute name on
        `src` matches a Component declared on this block, then the
        incoming attribute is passed to the local Component's
        `set_value()` method.  Attribute names appearing in this block's
        `_Block_reserved_words` set will not be transferred (although
        Components will be).

        Parameters
        ----------
        src: _BlockData or dict
            The Block or mapping that contains the new attributes to
            assign to this block.
        """
        if isinstance(src, _BlockData):
            # There is a special case where assigning a parent block to
            # this block creates a circular hierarchy
            if src is self:
                return
            p_block = self.parent_block()
            while p_block is not None:
                if p_block is src:
                    raise ValueError(
                        "_BlockData.transfer_attributes_from(): Cannot set a "
                        "sub-block (%s) to a parent block (%s): creates a "
                        "circular hierarchy" % (self, src))
                p_block = p_block.parent_block()
            # record the components and the non-component objects added
            # to the block
            src_comp_map = src.component_map()
            src_raw_dict = {k:v for k,v in iteritems(src.__dict__)
                            if k not in src_comp_map}
        elif isinstance(src, Mapping):
            src_comp_map = {}
            src_raw_dict = src
        else:
            raise ValueError(
                "_BlockData.transfer_attributes_from(): expected a "
                "Block or dict; received %s" % (type(src).__name__,))

        # Use component_map for the components to preserve decl_order
        for k,v in iteritems(src_comp_map):
            if k in self._decl:
                self.del_component(k)
            src.del_component(k)
            self.add_component(k,v)
        # Because Blocks are not slotized and we allow the
        # assignment of arbitrary data to Blocks, we will move over
        # any other unrecognized entries in the object's __dict__:
        for k in sorted(iterkeys(src_raw_dict)):
            if k not in self._Block_reserved_words or not hasattr(self, k) \
               or k in self._decl:
                setattr(self, k, src_raw_dict[k])

    def _add_implicit_sets(self, val):
        """TODO: This method has known issues (see tickets) and needs to be
        reviewed. [JDS 9/2014]"""

        _component_sets = getattr(val, '_implicit_subsets', None)
        #
        # FIXME: The name attribute should begin with "_", and None
        # should replace "_unknown_"
        #
        if _component_sets is not None:
            for ctr, tset in enumerate(_component_sets):
                if tset.parent_component().parent_block() is None \
                        and not isinstance(tset.parent_component(), GlobalSetBase):
                    self.add_component("%s_index_%d" % (val.local_name, ctr), tset)
        if getattr(val, '_index', None) is not None \
                and isinstance(val._index, _SetDataBase) \
                and val._index.parent_component().parent_block() is None \
                and not isinstance(val._index.parent_component(), GlobalSetBase):
            self.add_component("%s_index" % (val.local_name,), val._index.parent_component())
        if getattr(val, 'initialize', None) is not None \
                and isinstance(val.initialize, _SetDataBase) \
                and val.initialize.parent_component().parent_block() is None \
                and not isinstance(val.initialize.parent_component(), GlobalSetBase):
            self.add_component("%s_index_init" % (val.local_name,), val.initialize.parent_component())
        if getattr(val, 'domain', None) is not None \
                and isinstance(val.domain, _SetDataBase) \
                and val.domain.parent_block() is None \
                and not isinstance(val.domain, GlobalSetBase):
            self.add_component("%s_domain" % (val.local_name,), val.domain)

    def _flag_vars_as_stale(self):
        """
        Configure *all* variables (on active blocks) and
        their composite _VarData objects as stale. This
        method is used prior to loading solver
        results. Variable that did not particpate in the
        solution are flagged as stale.  E.g., it most cases
        fixed variables will be flagged as stale since they
        are compiled out of expressions; however, many
        solver plugins support including fixed variables in
        the output problem by overriding bounds in order to
        minimize preprocessing requirements, meaning fixed
        variables are not necessarily always stale.
        """
        for variable in self.component_objects(Var, active=True):
            variable.flag_as_stale()

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
                have been deactivated) should be counted.
            descend_into (bool): Indicates whether or not
                component types should be counted on
                sub-blocks. Default is True.

        Returns: A set of component types.
        """
        assert active in (True, None)
        ctypes = set()
        for block in self.block_data_objects(active=active,
                                             descend_into=descend_into,
                                             sort=SortComponents.unsorted):
            if active is None:
                ctypes.update(ctype for ctype in block._ctypes)
            else:
                assert active is True
                for ctype in block._ctypes:
                    for component in block.component_data_objects(
                            ctype=ctype,
                            active=True,
                            descend_into=False,
                            sort=SortComponents.unsorted):
                        ctypes.add(ctype)
                        break  # just need 1 or more
        return ctypes

    def model(self):
        #
        # Special case: the "Model" is always the top-level _BlockData,
        # so if this is the top-level block, it must be the model
        #
        # Also note the interesting and intentional characteristic for
        # an IndexedBlock that is not attached to anything:
        #   b = Block([1,2,3])
        #   b.model() is None
        #   b[1].model() is b[1]
        #   b[2].model() is b[2]
        #
        ans = self.parent_block()
        if ans is None:
            return self
        #
        # NOTE: This loop is probably OK, since
        #   1) most models won't be nested very deep and
        #   2) it is better than forcing everyone to maintain references
        #      to the top-level block from both the standpoint of memory
        #      use and update time).
        #
        next = ans.parent_block()
        while next is not None:
            ans = next
            next = next.parent_block()
        return ans

    def find_component(self, label_or_component):
        """
        Return a block component given a name.
        """
        return ComponentUID(label_or_component).find_component_on(self)

    def add_component(self, name, val):
        """
        Add a component 'name' to the block.

        This method assumes that the attribute is not in the model.
        """
        #
        # Error checks
        #
        if not val.valid_model_component():
            raise RuntimeError(
                "Cannot add '%s' as a component to a block" % str(type(val)))
        if name in self._Block_reserved_words and hasattr(self, name):
            raise ValueError("Attempting to declare a block component using "
                             "the name of a reserved attribute:\n\t%s"
                             % (name,))
        if name in self.__dict__:
            raise RuntimeError(
                "Cannot add component '%s' (type %s) to block '%s': a "
                "component by that name (type %s) is already defined."
                % (name, type(val), self.name, type(getattr(self, name))))
        #
        # Skip the add_component() logic if this is a
        # component type that is suppressed.
        #
        _component = self.parent_component()
        _type = val.ctype
        if _type in _component._suppress_ctypes:
            return
        #
        # Raise an exception if the component already has a parent.
        #
        if (val._parent is not None) and (val._parent() is not None):
            if val._parent() is self:
                msg = """
Attempting to re-assign the component '%s' to the same
block under a different name (%s).""" % (val.name, name)
            else:
                msg = """
Re-assigning the component '%s' from block '%s' to
block '%s' as '%s'.""" % (val._name, val._parent().name,
                          self.name, name)

            raise RuntimeError("""%s

This behavior is not supported by Pyomo; components must have a
single owning block (or model), and a component may not appear
multiple times in a block.  If you want to re-name or move this
component, use the block del_component() and add_component() methods.
""" % (msg.strip(),))
        #
        # If the new component is a Block, then there is the chance that
        # it is the model(), and assigning it would create a circular
        # hierarchy.  Note that we only have to check the model as the
        # check immediately above would catch any "internal" blocks in
        # the block hierarchy
        #
        if isinstance(val, Block) and val is self.model():
            raise ValueError(
                "Cannot assign the top-level block as a subblock of one of "
                "its children (%s): creates a circular hierarchy"
                % (self,))
        #
        # Set the name and parent pointer of this component.
        #
        val._name = name
        val._parent = weakref.ref(self)
        #
        # We want to add the temporary / implicit sets first so that
        # they get constructed before this component
        #
        # FIXME: This is sloppy and wasteful (most components trigger
        # this, even when there is no need for it).  We should
        # reconsider the whole _implicit_subsets logic to defer this
        # kind of thing to an "update_parent()" method on the
        # components.
        #
        self._add_implicit_sets(val)
        #
        # Add the component to the underlying Component store
        #
        _new_idx = len(self._decl_order)
        self._decl[name] = _new_idx
        self._decl_order.append((val, None))
        #
        # Add the component as an attribute.  Note that
        #
        #     self.__dict__[name]=val
        #
        # is inappropriate here.  The correct way to add the attribute
        # is to delegate the work to the next class up the MRO.
        #
        super(_BlockData, self).__setattr__(name, val)
        #
        # Update the ctype linked lists
        #
        if _type in self._ctypes:
            idx_info = self._ctypes[_type]
            tmp = idx_info[1]
            self._decl_order[tmp] = (self._decl_order[tmp][0], _new_idx)
            idx_info[1] = _new_idx
            idx_info[2] += 1
        else:
            self._ctypes[_type] = [_new_idx, _new_idx, 1]
        #
        # Propagate properties to sub-blocks:
        #   suppressed ctypes
        #
        if _type is Block:
            val._suppress_ctypes |= _component._suppress_ctypes
        #
        # Error, for disabled support implicit rule names
        #
        if '_rule' in val.__dict__ and val._rule is None:
            _found = False
            try:
                _test = val.local_name + '_rule'
                for i in (1, 2):
                    frame = sys._getframe(i)
                    _found |= _test in frame.f_locals
            except:
                pass
            if _found:
                # JDS: Do not blindly reformat this message.  The
                # formatter inserts arbitrarily-long names(), which can
                # cause the resulting logged message to be very poorly
                # formatted due to long lines.
                logger.warning(
                    """As of Pyomo 4.0, Pyomo components no longer support implicit rules.
You defined a component (%s) that appears
to rely on an implicit rule (%s).
Components must now specify their rules explicitly using 'rule=' keywords.""" %
                    (val.name, _test))
        #
        # Don't reconstruct if this component has already been constructed.
        # This allows a user to move a component from one block to
        # another.
        #
        if val._constructed is True:
            return
        #
        # If the block is Concrete, construct the component
        # Note: we are explicitly using getattr because (Scalar)
        #   classes that derive from Block may want to declare components
        #   within their __init__() [notably, pyomo.gdp's Disjunct).
        #   Those components are added *before* the _constructed flag is
        #   added to the class by Block.__init__()
        #
        if getattr(_component, '_constructed', False):
            # NB: we don't have to construct the temporary / implicit
            # sets here: if necessary, that happens when
            # _add_implicit_sets() calls add_component().
            if _BlockConstruction.data:
                data = _BlockConstruction.data.get(id(self), None)
                if data is not None:
                    data = data.get(name, None)
            else:
                data = None
            generate_debug_messages = is_debug_set(logger)
            if generate_debug_messages:
                # This is tricky: If we are in the middle of
                # constructing an indexed block, the block component
                # already has _constructed=True.  Now, if the
                # _BlockData.__init__() defines any local variables
                # (like pyomo.gdp.Disjunct's indicator_var), name(True)
                # will fail: this block data exists and has a parent(),
                # but it has not yet been added to the parent's _data
                # (so the idx lookup will fail in name).
                if self.parent_block() is None:
                    _blockName = "[Model]"
                else:
                    try:
                        _blockName = "Block '%s'" % self.name
                    except:
                        _blockName = "Block '%s[...]'" \
                            % self.parent_component().name
                logger.debug("Constructing %s '%s' on %s from data=%s",
                             val.__class__.__name__, name,
                             _blockName, str(data))
            try:
                val.construct(data)
            except:
                err = sys.exc_info()[1]
                logger.error(
                    "Constructing component '%s' from data=%s failed:\n%s: %s",
                    str(val.name), str(data).strip(),
                    type(err).__name__, err)
                raise
            if generate_debug_messages:
                if _blockName[-1] == "'":
                    _blockName = _blockName[:-1] + '.' + name + "'"
                else:
                    _blockName = "'" + _blockName + '.' + name + "'"
                _out = StringIO()
                val.pprint(ostream=_out)
                logger.debug("Constructed component '%s':\n%s"
                             % (_blockName, _out.getvalue()))

    def del_component(self, name_or_object):
        """
        Delete a component from this block.
        """
        obj = self.component(name_or_object)
        # FIXME: Is this necessary?  Should this raise an exception?
        if obj is None:
            return

        # FIXME: Is this necessary?  Should this raise an exception?
        # if name not in self._decl:
        #    return

        name = obj.local_name

        # Replace the component in the master list with a None placeholder
        idx = self._decl[name]
        del self._decl[name]
        self._decl_order[idx] = (None, self._decl_order[idx][1])

        # Update the ctype linked lists
        ctype_info = self._ctypes[obj.ctype]
        ctype_info[2] -= 1
        if ctype_info[2] == 0:
            del self._ctypes[obj.ctype]

        # Clear the _parent attribute
        obj._parent = None

        # Now that this component is not in the _decl map, we can call
        # delattr as usual.
        #
        #del self.__dict__[name]
        #
        # Note: 'del self.__dict__[name]' is inappropriate here.  The
        # correct way to add the attribute is to delegate the work to
        # the next class up the MRO.
        super(_BlockData, self).__delattr__(name)

    def reclassify_component_type(self, name_or_object, new_ctype,
                                  preserve_declaration_order=True):
        """
        TODO
        """
        obj = self.component(name_or_object)
        # FIXME: Is this necessary?  Should this raise an exception?
        if obj is None:
            return

        if obj.ctype is new_ctype:
            return

        name = obj.local_name
        if not preserve_declaration_order:
            # if we don't have to preserve the decl order, then the
            # easiest (and fastest) thing to do is just delete it and
            # re-add it.
            self.del_component(name)
            obj._ctype = new_ctype
            self.add_component(name, obj)
            return

        idx = self._decl[name]

        # Update the ctype linked lists
        ctype_info = self._ctypes[obj.ctype]
        ctype_info[2] -= 1
        if ctype_info[2] == 0:
            del self._ctypes[obj.ctype]
        elif ctype_info[0] == idx:
            ctype_info[0] = self._decl_order[idx][1]
        else:
            prev = None
            tmp = self._ctypes[obj.ctype][0]
            while tmp < idx:
                prev = tmp
                tmp = self._decl_order[tmp][1]

            self._decl_order[prev] = (self._decl_order[prev][0],
                                      self._decl_order[idx][1])
            if ctype_info[1] == idx:
                ctype_info[1] = prev

        obj._ctype = new_ctype

        # Insert into the new ctype list
        if new_ctype not in self._ctypes:
            self._ctypes[new_ctype] = [idx, idx, 1]
            self._decl_order[idx] = (obj, None)
        elif idx < self._ctypes[new_ctype][0]:
            self._decl_order[idx] = (obj, self._ctypes[new_ctype][0])
            self._ctypes[new_ctype][0] = idx
            self._ctypes[new_ctype][2] += 1
        elif idx > self._ctypes[new_ctype][1]:
            prev = self._ctypes[new_ctype][1]
            self._decl_order[prev] = (self._decl_order[prev][0], idx)
            self._decl_order[idx] = (obj, None)
            self._ctypes[new_ctype][1] = idx
            self._ctypes[new_ctype][2] += 1
        else:
            self._ctypes[new_ctype][2] += 1
            prev = None
            tmp = self._ctypes[new_ctype][0]
            while tmp < idx:
                # this test should be unnecessary: and tmp is not None:
                prev = tmp
                tmp = self._decl_order[tmp][1]
            self._decl_order[prev] = (self._decl_order[prev][0], idx)
            self._decl_order[idx] = (obj, tmp)

    def clone(self):
        """
        TODO
        """
        # FYI: we used to remove all _parent() weakrefs before
        # deepcopying and then restore them on the original and cloned
        # model.  It turns out that this was completely unnecessary and
        # wasteful.

        #
        # Note: Setting __block_scope__ determines which components are
        # deepcopied (anything beneath this block) and which are simply
        # preserved as references (anything outside this block
        # hierarchy).  We must always go through this effort to prevent
        # copying certain "reserved" components (like Any,
        # NonNegativeReals, etc) that are not "owned" by any blocks and
        # should be preserved as singletons.
        #
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

    def contains_component(self, ctype):
        """
        Return True if the component type is in _ctypes and ... TODO.
        """
        return ctype in self._ctypes and self._ctypes[ctype][2]

    def component(self, name_or_object):
        """
        Return a child component of this block.

        If passed a string, this will return the child component
        registered by that name.  If passed a component, this will
        return that component IFF the component is a child of this
        block. Returns None on lookup failure.
        """
        if isinstance(name_or_object, string_types):
            if name_or_object in self._decl:
                return self._decl_order[self._decl[name_or_object]][0]
        else:
            try:
                obj = name_or_object.parent_component()
                if obj.parent_block() is self:
                    return obj
            except AttributeError:
                pass
        return None

    def component_map(self, ctype=None, active=None, sort=False):
        """Returns a PseudoMap of the components in this block.

        Parameters
        ----------
        ctype:  None or type or iterable
            Specifies the component types (`ctypes`) to include in the
            resulting PseudoMap

                =============   ===================
                None            All components
                type            A single component type
                iterable        All component types in the iterable
                =============   ===================

        active: None or bool
            Filter components by the active flag

                =====  ===============================
                None   Return all components
                True   Return only active components
                False  Return only inactive components
                =====  ===============================

        sort: bool
            Iterate over the components in a sorted otder

                =====  ================================================
                True   Iterate using Block.alphabetizeComponentAndIndex
                False  Iterate using Block.declarationOrder
                =====  ================================================

        """
        return PseudoMap(self, ctype, active, sort)

    def _component_typemap(self, ctype=None, active=None, sort=False):
        """
        Return information about the block components.

        If ctype is None, return a dictionary that maps
           {component type -> {name -> component instance}}
        Otherwise, return a dictionary that maps
           {name -> component instance}
        for the specified component type.

        Note: The actual {name->instance} object is a PseudoMap that
        implements a lightweight interface to the underlying
        BlockComponents data structures.
        """
        if ctype is None:
            ans = {}
            for x in self._ctypes:
                ans[x] = PseudoMap(self, x, active, sort)
            return ans
        else:
            return PseudoMap(self, ctype, active, sort)

    def _component_data_iter(self, ctype=None, active=None, sort=False):
        """
        Generator that returns a 3-tuple of (component name, index value,
        and _ComponentData) for every component data in the block.
        """
        _sort_indices = SortComponents.sort_indices(sort)
        _subcomp = PseudoMap(self, ctype, active, sort)
        for name, comp in _subcomp.iteritems():
            # NOTE: Suffix has a dict interface (something other derived
            #   non-indexed Components may do as well), so we don't want
            #   to test the existence of iteritems as a check for
            #   component datas. We will rely on is_indexed() to catch
            #   all the indexed components.  Then we will do special
            #   processing for the scalar components to catch the case
            #   where there are "sparse scalar components"
            if comp.is_indexed():
                _items = comp.iteritems()
            elif hasattr(comp, '_data'):
                # This may be an empty Scalar component (e.g., from
                # Constraint.Skip on a scalar Constraint)
                assert len(comp._data) <= 1
                _items = iteritems(comp._data)
            else:
                _items = ((None, comp),)

            if _sort_indices:
                _items = sorted(_items, key=itemgetter(0))
            if active is None or not isinstance(comp, ActiveIndexedComponent):
                for idx, compData in _items:
                    yield (name, idx), compData
            else:
                for idx, compData in _items:
                    if compData.active == active:
                        yield (name, idx), compData

    @deprecated("The all_components method is deprecated.  "
                "Use the Block.component_objects() method.",
                version="4.1.10486")
    def all_components(self, *args, **kwargs):
        return self.component_objects(*args, **kwargs)

    @deprecated("The active_components method is deprecated.  "
                "Use the Block.component_objects() method.",
                version="4.1.10486")
    def active_components(self, *args, **kwargs):
        kwargs['active'] = True
        return self.component_objects(*args, **kwargs)

    @deprecated("The all_component_data method is deprecated.  "
                "Use the Block.component_data_objects() method.",
                version="4.1.10486")
    def all_component_data(self, *args, **kwargs):
        return self.component_data_objects(*args, **kwargs)

    @deprecated("The active_component_data method is deprecated.  "
                "Use the Block.component_data_objects() method.",
                version="4.1.10486")
    def active_component_data(self, *args, **kwargs):
        kwargs['active'] = True
        return self.component_data_objects(*args, **kwargs)

    def component_objects(self, ctype=None, active=None, sort=False,
                          descend_into=True, descent_order=None):
        """
        Return a generator that iterates through the
        component objects in a block.  By default, the
        generator recursively descends into sub-blocks.
        """
        if not descend_into:
            for x in self.component_map(ctype, active, sort).itervalues():
                yield x
            return
        for _block in self.block_data_objects(active, sort, descend_into, descent_order):
            for x in _block.component_map(ctype, active, sort).itervalues():
                yield x

    def component_data_objects(self,
                               ctype=None,
                               active=None,
                               sort=False,
                               descend_into=True,
                               descent_order=None):
        """
        Return a generator that iterates through the
        component data objects for all components in a
        block.  By default, this generator recursively
        descends into sub-blocks.
        """
        if descend_into:
            block_generator = self.block_data_objects(
                active=active,
                sort=sort,
                descend_into=descend_into,
                descent_order=descent_order)
        else:
            block_generator = (self,)

        for _block in block_generator:
            for x in _block._component_data_iter(ctype=ctype,
                                                 active=active,
                                                 sort=sort):
                yield x[1]

    def component_data_iterindex(self,
                                 ctype=None,
                                 active=None,
                                 sort=False,
                                 descend_into=True,
                                 descent_order=None):
        """
        Return a generator that returns a tuple for each
        component data object in a block.  By default, this
        generator recursively descends into sub-blocks.  The
        tuple is

            ((component name, index value), _ComponentData)

        """
        if descend_into:
            block_generator = self.block_data_objects(
                active=active,
                sort=sort,
                descend_into=descend_into,
                descent_order=descent_order)
        else:
            block_generator = (self,)

        for _block in block_generator:
            for x in _block._component_data_iter(ctype=ctype,
                                                 active=active,
                                                 sort=sort):
                yield x

    @deprecated("The all_blocks method is deprecated.  "
                "Use the Block.block_data_objects() method.",
                version="4.1.10486")
    def all_blocks(self, *args, **kwargs):
        return self.block_data_objects(*args, **kwargs)

    @deprecated("The active_blocks method is deprecated.  "
                "Use the Block.block_data_objects() method.",
                version="4.1.10486")
    def active_blocks(self, *args, **kwargs):
        kwargs['active'] = True
        return self.block_data_objects(*args, **kwargs)

    def block_data_objects(self,
                           active=None,
                           sort=False,
                           descend_into=True,
                           descent_order=None):

        """
        This method returns a generator that iterates
        through the current block and recursively all
        sub-blocks.  This is semantically equivalent to

            component_data_objects(Block, ...)

        """
        if descend_into is False:
            if active is not None and self.active != active:
                # Return an iterator over an empty tuple
                return ().__iter__()
            else:
                return (self,).__iter__()
        #
        # Rely on the _tree_iterator:
        #
        if descend_into is True:
            descend_into = (Block,)
        elif isclass(descend_into):
            descend_into = (descend_into,)
        return self._tree_iterator(ctype=descend_into,
                                   active=active,
                                   sort=sort,
                                   traversal=descent_order)

    def _tree_iterator(self,
                       ctype=None,
                       active=None,
                       sort=None,
                       traversal=None):

        # TODO: merge into block_data_objects
        if ctype is None:
            ctype = (Block,)
        elif isclass(ctype):
            ctype = (ctype,)

        # A little weird, but since we "normally" return a generator, we
        # will return a generator for an empty list instead of just
        # returning None or an empty list here (so that consumers can
        # count on us always returning a generator)
        if active is not None and self.active != active:
            return ().__iter__()

        # ALWAYS return the "self" Block, even if it does not match
        # ctype.  This is because we map this ctype to the
        # "descend_into" argument in public calling functions: callers
        # expect that the called thing will be iterated over.
        #
        # if self.parent_component().ctype not in ctype:
        #    return ().__iter__()

        if traversal is None or \
                traversal == TraversalStrategy.PrefixDepthFirstSearch:
            return self._prefix_dfs_iterator(ctype, active, sort)
        elif traversal == TraversalStrategy.BreadthFirstSearch:
            return self._bfs_iterator(ctype, active, sort)
        elif traversal == TraversalStrategy.PostfixDepthFirstSearch:
            return self._postfix_dfs_iterator(ctype, active, sort)
        else:
            raise RuntimeError("unrecognized traversal strategy: %s"
                               % (traversal, ))

    def _prefix_dfs_iterator(self, ctype, active, sort):
        """Helper function implementing a non-recursive prefix order
        depth-first search.  That is, the parent is returned before its
        children.

        Note: this method assumes it is called ONLY by the _tree_iterator
        method, which centralizes certain error checking and
        preliminaries.
        """
        PM = PseudoMap(self, ctype, active, sort)
        _stack = [(self,).__iter__(), ]
        while _stack:
            try:
                PM._block = _block = advance_iterator(_stack[-1])
                yield _block
                if not PM:
                    continue
                _stack.append(_block.component_data_objects(ctype=ctype,
                                                            active=active,
                                                            sort=sort,
                                                            descend_into=False))
            except StopIteration:
                _stack.pop()

    def _postfix_dfs_iterator(self, ctype, active, sort):
        """
        Helper function implementing a non-recursive postfix
        order depth-first search.  That is, the parent is
        returned after its children.

        Note: this method assumes it is called ONLY by the
        _tree_iterator method, which centralizes certain
        error checking and preliminaries.
        """
        _stack = [(self, self.component_data_iterindex(ctype, active, sort, False))]
        while _stack:
            try:
                _sub = advance_iterator(_stack[-1][1])[-1]
                _stack.append((_sub,
                               _sub.component_data_iterindex(ctype, active, sort, False)
                               ))
            except StopIteration:
                yield _stack.pop()[0]

    def _bfs_iterator(self, ctype, active, sort):
        """Helper function implementing a non-recursive breadth-first search.
        That is, all children at one level in the tree are returned
        before any of the children at the next level.

        Note: this method assumes it is called ONLY by the _tree_iterator
        method, which centralizes certain error checking and
        preliminaries.

        """
        if SortComponents.sort_indices(sort):
            if SortComponents.sort_names(sort):
                sorter = itemgetter(1, 2)
            else:
                sorter = itemgetter(0, 2)
        elif SortComponents.sort_names(sort):
            sorter = itemgetter(1)
        else:
            sorter = None

        _levelQueue = {0: (((None, None, self,),),)}
        while _levelQueue:
            _level = min(_levelQueue)
            _queue = _levelQueue.pop(_level)
            if not _queue:
                break
            if sorter is None:
                _queue = _levelWalker(_queue)
            else:
                _queue = sorted(_sortingLevelWalker(_queue), key=sorter)

            _level += 1
            _levelQueue[_level] = []
            # JDS: rework the _levelQueue logic so we don't need to
            # merge the key/value returned by the new
            # component_data_iterindex() method.
            for _items in _queue:
                yield _items[-1]  # _block
                _levelQueue[_level].append(
                    tmp[0] + (tmp[1],) for tmp in
                    _items[-1].component_data_iterindex(ctype=ctype,
                                                        active=active,
                                                        sort=sort,
                                                        descend_into=False))

    def fix_all_vars(self):
        # TODO: Simplify based on recursive logic
        for var in itervalues(self.component_map(Var)):
            var.fix()
        for block in itervalues(self.component_map(Block)):
            block.fix_all_vars()

    def unfix_all_vars(self):
        # TODO: Simplify based on recursive logic
        for var in itervalues(self.component_map(Var)):
            var.unfix()
        for block in itervalues(self.component_map(Block)):
            block.unfix_all_vars()

    def is_constructed(self):
        """
        A boolean indicating whether or not all *active* components of the
        input model have been properly constructed.
        """
        if not self.parent_component()._constructed:
            return False
        for x in self._decl_order:
            if x[0] is not None and x[0].active and not x[0].is_constructed():
                return False
        return True

    def _pprint_blockdata_components(self, ostream):
        #
        # We hard-code the order of the core Pyomo modeling
        # components, to ensure that the output follows the logical order
        # that expected by a user.
        #
        import pyomo.core.base.component_order
        items = list(pyomo.core.base.component_order.items)
        items_set = set(items)
        items_set.add(Block)
        #
        # Collect other model components that are registered
        # with the IModelComponent extension point.  These are appended
        # to the end of the list of the list.
        #
        dynamic_items = set()
        for item in self._ctypes:
            if not item in items_set:
                dynamic_items.add(item)
        # extra items get added alphabetically (so output is consistent)
        items.append(Block)
        items.extend(sorted(dynamic_items, key=lambda x: x.__name__))

        indented_ostream = StreamIndenter(ostream, self._PPRINT_INDENT)
        for item in items:
            keys = sorted(self.component_map(item))
            if not keys:
                continue
            #
            # NOTE: these conditional checks should not be hard-coded.
            #
            ostream.write("%d %s Declarations\n"
                          % (len(keys), item.__name__))
            for key in keys:
                self.component(key).pprint(ostream=indented_ostream)
            ostream.write("\n")
        #
        # Model Order
        #
        decl_order_keys = list(self.component_map().keys())
        ostream.write("%d Declarations: %s\n"
                      % (len(decl_order_keys),
                          ' '.join(str(x) for x in decl_order_keys)))

    def display(self, filename=None, ostream=None, prefix=""):
        """
        Print the Pyomo model in a verbose format.
        """
        if filename is not None:
            OUTPUT = open(filename, "w")
            self.display(ostream=OUTPUT, prefix=prefix)
            OUTPUT.close()
            return
        if ostream is None:
            ostream = sys.stdout
        if self.parent_block() is not None:
            ostream.write(prefix + "Block " + self.name + '\n')
        else:
            ostream.write(prefix + "Model " + self.name + '\n')
        #
        # FIXME: We should change the display order (to Obj, Var, Con,
        # Block) and change the printer to only display sections with
        # active components.  That will fix the need for the special
        # case for blocks below.  I am not implementing this now as it
        # would break tests just before a release.  [JDS 1/7/15]
        import pyomo.core.base.component_order
        for item in pyomo.core.base.component_order.display_items:
            #
            ostream.write(prefix + "\n")
            ostream.write(prefix + "  %s:\n" % pyomo.core.base.component_order.display_name[item])
            ACTIVE = self.component_map(item, active=True)
            if not ACTIVE:
                ostream.write(prefix + "    None\n")
            else:
                for obj in itervalues(ACTIVE):
                    obj.display(prefix=prefix + "    ", ostream=ostream)

        item = Block
        ACTIVE = self.component_map(item, active=True)
        if ACTIVE:
            ostream.write(prefix + "\n")
            ostream.write(
                prefix + "  %s:\n" %
                pyomo.core.base.component_order.display_name[item])
            for obj in itervalues(ACTIVE):
                obj.display(prefix=prefix + "    ", ostream=ostream)

    #
    # The following methods are needed to support passing blocks as
    # models to a solver.
    #

    def valid_problem_types(self):
        """This method allows the pyomo.opt convert function to work with a
        Model object."""
        return [ProblemFormat.pyomo]

    def write(self,
              filename=None,
              format=None,
              solver_capability=None,
              io_options={}):
        """
        Write the model to a file, with a given format.
        """
        #
        # Guess the format if none is specified
        #
        if (filename is None) and (format is None):
            # Preserving backwards compatibility here.
            # The function used to be defined with format='lp' by
            # default, but this led to confusing behavior when a
            # user did something like 'model.write("f.nl")' and
            # expected guess_format to create an NL file.
            format = ProblemFormat.cpxlp
        if filename is not None:
            try:
                _format = guess_format(filename)
            except AttributeError:
                # End up here if an ostream is passed to the filename argument
                _format = None
            if format is None:
                if _format is None:
                    raise ValueError(
                        "Could not infer file format from file name '%s'.\n"
                        "Either provide a name with a recognized extension "
                        "or specify the format using the 'format' argument."
                        % filename)
                else:
                    format = _format
            elif format != _format and _format is not None:
                logger.warning(
                    "Filename '%s' likely does not match specified "
                    "file format (%s)" % (filename, format))
        problem_writer = WriterFactory(format)
        if problem_writer is None:
            raise ValueError(
                "Cannot write model in format '%s': no model "
                "writer registered for that format"
                % str(format))

        if solver_capability is None:
            def solver_capability(x): return True
        (filename, smap) = problem_writer(self,
                                          filename,
                                          solver_capability,
                                          io_options)
        smap_id = id(smap)
        if not hasattr(self, 'solutions'):
            # This is a bit of a hack.  The write() method was moved
            # here from PyomoModel to support the solution of arbitrary
            # blocks in a hierarchical model.  However, we cannot import
            # PyomoModel at the beginning of the file due to a circular
            # import.  When we rearchitect the solution writers/solver
            # API, we should revisit this and remove the circular
            # dependency (we only need it here because we store the
            # SymbolMap returned by the writer in the solutions).
            from pyomo.core.base.PyomoModel import ModelSolutions
            self.solutions = ModelSolutions(self)
        self.solutions.add_symbol_map(smap)

        if is_debug_set(logger):
            logger.debug(
                "Writing model '%s' to file '%s' with format %s",
                self.name,
                str(filename),
                str(format))
        return filename, smap_id


@ModelComponentFactory.register("A component that contains one or more model components.")
class Block(ActiveIndexedComponent):
    """
    Blocks are indexed components that contain other components
    (including blocks).  Blocks have a global attribute that defines
    whether construction is deferred.  This applies to all components
    that they contain except blocks.  Blocks contained by other
    blocks use their local attribute to determine whether construction
    is deferred.
    """

    _ComponentDataClass = _BlockData

    def __new__(cls, *args, **kwds):
        if cls != Block:
            return super(Block, cls).__new__(cls)
        if not args or (args[0] is UnindexedComponent_set and len(args) == 1):
            return SimpleBlock.__new__(SimpleBlock)
        else:
            return IndexedBlock.__new__(IndexedBlock)

    def __init__(self, *args, **kwargs):
        """Constructor"""
        self._suppress_ctypes = set()
        _rule = kwargs.pop('rule', None)
        _options = kwargs.pop('options', None)
        # As concrete applies to the Block at declaration time, we will
        # not use an initializer.
        _concrete = kwargs.pop('concrete', False)
        # As dense applies to the whole container, we will not use an
        # initializer
        self._dense = kwargs.pop('dense', True)
        kwargs.setdefault('ctype', Block)
        ActiveIndexedComponent.__init__(self, *args, **kwargs)
        if _options is not None:
            deprecation_warning(
                "The Block 'options=' keyword is deprecated.  "
                "Equivalent functionality can be obtained by wrapping "
                "the rule function to add the options dictionary to "
                "the function arguments", version='5.7.2')
            if self.is_indexed():
                def rule_wrapper(model, *_idx):
                    return _rule(model, *_idx, **_options)
            else:
                def rule_wrapper(model):
                    return _rule(model, **_options)
            self._rule = Initializer(rule_wrapper)
        else:
            self._rule = Initializer(_rule)
        if _concrete:
            # Call self.construct() as opposed to just setting the _constructed
            # flag so that the base class construction procedure fires (this
            # picks up any construction rule that the user may provide)
            self.construct()

    def _getitem_when_not_present(self, idx):
        _block = self._setitem_when_not_present(idx)
        if self._rule is None:
            return _block

        if _BlockConstruction.data:
            data = _BlockConstruction.data.get(id(self), None)
            if data is not None:
                data = data.get(idx, None)
            if data is not None:
                # Note that for scalar Blocks, this will override the
                # entry for _BlockConstruction.data[id(self)], as _block
                # is self.
                _BlockConstruction.data[id(_block)] = data
        else:
            data = None

        try:
            obj = self._rule(_block, idx)
            # If the user returns a block, transfer over everything
            # they defined into the empty one we created.  We do
            # this inside the try block so that any abstract
            # components declared by the rule have the opportunity
            # to be initialized with data from
            # _BlockConstruction.data as they are transferred over.
            if obj is not _block and isinstance(obj, _BlockData):
                _block.transfer_attributes_from(obj)
        finally:
            if data is not None and _block is not self:
                del _BlockConstruction.data[id(_block)]

        # TBD: Should we allow skipping Blocks???
        # if obj is Block.Skip and idx is not None:
        #   del self._data[idx]
        return _block

    def construct(self, data=None):
        """
        Initialize the block
        """
        if is_debug_set(logger):
            logger.debug("Constructing %s '%s', from data=%s",
                         self.__class__.__name__, self.name, str(data))
        if self._constructed:
            return
        timer = ConstructionTimer(self)
        self._constructed = True

        # Constructing blocks is tricky.  Scalar blocks are already
        # partially constructed (they have _data[None] == self) in order
        # to support Abstract blocks.  The block may therefore already
        # have components declared on it.  In order to preserve
        # decl_order, we must construct those components *first* before
        # firing any rule.  Indexed blocks should be empty, so we only
        # need to fire the rule in order.
        #
        #  Since the rule does not pass any "data" on, we build a scalar
        #  "stack" of pointers to block data (_BlockConstruction.data)
        #  that the individual blocks' add_component() can refer back to
        #  to handle component construction.
        if data is not None:
            _BlockConstruction.data[id(self)] = data
        try:
            if self.is_indexed():
                # We can only populate Blocks with finite indexing sets
                if self.index_set().isfinite() and (
                        self._dense or self._rule is not None):
                    for _idx in self.index_set():
                        # Trigger population & call the rule
                        self._getitem_when_not_present(_idx)
            else:
                # We must check that any pre-existing components are
                # constructed.  This catches the case where someone is
                # building a Concrete model by building (potentially
                # pseudo-abstract) sub-blocks and then adding them to a
                # Concrete model block.
                _idx = next(iter(UnindexedComponent_set))
                _predefined_components = self.component_map()
                if _predefined_components:
                    if _idx not in self._data:
                        # Derived block classes may not follow the scalar
                        # Block convention of initializing _data to point to
                        # itself (i.e., they are not set up to support
                        # Abstract models)
                        self._data[_idx] = self
                    if data is not None:
                        data = data.get(_idx, None)
                    if data is None:
                        data = {}
                    for name, obj in iteritems(_predefined_components):
                        if not obj._constructed:
                            obj.construct(data.get(name, None))
                # Trigger the (normal) initialization of the block
                self._getitem_when_not_present(_idx)
        finally:
            # We must allow that id(self) may no longer be in
            # _BlockConstruction.data, as _getitem_when_not_present will
            # have already removed the entry for scalar blocks (as the
            # BlockData and the Block component are the same object)
            if data is not None:
                _BlockConstruction.data.pop(id(self), None)
            timer.report()

    def _pprint_callback(self, ostream, idx, data):
        if not self.is_indexed():
            data._pprint_blockdata_components(ostream)
        else:
            ostream.write("%s : Active=%s\n" % (data.name, data.active))
            ostream = StreamIndenter(ostream, self._PPRINT_INDENT)
            data._pprint_blockdata_components(ostream)

    def _pprint(self):
        _attrs = [
            ("Size", len(self)),
            ("Index", self._index if self.is_indexed() else None),
            ('Active', self.active),
        ]
        # HACK: suppress the top-level block header (for historical reasons)
        if self.parent_block() is None and not self.is_indexed():
            return None, iteritems(self._data), None, self._pprint_callback
        else:
            return _attrs, iteritems(self._data), None, self._pprint_callback

    def display(self, filename=None, ostream=None, prefix=""):
        """
        Display values in the block
        """
        if filename is not None:
            OUTPUT = open(filename, "w")
            self.display(ostream=OUTPUT, prefix=prefix)
            OUTPUT.close()
            return
        if ostream is None:
            ostream = sys.stdout

        for key in sorted(self):
            _BlockData.display(self[key], filename, ostream, prefix)


class SimpleBlock(_BlockData, Block):

    def __init__(self, *args, **kwds):
        _BlockData.__init__(self, component=self)
        Block.__init__(self, *args, **kwds)
        # Initialize the data dict so that (abstract) attribute
        # assignment will work.  Note that we do not trigger
        # get/setitem_when_not_present so that we do not (implicitly)
        # trigger the Block rule
        self._data[None] = self

    # We want scalar Blocks to pick up the Block display method
    display = Block.display


class IndexedBlock(Block):

    def __init__(self, *args, **kwds):
        Block.__init__(self, *args, **kwds)


#
# Deprecated functions.
#
@deprecated("generate_cuid_names() is deprecated. "
            "Use the ComponentUID.generate_cuid_string_map() static method",
            version="5.7.2")
def generate_cuid_names(block, ctype=None, descend_into=True):
    return ComponentUID.generate_cuid_string_map(block, ctype, descend_into)

@deprecated("The active_components function is deprecated.  "
            "Use the Block.component_objects() method.",
            version="4.1.10486")
def active_components(block, ctype, sort_by_names=False, sort_by_keys=False):
    return block.component_objects(ctype, active=True, sort=sort_by_names)


@deprecated("The components function is deprecated.  "
            "Use the Block.component_objects() method.",
            version="4.1.10486")
def components(block, ctype, sort_by_names=False, sort_by_keys=False):
    return block.component_objects(ctype, active=False, sort=sort_by_names)


@deprecated("The active_components_data function is deprecated.  "
            "Use the Block.component_data_objects() method.",
            version="4.1.10486")
def active_components_data(block, ctype,
                           sort=None, sort_by_keys=False, sort_by_names=False):
    return block.component_data_objects(ctype=ctype, active=True, sort=sort)


@deprecated("The components_data function is deprecated.  "
            "Use the Block.component_data_objects() method.",
            version="4.1.10486")
def components_data(block, ctype,
                    sort=None, sort_by_keys=False, sort_by_names=False):
    return block.component_data_objects(ctype=ctype, active=False, sort=sort)


#
# Create a Block and record all the default attributes, methods, etc.
# These will be assumes to be the set of illegal component names.
#
_BlockData._Block_reserved_words = set(dir(Block()))


class _IndexedCustomBlockMeta(type):
    """Metaclass for creating an indexed custom block.
    """

    pass


class _ScalarCustomBlockMeta(type):
    """Metaclass for creating a scalar custom block.
    """

    def __new__(meta, name, bases, dct):
        def __init__(self, *args, **kwargs):
            # bases[0] is the custom block data object
            bases[0].__init__(self, component=self)
            # bases[1] is the custom block object that
            # is used for declaration
            bases[1].__init__(self, *args, **kwargs)

        dct["__init__"] = __init__
        return type.__new__(meta, name, bases, dct)


class CustomBlock(Block):
    """ The base class used by instances of custom block components
    """

    def __init__(self, *args, **kwds):
        if self._default_ctype is not None:
            kwds.setdefault('ctype', self._default_ctype)
        Block.__init__(self, *args, **kwds)


    def __new__(cls, *args, **kwds):
        if cls.__name__.startswith('_Indexed') or \
                cls.__name__.startswith('_Scalar'):
            # we are entering here the second time (recursive)
            # therefore, we need to create what we have
            return super(CustomBlock, cls).__new__(cls)
        if not args or (args[0] is UnindexedComponent_set and len(args) == 1):
            n = _ScalarCustomBlockMeta(
                "_Scalar%s" % (cls.__name__,),
                (cls._ComponentDataClass, cls),
                {}
            )
            return n.__new__(n)
        else:
            n = _IndexedCustomBlockMeta(
                "_Indexed%s" % (cls.__name__,),
                (cls,),
                {}
            )
            return n.__new__(n)


def declare_custom_block(name, new_ctype=None):
    """ Decorator to declare components for a custom block data class

    >>> @declare_custom_block(name=FooBlock)
    ... class FooBlockData(_BlockData):
    ...    # custom block data class
    ...    pass
    """

    def proc_dec(cls):
        # this is the decorator function that
        # creates the block component class

        # Default (derived) Block attributes
        clsbody = {
            "__module__": cls.__module__,  # magic to fix the module
            # Default IndexedComponent data object is the decorated class:
            "_ComponentDataClass": cls,
            # By default this new block does not declare a new ctype
            "_default_ctype": None,
        }

        c = type(
            name,  # name of new class
            (CustomBlock,),  # base classes
            clsbody,  # class body definitions (will populate __dict__)
        )

        if new_ctype is not None:
            if new_ctype is True:
                c._default_ctype = c
            elif type(new_ctype) is type:
                c._default_ctype = new_ctype
            else:
                raise ValueError("Expected new_ctype to be either type "
                                 "or 'True'; received: %s" % (new_ctype,))

        # Register the new Block type in the same module as the BlockData
        setattr(sys.modules[cls.__module__], name, c)
        # TODO: can we also register concrete Indexed* and Scalar*
        # classes into the original BlockData module (instead of relying
        # on metaclasses)?

        # are these necessary?
        setattr(cls, '_orig_name', name)
        setattr(cls, '_orig_module', cls.__module__)
        return cls

    return proc_dec

