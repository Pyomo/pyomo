#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from __future__ import annotations
import copy
import functools
import logging
import sys
import weakref
import textwrap

from collections import defaultdict
from contextlib import contextmanager
from inspect import isclass, currentframe
from io import StringIO
from itertools import filterfalse, chain
from operator import itemgetter, attrgetter
from typing import Union, Any, Type

from pyomo.common.autoslots import AutoSlots
from pyomo.common.collections import Mapping
from pyomo.common.deprecation import deprecated, deprecation_warning, RenamedClass
from pyomo.common.formatting import StreamIndenter
from pyomo.common.gc_manager import PauseGC
from pyomo.common.log import is_debug_set
from pyomo.common.pyomo_typing import overload
from pyomo.common.timing import ConstructionTimer
from pyomo.core.base.component import (
    Component,
    ComponentData,
    ActiveComponentData,
    ModelComponentFactory,
)
from pyomo.core.base.enums import SortComponents, TraversalStrategy
from pyomo.core.base.global_set import UnindexedComponent_index
from pyomo.core.base.componentuid import ComponentUID
from pyomo.core.base.set import Any
from pyomo.core.base.var import Var
from pyomo.core.base.initializer import Initializer
from pyomo.core.base.indexed_component import (
    ActiveIndexedComponent,
    UnindexedComponent_set,
    IndexedComponent,
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
            self._component(*self._args, rule=rule, **(self._kwds)),
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
        return _generic_component_decorator(self._component, self._block, *args, **kwds)


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
        self.__name__ = 'SubclassOf(%s)' % (','.join(x.__name__ for x in ctype),)

    def __contains__(self, item):
        return issubclass(item, self.ctype)

    def __len__(self):
        return 1

    def __getitem__(self, item):
        return self

    def __iter__(self):
        return iter((self,))


class _DeduplicateInfo(object):
    """Class implementing a unique component data object filter

    This class implements :py:meth:`unique()`, which is an efficient
    Reference-aware filter that wraps a generator and returns only
    unique component data objects.  This is nominally the same as:

        seen = set()
        for data in iterator:
            if id(data) not in seen:
                seen.add(id(data))
                yield data

    However, it is aware of the existence of Reference components (and
    that the only way you should ever encounter a duplicate is through a
    Reference).  This allows it to avoid generating and storing the id()
    of every data object.

    """

    __slots__ = ('seen_components', 'seen_comp_thru_reference', 'seen_data')

    def __init__(self):
        self.seen_components = set()
        self.seen_comp_thru_reference = set()
        self.seen_data = set()

    def unique(self, comp, items, are_values):
        """Returns generator that filters duplicate ComponentData objects from items

        Parameters
        ----------
        comp: ComponentBase
           The Component (indexed or scalar) that contains all
           ComponentData returned by the `items` generator.  `comp` may
           be an IndexedComponent generated by :py:func:`Reference` (and
           hence may not own the component datas in `items`)

        items: generator
            Generator yielding either the values or the items from the
            `comp` Component.

        are_values: bool
            If `True`, `items` yields ComponentData objects, otherwise,
            `items` yields `(index, ComponentData)` tuples.

        """
        if comp.is_reference():
            seen_components_contains = self.seen_components.__contains__
            seen_comp_thru_reference_contains = (
                self.seen_comp_thru_reference.__contains__
            )
            seen_comp_thru_reference_add = self.seen_comp_thru_reference.add
            seen_data_contains = self.seen_data.__contains__
            seen_data_add = self.seen_data.add

            def has_been_seen(data):
                # If the data is contained in a component we have
                # already processed, then it is a duplicate and we can
                # bypass further checks.
                _id = id(data.parent_component())
                if seen_components_contains(_id):
                    return True
                # Remember that this component has already been
                # partially visited (important for the case that we hit
                # the "natural" component later in the generator)
                if not seen_comp_thru_reference_contains(_id):
                    seen_comp_thru_reference_add(_id)
                # Yield any data objects we haven't seen yet (and
                # remember them)
                _id = id(data)
                if seen_data_contains(_id):
                    return True
                else:
                    seen_data_add(_id)
                    return False

            if are_values:
                return filterfalse(has_been_seen, items)
            else:
                return filterfalse(lambda item: has_been_seen(item[1]), items)

        else:  # this is a "natural" component
            # Remember that we have completely processed this component
            _id = id(comp)
            self.seen_components.add(_id)
            if _id not in self.seen_comp_thru_reference:
                # No data in this component has yet been emitted
                # (through a Reference), so we can just yield all the
                # values.
                return items
            else:
                # This component has had some data yielded (through
                # References).  We need to check for conflicts before
                # yielding each data.  Note that since we have already
                # marked the entire component as processed and data can
                # not reappear in natural components, we only need to
                # check for duplicates and not remember them.
                seen_data_contains = self.seen_data.__contains__
                if are_values:
                    has_been_seen = lambda item: seen_data_contains(id(item))
                else:
                    has_been_seen = lambda item: seen_data_contains(id(item[1]))
                return filterfalse(has_been_seen, items)


def _isNotNone(val):
    return val is not None


class _BlockConstruction(object):
    """
    This class holds a "global" dict used when constructing
    (hierarchical) models.
    """

    data = {}


class PseudoMap(AutoSlots.Mixin):
    """
    This class presents a "mock" dict interface to the internal
    BlockData data structures.  We return this object to the
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
            self._ctypes = {ctype}
        elif ctype is None:
            self._ctypes = Any
        elif ctype.__class__ is SubclassOf:
            self._ctypes = ctype
        else:
            self._ctypes = set(ctype)
        self._active = active
        self._sorted = SortComponents.ALPHABETICAL in SortComponents(sort)

    def __iter__(self):
        """
        TODO
        """
        return self.keys()

    def __getitem__(self, key):
        """
        TODO
        """
        if key in self._block._decl:
            x = self._block._decl_order[self._block._decl[key]]
            if x[0].ctype in self._ctypes:
                if self._active is None or x[0].active == self._active:
                    return x[0]
        msg = ""
        if self._active is not None:
            msg += self._active and "active " or "inactive "
        if self._ctypes is not Any:
            if len(self._ctypes) == 1:
                msg += next(iter(self._ctypes)).__name__ + ' '
            else:
                types = sorted(x.__name__ for x in self._ctypes)
                msg += '%s or %s ' % (', '.join(types[:-1]), types[-1])
        raise KeyError(
            "%scomponent '%s' not found in block %s" % (msg, key, self._block.name)
        )

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
            for x in self.values():
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
            if self._ctypes is Any:
                return sum(x[2] for x in self._block._ctypes.values())
            else:
                # Note that because of SubclassOf, we cannot iterate
                # over self._ctypes.
                return sum(
                    self._block._ctypes[x][2]
                    for x in self._block._ctypes
                    if x in self._ctypes
                )
        #
        # If _active is True or False, then we have to count by brute force.
        #
        ans = 0
        for x in self.values():
            ans += 1
        return ans

    def __contains__(self, key):
        """
        TODO
        """
        # Return True is the underlying Block contains the component
        # name.  Note, if this Pseudomap specifies a ctype or the
        # active flag, we need to check that the underlying
        # component matches those flags
        if key in self._block._decl:
            x = self._block._decl_order[self._block._decl[key]]
            if x[0].ctype in self._ctypes:
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
        # Note that because of SubclassOf, we cannot iterate over
        # self._ctypes. But this gets called a lot with a single type as
        # the ctypes set, so we will special case the set intersection.
        if self._ctypes.__class__ is set:
            _idx_list = [
                self._block._ctypes[x][0]
                for x in self._ctypes
                if x in self._block._ctypes
            ]
        else:
            _idx_list = [
                self._block._ctypes[x][0]
                for x in self._block._ctypes
                if x in self._ctypes
            ]
        _idx_list.sort(reverse=True)
        while _idx_list:
            _idx = _idx_list.pop()
            _next_ctype = _idx_list[-1] if _idx_list else None
            while 1:
                _obj, _idx = _decl_order[_idx]
                if _obj is not None:
                    yield _obj
                if _idx is None:
                    break
                if _next_ctype is not None and _idx > _next_ctype:
                    _idx_list.append(_idx)
                    _idx_list.sort(reverse=True)
                    break

    def keys(self):
        """
        Generator returning the component names defined on the Block
        """
        # Iterate over the PseudoMap keys (the component names) in
        # declaration order
        #
        # Ironically, the values are the fundamental thing that we
        # can (efficiently) iterate over in decl_order.  keys()
        # just wraps values().
        return map(attrgetter('_name'), self.values())

    def values(self):
        """
        Generator returning the components defined on the Block
        """
        # Iterate over the PseudoMap values (the component objects) in
        # declaration order
        if self._ctypes is Any:
            # If there is no ctype, then we will just iterate over
            # all components and return them all
            walker = filter(_isNotNone, map(itemgetter(0), self._block._decl_order))
        else:
            # The user specified a desired ctype; we will leverage
            # the _ctypewalker generator to walk the underlying linked
            # list and just return the desired objects (again, in
            # decl order)
            walker = self._ctypewalker()

        if self._active:
            walker = filter(attrgetter('active'), walker)
        elif self._active is not None:
            walker = filterfalse(attrgetter('active'), walker)

        # If the user wants this sorted by name, then there is
        # nothing we can do to save memory: we must create the whole
        # list (so we can sort it) and then iterate over the sorted
        # temporary list
        if self._sorted:
            return iter(sorted(walker, key=attrgetter('_name')))
        else:
            return walker

    def items(self):
        """
        Generator returning (name, component) tuples for components
        defined on the Block
        """
        # Ironically, the values are the fundamental thing that we
        # can (efficiently) iterate over in decl_order.  items()
        # just wraps values().
        for obj in self.values():
            yield (obj._name, obj)

    @deprecated('The iterkeys method is deprecated. Use dict.keys().', version='6.0')
    def iterkeys(self):
        """
        Generator returning the component names defined on the Block
        """
        return self.keys()

    @deprecated(
        'The itervalues method is deprecated. Use dict.values().', version='6.0'
    )
    def itervalues(self):
        """
        Generator returning the components defined on the Block
        """
        return self.values()

    @deprecated('The iteritems method is deprecated. Use dict.items().', version='6.0')
    def iteritems(self):
        """
        Generator returning (name, component) tuples for components
        defined on the Block
        """
        return self.items()


class BlockData(ActiveComponentData):
    """
    This class holds the fundamental block data.
    """

    _Block_reserved_words = set()

    # If a writer cached a repn on this block, remove it when cloning
    #  TODO: remove repn caching from the model
    __autoslot_mappers = {'_repn': AutoSlots.encode_as_none}

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
        super(BlockData, self).__setattr__('_ctypes', {})
        super(BlockData, self).__setattr__('_decl', {})
        super(BlockData, self).__setattr__('_decl_order', [])
        self._private_data = None

    def __getattr__(self, val) -> Union[Component, IndexedComponent, Any]:
        if val in ModelComponentFactory:
            return _component_decorator(self, ModelComponentFactory.get_class(val))
        # Since the base classes don't support getattr, we can just
        # throw the "normal" AttributeError
        raise AttributeError(
            "'%s' object has no attribute '%s'" % (self.__class__.__name__, val)
        )

    def __setattr__(self, name: str, val: Union[Component, IndexedComponent, Any]):
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
                super(BlockData, self).__setattr__(name, val)
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
                    % (name, type(self.component(name)), self.name, type(val))
                )
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
                        "'set_value' method, but none was found."
                        % (name, type(self.component(name)), self.name)
                    )
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
            # NB: This is important: the BlockData is either a scalar
            # Block (where _parent and _component are defined) or a
            # single block within an Indexed Block (where only
            # _component is defined).  Regardless, the
            # BlockData.__init__() method declares these methods and
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
                if val is not None and not isinstance(val(), BlockData):
                    raise ValueError(
                        "Cannot set the '_parent' attribute of Block '%s' "
                        "to a non-Block object (with type=%s); Did you "
                        "try to create a model component named '_parent'?"
                        % (self.name, type(val))
                    )
                super(BlockData, self).__setattr__(name, val)
            elif name == '_component':
                if val is not None and not isinstance(val(), BlockData):
                    raise ValueError(
                        "Cannot set the '_component' attribute of Block '%s' "
                        "to a non-Block object (with type=%s); Did you "
                        "try to create a model component named '_component'?"
                        % (self.name, type(val))
                    )
                super(BlockData, self).__setattr__(name, val)
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
                    "    del %s.%s" % (name, self.name, type(val), self.name, name)
                )
                delattr(self, name)
                self.add_component(name, val)
            else:
                super(BlockData, self).__setattr__(name, val)

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
            super(BlockData, self).__delattr__(name)

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
        self._decl = {k: idxMap[idx] for k, idx in self._decl.items()}
        # Update the ctypes, _decl_order linked lists
        for ctype, info in self._ctypes.items():
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
        raise RuntimeError(
            textwrap.dedent(
                """
                Block components do not support assignment or set_value().
                Use the transfer_attributes_from() method to transfer the
                components and public attributes from one block to another:
                    model.b[1].transfer_attributes_from(other_block)
                """
            ).strip()
        )

    def clear(self):
        for name in self.component_map().keys():
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
        src: BlockData or dict
            The Block or mapping that contains the new attributes to
            assign to this block.
        """
        if isinstance(src, BlockData):
            # There is a special case where assigning a parent block to
            # this block creates a circular hierarchy
            if src is self:
                return
            p_block = self.parent_block()
            while p_block is not None:
                if p_block is src:
                    raise ValueError(
                        "BlockData.transfer_attributes_from(): Cannot set a "
                        "sub-block (%s) to a parent block (%s): creates a "
                        "circular hierarchy" % (self, src)
                    )
                p_block = p_block.parent_block()
            # record the components and the non-component objects added
            # to the block
            src_comp_map = dict(src.component_map().items())
            src_raw_dict = src.__dict__
            del_src_comp = src.del_component
        elif isinstance(src, Mapping):
            src_comp_map = {k: v for k, v in src.items() if isinstance(v, Component)}
            src_raw_dict = src
            del_src_comp = lambda x: None
        else:
            raise ValueError(
                "BlockData.transfer_attributes_from(): expected a "
                "Block or dict; received %s" % (type(src).__name__,)
            )

        if src_comp_map:
            # Filter out any components from src
            src_raw_dict = {
                k: v for k, v in src_raw_dict.items() if k not in src_comp_map
            }

        # Use component_map for the components to preserve decl_order
        # Note that we will move any reserved components over as well as
        # any user-defined components.  There is a bit of trust here
        # that the user knows what they are doing.
        with self._declare_reserved_components():
            for k, v in src_comp_map.items():
                if k in self._decl:
                    self.del_component(k)
                del_src_comp(k)
                self.add_component(k, v)
        # Because Blocks are not slotized and we allow the
        # assignment of arbitrary data to Blocks, we will move over
        # any other unrecognized entries in the object's __dict__:
        for k, v in src_raw_dict.items():
            if (
                k not in self._Block_reserved_words  # user-defined
                or not hasattr(self, k)  # reserved, but not present
                or k in self._decl  # reserved, but a component and the
                # incoming thing is data (attempt to
                # set the value)
            ):
                setattr(self, k, v)

    def collect_ctypes(self, active=None, descend_into=True):
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
        for block in self.block_data_objects(
            active=active, descend_into=descend_into, sort=SortComponents.UNSORTED
        ):
            if active is None:
                ctypes.update(block._ctypes)
            else:
                assert active is True
                for ctype in block._ctypes:
                    for component in block.component_data_objects(
                        ctype=ctype,
                        active=True,
                        descend_into=False,
                        sort=SortComponents.UNSORTED,
                    ):
                        # We only need to verify that there is at least
                        # one active data member
                        ctypes.add(ctype)
                        break
        return ctypes

    def model(self):
        #
        # Special case: the "Model" is always the top-level BlockData,
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
        Returns a component in the block given a name.

        Parameters
        ----------
        label_or_component : str, Component, or ComponentUID
            The name of the component to find in this block. String or
            Component arguments are first converted to ComponentUID.

        Returns
        -------
        Component
            Component on the block identified by the ComponentUID. If
            a matching component is not found, None is returned.

        """
        return ComponentUID(label_or_component).find_component_on(self)

    @contextmanager
    def _declare_reserved_components(self):
        # Temporarily mask the class reserved words like with a local
        # instance attribute
        self._Block_reserved_words = ()
        yield
        del self._Block_reserved_words

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
                "Cannot add '%s' as a component to a block" % str(type(val))
            )
        if name in self._Block_reserved_words:
            raise ValueError(
                "Attempting to declare a block component using "
                "the name of a reserved attribute:\n\t%s" % (name,)
            )
        if name in self.__dict__:
            raise RuntimeError(
                "Cannot add component '%s' (type %s) to block '%s': a "
                "component by that name (type %s) is already defined."
                % (name, type(val), self.name, type(getattr(self, name)))
            )
        #
        _component = self.parent_component()
        _type = val.ctype
        #
        # Raise an exception if the component already has a parent.
        #
        if (val._parent is not None) and (val._parent() is not None):
            if val._parent() is self:
                msg = """
Attempting to re-assign the component '%s' to the same
block under a different name (%s).""" % (
                    val.name,
                    name,
                )
            else:
                msg = """
Re-assigning the component '%s' from block '%s' to
block '%s' as '%s'.""" % (
                    val._name,
                    val._parent().name,
                    self.name,
                    name,
                )

            raise RuntimeError(
                """%s

This behavior is not supported by Pyomo; components must have a
single owning block (or model), and a component may not appear
multiple times in a block.  If you want to re-name or move this
component, use the block del_component() and add_component() methods.
"""
                % (msg.strip(),)
            )
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
                "its children (%s): creates a circular hierarchy" % (self,)
            )
        #
        # Set the name and parent pointer of this component.
        #
        val._parent = weakref.ref(self)
        val._name = name
        #
        # Update the context of any anonymous sets
        #
        if getattr(val, '_anonymous_sets', None) is not None:
            for _set in val._anonymous_sets:
                _set._parent = val._parent
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
        super(BlockData, self).__setattr__(name, val)
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
        # Error, for disabled support implicit rule names
        #
        if '_rule' in val.__dict__ and val._rule is None:
            try:
                _test = val.local_name + '_rule'
                for i in (1, 2):
                    frame = sys._getframe(i)
            except:
                pass

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
            # NB: we don't have to construct the anonymous sets here: if
            # necessary, that happens in component.construct()
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
                # BlockData.__init__() defines any local variables
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
                        _blockName = "Block '%s[...]'" % self.parent_component().name
                logger.debug(
                    "Constructing %s '%s' on %s from data=%s",
                    val.__class__.__name__,
                    name,
                    _blockName,
                    str(data),
                )
            try:
                val.construct(data)
            except:
                err = sys.exc_info()[1]
                logger.error(
                    "Constructing component '%s' from data=%s failed:\n    %s: %s",
                    str(val.name),
                    str(data).strip(),
                    type(err).__name__,
                    err,
                    extra={'cleandoc': False},
                )
                raise
            if generate_debug_messages:
                if _blockName[-1] == "'":
                    _blockName = _blockName[:-1] + '.' + name + "'"
                else:
                    _blockName = "'" + _blockName + '.' + name + "'"
                _out = StringIO()
                val.pprint(ostream=_out)
                logger.debug(
                    "Constructed component '%s':\n%s" % (_blockName, _out.getvalue())
                )

    def del_component(self, name_or_object):
        """
        Delete a component from this block.
        """
        # in-lining self.component(name_or_object) so that we can add the
        # additional check of whether or not name_or_object is a ComponentData
        obj = None
        if isinstance(name_or_object, str):
            if name_or_object in self._decl:
                obj = self._decl_order[self._decl[name_or_object]][0]
            else:
                # Maintaining current behavior, but perhaps this should raise an
                # exception?
                return
        else:
            try:
                obj = name_or_object.parent_component()
            except AttributeError:
                # Maintaining current behavior, but perhaps this should raise an
                # exception?
                return
            if obj is not name_or_object:
                raise ValueError(
                    "Argument '%s' to del_component is a ComponentData object. "
                    "Please use the Python 'del' function to delete members of "
                    "indexed Pyomo components. The del_component function can "
                    "only be used to delete IndexedComponents and "
                    "ScalarComponents." % name_or_object.local_name
                )
            if obj.parent_block() is not self:
                return

        name = obj.local_name

        if name in self._Block_reserved_words:
            raise ValueError(
                "Attempting to delete a reserved block component:\n\t%s" % (obj.name,)
            )

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
        # Update the context of any anonymous sets
        if getattr(obj, '_anonymous_sets', None) is not None:
            for _set in obj._anonymous_sets:
                _set._parent = None

        # Now that this component is not in the _decl map, we can call
        # delattr as usual.
        #
        # del self.__dict__[name]
        #
        # Note: 'del self.__dict__[name]' is inappropriate here.  The
        # correct way to add the attribute is to delegate the work to
        # the next class up the MRO.
        super(BlockData, self).__delattr__(name)

    def reclassify_component_type(
        self, name_or_object, new_ctype, preserve_declaration_order=True
    ):
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

            self._decl_order[prev] = (
                self._decl_order[prev][0],
                self._decl_order[idx][1],
            )
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

    def clone(self, memo=None):
        """Make a copy of this block (and all components contained in it).

        Pyomo models use :py:class:`Block` components to define a
        hierarchical structure and provide model scoping.  When modeling
        :py:class:`~pyomo.core.base.component.Component` objects are
        assigned to a block, they are automatically added to that block's
        scope.

        :py:meth:`clone()` implements a specialization of
        :py:func:`copy.deepcopy` that will deep copy the
        :py:class:`BlockData` using that block's scope: that is, copy
        the :py:class:`BlockData` and (recursively) all
        :py:class:`Component` objects attached to it (including any
        sub-blocks).  Pyomo
        :py:class:`~pyomo.core.base.component.Component` /
        :py:class:`~pyomo.core.base.component.ComponentData` objects
        that are referenced through objects on this block but are not in
        this block scope (i.e., are not owned by this block or a
        subblock of this block) are not duplicated.

        Parameters
        ----------
        memo : dict
            A user-defined memo dictionary.  The dictionary will be
            updated by :py:meth:`clone` and :py:func:`copy.deepcopy`.
            See :py:meth:`object.__deepcopy__` for more information.

        Examples
        --------
        Given the following model:

        >>> m = pyo.ConcreteModel()
        >>> m.I = pyo.RangeSet(3)
        >>> m.x = pyo.Var()
        >>> m.b1 = pyo.Block()
        >>> m.b1.J = pyo.RangeSet(3)
        >>> m.b1.y = pyo.Var(domain=pyo.Reals)
        >>> m.b1.z = pyo.Var(m.I)
        >>> m.b1.c = pyo.Constraint(expr=m.x >= m.b1.y + sum(m.b1.z[:]))
        >>> m.b1.b2 = pyo.Block()
        >>> m.b1.b2.w = pyo.Var(m.b1.J)
        >>> m.b1.d = pyo.Constraint(expr=m.b1.y + sum(m.b1.b2.w[:]) == 5)

        If we clone a block:

        >>> i = m.b1.clone()

        All local components are copied:

        >>> assert m.b1 is not i
        >>> assert m.b1.J is not i.J
        >>> assert m.b1.y is not i.y
        >>> assert m.b1.z is not i.z
        >>> assert m.b1.b2 is not i.b2
        >>> assert m.b1.b2.w is not i.b2.w

        References to local components (in this case, Sets) are copied
        and updated:

        >>> assert m.b1.b2.w.index_set() is not i.b2.w.index_set()

        But references to out-of-scope Sets (either global or in a
        different block scope) are preserved:

        >>> assert m.b1.y.index_set() is i.y.index_set()
        >>> assert m.b1.z.index_set() is i.z.index_set()
        >>> assert m.b1.y.domain is i.y.domain

        Expressions are also updated in a similar manner: the new
        expression will reference the new (copied) components for any
        components in scope, but references to out-of-scope components
        will be preserved:

        >>> from pyomo.core.expr.compare import compare_expressions
        >>> assert compare_expressions(i.c.expr, m.x >= i.y + sum(i.z[:]))
        >>> assert compare_expressions(i.d.expr, i.y + sum(i.b2.w[:]) == 5)

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
        pc = self.parent_component()
        if pc is self:
            parent = self.parent_block()
        else:
            parent = pc

        if memo is None:
            memo = {}
        memo['__block_scope__'] = {id(self): True, id(None): False}
        memo[id(parent)] = parent

        with PauseGC():
            new_block = copy.deepcopy(self, memo)

        # We need to "detangle" the new block from the original block
        # hierarchy
        if pc is self:
            new_block._parent = None
        else:
            new_block._component = None

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
        if isinstance(name_or_object, str):
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

                =============   ===================================
                None            All components
                type            A single component type
                iterable        All component types in the iterable
                =============   ===================================

        active: None or bool
            Filter components by the active flag

                =====  ===============================
                None   Return all components
                True   Return only active components
                False  Return only inactive components
                =====  ===============================

        sort: bool
            Iterate over the components in a sorted order

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

    def _component_data_iteritems(self, ctype, active, sort, dedup):
        """return the name, index, and component data for matching ctypes

        Generator that returns a nested 2-tuple of

            ((component name, index value), ComponentData)

        for every component data in the block matching the specified
        ctype(s).

        Parameters
        ----------
        ctype:  None or type or iterable
            Specifies the component types (`ctypes`) to include

        active: None or bool
            Filter components by the active flag

        sort: None or bool or SortComponents
            Iterate over the components in a specified sorted order

        dedup: _DeduplicateInfo
            Deduplicator to prevent returning the same ComponentData twice
        """
        for name, comp in PseudoMap(self, ctype, active, sort).items():
            # NOTE: Suffix has a dict interface (something other derived
            #   non-indexed Components may do as well), so we don't want
            #   to test the existence of iteritems as a check for
            #   component datas. We will rely on is_indexed() to catch
            #   all the indexed components.  Then we will do special
            #   processing for the scalar components to catch the case
            #   where there are "sparse scalar components"
            if comp.is_indexed():
                _items = comp.items(sort)
            elif hasattr(comp, '_data'):
                # This is a Scalar component, which may be empty (e.g.,
                # from Constraint.Skip on a scalar Constraint).  Only
                # return a ComponentData if one officially exists.
                # Sorting is not a concern as this component has either
                # 0 or 1 datas
                assert len(comp._data) <= 1
                _items = comp._data.items()
            else:
                # This is a non-IndexedComponent Component.  Return it.
                _items = ((None, comp),)

            if active is None or not isinstance(comp, ActiveIndexedComponent):
                _items = (((name, idx), compData) for idx, compData in _items)
            else:
                _items = (
                    ((name, idx), compData)
                    for idx, compData in _items
                    if compData.active == active
                )

            yield from dedup.unique(comp, _items, False)

    def _component_data_itervalues(self, ctype, active, sort, dedup):
        """Generator that returns the ComponentData for every component data
        in the block.

        Parameters
        ----------
        ctype:  None or type or iterable
            Specifies the component types (`ctypes`) to include

        active: None or bool
            Filter components by the active flag

        sort: None or bool or SortComponents
            Iterate over the components in a specified sorted order

        dedup: _DeduplicateInfo
            Deduplicator to prevent returning the same ComponentData twice
        """
        for comp in PseudoMap(self, ctype, active, sort).values():
            # NOTE: Suffix has a dict interface (something other derived
            #   non-indexed Components may do as well), so we don't want
            #   to test the existence of iteritems as a check for
            #   component datas. We will rely on is_indexed() to catch
            #   all the indexed components.  Then we will do special
            #   processing for the scalar components to catch the case
            #   where there are "sparse scalar components"
            if comp.is_indexed():
                _values = comp.values(sort)
            elif hasattr(comp, '_data'):
                # This is a Scalar component, which may be empty (e.g.,
                # from Constraint.Skip on a scalar Constraint).  Only
                # return a ComponentData if one officially exists.
                assert len(comp._data) <= 1
                _values = comp._data.values()
            else:
                # This is a non-IndexedComponent Component.  Return it.
                _values = (comp,)

            if active is not None and isinstance(comp, ActiveIndexedComponent):
                _values = (filter if active else filterfalse)(
                    attrgetter('active'), _values
                )

            yield from dedup.unique(comp, _values, True)

    @deprecated(
        "The all_components method is deprecated.  "
        "Use the Block.component_objects() method.",
        version="4.1.10486",
    )
    def all_components(self, *args, **kwargs):
        return self.component_objects(*args, **kwargs)

    @deprecated(
        "The active_components method is deprecated.  "
        "Use the Block.component_objects() method.",
        version="4.1.10486",
    )
    def active_components(self, *args, **kwargs):
        kwargs['active'] = True
        return self.component_objects(*args, **kwargs)

    @deprecated(
        "The all_component_data method is deprecated.  "
        "Use the Block.component_data_objects() method.",
        version="4.1.10486",
    )
    def all_component_data(self, *args, **kwargs):
        return self.component_data_objects(*args, **kwargs)

    @deprecated(
        "The active_component_data method is deprecated.  "
        "Use the Block.component_data_objects() method.",
        version="4.1.10486",
    )
    def active_component_data(self, *args, **kwargs):
        kwargs['active'] = True
        return self.component_data_objects(*args, **kwargs)

    def component_objects(
        self, ctype=None, active=None, sort=False, descend_into=True, descent_order=None
    ):
        """
        Return a generator that iterates through the
        component objects in a block.  By default, the
        generator recursively descends into sub-blocks.
        """
        for _block in self.block_data_objects(
            active, sort, descend_into, descent_order
        ):
            yield from _block.component_map(ctype, active, sort).values()

    def component_data_objects(
        self, ctype=None, active=None, sort=False, descend_into=True, descent_order=None
    ):
        """
        Return a generator that iterates through the
        component data objects for all components in a
        block.  By default, this generator recursively
        descends into sub-blocks.
        """
        dedup = _DeduplicateInfo()
        for _block in self.block_data_objects(
            active, sort, descend_into, descent_order
        ):
            yield from _block._component_data_itervalues(ctype, active, sort, dedup)

    @deprecated(
        "The component_data_iterindex method is deprecated.  "
        "Components now know their index, so it is more efficient to use the "
        "Block.component_data_objects() method followed by .index().",
        version="6.6.0",
    )
    def component_data_iterindex(
        self, ctype=None, active=None, sort=False, descend_into=True, descent_order=None
    ):
        """
        Return a generator that returns a tuple for each
        component data object in a block.  By default, this
        generator recursively descends into sub-blocks.  The
        tuple is

            ((component name, index value), ComponentData)

        """
        dedup = _DeduplicateInfo()
        for _block in self.block_data_objects(
            active, sort, descend_into, descent_order
        ):
            yield from _block._component_data_iteritems(ctype, active, sort, dedup)

    @deprecated(
        "The all_blocks method is deprecated.  "
        "Use the Block.block_data_objects() method.",
        version="4.1.10486",
    )
    def all_blocks(self, *args, **kwargs):
        return self.block_data_objects(*args, **kwargs)

    @deprecated(
        "The active_blocks method is deprecated.  "
        "Use the Block.block_data_objects() method.",
        version="4.1.10486",
    )
    def active_blocks(self, *args, **kwargs):
        kwargs['active'] = True
        return self.block_data_objects(*args, **kwargs)

    def block_data_objects(
        self, active=None, sort=False, descend_into=True, descent_order=None
    ):
        """Returns this block and any matching sub-blocks.

        This is roughly equivalent to

        .. code-block:: python

            iter(block for block in itertools.chain(
                 [self], self.component_data_objects(descend_into, ...))
                 if block.active == active)

        Notes
        -----
        The `self` block is *always* returned, regardless of the types
        indicated by `descend_into`.

        The active flag is enforced on *all* blocks, including `self`.

        Parameters
        ----------
        active: None or bool
            If not None, filter components by the active flag

        sort: None or bool or SortComponents
            Iterate over the components in a specified sorted order

        descend_into:  None or type or iterable
            Specifies the component types (`ctypes`) to return and to
            descend into.  If `True` or `None`, defaults to `(Block,)`.
            If `False`, only `self` is returned.

        descent_order: None or TraversalStrategy
            The strategy used to walk the block hierarchy.  Defaults to
            `TraversalStrategy.PrefixDepthFirstSearch`.

        Returns
        -------
        tuple or generator

        """
        # TODO: we should determine if that is desirable behavior(it is
        # historical, so there are backwards compatibility arguments to
        # not change it, but because of block_data_objects() use in
        # component_data_objects, it might be desirable to always return
        # self.
        if active is not None and self.active != active:
            return ()
        if not descend_into:
            return (self,)

        if descend_into is True:
            ctype = (Block,)
        elif isclass(descend_into):
            ctype = (descend_into,)
        else:
            ctype = descend_into
        dedup = _DeduplicateInfo()

        if (
            descent_order is None
            or descent_order == TraversalStrategy.PrefixDepthFirstSearch
        ):
            walker = self._prefix_dfs_iterator(ctype, active, sort, dedup)
        elif descent_order == TraversalStrategy.BreadthFirstSearch:
            walker = self._bfs_iterator(ctype, active, sort, dedup)
        elif descent_order == TraversalStrategy.PostfixDepthFirstSearch:
            walker = self._postfix_dfs_iterator(ctype, active, sort, dedup)
        else:
            raise RuntimeError("unrecognized traversal strategy: %s" % (descent_order,))
        return walker

    def _prefix_dfs_iterator(self, ctype, active, sort, dedup):
        """Helper function implementing a non-recursive prefix order
        depth-first search.  That is, the parent is returned before its
        children.

        Note: this method assumes it is called ONLY by the _tree_iterator
        method, which centralizes certain error checking and
        preliminaries.
        """
        # We will unconditionally return self, so preemptively add it to
        # the list of "seen" IDs
        dedup.seen_data.add(id(self))

        PM = PseudoMap(self, ctype, active, sort)
        _stack = (None, (self,).__iter__())
        while _stack is not None:
            try:
                PM._block = _block = next(_stack[1])
                yield _block
                if not PM:
                    continue
                _stack = (
                    _stack,
                    _block._component_data_itervalues(ctype, active, sort, dedup),
                )
            except StopIteration:
                _stack = _stack[0]

    def _postfix_dfs_iterator(self, ctype, active, sort, dedup):
        """
        Helper function implementing a non-recursive postfix
        order depth-first search.  That is, the parent is
        returned after its children.

        Note: this method assumes it is called ONLY by the
        _tree_iterator method, which centralizes certain
        error checking and preliminaries.
        """
        # We will unconditionally return self, so preemptively add it to
        # the list of "seen" IDs
        dedup.seen_data.add(id(self))

        _stack = (
            None,
            self,
            self._component_data_itervalues(ctype, active, sort, dedup),
        )
        while _stack is not None:
            try:
                _sub = next(_stack[2])
                _stack = (
                    _stack,
                    _sub,
                    _sub._component_data_itervalues(ctype, active, sort, dedup),
                )
            except StopIteration:
                yield _stack[1]
                _stack = _stack[0]

    def _bfs_iterator(self, ctype, active, sort, dedup):
        """Helper function implementing a non-recursive breadth-first search.
        That is, all children at one level in the tree are returned
        before any of the children at the next level.

        Note: this method assumes it is called ONLY by the _tree_iterator
        method, which centralizes certain error checking and
        preliminaries.

        """
        # We will unconditionally return self, so preemptively add it to
        # the list of "seen" IDs
        dedup.seen_data.add(id(self))

        _thisLevel = None
        _nextLevel = [(self,)]
        while _nextLevel:
            _thisLevel = _nextLevel
            _nextLevel = []
            for block in chain(*_thisLevel):
                yield block
                _nextLevel.append(
                    block._component_data_itervalues(ctype, active, sort, dedup)
                )

    def fix_all_vars(self):
        # TODO: Simplify based on recursive logic
        for var in self.component_map(Var).values():
            var.fix()
        for block in self.component_map(Block).values():
            block.fix_all_vars()

    def unfix_all_vars(self):
        # TODO: Simplify based on recursive logic
        for var in self.component_map(Var).values():
            var.unfix()
        for block in self.component_map(Block).values():
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
            ostream.write("%d %s Declarations\n" % (len(keys), item.__name__))
            for key in keys:
                self.component(key).pprint(ostream=indented_ostream)
            ostream.write("\n")
        #
        # Model Order
        #
        decl_order_keys = list(self.component_map().keys())
        ostream.write(
            "%d Declarations: %s\n"
            % (len(decl_order_keys), ' '.join(str(x) for x in decl_order_keys))
        )

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
            ostream.write(
                prefix + "  %s:\n" % pyomo.core.base.component_order.display_name[item]
            )
            ACTIVE = self.component_map(item, active=True)
            if not ACTIVE:
                ostream.write(prefix + "    None\n")
            else:
                for obj in ACTIVE.values():
                    obj.display(prefix=prefix + "    ", ostream=ostream)

        item = Block
        ACTIVE = self.component_map(item, active=True)
        if ACTIVE:
            ostream.write(prefix + "\n")
            ostream.write(
                prefix + "  %s:\n" % pyomo.core.base.component_order.display_name[item]
            )
            for obj in ACTIVE.values():
                obj.display(prefix=prefix + "    ", ostream=ostream)

    #
    # The following methods are needed to support passing blocks as
    # models to a solver.
    #

    def valid_problem_types(self):
        """This method allows the pyomo.opt convert function to work with a
        Model object."""
        return [ProblemFormat.pyomo]

    def write(
        self,
        filename=None,
        format=None,
        solver_capability=None,
        io_options={},
        int_marker=False,
    ):
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
                        "or specify the format using the 'format' argument." % filename
                    )
                else:
                    format = _format
            elif format != _format and _format is not None:
                logger.warning(
                    "Filename '%s' likely does not match specified "
                    "file format (%s)" % (filename, format)
                )
        int_marker_kwds = {"int_marker": int_marker} if int_marker else {}
        problem_writer = WriterFactory(format, **int_marker_kwds)
        if problem_writer is None:
            raise ValueError(
                "Cannot write model in format '%s': no model "
                "writer registered for that format" % str(format)
            )

        if solver_capability is None:

            def solver_capability(x):
                return True

        (filename, smap) = problem_writer(self, filename, solver_capability, io_options)
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
                str(format),
            )
        return filename, smap_id

    def _create_objects_for_deepcopy(self, memo, component_list):
        _new = self.__class__.__new__(self.__class__)
        _ans = memo.setdefault(id(self), _new)
        if _ans is _new:
            component_list.append((self, _new))
            # Blocks (and block-like things) need to pre-populate all
            # Components / ComponentData objects to help prevent
            # deepcopy() from violating the Python recursion limit.
            # This step is recursive; however, we do not expect "super
            # deep" Pyomo block hierarchies, so should be okay.
            for comp, _ in self._decl_order:
                if comp is not None:
                    comp._create_objects_for_deepcopy(memo, component_list)
        return _ans

    def private_data(self, scope=None):
        mod = currentframe().f_back.f_globals['__name__']
        if scope is None:
            scope = mod
        elif not mod.startswith(scope):
            raise ValueError(
                "All keys in the 'private_data' dictionary must "
                "be substrings of the caller's module name. "
                "Received '%s' when calling private_data on Block "
                "'%s'." % (scope, self.name)
            )
        if self._private_data is None:
            self._private_data = {}
        if scope not in self._private_data:
            self._private_data[scope] = Block._private_data_initializers[scope]()
        return self._private_data[scope]


class _BlockData(metaclass=RenamedClass):
    __renamed__new_class__ = BlockData
    __renamed__version__ = '6.7.2'


@ModelComponentFactory.register(
    "A component that contains one or more model components."
)
class Block(ActiveIndexedComponent):
    """
    Blocks are indexed components that contain other components
    (including blocks).  Blocks have a global attribute that defines
    whether construction is deferred.  This applies to all components
    that they contain except blocks.  Blocks contained by other
    blocks use their local attribute to determine whether construction
    is deferred.
    """

    _ComponentDataClass = BlockData
    _private_data_initializers = defaultdict(lambda: dict)

    @overload
    def __new__(
        cls: Type[Block], *args, **kwds
    ) -> Union[ScalarBlock, IndexedBlock]: ...

    @overload
    def __new__(cls: Type[ScalarBlock], *args, **kwds) -> ScalarBlock: ...

    @overload
    def __new__(cls: Type[IndexedBlock], *args, **kwds) -> IndexedBlock: ...

    def __new__(cls, *args, **kwds):
        if cls != Block:
            return super(Block, cls).__new__(cls)
        if not args or (args[0] is UnindexedComponent_set and len(args) == 1):
            return ScalarBlock.__new__(ScalarBlock)
        else:
            return IndexedBlock.__new__(IndexedBlock)

    # `options` is ignored since it is deprecated
    @overload
    def __init__(
        self, *indexes, rule=None, concrete=False, dense=True, name=None, doc=None
    ): ...

    def __init__(self, *args, **kwargs):
        """Constructor"""
        _rule = kwargs.pop('rule', None)
        _options = kwargs.pop('options', None)
        # As concrete applies to the Block at declaration time, we will
        # not use an initializer.
        _concrete = kwargs.pop('concrete', False)
        # As dense applies to the whole container, we will not use an
        # initializer
        self._dense = kwargs.pop('dense', True)
        kwargs.setdefault('ctype', Block)
        if _options is not None:
            deprecation_warning(
                "The Block 'options=' keyword is deprecated.  "
                "Equivalent functionality can be obtained by wrapping "
                "the rule function to add the options dictionary to "
                "the function arguments",
                version='5.7.2',
            )
            self._rule = Initializer(functools.partial(_rule, **_options))
        else:
            self._rule = Initializer(_rule)
        ActiveIndexedComponent.__init__(self, *args, **kwargs)
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
            if obj is not _block and isinstance(obj, BlockData):
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
        if self._constructed:
            return
        self._constructed = True

        timer = ConstructionTimer(self)
        if is_debug_set(logger):
            logger.debug(
                "Constructing %s '%s', from data=%s",
                self.__class__.__name__,
                self.name,
                str(data),
            )

        if self._anonymous_sets is not None:
            for _set in self._anonymous_sets:
                _set.construct()

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
                    self._dense or self._rule is not None
                ):
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
                    for name, obj in _predefined_components.items():
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
            ("Index", self._index_set if self.is_indexed() else None),
            ('Active', self.active),
        ]
        # HACK: suppress the top-level block header (for historical reasons)
        if self.parent_block() is None and not self.is_indexed():
            return None, self._data.items(), None, self._pprint_callback
        else:
            return _attrs, self._data.items(), None, self._pprint_callback

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
            BlockData.display(self[key], filename, ostream, prefix)

    @staticmethod
    def register_private_data_initializer(initializer, scope=None):
        mod = currentframe().f_back.f_globals['__name__']
        if scope is None:
            scope = mod
        elif not mod.startswith(scope):
            raise ValueError(
                "'private_data' scope must be substrings of the caller's module name. "
                f"Received '{scope}' when calling register_private_data_initializer()."
            )
        if scope in Block._private_data_initializers:
            raise RuntimeError(
                "Duplicate initializer registration for 'private_data' dictionary "
                f"(scope={scope})"
            )
        Block._private_data_initializers[scope] = initializer


class ScalarBlock(BlockData, Block):
    def __init__(self, *args, **kwds):
        BlockData.__init__(self, component=self)
        Block.__init__(self, *args, **kwds)
        # Initialize the data dict so that (abstract) attribute
        # assignment will work.  Note that we do not trigger
        # get/setitem_when_not_present so that we do not (implicitly)
        # trigger the Block rule
        self._data[None] = self
        self._index = UnindexedComponent_index

    # We want scalar Blocks to pick up the Block display method
    display = Block.display


class SimpleBlock(metaclass=RenamedClass):
    __renamed__new_class__ = ScalarBlock
    __renamed__version__ = '6.0'


class IndexedBlock(Block):
    def __init__(self, *args, **kwds):
        Block.__init__(self, *args, **kwds)

    @overload
    def __getitem__(self, index) -> BlockData: ...

    __getitem__ = IndexedComponent.__getitem__  # type: ignore


#
# Deprecated functions.
#
@deprecated(
    "generate_cuid_names() is deprecated. "
    "Use the ComponentUID.generate_cuid_string_map() static method",
    version="5.7.2",
)
def generate_cuid_names(block, ctype=None, descend_into=True):
    return ComponentUID.generate_cuid_string_map(block, ctype, descend_into)


@deprecated(
    "The active_components function is deprecated.  "
    "Use the Block.component_objects() method.",
    version="4.1.10486",
)
def active_components(block, ctype, sort_by_names=False, sort_by_keys=False):
    return block.component_objects(ctype, active=True, sort=sort_by_names)


@deprecated(
    "The components function is deprecated.  "
    "Use the Block.component_objects() method.",
    version="4.1.10486",
)
def components(block, ctype, sort_by_names=False, sort_by_keys=False):
    return block.component_objects(ctype, active=False, sort=sort_by_names)


@deprecated(
    "The active_components_data function is deprecated.  "
    "Use the Block.component_data_objects() method.",
    version="4.1.10486",
)
def active_components_data(
    block, ctype, sort=None, sort_by_keys=False, sort_by_names=False
):
    return block.component_data_objects(ctype=ctype, active=True, sort=sort)


@deprecated(
    "The components_data function is deprecated.  "
    "Use the Block.component_data_objects() method.",
    version="4.1.10486",
)
def components_data(block, ctype, sort=None, sort_by_keys=False, sort_by_names=False):
    return block.component_data_objects(ctype=ctype, active=False, sort=sort)


#
# Create a Block and record all the default attributes, methods, etc.
# These will be assumed to be the set of illegal component names.
#
BlockData._Block_reserved_words = set(dir(Block()))


class ScalarCustomBlockMixin(object):
    def __init__(self, *args, **kwargs):
        # __bases__ for the ScalarCustomBlock is
        #
        #    (ScalarCustomBlockMixin, {custom_data}, {custom_block})
        #
        # Unfortunately, we cannot guarantee that this is being called
        # from the ScalarCustomBlock (someone could have inherited from
        # that class to make another scalar class).  We will walk up the
        # MRO to find the Scalar class (which should be the only class
        # that has this Mixin as the first base class)
        for cls in self.__class__.__mro__:
            if cls.__bases__[0] is ScalarCustomBlockMixin:
                _mixin, _data, _block = cls.__bases__
                _data.__init__(self, component=self)
                _block.__init__(self, *args, **kwargs)
                break


class CustomBlock(Block):
    """The base class used by instances of custom block components"""

    def __init__(self, *args, **kwargs):
        if self._default_ctype is not None:
            kwargs.setdefault('ctype', self._default_ctype)
        kwargs.setdefault("rule", getattr(self, '_default_rule', None))
        Block.__init__(self, *args, **kwargs)

    def __new__(cls, *args, **kwargs):
        if cls.__bases__[0] is not CustomBlock:
            # we are creating a class other than the "generic" derived
            # custom block class.  We can assume that the routing of the
            # generic block class to the specific Scalar or Indexed
            # subclass has already occurred and we can pass control up
            # to (toward) object.__new__()
            return super().__new__(cls, *args, **kwargs)
        # If the first base class is this CustomBlock class, then the
        # user is attempting to create the "generic" block class.
        # Depending on the arguments, we need to map this to either the
        # Scalar or Indexed block subclass.
        if not args or (args[0] is UnindexedComponent_set and len(args) == 1):
            return super().__new__(cls._scalar_custom_block, *args, **kwargs)
        else:
            return super().__new__(cls._indexed_custom_block, *args, **kwargs)


class _custom_block_rule_redirect(object):
    """Functor to redirect the default rule to a BlockData method"""

    def __init__(self, cls, name):
        self.cls = cls
        self.name = name

    def __call__(self, block, *args, **kwargs):
        return getattr(self.cls, self.name)(block, *args, **kwargs)


def declare_custom_block(name, new_ctype=None, rule=None):
    """Decorator to declare components for a custom block data class

    This decorator simplifies the definition of custom derived Block
    classes.  With this decorator, developers must only implement the
    derived "Data" class.  The decorator automatically creates the
    derived containers using the provided name, and adds them to the
    current module:

    >>> @declare_custom_block(name="FooBlock")
    ... class FooBlockData(BlockData):
    ...    pass

    >>> s = FooBlock()
    >>> type(s)
    <class 'ScalarFooBlock'>

    >>> s = FooBlock([1,2])
    >>> type(s)
    <class 'IndexedFooBlock'>

    It is frequently desirable for the custom class to have a default
    ``rule`` for constructing and populating new instances.  The default
    rule can be provided either as an explicit function or a string.  If
    a string, the rule is obtained by attribute lookup on the derived
    Data class:

    >>> @declare_custom_block(name="BarBlock", rule="build")
    ... class BarBlockData(BlockData):
    ...    def build(self, *args):
    ...        self.x = Var(initialize=5)

    >>> m = pyo.ConcreteModel()
    >>> m.b = BarBlock([1,2])
    >>> print(m.b[1].x.value)
    5
    >>> print(m.b[2].x.value)
    5

    """

    def block_data_decorator(block_data):
        # this is the decorator function that creates the block
        # component classes

        # Declare the new Block component (derived from CustomBlock)
        # corresponding to the BlockData that we are decorating
        #
        # Note the use of `type(CustomBlock)` to pick up the metaclass
        # that was used to create the CustomBlock (in general, it should
        # be `type`)
        comp = type(CustomBlock)(
            name,  # name of new class
            (CustomBlock,),  # base classes
            # class body definitions (populate the new class' __dict__)
            {
                # ensure the created class is associated with the calling module
                "__module__": block_data.__module__,
                # Default IndexedComponent data object is the decorated class:
                "_ComponentDataClass": block_data,
                # By default this new block does not declare a new ctype
                "_default_ctype": None,
                # Define the default rule (may be None)
                "_default_rule": rule,
            },
        )

        # If the default rule is a string, then replace it with a
        # function that will look up the attribute on the data class.
        if type(rule) is str:
            comp._default_rule = _custom_block_rule_redirect(block_data, rule)

        if new_ctype is not None:
            if new_ctype is True:
                comp._default_ctype = comp
            elif isinstance(new_ctype, type):
                comp._default_ctype = new_ctype
            else:
                raise ValueError(
                    "Expected new_ctype to be either type "
                    "or 'True'; received: %s" % (new_ctype,)
                )

        # Declare Indexed and Scalar versions of the custom block.  We
        # will register them both with the calling module scope, and
        # with the CustomBlock (so that CustomBlock.__new__ can route
        # the object creation to the correct class)
        comp._indexed_custom_block = type(comp)(
            "Indexed" + name,
            (comp,),
            {  # ensure the created class is associated with the calling module
                "__module__": block_data.__module__
            },
        )
        comp._scalar_custom_block = type(comp)(
            "Scalar" + name,
            (ScalarCustomBlockMixin, block_data, comp),
            {  # ensure the created class is associated with the calling module
                "__module__": block_data.__module__
            },
        )

        # Register the new Block types in the same module as the BlockData
        for _cls in (comp, comp._indexed_custom_block, comp._scalar_custom_block):
            setattr(sys.modules[block_data.__module__], _cls.__name__, _cls)
        return block_data

    return block_data_decorator
