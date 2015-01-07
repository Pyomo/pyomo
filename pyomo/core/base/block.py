#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

__all__ = ['Block',
           'TraversalStrategy',
           'SortComponents',
           'active_components',
           'components',
           'active_components_data',
           'components_data']

import copy
import sys
import weakref
import logging
from inspect import isclass
from operator import itemgetter, attrgetter
from six import iterkeys, iteritems, itervalues, StringIO, string_types, \
    advance_iterator

from pyutilib.misc import Container

from pyomo.core.base.plugin import *
from pyomo.core.base.component import Component, ActiveComponentData, \
    ComponentUID, register_component
from pyomo.core.base.sets import Set, SimpleSet, _SetDataBase
from pyomo.core.base.var import Var
from pyomo.core.base.misc import apply_indexed_rule
from pyomo.core.base.sparse_indexed_component import SparseIndexedComponent, \
    ActiveSparseIndexedComponent

logger = logging.getLogger('pyomo.core')


# 
#
#

class SortComponents(object):
    """
    This class is a convenient wrapper for specifying various sort
    ordering.  We pass these objects to the "sort" argument to various
    accessors / iterators to control how much work we perform sorting
    the resultant list.  The idea is that
    "sort=SortComponents.deterministic" is more descriptive than
    "sort=True".
    """
    unsorted          = set()
    indices           = set([1])
    declOrder         = set([2])
    declarationOrder  = declOrder
    alphaOrder        = set([3])
    alphabeticalOrder = alphaOrder
    alphabetical      = alphaOrder
    # both alpha and decl orders are deterministic, so only must sort indices
    deterministic     = indices 
    sortBoth          = indices | alphabeticalOrder         # Same as True
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
                return SortComponents.alphaOrder.issubset( flag )
            except:
                return False

    @staticmethod
    def sort_indices(flag):
        if type(flag) is bool:
            return flag
        else:
            try:
                return SortComponents.indices.issubset( flag )
            except:
                return False


class TraversalStrategy(object):
    BreadthFirstSearch       = (1,)
    PrefixDepthFirstSearch   = (2,)
    PostfixDepthFirstSearch  = (3,)
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
        nameCounter += 1 # Each generator starts a new component name
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


class _BlockData(ActiveComponentData):
    """
    This class holds the fundamental block data.
    """

    class PseudoMap(object):
        """
        This class presents a "mock" dict interface to the internal
        _BlockData data structures.  We return this object to the
        user to preserve the historical "{ctype : {name : obj}}"
        interface without actually regenerating that dict-of-dicts data
        structure.

        We now support {ctype : PseudoMap()}
        """

        __slots__ = ( '_block', '_ctypes', '_active', '_sorted' )
        
        #TODO: add documentation!

        def __init__(self, block, ctype, active=None, sort=False):
            self._block = block
            if isclass(ctype):
                self._ctypes = (ctype,)
            else:
                self._ctypes = ctype
            self._active = active
            self._sorted = SortComponents.sort_names(sort)

        def __iter__(self):
            return self.iterkeys()

        def __getitem__(self, key):
            if key in self._block._decl:
                x = self._block._decl_order[self._block._decl[key]]
                if self._ctypes is None or x[0].type() in self._ctypes:
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
            raise KeyError( "%scomponent '%s' not found in block %s" 
                            % (msg, key, self._block.cname(True)))

        def __nonzero__(self):
            # Shortcut: this will bail after finding the first
            # (non-None) item
            for x in itervalues(self):
                return True
            return False

        __bool__ = __nonzero__

        def __len__(self):
            # If we don't care about the active flag, then it is
            # sufficient to rely on the count flags we store for each
            # ctype
            if self._active is None:
                if self._ctypes is None:
                    return sum(x[2] for x in itervalues(self._block._ctypes))
                else:
                    return sum( self._block._ctypes.get(x,(0,0,0))[2]
                                for x in self._ctypes )
            # For active==True/False, we have to count by brute force
            ans = 0
            for x in itervalues(self):
                ans += 1
            return ans

        def __contains__(self, key):
            # Return True is the underlying Block contains the component
            # name.  Note, if this Pseudomap soecifies a ctype or the
            # active flag, we need to check that the underlying
            # component matches those flags
            if key in self._block._decl:
                x = self._block._decl_order[self._block._decl[key]]
                if self._ctypes is None or x[0].type() in self._ctypes:
                    return self._active is None or x[0].active == self._active
            return False

        def _ctypewalker(self):
            # Note: since push/pop from the end of lists is slightly more
            # efficient, we will reverse-sort so the next ctype index is
            # at the end of the list.
            _decl_order = self._block._decl_order
            _idx_list = sorted( ( self._block._ctypes.get(x, [None])[0]
                                  for x in self._ctypes), 
                                reverse=True, 
                                key=lambda x: -1 if x is None else x )
            while _idx_list:
                _idx = _idx_list.pop()
                while _idx is not None:
                    x = _decl_order[_idx]
                    if x[0] is not None:
                        yield x[0]
                    _idx = x[1]
                    if _idx is not None and _idx_list and _idx > _idx_list[-1]:
                        _idx_list.append(_idx)
                        _idx_list.sort(reverse=True)
                        break

        def iterkeys(self):
            # Iterate over the PseudoMap keys (the component names) in
            # declaration order
            #
            # Ironically, the values are the fundamental thing that we
            # can (efficiently) iterate over in decl_order.  iterkeys
            # just wraps itervalues.
            for obj in self.itervalues():
                yield obj.name

        def itervalues(self):
            # Iterate over the PseudoMap values (the component objects) in
            # declaration order
            _active = self._active
            if self._ctypes is None:
                # If there is no ctype, then we will just iterate over
                # all components and return them all
                if _active is None:
                    walker = ( obj for obj,idx in self._block._decl_order 
                               if obj is not None )
                else:
                    walker = ( obj for obj,idx in self._block._decl_order 
                               if obj is not None and obj.active == _active )
            else:
                # The user specified a desired ctype; we will leverage
                # the _ctypewalker generator to walk the underlying linked
                # list and just return the desired objects (again, in
                # decl order)
                if _active is None:
                    walker = ( obj for obj in self._ctypewalker() )
                else:
                    walker = ( obj for obj in self._ctypewalker() 
                               if obj.active == _active )
            # If the user wants this sorted by name, then there is
            # nothing we can do to save memory: we must create the whole
            # list (so we can sort it) and then iterate over the sorted
            # temporary list
            if self._sorted:
                return ( obj for obj in sorted(walker, key=attrgetter('name')) )
            else:
                return walker

        def iteritems(self):
            # Ironically, the values are the fundamental thing that we
            # can (efficiently) iterate over in decl_order.  iteritems
            # just wraps itervalues.
            for obj in self.itervalues():
                yield (obj.name, obj)

        def keys(self):
            return list(self.iterkeys())

        def values(self):
            return list(self.itervalues())

        def items(self):
            return list(self.iteritems())

    # In Python3, the items(), etc methods of dict-like things return
    # generator-like objects.
    if sys.version_info >= (3, 0):
        PseudoMap.keys   = PseudoMap.iterkeys
        PseudoMap.values = PseudoMap.itervalues
        PseudoMap.items  = PseudoMap.iteritems


    def __init__(self, owner):
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
        ActiveComponentData.__init__(self, owner)
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
        ans =  dict(self.__dict__)
        ans.update(super(_BlockData, self).__getstate__())
        return ans

    def __setstate__(self, state):
        # We want the base class's __setstate__ to override our blanket
        # approach here (i.e., it will handle the _component weakref).
        for (slot_name, value) in iteritems(state):
            super(_BlockData, self).__setattr__(slot_name, value)
        super(_BlockData, self).__setstate__(state)

    #
    # NOTE: __getitem__() is no longer overridden so that Blocks now
    # behave as all other components: __getitem__ looks up indexed data
    # objects.
    #
    #def __getitem__(self, name):
    #    # This supports two key functionalities:
    #    #  1. block["component_name"] as a convenient alternative to
    #    #     block.component_name
    #    #     NOTE: use block.component("component_name") now
    #    #  2. block[Var] as an alternative to block.components(Var)
    #    #     NOTE: This usage is no longer supported.
    #
    #    if name in self._decl:
    #        return self._decl_order[self._decl[name]][0]
    #    elif name in self._ctypes:
    #        return BlockComponents.PseudoMap(self, name)
    #    elif isinstance(name,basestring):
    #        return None
    #    else:
    #        return {}

    def __setattr__(self, name, val):
        #
        # Set an attribute on this Block.  In general, the most common
        # case for this is setting a *new* attribute.  After that, there
        # is updating an existing Component value, with the least common
        # case being resetting an existing general attribute.  Given
        # that, we will:
        #
        # 1) handle new attributes, with special handling of Components
        if name not in self.__dict__:
            if isinstance(val, Component):
                self.add_component(name, val)
            else:
                super(_BlockData, self).__setattr__(name, val)
                #self.__dict__[name] = val
        # 2) Update the value of existing [scalar] Components
        # (through the component's set_value())
        elif name in self._decl:
            if isinstance(val, Component):
                logger.warn(
                    "Implicitly replacing the Component attribute "
                    "%s (type=%s) on block %s with a new Component (type=%s)."
                    "\nThis is usually indicative of a modelling error.\n"
                    "To avoid this warning, use block.del_component() and "
                    "block.add_component()."
                    % ( name, type(self.component(name)), self.cname(True), 
                        type(val) ) )
                self.del_component(name)
                self.add_component(name, val)
            else:
                # Because we want to raise a special error if the
                # set_value attribute is missing, we will fetch the
                # attribute first and then call the method outside of
                # the try-except so as to not suppress any exceptions
                # generated while setting the value.
                try:
                    _set_value = self._decl_order[self._decl[name]][0].set_value
                except AttributeError:
                    logger.error(
                        "Expected component %s (type=%s) on block %s to have a "
                        "'set_value' method, but none was found." % 
                        ( name, type(self.component(name)), 
                          self.cname(True) ) )
                    raise
                _set_value(val)
        # 3) handle setting non-Component attributes
        else:
            # NB: This is important: the _BlockData is either a
            # scalar Block (where _parent and _component are defined)
            # or a single block within an Indexed Block (where only
            # _component is defined).  Regardless, the __init__()
            # declares these attributes and sets them to either None or
            # a weakref.  This means that we will never have a problem
            # converting these objects from weakrefs into Blocks and
            # back (when pickling): because the attribute is already
            # in __dict__, we will not hit the add_component /
            # del_component branches above.  It also means that any
            # error checking we want to do when assigning these
            # attributes should be done here.

            # NB: isintance() can be slow; however, it is only slow when
            # it returns False.  Since the common paths on this branch
            # should return True, this shouldn't be too inefficient.
            if name == '_parent':
                if val is not None and not isinstance(val(), _BlockData):
                    raise ValueError(
                        "Cannot set the '_parent' attribute of Block '%s' "
                        "to a non-Block object (with type=%s); Did you "
                        "try to create a model component named '_parent'?" 
                        % (self.cname(), type(val)) )
                super(_BlockData, self).__setattr__(name, val)
            elif name == '_component':
                if val is not None and not isinstance(val(), _BlockData):
                    raise ValueError(
                        "Cannot set the '_component' attribute of Block '%s' "
                        "to a non-Block object (with type=%s); Did you "
                        "try to create a model component named '_component'?" 
                        % (self.cname(), type(val)) )
                super(_BlockData, self).__setattr__(name, val)
            # At this point, we should only be seeing non-component data
            # the user is hanging on the blocks (uncommon) or the
            # initial setup of the object data (in __init__).
            elif isinstance(val, Component):
                logger.warn(
                    "Reassigning the non-component attribute %s "
                    "on block %s with a new Component with type %s."
                    "\nThis is usually indicative of a modelling error."
                    % (name, self.cname(True), type(val)) )
                delattr(self, name)
                self.add_component(name, val)
            else:
                super(_BlockData, self).__setattr__(name, val)
 
    def __delattr__(self, name):
        # We need to make sure that del_component() gets called if a
        # user attempts to remove an attribute from a block that happens
        # to be a component
        if name in self._decl:
            self.del_component(name)
        else:
            super(_BlockData, self).__delattr__(name)


    def _add_temporary_set(self,val):
        """TODO: This method has known issues (see tickets) and needs to be
        reviewed. [JDS 9/2014]"""

        _component_sets = getattr(val, '_implicit_subsets', None)
        #
        # FIXME: The name attribute should begin with "_", and None
        # should replace "_unknown_"
        #
        if _component_sets is not None:
            for ctr, tset in enumerate(_component_sets):
                if tset.name == "_unknown_":
                    self._construct_temporary_set(
                      tset,
                      val.cname()+"_index_"+str(ctr)
                    )
        if isinstance(val._index, _SetDataBase) and \
                val._index.parent_component().cname() == "_unknown_":
            self._construct_temporary_set(val._index,val.cname()+"_index")
        if isinstance(getattr(val,'initialize',None), _SetDataBase) and \
                val.initialize.parent_component().cname() == "_unknown_":
            self._construct_temporary_set(val.initialize, val.cname()+"_index_init")
        if getattr(val,'domain',None) is not None and val.domain.cname() == "_unknown_":
            self._construct_temporary_set(val.domain,val.cname()+"_domain")

    def _construct_temporary_set(self, obj, name):
        """TODO: This method has known issues (see tickets) and needs to be
        reviewed. [JDS 9/2014]"""
        if type(obj) is tuple:
            if len(obj) == 1:                #pragma:nocover
                raise Exception(
                    "Unexpected temporary set construction for set "
                    "%s on block %s" % ( name, self.cname()) )
            else:
                tobj = obj[0]
                for t in obj[1:]:
                    tobj = tobj*t
                self.add_component(name,tobj)
                tobj.virtual=True
                return tobj
        elif isinstance(obj,Set):
            self.add_component(name,obj)
            return obj
        raise Exception("BOGUS")


    def add_component(self, name, val):
        if not val.valid_model_component():
            raise RuntimeError(
                "Cannot add '%s' as a component to a model" % str(type(val)) )
        if name in self.__dict__:
            raise RuntimeError(
                "Cannot add component '%s' (type %s) to block '%s': a "
                "component by that name (type %s) is already defined."
                % (name, type(val), self.cname(), type(getattr(self, name))))

        _component = self.parent_component()
        _type = val.type()
        if _type in _component._suppress_ctypes:
            return

        # all Pyomo components have Parents
        if (val._parent is not None) and (val._parent() is not None):
            if val._parent() is self:
                msg = """
Attempting to re-assign the component '%s' to the same 
block under a different name (%s).""" % (val.name, name)
            else:
                msg = """
Re-assigning the component '%s' from block '%s' to 
block '%s' as '%s'.""" % ( val.name, val._parent().cname(True), 
                           self.cname(True), name )

            raise RuntimeError("""%s

This behavior is not supported by Pyomo: components must have a single
owning block (or model) and a component may not appear multiple times in
a block.  If you want to re-name or move this component, use the block
del_component() and add_component() methods.  We are cowardly refusing
to do this automatically as renaming the component will change its
position in the construction order for Abstract Models; potentially
leading to unintuitive data validation and construction errors.
""" % (msg.strip(),) ) 

        # all Pyomo components have names. 
        val.name = name
        # set the parent pointer (weakref)
        val._parent = weakref.ref(self)

        # We want to add the temporary / implicit sets first so that
        # they get constructed before this component
        #
        # FIXME: This is sloppy and wasteful (most components trigger
        # this, even when there is no need for it).  We should
        # reconsider the whole _implicit_subsets logic to defer this
        # kind of thing to an "update_parent()" method on the
        # components.
        if hasattr(val,'_index'): 
            self._add_temporary_set(val)

        # Add the component to the underlying Component store
        _new_idx = len(self._decl_order)
        self._decl[name] = _new_idx
        self._decl_order.append( (val, None) )

        # Add the component as an attribute
        #
        # Note: 'self.__dict__[name]=val' is inappropriate here.  The
        # correct way to add the attribute is to delegate the work to
        # the next class up the MRO
        super(_BlockData, self).__setattr__(name, val)

        # Update the ctype linked lists
        if _type in self._ctypes:
            idx_info = self._ctypes[_type]
            tmp = idx_info[1]
            self._decl_order[tmp] = (self._decl_order[tmp][0], _new_idx)
            idx_info[1] = _new_idx
            idx_info[2] += 1
        else:
            self._ctypes[_type] = [_new_idx, _new_idx, 1]

        # There are some properties that need to be propagated to sub-blocks:
        if _type is Block:
            val._suppress_ctypes |= _component._suppress_ctypes

        # WEH - disabled support implicit rule names
        if False and '_rule' in val.__dict__:
            if val._rule is None:
                frame = sys._getframe(2)
                locals_ = frame.f_locals
                if val.cname()+'_rule' in locals_:
                    val._rule = locals_[val.cname()+'_rule']

        # FIXME: This is a HACK to support the way old Blocks and legacy
        # IndexedComponents (like Set) behave.  In particular, Set does
        # not define a "rule" attribute.  I put the hack back in to get
        # some tests passing again, but in all honesty, I am amazed this
        # ever worked properly. [JDS]
        elif False and getattr(val, 'rule', None) is None:
            try:
                frame = sys._getframe(2)
                locals_ = frame.f_locals
            except:
                locals_ = ()
            if val.cname()+'_rule' in locals_:
                val.rule = locals_[val.cname()+'_rule']

        # Don't reconstruct if this component has already been constructed,
        # the user may just want to transer it to another parent component
        if val._constructed is True:
            return

        # If the block is Concrete, construct the component
        # Note: we are explicitly using getattr because (Scalar)
        #   classes that derive from Block may want to declare components
        #   within their __init__() [notably, pyomo.gdp's Disjunct).
        #   Those components are added *before* the _constructed flag is
        #   added to the class by Block.__init__()
        if getattr(_component, '_constructed', False):
            # NB: we don't have to construct the temporary / implicit
            # sets here: if necessary, that happens when
            # _add_temporary_set() calls add_component().
            if id(self) in _BlockConstruction.data:
                data = _BlockConstruction.data[id(self)].get(name,None)
            else:
                data = None
            if __debug__ and logger.isEnabledFor(logging.DEBUG):
                # This is tricky: If we are in the middle of
                # constructing an indexed block, the block component
                # already has _constructed=True.  Now, if the
                # _BlockData.__init__() defines any local variables
                # (like pyomo.gdp.Disjunct's indicator_var), cname(True)
                # will fail: this block data exists and has a parent(),
                # but it has not yet been added to the parent's _data
                # (so the idx lookup will fail in cname()).
                if self.parent_block() is None:
                    _blockName = "[Model]"
                else:
                    try:
                        _blockName = "Block '%s'" % self.cname(True)
                    except:
                        _blockName = "Block '%s[...]'" \
                            % self.parent_component().cname(True)
                logger.debug( "Constructing %s '%s' on %s from data=%s",
                              val.__class__.__name__, val.cname(), 
                              _blockName, str(data) )
            try:
                val.construct(data)
            except:
                err = sys.exc_info()[1]
                logger.error(
                    "Constructing component '%s' from data=%s failed:\n%s: %s",
                    str(val.cname(True)), str(data).strip(),
                    type(err).__name__, err )
                raise
            if __debug__ and logger.isEnabledFor(logging.DEBUG):
                if _blockName[-1] == "'":
                    _blockName = _blockName[:-1] + '.' + val.cname() + "'"
                else:
                    _blockName = "'" + _blockName + '.' + val.cname() + "'"
                _out = StringIO()
                val.pprint(ostream=_out)
                logger.debug( "Constructed component '%s':\n%s" 
                              % ( _blockName, _out.getvalue() ) )

    def del_component(self, name_or_object):
        obj = self.component(name_or_object)
        # FIXME: Is this necessary?  Should this raise an exception?
        if obj is None:
            return

        # FIXME: Is this necessary?  Should this raise an exception?
        #if name not in self._decl:
        #    return

        name = obj.cname()

        # Replace the component in the master list with a None placeholder
        idx = self._decl[name]
        del self._decl[name]
        self._decl_order[idx] = (None, self._decl_order[idx][1])

        # Update the ctype linked lists
        ctype_info = self._ctypes[obj.type()]
        ctype_info[2] -= 1
        if ctype_info[2] == 0:
            del self._ctypes[obj.type()]

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

    def reclassify_component_type( self, name_or_object, new_ctype, 
                                   preserve_declaration_order=True ):
        obj = self.component(name_or_object)
        # FIXME: Is this necessary?  Should this raise an exception?
        if obj is None:
            return

        if obj._type is new_ctype:
            return

        name = obj.cname()
        if not preserve_declaration_order:
            # if we don't have to preserve the decl order, then the
            # easiest (and fastest) thing to do is just delete it and
            # re-add it.
            self.del_component(name)
            obj._type = new_ctype
            self.add_component(name, obj)
            return
            
        idx = self._decl[name]

        # Update the ctype linked lists
        ctype_info = self._ctypes[obj.type()]
        ctype_info[2] -= 1
        if ctype_info[2] == 0:
            del self._ctypes[obj.type()]
        elif ctype_info[0] == idx:
            ctype_info[0] = self._decl_order[idx][1]
        else:
            prev = None
            tmp = self._ctypes[obj.type()][0]
            while tmp < idx: 
                prev = tmp
                tmp = self._decl_order[tmp][1]
            
            self._decl_order[prev] = ( self._decl_order[prev][0], 
                                       self._decl_order[idx][1] )
            if ctype_info[1] == idx:
                ctype_info[1] = prev

        obj._type = new_ctype

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
        # FYI: we used to remove all _parent() weakrefs before
        # deepcopying and then restore them on the original and cloned
        # model.  It turns out that this was completely unnecessary and
        # wasteful.

        # Monkey-patch for deepcopying weakrefs
        # Only required on Python <= 2.6
        #
        # TODO: can we verify that this is really needed? [JDS 7/8/14]
        if sys.version_info[0] == 2 and sys.version_info[1] <= 6:
            copy._copy_dispatch[weakref.ref] = copy._copy_immutable
            copy._deepcopy_dispatch[weakref.ref] = copy._deepcopy_atomic
            copy._deepcopy_dispatch[weakref.KeyedRef] = copy._deepcopy_atomic

            def dcwvd(self, memo):
                """Deepcopy implementation for WeakValueDictionary class"""
                from copy import deepcopy
                new = self.__class__()
                for key, wr in self.data.items():
                    o = wr()
                    if o is not None:
                        new[deepcopy(key, memo)] = o
                return new
            weakref.WeakValueDictionary.__copy__ = \
                    weakref.WeakValueDictionary.copy
            weakref.WeakValueDictionary.__deepcopy__ = dcwvd

            def dcwkd(self, memo):
                """Deepcopy implementation for WeakKeyDictionary class"""
                from copy import deepcopy
                new = self.__class__()
                for key, value in self.data.items():
                    o = key()
                    if o is not none:
                        new[o] = deepcopy(value, memo)
                return new
            weakref.WeakKeyDictionary.__copy__ = weakref.WeakKeyDictionary.copy
            weakref.WeakKeyDictionary.__deepcopy__ = dcwkd

        #
        # Actually do the copy
        #
        # Note: Setting __block_scope__ determines which components are
        # deepcopied (anything beneath this block) and which are simply
        # preserved as references (anything outside this block hierarchy)
        return copy.deepcopy(self, {'__block_scope__': id(self)})

    def contains_component(self, ctype):
        return ctype in self._ctypes and self._ctypes[ctype][2]

    def component_map(self, ctype=None, active=None):
        """
        Return information about the block components.  If ctype is
        None, return the dictionary that maps
           {component type -> {name -> component instance}}
        Otherwise, return the dictionary that maps
           {name -> component instance}
        for the specified component type.

        Note: the actual {name->instance} object is a PseudoMap that
        implements a lightweight interface to the underlying
        BlockComponents data structures.
        """

        if ctype is None:
            ans = {}
            for x in self._ctypes:
                ans[x] = _BlockData.PseudoMap(self, x, active)
            return ans
        else:
            return _BlockData.PseudoMap(self, ctype, active)

    def component(self, name_or_object):
        """Return a component instance with the specified name

        TODO: verify why we want to support a check for the component
        with a given component data

        """
        if isinstance(name_or_object, string_types):
            if name_or_object in self._decl:
                return self._decl_order[self._decl[name_or_object]][0]
        else:
            try:
                obj = name_or_object.parent_component()
                # FIXME: Is this necessary?  Should this raise an exception?
                if obj.parent_block() is self:
                    return obj
            except AttributeError:
                pass
        return None

    def components( self, ctype=None, active=None, sort=False ):
        """
            ctype is None, ComponentType, Iterable of ComponentTypes
            active is None, True, False
                None - All
                True - Active
                False - Inactive
            sort is True, False
                True - Maps to Block.alphabetizeComponentAndIndex
                False - Maps to Block.declarationOrder
                SortComponents object (e.g. Block.deterministic)
            descend_into
                None/False - Block.localBlock
                True - allBlocks_dfsOrder   (prefix dfs)

        Returns the components in this block as a PseudoMap.  Callers should
        treat the return value as a normal dictionary.
        """
        return _BlockData.PseudoMap(self, ctype, active, sort)

    def active_components(self, ctype=None, sort=False):
        """
        Returns the active components in this block.  Return values
        match those of .components().
        """
        return self.components(ctype=ctype, active=True, sort=sort)


    def _component_data_iter( self, ctype=None, active=None, sort=False ):
        """
        Generator that returns a 3-tuple of (component name, index value,
        and _ComponentData) for every component data in the model
        """

        _sort_indices = SortComponents.sort_indices(sort)
        _subcomp = _BlockData.PseudoMap(self, ctype, active, sort)
        for name, comp in _subcomp.iteritems():
            # _NOTE_: Suffix has a dict interface (something other
            #         derived non-indexed Components may do as well),
            #         so we don't want to test the existence of
            #         iteritems as a check for components. Also,
            #         the case where we test len(comp) after seeing
            #         that comp.is_indexed is False is a hack for a
            #         SimpleConstraint whose expression resolved to
            #         Constraint.skip or Constraint.feasible (in which
            #         case its data is empty and iteritems would have
            #         been empty as well)
            #try:
            #    _items = comp.iteritems() 
            #except AttributeError:
            #    _items = [ (None, comp) ] 
            if comp.is_indexed():
                _items = comp.iteritems()
            # This is a hack (see _NOTE_ above).
            elif len(comp) or not hasattr(comp,'_data'):
                _items = ((None, comp),)
            else:
                _items = tuple()

            if _sort_indices:
                _items = sorted(_items, key=itemgetter(0))
            if active is None or not isinstance(comp, ActiveSparseIndexedComponent):
                for idx, compData in _items:
                    yield (name, idx, compData)
            else:
                for idx, compData in _items:
                    if compData.active == active:
                        yield (name, idx, compData)

    def all_components( self, ctype=None, active=None, sort=False, 
                        descend_into=None, descent_order=None ):
        if descent_into is None:
            for x in self.components( ctype, active, sort ).itervalues():
                yield x
            return
        for _block in self.all_blocks( active, sort, 
                                       descend_into, descent_order ):
            for x in _block.components( ctype, active, sort ).itervalues():
                yield x

    def active_components( self, ctype=None, sort=False, 
                           descend_into=None, descent_order=None ):
        return self.all_components( ctype, True, sort, 
                                    descend_into, descent_order )

    def all_component_data( self, ctype=None, active=None, sort=False, 
                            descend_into=None, descent_order=None ):
        if descend_into is None:
            for x in self._component_data_iter( ctype, active, sort ):
                yield x
            return
        for _block in self.all_blocks( active, sort, 
                                       descend_into, descent_order ):
            for x in _block._component_data_iter( ctype, active, sort ):
                yield x

    def active_component_data( self, ctype=None, sort=False, 
                               descend_into=None, descent_order=None ):
        """
        Generator that returns a 3-tuple of (component name, index value,
        and _ComponentData) for every active component data in the model
        """
        return self.all_component_data( ctype, True, sort,
                                         descend_into, descent_order )

    def all_blocks( self, active=None, sort=False, 
                    descend_into=True, descent_order=None ):
    #def all_blocks( self, active=None, filter_active=None, 
                    #sort_by_keys=False, sort_by_names=False ):
        # FIXME: all_blocks() used to always default to active only, and
        # filter_active=False was equivalent to active=None (no
        # filtering).  Until we edit all consumers to move their logic
        # to use the active flag, we will maintain backwards
        # compatibility
        #if active is not None:
        #    if filter_active is not None:
        #        raise RuntimeError(
        #            "cannot specify both active and filter_active arguments" )
        #else:
        #    if filter_active is None:
        #        active = True
        #    elif not filter_active:
        #        active = None
        #    else:
        #        active = filter_active
        # END FIX

        #sorter = SortComponents.sorter( sort_by_keys=sort_by_keys, 
        #                                sort_by_names=sort_by_names )

        #yield self
        #for name, idx, subblock in self.all_component_data( 
        #        ctype=Block, active=active, sort=sorter ):
        #    for b in subblock.all_blocks( active=active, 
        #                                  sort_by_keys=sort_by_keys,
        #                                  sort_by_names=sort_by_names ):
        #        yield b

        if descend_into is False:
            if active is not None and self.active != active:
                return ().__iter__()
            else:
                return (self,).__iter__()

        # Rely on the tree_iterator:
        return self.tree_iterator( descend_into, active, sort, descent_order )

    def active_blocks( self, active=None, sort=False, 
                       descend_into=True, descent_order=None ):
        return self.all_blocks(True, sort, descend_into, descent_order)

    def tree_iterator(self, ctype=None, active=None, sort=None, traversal=None):
        # TODO: merge into all_blocks
        if ctype is None or ctype is True:
            ctype = (Block,)
        elif isclass(ctype):
            ctype = (ctype,)

        # A little weird, but since we "normally" return a generator, we
        # will return a generator for an empty list instead of just
        # returning None or an empty list here (so that consumers can
        # count on us always returning a generator)
        if active is not None and self.active != active:
            return ().__iter__()
        if self._type not in ctype:
            return ().__iter__()

        if traversal is None or \
                traversal == TraversalStrategy.PrefixDepthFirstSearch:
            return self._prefix_dfs_iterator(ctype, active, sort)
        elif traversal == TraversalStrategy.BreadthFirstSearch:
            return self._bfs_iterator(ctype, active, sort)
        elif traversal == TraversalStrategy.PostfixDepthFirstSearch:
            return self._postfix_dfs_iterator(ctype, active, sort)
        else:
            raise RuntimeError( "unrecognized traversal strategy: %s" 
                                % ( traversal, ))

    def _prefix_dfs_iterator(self, ctype, active, sort):
        """Helper function implementing a non-recursive prefix order
        depth-first search.  That is, the parent is returned before its
        children.

        Note: this method assumes it is called ONLY by the tree_iterator
        method, which centralizes certain error checking and
        preliminaries.
        """
        _stack = [ self ]
        while _stack:
            _block = _stack.pop()
            yield _block
            _stack.extend( reversed( list(
                x[-1] for x in _block.all_component_data(ctype, active, sort)
            ) ) )
             

    def _postfix_dfs_iterator(self, ctype, active, sort):
        """Helper function implementing a non-recursive postfix order
        depth-first search.  That is, the parent is returned after its
        children.

        Note: this method assumes it is called ONLY by the tree_iterator
        method, which centralizes certain error checking and
        preliminaries.
        """
        _stack = [ (self, self.all_component_data(ctype, active, sort)) ]
        while _stack:
            try:
                _sub = advance_iterator(_stack[-1][1])[-1]
                _stack.append(( _sub, 
                                _sub.all_component_data(ctype, active, sort)
                            ))
            except StopIteration:
                yield _stack.pop()[0]

    def _bfs_iterator(self, ctype, active, sort):
        """Helper function implementing a non-recursive breadth-first search.
        That is, all children at one level in the tree are returned
        before any of the children at the next level.

        Note: this method assumes it is called ONLY by the tree_iterator
        method, which centralizes certain error checking and
        preliminaries.

        """
        if SortComponents.sort_indices(sort):
            if SortComponents.sort_names(sort):
                sorter = itemgetter(1,2)
            else:
                sorter = itemgetter(0,2)
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
                _queue = sorted( _sortingLevelWalker(_queue), key=sorter )

            _level += 1
            _levelQueue[_level] = []
            for _items in _queue:
                yield _items[-1] # _block
                _levelQueue[_level].append(
                    _items[-1].all_component_data(ctype, active, sort) )

    def fix_all_vars(self):
        # TODO: Simplify based on recursive logic
        for var in itervalues(self.components(Var)):
            var.fix()
        for block in itervalues(self.components(Block)):
            block.fix_all_vars()

    def unfix_all_vars(self):
        # TODO: Simplify based on recursive logic
        for var in itervalues(self.components(Var)):
            var.unfix()
        for block in itervalues(self.components(Block)):
            block.unfix_all_vars()

    def is_constructed(self):
        """
        A boolean indicating whether or not all *active* components of the
        input model have been properly constructed.
        """
        if not self._constructed:
            return False
        for x in self._decl_order:
            if x[0] is not None and x[0].active and not x[0].is_constructed():
                return False
        return True

    def pprint(self, ostream=None, verbose=False, prefix=""):
        """
        Print a summary of the block info
        """
        if ostream is None:
            ostream = sys.stdout
        #
        # We hard-code the order of the core Pyomo modeling
        # components, to ensure that the output follows the logical order
        # that expected by a user.
        #
        import pyomo.core.base.component_order
        items = pyomo.core.base.component_order.items + [Block]
        #
        # Collect other model components that are registered
        # with the IModelComponent extension point.  These are appended
        # to the end of the list of the list.
        #
        dynamic_items = set()
        for item in [ModelComponentFactory.get_class(name).component for name in ModelComponentFactory.services()]:
            if not item in items:
                dynamic_items.add(item)
        # extra items get added alphabetically (so output is consistent)
        items.extend(sorted(dynamic_items, key=lambda x: x.__name__))

        for item in items:
            keys = sorted(self.components(item))
            if not keys:
                continue
            #
            # NOTE: these conditional checks should not be hard-coded.
            #
            ostream.write( "%s%d %s Declarations\n" 
                           % (prefix, len(keys), item.__name__) )
            for key in keys:
                self.component(key).pprint(
                    ostream=ostream, verbose=verbose, prefix=prefix+'    ' )
            ostream.write("\n")
        #
        # Model Order
        #
        decl_order_keys = list(self.components().keys())
        ostream.write("%s%d Declarations: %s\n" 
                      % ( prefix, len(decl_order_keys),
                          ' '.join(str(x) for x in decl_order_keys) ))


class Block(ActiveSparseIndexedComponent):
    """
    Blocks are indexed components that contain other components
    (including blocks).  Blocks have a global attribute that defines
    whether construction is deferred.  This applies to all components
    that they contain except blocks.  Blocks contained by other
    blocks use their local attribute to determine whether construction
    is deferred.

    NOTE: Blocks do not currently maintain statistics about the sets,
    parameters, constraints, etc that they contain, including these
    components from subblocks.
    """

    def __new__(cls, *args, **kwds):
        if cls != Block:
            return super(Block, cls).__new__(cls)
        if args == ():
            return SimpleBlock.__new__(SimpleBlock)
        else:
            return IndexedBlock.__new__(IndexedBlock)

    def __init__(self, *args, **kwargs):
        """Constructor"""
        self._suppress_ctypes = set()
        self._rule = kwargs.pop('rule', None )
        self._options = kwargs.pop('options', None )
        kwargs.setdefault('ctype', Block)
        SparseIndexedComponent.__init__(self, *args, **kwargs)

    def _default(self, idx):
        return self._data.setdefault(idx, _BlockData(self))

    # TODO: review if this is necessary
    def concrete_mode(self):
        """Configure block to immediately construct components"""
        self.construct()

    # TODO: review if this is necessary
    def symbolic_mode(self):
        """Configure block to defer construction of components"""
        if self._constructed:
            logger.error(
                "Returning a Concrete model to Symbolic (Abstract) mode: "
                "you will likely experience unexpected and potentially "
                "erroneous behavior.")
        self._constructed=False

    # flags *all* active variables (on active blocks) and their
    # composite _VarData objects as stale.  fixed variables are flagged
    # as non-stale. the state of all inactive variables is left
    # unchanged.
    def flag_vars_as_stale(self):
        for block in self.all_blocks(active=True):
            for variable in active_components(block,Var):
                variable.flag_as_stale()

    def find_component(self, label_or_component):
        return ComponentUID(label_or_component).find_component_on(self)

    def construct(self, data=None):
        """ TODO """
        if __debug__ and logger.isEnabledFor(logging.DEBUG):
            logger.debug( "Constructing %s '%s', from data=%s",
                          self.__class__.__name__, self.cname(), str(data) )
        if self._constructed:
            return
        self._constructed = True

        # We must check that any pre-existing components are
        # constructed.  This catches the case where someone is building
        # a Concrete model by building (potentially pseudo-abstract)
        # sub-blocks and then adding them to a Concrete model block.
        for idx in self._data:
            _block = self[idx]
            for name, obj in iteritems( _block.components() ):
                if not obj._constructed:
                    if data is None:
                        _data = None
                    else:
                        _data = data.get(name, None)
                    obj.construct(_data)
        
        if self._rule is None:
            return
        # If we have a rule, fire the rule for all indices.  
        # Notes:
        #  - Since this block is now concrete, any components added to
        #    it will be immediately constructed by
        #    block.add_component().
        #  - Since the rule does not pass any "data" on, we build a
        #    scalar "stack" of pointers to block data
        #    (_BlockConstruction.data) that the individual blocks'
        #    add_component() can refer back to to handle component
        #    construction.
        for idx in self._index:
            _block = self[idx]
            if data is not None and idx in data:
                _BlockConstruction.data[id(_block)] = data[idx]
            obj = apply_indexed_rule(
                None, self._rule, _block, idx, self._options )
            if id(_block) in _BlockConstruction.data:
                del _BlockConstruction.data[id(_block)]

            # TBD: Should we allow skipping Blocks???
            #if obj is Block.Skip and idx is not None:
            #   del self._data[idx]

    def pprint(self, ostream=None, verbose=False, prefix=""):
        if ostream is None:
            ostream = sys.stdout
        subblock = (self._parent is not None) and (self.parent_block() is not None)
        for key in sorted(self):
            b = self[key]
            if subblock:
                ostream.write( "%s%s : Active=%s\n" % 
                              ( prefix, b.cname(True), b.active ))
            _BlockData.pprint( b, ostream=ostream, verbose=verbose, 
                               prefix=prefix+'    ' if subblock else prefix )

    def display(self, filename=None, ostream=None):
        """
        Print the Pyomo model in a verbose format.
        """
        if filename is not None:
            OUTPUT=open(filename,"w")
            self.display(ostream=OUTPUT)
            OUTPUT.close()
            return
        if ostream is None:
            ostream = sys.stdout
        if (self._parent is not None) and (self._parent() is not None):
            ostream.write("Block "+self.cname()+'\n')
        else:
            ostream.write("Model "+self.cname()+'\n')
        #
        import pyomo.core.base.component_order
        for item in pyomo.core.base.component_order.display_items:
            #
            ostream.write("\n")
            ostream.write("  %s:\n" % pyomo.core.base.component_order.display_name[item])
            ACTIVE = self.active_components(item)
            if not ACTIVE:
                ostream.write("    None\n")
            else:
                for obj in itervalues(ACTIVE):
                    obj.display(prefix="    ",ostream=ostream)


class SimpleBlock(_BlockData, Block):

    def __init__(self, *args, **kwds):
        _BlockData.__init__(self, self)
        Block.__init__(self, *args, **kwds)
        self._data[None] = self

    def pprint(self, ostream=None, verbose=False, prefix=""):
        Block.pprint(self, ostream, verbose, prefix)


class IndexedBlock(Block):

    def __init__(self, *args, **kwds):
        Block.__init__(self, *args, **kwds)


#TODO:  delete all of these functions

#
# Iterate over all active components (e.g., Constraint) of the
# specified type in the input block by declaration order
#
def active_components(block, ctype, sort_by_names=False, sort_by_keys=False):
    sorter = SortComponents.sorter( 
        sort_by_names=sort_by_names, sort_by_keys=sort_by_keys ) 
    return itervalues( block.active_components( ctype, sort=sorter ) )

#
# Iterate over all components (e.g., Var) of the specified type
# in the input block by declaration order
#
def components(block, ctype, sort_by_names=False, sort_by_keys=False):
    sorter = SortComponents.sorter( 
        sort_by_names=sort_by_names, sort_by_keys=sort_by_keys ) 
    return itervalues( block.components( ctype, sort=sorter ) )

#
# Iterate over all active components_datas (e.g., _VarData) of all
# active components of the specified type in the input block.
# 
# If sort_by_names is set to True, the objects will be returned in a
# deterministic order sorted by name otherwise objects will be return by
# declartion order. Within each component, data objects are returned
# in arbitrary (non-determinisitic) order, unless sort_by_keys is set to
# True
#
def active_components_data( block, ctype, 
                            sort=None, sort_by_keys=False, sort_by_names=False ):
    if sort is None:
        sort = SortComponents.sorter( 
            sort_by_names=sort_by_names, sort_by_keys=sort_by_keys ) 
    else:
        assert(sort_by_keys==False and sort_by_names==False)
    return ( obj for name,idx,obj in
             block.active_component_data( ctype, sort=sort ) )

#
# Same as above but don't check the .active flag
#
def components_data( block, ctype, sort=None, sort_by_keys=False, sort_by_names=False ):
    if sort is None:
        sorter = SortComponents.sorter( 
            sort_by_names=sort_by_names, sort_by_keys=sort_by_keys ) 
    else:
        assert(sort_by_keys==False and sort_by_names==False)
    return ( obj for (name,idx,obj) in 
        block.all_component_data( ctype, sort=sort ) )


register_component(
    Block, "A component that contains one or more model components." )

