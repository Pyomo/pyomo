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
           'components_data']

import copy
import sys
import weakref
import logging
from inspect import isclass
from operator import itemgetter, attrgetter
from six import iteritems, iterkeys, itervalues, StringIO, string_types, \
    advance_iterator, PY3

from pyomo.common.timing import ConstructionTimer
from pyomo.core.base.plugin import *  # ModelComponentFactory
from pyomo.core.base.component import Component, ActiveComponentData, \
    ComponentUID
from pyomo.core.base.sets import Set,  _SetDataBase
from pyomo.core.base.var import Var
from pyomo.core.base.misc import apply_indexed_rule
from pyomo.core.base.suffix import ComponentMap
from pyomo.core.base.indexed_component import IndexedComponent, \
    ActiveIndexedComponent, UnindexedComponent_set
import collections

from pyomo.opt.base import ProblemFormat, guess_format
from pyomo.opt import WriterFactory

logger = logging.getLogger('pyomo.core')


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


class _BlockData(ActiveComponentData):
    """
    This class holds the fundamental block data.
    """
    _Block_reserved_words = set()

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
            self._sorted = False
            for x in itervalues(self):
                self._sorted = sort_order
                return True
            self._sorted = sort_order
            return False

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
                if self._ctypes is None or x[0].type() in self._ctypes:
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

    def set_value(self, val):
        for k in list(getattr(self, '_decl', {})):
            self.del_component(k)
        self._ctypes = {}
        self._decl = {}
        self._decl_order = []
        if val:
            for k in sorted(iterkeys(val)):
                self.add_component(k,val[k])

    def _add_temporary_set(self, val):
        """TODO: This method has known issues (see tickets) and needs to be
        reviewed. [JDS 9/2014]"""

        _component_sets = getattr(val, '_implicit_subsets', None)
        #
        # FIXME: The name attribute should begin with "_", and None
        # should replace "_unknown_"
        #
        if _component_sets is not None:
            for ctr, tset in enumerate(_component_sets):
                if tset._name == "_unknown_":
                    self._construct_temporary_set(
                        tset,
                        val.local_name + "_index_" + str(ctr)
                    )
        if isinstance(val._index, _SetDataBase) and \
                val._index.parent_component().local_name == "_unknown_":
            self._construct_temporary_set(val._index, val.local_name + "_index")
        if isinstance(getattr(val, 'initialize', None), _SetDataBase) and \
                val.initialize.parent_component().local_name == "_unknown_":
            self._construct_temporary_set(val.initialize, val.local_name + "_index_init")
        if getattr(val, 'domain', None) is not None and \
           getattr(val.domain, 'local_name', None) == "_unknown_":
            self._construct_temporary_set(val.domain, val.local_name + "_domain")

    def _construct_temporary_set(self, obj, name):
        """TODO: This method has known issues (see tickets) and needs to be
        reviewed. [JDS 9/2014]"""
        if type(obj) is tuple:
            if len(obj) == 1:  # pragma:nocover
                raise Exception(
                    "Unexpected temporary set construction for set "
                    "%s on block %s" % (name, self.name))
            else:
                tobj = obj[0]
                for t in obj[1:]:
                    tobj = tobj * t
                self.add_component(name, tobj)
                tobj.virtual = True
                return tobj
        elif isinstance(obj, Set):
            self.add_component(name, obj)
            return obj
        raise Exception("BOGUS")

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
        if name in self._Block_reserved_words:
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
        _type = val.type()
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
        if hasattr(val, '_index'):
            self._add_temporary_set(val)
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
            # _add_temporary_set() calls add_component().
            if id(self) in _BlockConstruction.data:
                data = _BlockConstruction.data[id(self)].get(name, None)
            else:
                data = None
            if __debug__ and logger.isEnabledFor(logging.DEBUG):
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
                             val.__class__.__name__, val.name,
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
            if __debug__ and logger.isEnabledFor(logging.DEBUG):
                if _blockName[-1] == "'":
                    _blockName = _blockName[:-1] + '.' + val.name + "'"
                else:
                    _blockName = "'" + _blockName + '.' + val.name + "'"
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

    def reclassify_component_type(self, name_or_object, new_ctype,
                                  preserve_declaration_order=True):
        """
        TODO
        """
        obj = self.component(name_or_object)
        # FIXME: Is this necessary?  Should this raise an exception?
        if obj is None:
            return

        if obj._type is new_ctype:
            return

        name = obj.local_name
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

            self._decl_order[prev] = (self._decl_order[prev][0],
                                      self._decl_order[idx][1])
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
        """
        Returns a PseudoMap of the components in this block.

            ctype
                None            - All components
                ComponentType   - A single ComponentType
                Iterable        - Iterate to generate ComponentTypes

            active is None, True, False
                None  - All
                True  - Active
                False - Inactive

            sort is True, False
                True - Maps to Block.alphabetizeComponentAndIndex
                False - Maps to Block.declarationOrder
        """
        return _BlockData.PseudoMap(self, ctype, active, sort)

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
                ans[x] = _BlockData.PseudoMap(self, x, active, sort)
            return ans
        else:
            return _BlockData.PseudoMap(self, ctype, active, sort)

    def _component_data_iter(self, ctype=None, active=None, sort=False):
        """
        Generator that returns a 3-tuple of (component name, index value,
        and _ComponentData) for every component data in the block.
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
            # try:
            #    _items = comp.iteritems()
            # except AttributeError:
            #    _items = [ (None, comp) ]
            if comp.is_indexed():
                _items = comp.iteritems()
            # This is a hack (see _NOTE_ above).
            elif len(comp) or not hasattr(comp, '_data'):
                _items = ((None, comp),)
            else:
                _items = tuple()

            if _sort_indices:
                _items = sorted(_items, key=itemgetter(0))
            if active is None or not isinstance(comp, ActiveIndexedComponent):
                for idx, compData in _items:
                    yield (name, idx), compData
            else:
                for idx, compData in _items:
                    if compData.active == active:
                        yield (name, idx), compData

    def all_components(self, *args, **kwargs):
        logger.warning(
            "DEPRECATED: The all_components method is deprecated.  Use the Block.component_objects() method.")
        return self.component_objects(*args, **kwargs)

    def active_components(self, *args, **kwargs):
        logger.warning(
            "DEPRECATED: The active_components method is deprecated.  Use the Block.component_objects() method.")
        kwargs['active'] = True
        return self.component_objects(*args, **kwargs)

    def all_component_data(self, *args, **kwargs):
        logger.warning(
            "DEPRECATED: The all_component_data method is deprecated.  Use the Block.component_data_objects() method.")
        return self.component_data_objects(*args, **kwargs)

    def active_component_data(self, *args, **kwargs):
        logger.warning(
            "DEPRECATED: The active_component_data method is deprecated.  Use the Block.component_data_objects() method.")
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

    def all_blocks(self, *args, **kwargs):
        logger.warning(
            "DEPRECATED: The all_blocks method is deprecated.  Use the Block.block_data_objects() method.")
        return self.block_data_objects(*args, **kwargs)

    def active_blocks(self, *args, **kwargs):
        logger.warning(
            "DEPRECATED: The active_blocks method is deprecated.  Use the Block.block_data_objects() method.")
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
        # if self.parent_component().type() not in ctype:
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
        PM = _BlockData.PseudoMap(self, ctype, active, sort)
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

    def pprint(self, filename=None, ostream=None, verbose=False, prefix=""):
        """
        Print a summary of the block info
        """
        if filename is not None:
            OUTPUT = open(filename, "w")
            self.pprint(ostream=OUTPUT, verbose=verbose, prefix=prefix)
            OUTPUT.close()
            return
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
        #for item in [ModelComponentFactory.get_class(name).component for name in ModelComponentFactory]:
        for item in [ModelComponentFactory.get_class(name) for name in ModelComponentFactory]:
            if not item in items:
                dynamic_items.add(item)
        # extra items get added alphabetically (so output is consistent)
        items.extend(sorted(dynamic_items, key=lambda x: x.__name__))

        for item in items:
            keys = sorted(self.component_map(item))
            if not keys:
                continue
            #
            # NOTE: these conditional checks should not be hard-coded.
            #
            ostream.write("%s%d %s Declarations\n"
                          % (prefix, len(keys), item.__name__))
            for key in keys:
                self.component(key).pprint(
                    ostream=ostream, verbose=verbose, prefix=prefix + '    ')
            ostream.write("\n")
        #
        # Model Order
        #
        decl_order_keys = list(self.component_map().keys())
        ostream.write("%s%d Declarations: %s\n"
                      % (prefix, len(decl_order_keys),
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
        if (filename is not None) and (format is None):
            format = guess_format(filename)
            if format is None:
                raise ValueError(
                    "Could not infer file format from file name '%s'.\n"
                    "Either provide a name with a recognized extension "
                    "or specify the format using the 'format' argument."
                    % filename)
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

        if __debug__ and logger.isEnabledFor(logging.DEBUG):
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
        self._rule = kwargs.pop('rule', None)
        self._options = kwargs.pop('options', None)
        _concrete = kwargs.pop('concrete', False)
        kwargs.setdefault('ctype', Block)
        ActiveIndexedComponent.__init__(self, *args, **kwargs)
        if _concrete:
            # Call self.construct() as opposed to just setting the _constructed
            # flag so that the base class construction procedure fires (this
            # picks up any construction rule that the user may provide)
            self.construct()

    def _getitem_when_not_present(self, idx):
        return self._setitem_when_not_present(idx, None)

    def find_component(self, label_or_component):
        """
        Return a block component given a name.
        """
        return ComponentUID(label_or_component).find_component_on(self)

    def construct(self, data=None):
        """
        Initialize the block
        """
        if __debug__ and logger.isEnabledFor(logging.DEBUG):
            logger.debug("Constructing %s '%s', from data=%s",
                         self.__class__.__name__, self.name, str(data))
        if self._constructed:
            return
        timer = ConstructionTimer(self)
        self._constructed = True

        # We must check that any pre-existing components are
        # constructed.  This catches the case where someone is building
        # a Concrete model by building (potentially pseudo-abstract)
        # sub-blocks and then adding them to a Concrete model block.
        for idx in self._data:
            _block = self[idx]
            for name, obj in iteritems(_block.component_map()):
                if not obj._constructed:
                    if data is None:
                        _data = None
                    else:
                        _data = data.get(name, None)
                    obj.construct(_data)

        if self._rule is None:
            # Ensure the _data dictionary is populated for singleton
            # blocks
            if not self.is_indexed():
                self[None]
            timer.report()
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
                self, self._rule, _block, idx, self._options)
            if id(_block) in _BlockConstruction.data:
                del _BlockConstruction.data[id(_block)]

            if isinstance(obj, _BlockData) and obj is not _block:
                # If the user returns a block, use their block instead
                # of the empty one we just created.
                for c in list(obj.component_objects(descend_into=False)):
                    obj.del_component(c)
                    _block.add_component(c.local_name, c)
                # transfer over any other attributes that are not components
                for name, val in iteritems(obj.__dict__):
                    if not hasattr(_block, name) and not hasattr(self, name):
                        super(_BlockData, _block).__setattr__(name, val)

            # TBD: Should we allow skipping Blocks???
            # if obj is Block.Skip and idx is not None:
            #   del self._data[idx]
        timer.report()

    def pprint(self, filename=None, ostream=None, verbose=False, prefix=""):
        """
        Print block information
        """
        if filename is not None:
            OUTPUT = open(filename, "w")
            self.pprint(ostream=OUTPUT, verbose=verbose, prefix=prefix)
            OUTPUT.close()
            return
        if ostream is None:
            ostream = sys.stdout

        subblock = self._parent is not None and self.parent_block() is not None
        if subblock:
            super(Block, self).pprint(ostream=ostream, verbose=verbose,
                                      prefix=prefix)

        if not len(self):
            return
        if not self.is_indexed():
            _BlockData.pprint(self, ostream=ostream, verbose=verbose,
                              prefix=prefix+'    ' if subblock else prefix)
            return

        # Note: all indexed blocks must be sub-blocks (if they aren't
        # then you will run into problems constructing them as there is
        # nowhere to put (or find) the indexing set!).
        prefix += '    '
        for key in sorted(self):
            b = self[key]
            ostream.write("%s%s : Active=%s\n" %
                          (prefix, b.name, b.active))
            _BlockData.pprint(b, ostream=ostream, verbose=verbose,
                              prefix=prefix + '    ' if subblock else prefix)

    def _pprint(self):
        return [("Size", len(self)),
                ("Index", self._index if self.is_indexed() else None),
                ('Active', self.active),
                ], ().__iter__(), (), ()

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
        self._data[None] = self

    def pprint(self, filename=None, ostream=None, verbose=False, prefix=""):
        """
        Print block information
        """
        Block.pprint(self, filename, ostream, verbose, prefix)

    def display(self, filename=None, ostream=None, prefix=""):
        """
        Display values in the block
        """
        Block.display(self, filename, ostream, prefix)


class IndexedBlock(Block):

    def __init__(self, *args, **kwds):
        Block.__init__(self, *args, **kwds)


def generate_cuid_names(block,
                        ctype=None,
                        descend_into=True,
                        cuid_names_=None):
    """
    Bulk generation of CUID strings for all components stored on a block.

    Args:
        block: The block to generate CUID strings for.
        ctype: The ctype to generate CUID strings for (e.g.,
            Var). This keyword is optional and if left to
            its default value of None, the function will
            generate CUID strings for all component
            types. Note that if ctype is not None, this
            function will still generate CUID strings for
            any parent containers (such as blocks) that
            prefix the components requested even though the
            parent ctype may not match the input ctype.
        descend_into (bool, component type, or iterable of component types):
            Indicates whether or not the function should descend
            into subblocks. Default is True.
            Example usage: descend_into=(Block, Disjunct)
        cuid_names_: Used internally by the function.

    Returns:
        A dictionary-like object that maps model components
        to their CUID string.

    """
    # get the current blocks label, if it has one
    if cuid_names_ is None:
        cuid_names_ = ComponentMap()
        block_prefix = ''
    else:
        block_prefix = cuid_names_[block] + '.'

    # determine if we need to generate labels on
    # subblocks
    if descend_into is True:
        descend_ctype = (Block,)
    elif descend_into is False:
        descend_ctype = False
    elif type(descend_into) == type:
        descend_ctype = (descend_into,)
    elif isinstance(descend_into, collections.Iterable):
        for i in descend_into:
            assert type(i) == type
        descend_ctype = tuple(descend_into)
    else:
        raise ValueError('Unrecognized value passed to descend_into: %s. '
                         'We support True, False, types, or '
                         'iterables of types.'
                         % descend_into)

    if type(ctype) in (tuple, list, set):
        ctypes = tuple(ctype)
    elif ctype is None:
        ctypes = None
    else:
        ctypes = (ctype,)

    if descend_into and ctype is not None:
        ctypes = tuple(set(descend_ctype) | set(ctypes))

    for key, obj in block.component_map(ctype=ctypes).items():
        obj_cuid = block_prefix + key
        if obj.is_indexed():
            for data_key, obj_data in obj.items():
                if data_key.__class__ is tuple:
                    key_cuid = ','.join(
                        ComponentUID.tDict.get(type(x), '?') + str(x)
                        for x in data_key)
                else:
                    key_cuid = ComponentUID.tDict.get(type(data_key), '?') + \
                        str(data_key)
                cuid_names_[obj_data] = obj_cuid + ":" + key_cuid
            obj_cuid += ":**"
        cuid_names_[obj] = obj_cuid

    # Now recurse into subblocks
    if descend_into:
        sub_blocks = block.component_data_objects(descend_ctype,
                                                  descend_into=descend_into)
        for block_ in sub_blocks:
            generate_cuid_names(block_,
                                ctype=ctypes,
                                descend_into=False,
                                cuid_names_=cuid_names_)

    return cuid_names_


#
# Deprecated functions.
#
def active_components(block, ctype, sort_by_names=False, sort_by_keys=False):
    """DEPRECATED: The active_components function is deprecated.

    Use the Block.component_objects() method.
    """
    logger.warning(active_components.__doc__)
    return block.component_objects(ctype, active=True, sort=sort_by_names)


def components(block, ctype, sort_by_names=False, sort_by_keys=False):
    """DEPRECATED: The components function is deprecated.

    Use the Block.component_objects() method.
    """
    logger.warning(components.__doc__)
    return block.component_objects(ctype, active=False, sort=sort_by_names)


def active_components_data(block, ctype,
                           sort=None, sort_by_keys=False, sort_by_names=False):
    """DEPRECATED: The active_components_data function is deprecated.

    Use the Block.component_data_objects() method.
    """
    logger.warning(active_components_data.__doc__)
    return block.component_data_objects(ctype=ctype, active=True, sort=sort)


def components_data(block, ctype,
                    sort=None, sort_by_keys=False, sort_by_names=False):
    """DEPRECATED: The components_data function is deprecated.

    Use the Block.component_data_objects() method.
    """
    logger.warning(components_data.__doc__)
    return block.component_data_objects(ctype=ctype, active=False, sort=sort)


#
# Create a Block and record all the default attributes, methods, etc.
# These will be assumes to be the set of illegal component names.
#
_BlockData._Block_reserved_words = set(dir(Block()))

