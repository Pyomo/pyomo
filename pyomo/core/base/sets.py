#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

# TODO
# . rename 'filter' to something else
# . confirm that filtering is efficient

__all__ = ['Set', 'set_options', 'simple_set_rule', 'SetOf']

import logging
import sys
import types
import copy
import itertools
from weakref import ref as weakref_ref

from pyutilib.misc import flatten_tuple as pyutilib_misc_flatten_tuple

from pyomo.common.timing import ConstructionTimer
from pyomo.core.base.misc import apply_indexed_rule, \
    apply_parameterized_indexed_rule, sorted_robust
from pyomo.core.base.plugin import ModelComponentFactory
from pyomo.core.base.component import Component, ComponentData
from pyomo.core.base.indexed_component import IndexedComponent, \
    UnindexedComponent_set
from pyomo.core.base.numvalue import native_numeric_types

from six import itervalues, iteritems, string_types
from six.moves import xrange

logger = logging.getLogger('pyomo.core')

def process_setarg(arg):
    """
    Process argument and return an associated set object.

    This method is used by IndexedComponent
    """
    if isinstance(arg,_SetDataBase):
        # Argument is a non-indexed Set instance
        return arg
    elif isinstance(arg,IndexedSet):
        # Argument is an indexed Set instance
        raise TypeError("Cannot index a component with an indexed set")
    elif isinstance(arg,Component):
        # Argument is some other component
        raise TypeError("Cannot index a component with a non-set "
                        "component: %s" % (arg.name))
    else:
        try:
            #
            # If the argument has a set_options attribute, then use
            # it to initialize a set
            #
            options = getattr(arg,'set_options')
            options['initialize'] = arg
            return Set(**options)
        except:
            pass
    # Argument is assumed to be an initialization function
    return Set(initialize=arg)


def set_options(**kwds):
    """
    This is a decorator for set initializer functions.  This
    decorator allows an arbitrary dictionary of values to passed
    through to the set constructor.

    Examples:
        @set_options(dimen=3)
        def B_index(model):
            return [(i,i+1,i*i) for i in model.A]

        @set_options(domain=Integers)
        def B_index(model):
            return range(10)
    """
    def decorator(func):
        func.set_options = kwds
        return func
    return decorator


def simple_set_rule( fn ):
    """
    This is a decorator that translates None into Set.End.
    This supports a simpler syntax in set rules, though these can be
    more difficult to debug when errors occur.

    Example:

    @simple_set_rule
    def A_rule(model, i, j):
        ...
    """

    def wrapper_function ( *args, **kwargs ):
        value = fn( *args, **kwargs )
        if value is None:
            return Set.End
        return value
    return wrapper_function


def _value_sorter(self, obj):
    """Utility to sort the values of a Set.

    This returns the values of the Set in a consistent order.  For
    ordered Sets, simply return the ordered list.  For unordered Sets,
    first try the standard sorted order, and if that fails (for example
    with mixed-type Sets in Python3), use the sorted_robust utility to
    generate sortable keys.

    """
    if self.ordered:
        return obj.value
    else:
        return sorted_robust(obj)


# A trivial class that we can use to test if an object is a "legitimate"
# set (either SimpleSet, or a member of an IndexedSet)
class _SetDataBase(ComponentData):
    __slots__ = tuple()


class _SetData(_SetDataBase):
    """
    This class defines the data for an unordered set.

    Constructor Arguments:
        owner       The Set object that owns this data.
        bounds      A tuple of bounds for set values: (lower, upper)

    Public Class Attributes:
        value_list  The list of values
        value       The set of values
        _bounds     The tuple of bound values
    """

    __slots__ = ('value_list', 'value', '_bounds')

    def __init__(self, owner, bounds):
        #
        # The following is equivalent to calling
        # the base ComponentData constructor.
        #
        self._component = weakref_ref(owner)
        #
        self._clear()
        self._bounds = bounds

    def __getstate__(self):
        """
        This method must be defined because this class uses slots.
        """
        state = super(_SetData, self).__getstate__()
        for i in _SetData.__slots__:
            state[i] = getattr(self, i)
        return state

    # Note: because None of the slots on this class need to be edited,
    # we don't need to implement a specialized __setstate__ method.

    def __getitem__(self, key):
        """
        Return the specified member of the set.

        This method generates an exception because the set is unordered.
        """
        raise ValueError("Cannot index an unordered set '%s'" % self._component().name)

    def bounds(self):
        """
        Return bounds information.  The default value is 'None', which
        indicates that this set does not contain bounds.  Otherwise, this is
        assumed to be a tuple: (lower, upper).
        """
        return self._bounds

    def data(self):
        """
        The underlying set data.

        Note that this method is preferred to the direct use of the
        'value' attribute in most cases.  The reason is that the
        underlying set values may not be stored as a Python set() object.
        In fact, the underlying set values may not be explicitly stored
        in the Set() object at all!
        """
        return self.value

    def _clear(self):
        """
        Reset the set data
        """
        self.value = set()
        self.value_list = []

    def _add(self, val, verify=True):
        """
        Add an element, and optionally verify that it is a valid type.

        The type verification is done by the owning component.
        """
        if verify:
            self._component()._verify(val)
        if not val in self.value:
            self.value.add(val)
            self.value_list.append(val)

    def _discard(self, val):
        """
        Discard an element of this set.  This does not return an error
        if the element does not already exist.

        NOTE: This operation is probably expensive, as it should require a walk through a list.  An
        OrderedDict object might be more efficient, but it's notoriously slow in Python 2.x

        NOTE: We could make this more efficient by mimicing the logic in the _OrderedSetData class.
        But that would make the data() method expensive (since it is creating a set).  It's
        not obvious which is the better choice.
        """
        try:
            self.value.remove(val)
            self.value_list.remove(val)
        except KeyError:
            pass

    def __len__(self):
        """
        Return the number of elements in the set.
        """
        return len(self.value)

    def __iter__(self):
        """
        Return an iterator for the set.
        """
        return self.value_list.__iter__()

    def __contains__(self, val):
        """
        Return True if the set contains a given value.
        """
        return val in self.value


class _OrderedSetData(_SetDataBase):
    """
    This class defines the data for an ordered set.

    Constructor Arguments:
        owner       The Set object that owns this data.
        bounds      A tuple of bounds for set values: (lower, upper)

    Public Class Attributes:
        value       The set values
        _bounds     The tuple of bound values
        order_dict  A dictionary that maps from element value to element id.
                        Indices in this dictionary start with 1 (not 0).

    The ordering supported in this class depends on the 'ordered' attribute
    of the owning component:
        InsertionOrder      The order_dict maps from the insertion order
                                back to the member of the value array.
        SortedOrder         The ordered attribute of the owning component can
                                be used to define the sort order.  By default,
                                the Python ordering of the set types is used.
                                Note that a _stable_ sort method is required
                                if the discard method is used.
    """

    __slots__ = ('value', 'order_dict', '_bounds', '_is_sorted')

    def __init__(self, owner, bounds):
        #
        # The following is equivalent to calling
        # the base ComponentData constructor.
        #
        self._component = weakref_ref(owner)
        #
        self._bounds = bounds
        if self.parent_component().ordered is Set.InsertionOrder:
            self._is_sorted = 0
        else:
            self._is_sorted = 1
        self._clear()

    def __getstate__(self):
        """
        This method must be defined because this class uses slots.
        """
        state = super(_OrderedSetData, self).__getstate__()
        for i in _OrderedSetData.__slots__:
            state[i] = getattr(self, i)
        return state

    # Note: because None of the slots on this class need to be edited,
    # we don't need to implement a specialized __setstate__ method.

    def bounds(self):
        """
        Return bounds information.  The default value is 'None', which
        indicates that this set does not contain bounds.  Otherwise, this is
        assumed to be a tuple: (lower, upper).
        """
        return self._bounds

    def data(self):
        """
        Return the underlying set data.

        Note that this method returns a value that is different from the
        'value' attribute.  The underlying set values are not be stored
        as a Python set() object.
        """
        return set(self.value)

    def _sort(self):
        """
        Sort the set using the 'ordered' attribute of the owning
        component.  This recreates the order_dict dictionary, which indicates
        that the set is sorted.
        """
        _sorter = self.parent_component().ordered
        self.value = sorted(self.value, key=None if _sorter is Set.SortedOrder else _sorter)
        self.order_dict = dict((j,i) for i,j in enumerate(self.value))
        self._is_sorted = 1

    def _clear(self):
        """
        Reset the set data
        """
        self.value = []
        self.order_dict = {}
        if self._is_sorted:
            self._is_sorted = 1

    def _add(self, val, verify=True):
        """
        Add an element, and optionally verify that it is a valid type.

        The type verification is done by the owning component.
        """
        if verify:
            self._component()._verify(val)
        self.order_dict[val] = len(self.value)
        self.value.append(val)
        if self._is_sorted:
            self._is_sorted = 2

    def _discard(self, val):
        """
        Discard an element of this set.  This does not return an error
        if the element does not already exist.
        """
        try:
            _id = self.order_dict.pop(val)
        except KeyError:
            return
        del self.value[_id]
        #
        # Update the order_dict: this assumes the user-specified sorter
        # (if one was used) is stable.
        #
        for i in xrange(_id,len(self.value)):
            self.order_dict[self.value[i]] = i

    def __len__(self):
        """
        Return the number of elements in the set.
        """
        return len(self.value)

    def __iter__(self):
        """
        Return an iterator for the set.
        """
        if self._is_sorted == 2:
            self._sort()
        return self.value.__iter__()

    def __contains__(self, val):
        """
        Return True if the set contains a given value.
        """
        return val in self.order_dict

    def first(self):
        """
        Return the first element of the set.
        """
        if self._is_sorted == 2:
            self._sort()
        return self[1]

    def last(self):
        """
        Return the last element of the set.
        """
        if self._is_sorted == 2:
            self._sort()
        return self[len(self)]

    def __getitem__(self, idx):
        """
        Return the specified member of the set.

        The public Set API is 1-based, even though the
        internal order_dict is (pythonically) 0-based.
        """
        if self._is_sorted == 2:
            self._sort()
        if idx >= 1:
            if idx > len(self):
                raise IndexError("Cannot index a RangeSet past the last element")
            return self.value[idx-1]
        elif idx < 0:
            if len(self)+idx < 0:
                raise IndexError("Cannot index a RangeSet past the first element")
            return self.value[idx]
        else:
            raise IndexError("Valid index values for sets are 1 .. len(set) or -1 .. -len(set)")


    def ord(self, match_element):
        """
        Return the position index of the input value.  The
        position indices start at 1.
        """
        if self._is_sorted == 2:
            self._sort()
        try:
            return self.order_dict[match_element] + 1
        except IndexError:
            raise IndexError("Unknown input element="+str(match_element)+" provided as input to ord() method for set="+self.name)

    def next(self, match_element, k=1):
        """
        Return the next element in the set. The default
        behavior is to return the very next element. The k
        option can specify how many steps are taken to get
        the next element.

        If the next element is beyond the end of the set,
        then an exception is raised.
        """
        try:
            element_position = self.ord(match_element)
        except IndexError:
            raise KeyError("Cannot obtain next() member of set="+self.name+"; input element="+str(match_element)+" is not a member of the set!")
        #
        try:
            return self[element_position+k]
        except KeyError:
            raise KeyError("Cannot obtain next() member of set="+self.name+"; failed to access item in position="+str(element_position+k))

    def nextw(self, match_element, k=1):
        """
        Return the next element in the set.  The default
        behavior is to return the very next element.  The k
        option can specify how many steps are taken to get
        the next element.

        If the next element goes beyond the end of the list
        of elements in the set, then this wraps around to
        the beginning of the list.
        """
        try:
            element_position = self.ord(match_element)
        except KeyError:
            raise KeyError("Cannot obtain nextw() member of set="+self.name+"; input element="+str(match_element)+" is not a member of the set!")
        #
        return self[(element_position+k-1) % len(self.value) + 1]

    def prev(self, match_element, k=1):
        """
        Return the previous element in the set. The default
        behavior is to return the element immediately prior
        to the specified element.  The k option can specify
        how many steps are taken to get the previous
        element.

        If the previous element is before the start of the
        set, then an exception is raised.
        """
        return self.next(match_element, k=-k)

    def prevw(self, match_element, k=1):
        """
        Return the previous element in the set. The default
        behavior is to return the element immediately prior
        to the specified element.  The k option can specify
        how many steps are taken to get the previous
        element.

        If the previous element is before the start of the
        set, then this wraps around to the end of the list.
        """
        return self.nextw(match_element, k=-k)

class _IndexedSetData(_SetData):
    """
    This class adds the __call__ method, which is expected
    for indexed component data. But we omit this from
    _SetData because we do not want to treat scalar sets as
    functors.
    """

    __slots__ = tuple()

    def __call__(self):
        """
        Return the underlying set data.
        """
        return self.value

    def clear(self):
        """
        Reset this data.
        """
        self._clear()

    def add(self, val):
        """
        Add an element to the set.
        """
        self._add(val)

    def discard(self, val):
        """
        Discard an element from the set.
        """
        self._discard(val)


class _IndexedOrderedSetData(_OrderedSetData):
    """
    This class adds the __call__ method, which is expected
    for indexed component data. But we omit this from
    _OrderedSetData because we do not want to treat scalar
    sets as functors.
    """

    __slots__ = tuple()

    def __call__(self):
        """
        Return the underlying set data.
        """
        return self.value

    def clear(self):
        """
        Reset this data.
        """
        self._clear()

    def add(self, val):
        """
        Add an element to the set.
        """
        self._add(val)

    def discard(self, val):
        """
        Discard an element from the set.
        """
        self._discard(val)


@ModelComponentFactory.register("Set data that is used to define a model instance.")
class Set(IndexedComponent):
    """
    A set object that is used to index other Pyomo objects.

    This class has a similar look-and-feel as a Python set class.
    However, the set operations defined in this class return another
    abstract Set object. This class contains a concrete set, which
    can be initialized by the load() method.

    Constructor Arguments:
        name            The name of the set
        doc             A text string describing this component
        within          A set that defines the type of values that can
                            be contained in this set
        domain          A set that defines the type of values that can
                            be contained in this set
        initialize      A dictionary or rule for setting up this set
                            with existing model data
        validate        A rule for validating membership in this set. This has
                            the functional form:
                                f: data -> bool
                            and returns true if the data belongs in the set
        dimen           Specify the set's arity, or None if no arity is enforced
        virtual         If true, then this is a virtual set that does not
                            store data using the class dictionary
        bounds          A 2-tuple that specifies the range of possible set values.
        ordered         Specifies whether the set is ordered. Possible values are:
                            False           Unordered
                            True            Ordered by insertion order
                            InsertionOrder  Ordered by insertion order
                            SortedOrder     Ordered by sort order
                            <function>      Ordered with this comparison function
        filter          A function that is used to filter set entries.

    Public class attributes:
        concrete        If True, then this set contains elements.(TODO)
        dimen           The dimension of the data in this set.
        doc             A text string describing this component
        domain          A set that defines the type of values that can
                            be contained in this set
        filter          A function that is used to filter set entries.
        initialize      A dictionary or rule for setting up this set
                            with existing model data
        ordered         Specifies whether the set is ordered.
        validate        A rule for validating membership in this set.
        virtual         If True, then this set does not store data using the class
                             dictionary
    """

    End             = (1003,)
    InsertionOrder  = (1004,)
    SortedOrder     = (1005,)

    def __new__(cls, *args, **kwds):
        if cls != Set:
            return super(Set, cls).__new__(cls)
        if not args or (args[0] is UnindexedComponent_set and len(args)==1):
            if kwds.get('ordered',False) is False:
                return SimpleSet.__new__(SimpleSet)
            else:
                return OrderedSimpleSet.__new__(OrderedSimpleSet)
        else:
            return IndexedSet.__new__(IndexedSet)

    def __init__(self, *args, **kwds):
        #
        # Default keyword values
        #
        kwds.setdefault("name", "_unknown_")
        self.initialize = kwds.pop("rule", None)
        self.initialize = kwds.pop("initialize", self.initialize)
        self.validate   = kwds.pop("validate", None)
        self.ordered    = kwds.pop("ordered", False)
        self.filter     = kwds.pop("filter", None)
        self.domain     = kwds.pop("within", None)
        self.domain     = kwds.pop('domain', self.domain )
        #
        if self.ordered is True:
            self.ordered = Set.InsertionOrder

        # We can't access self.dimen after its been written, so we use
        # tmp_dimen until the end of __init__
        tmp_dimen = 0

        # Get dimen from domain, if possible
        if self.domain is not None:
            tmp_dimen = getattr(self.domain, 'dimen', 0)
        if self._bounds is None and not self.domain is None:
            self._bounds = copy.copy(self.domain._bounds)

        # Make sure dimen and implied dimensions don't conflict
        kwd_dimen = kwds.pop("dimen", 0)
        if kwd_dimen != 0:
            if self.domain is not None and tmp_dimen != kwd_dimen:
                raise ValueError(\
                      ("Value of keyword 'dimen', %s, differs from the " + \
                       "dimension of the superset '%s', %s") % \
                       (str(kwd_dimen), str(self.domain.name), str(tmp_dimen)))
            else:
                tmp_dimen = kwd_dimen

        kwds.setdefault('ctype', Set)
        IndexedComponent.__init__(self, *args, **kwds)

        if tmp_dimen == 0:
            # We set the default to 1
            tmp_dimen = 1
        if self.initialize is not None:
            #
            # Convert initialization value to a list (which are
            # copyable).  There are subtlies here: dict should be left
            # alone (as dict's are used for initializing indezed Sets),
            # and lists should be left alone (for efficiency).  tuples,
            # generators, and iterators like dict.keys() [in Python 3.x]
            # should definitely be converted to lists.
            #
            if type(self.initialize) is tuple \
                    or ( hasattr(self.initialize, "__iter__")
                         and not hasattr(self.initialize, "__getitem__") ):
                self.initialize = list(self.initialize)
            #
            # Try to guess dimen from the initialize list
            #
            if not tmp_dimen is None:
                tmp=0
                if type(self.initialize) is tuple:
                    tmp = len(self.initialize)
                elif type(self.initialize) is list and len(self.initialize) > 0 \
                         and type(self.initialize[0]) is tuple:
                    tmp = len(self.initialize[0])
                else:
                    tmp = getattr(self.initialize, 'dimen', tmp)
                if tmp != 0:
                    if kwd_dimen != 0 and tmp != kwd_dimen:
                        raise ValueError("Dimension argument differs from the data in the initialize list")
                    tmp_dimen = tmp

        self.dimen = tmp_dimen

    def _verify(self, element):
        """
        Verify that the element is valid for this set.
        """
        if self.domain is not None and element not in self.domain:
            raise ValueError(
                "The value=%s is not valid for set=%s\n"
                "because it is not within the domain=%s"
                % ( element, self.name, self.domain.name ) )
        if self.validate is not None:
            flag = False
            try:
                if self._parent is not None:
                    flag = apply_indexed_rule(self, self.validate, self._parent(), element)
                else:
                    flag = apply_indexed_rule(self, self.validate, None, element)
            except:
                pass
            if not flag:
                raise ValueError("The value="+str(element)+" violates the validation rule of set="+self.name)
        if not self.dimen is None:
            if self.dimen > 1 and type(element) is not tuple:

                raise ValueError("The value="+str(element)+" is not a tuple for set="+self.name+", which has dimen="+str(self.dimen))
            elif self.dimen == 1 and type(element) is tuple:
                raise ValueError("The value="+str(element)+" is a tuple for set="+self.name+", which has dimen="+str(self.dimen))
            elif type(element) is tuple and len(element) != self.dimen:
                raise ValueError("The value="+str(element)+" does not have dimension="+str(self.dimen)+", which is needed for set="+self.name)
        return True


class SimpleSetBase(Set):
    """
    A derived Set object that contains a single set.
    """

    def __init__(self, *args, **kwds):
        self.virtual    = kwds.pop("virtual", False)
        self.concrete   = not self.virtual
        Set.__init__(self, *args, **kwds)

    def valid_model_component(self):
        """
        Return True if this can be used as a model component.
        """
        if self.virtual and not self.concrete:
            return False
        return True

    def clear(self):
        """
        Clear that data in this component.
        """
        if self.virtual:
            raise TypeError("Cannot clear virtual set object `"+self.name+"'")
        self._clear()

    def check_values(self):
        """
        Verify that the values in this set are valid.
        """
        if not self.concrete:
            return
        for val in self:
            self._verify(val)

    def add(self, *args):
        """
        Add one or more elements to a set.
        """
        if self.virtual:
            raise TypeError("Cannot add elements to virtual set `"+self.name+"'")
        for val in args:
            tmp = pyutilib_misc_flatten_tuple(val)
            self._verify(tmp)
            try:
                if tmp in self:
                    #
                    # Generate a warning, since we expect that users will not plan to
                    # re-add the same element to a set.
                    #
                    logger.warning("Element "+str(tmp)+" already exists in set "+self.name+"; no action taken.")
                    continue
                self._add(tmp, False)
            except TypeError:
                raise TypeError("Problem inserting "+str(tmp)+" into set "+self.name)

    def remove(self, element):
        """
        Remove an element from the set.

        If the element is not a member, raise an error.
        """
        if self.virtual:
            raise KeyError("Cannot remove element `"+str(element)+"' from virtual set "+self.name)
        if element not in self:
            raise KeyError("Cannot remove element `"+str(element)+"' from set "+self.name)
        self._discard(element)

    def discard(self, element):
        """
        Remove an element from the set.

        If the element is not a member, do nothing.
        """
        if self.virtual:
            raise KeyError("Cannot discard element `"+str(element)+"' from virtual set "+self.name)
        self._discard(element)

    def _pprint(self):
        """
        Return data that will be printed for this component.
        """
        _ordered = self.ordered
        if type(_ordered) is bool:
            pass
        elif _ordered is Set.InsertionOrder:
            _ordered = 'Insertion'
        elif _ordered is Set.SortedOrder:
            _ordered = 'Sorted'
        else:
            _ordered = '{user}'
        return (
            [("Dim", self.dim()),
             ("Dimen", self.dimen),
             ("Size", len(self)),
             ("Domain", None if self.domain is None else self.domain.name),
             ("Ordered", _ordered),
             ("Bounds", self._bounds)],
            iteritems( {None: self} ),
            None, #("Members",),
            lambda k, v: [
                "Virtual" if not self.concrete or v.virtual \
                    else v.value if v.ordered \
                    else sorted(v), ] )

    def _set_repn(self, other):
        """
        Return a Set subset for 'other'
        """
        if isinstance(other, SimpleSet):
            return other
        if isinstance(other, OrderedSimpleSet):
            return other
        return SetOf(other)

    def __len__(self):
        """
        Return the number of elements in this set.
        """
        if not self.concrete:
            raise ValueError("The size of a non-concrete set is unknown")
        return len(self.value)

    def __iter__(self):
        """
        Return an iterator for the underlying set
        """
        if not self._constructed:
            raise RuntimeError(
                "Cannot iterate over abstract Set '%s' before it has "
                "been constructed (initialized)." % (self.name,) )
        if not self.concrete:
            raise TypeError("Cannot iterate over a non-concrete set '%s'" % self.name)
        return self.value.__iter__()

    def __reversed__(self):
        """
        Return a reversed iterator
        """
        return reversed(self.__iter__())

    def __hash__(self):
        """
        Hash this object
        """
        return Set.__hash__(self)

    def __eq__(self,other):
        """
        Equality comparison
        """
        # the obvious test: two references to the same set are the same
        if id(self) == id(other):
            return True
        # easy cases: if other isn't a Set-like thing, then we aren't equal
        if other is None:
            return False
        try:
            tmp = self._set_repn(other)
        except:
            return False
        # if we are both concrete, then we should compare elements
        if self.concrete and tmp.concrete:
            if self.dimen != tmp.dimen:
                return False
            if self.virtual or tmp.virtual:
                # optimization: usually len() is faster than checking
                # all elements... if the len() are different, then we
                # are obviously not equal.  We only do this test here
                # because we assume that the __eq__() method for native
                # types (in the case of non-virtual sets) is already
                # smart enough to do this optimization internally if it
                # is applicable.
                if len(self) != len(other):
                    return False
                for i in other:
                    if not i in self:
                        return False
                return True
            else:
                return self.data().__eq__( tmp.data() )

        # if we are both virtual, compare hashes
        if self.virtual and tmp.virtual:
            return hash(self) == hash(tmp)

        # I give... not equal!
        return False

    def __ne__(self,other):
        """
        Inequality comparison
        """
        return not self.__eq__(other)

    def __contains__(self, element):
        """
        Return True if element is a member of this set.
        """
        #
        # If the element is a set, then see if this is a subset.
        # We first test if the element is a number or tuple, before
        # doing the expensive calls to isinstance().
        #
        element_t = type(element)
        if not element_t in native_numeric_types and element_t is not tuple:
            if isinstance(element,SimpleSet) or isinstance(element,OrderedSimpleSet):
                return element.issubset(self)
        #    else:
        #        set_ = SetOf(element)
        #        return set_.issubset(self)

        #
        # When dealing with a concrete set, just check if the element is
        # in the set. There is no need for extra validation.
        #
        if self._constructed and self.concrete is True:
           return self._set_contains(element)
        #
        # If this is not a valid element, then return False
        #
        try:
            self._verify(element)
        except:
            return False
        #
        # If the validation rule is used then we do not actually
        # check whether the data is in self.value.
        #
        if self.validate is not None and not self.concrete:
            return True
        #
        # The final check: return true if self.concrete is False, since we should
        # have already validated this value.  The following, or at least one of
        # the execution paths - is probably redundant with the above.
        #
        return not self.concrete or self._set_contains(element)

    def isdisjoint(self, other):
        """
        Return True if the set has no elements in common with 'other'.
        Sets are disjoint if and only if their intersection is the empty set.
        """
        other = self._set_repn(other)
        tmp = self & other
        for elt in tmp:
            return False
        return True

    def issubset(self,other):
        """
        Return True if the set is a subset of 'other'.
        """
        if not self.concrete:
            raise TypeError("ERROR: cannot perform \"issubset\" test because the current set is not a concrete set.")
        other = self._set_repn(other)
        if self.dimen != other.dimen:
            raise ValueError("Cannot perform set operation with sets "+self.name+" and "+other.name+" that have different element dimensions: "+str(self.dimen)+" "+str(other.dimen))
        for val in self:
            if val not in other:
                return False
        return True

    def issuperset(self, other):
        """
        Return True if the set is a superset of 'other'.

        Note that we do not simply call other.issubset(self) because
        'other' may not be a Set instance.
        """
        other = self._set_repn(other)
        if self.dimen != other.dimen:
            raise ValueError("Cannot perform set operation with sets "+self.name+" and "+other.name+" that have different element dimensions: "+str(self.dimen)+" "+str(other.dimen))
        if not other.concrete:
            raise TypeError("ERROR: cannot perform \"issuperset\" test because the target set is not a concrete set.")
        for val in other:
            if val not in self:
                return False
        return True

    def union(self, *args):
        """
        Return the union of this set with one or more sets.
        """
        tmp = self
        for arg in args:
            tmp = _SetUnion(tmp, arg)
        return tmp

    def intersection(self, *args):
        """
        Return the intersection of this set with one or more sets
        """
        tmp = self
        for arg in args:
            tmp = _SetIntersection(tmp, arg)
        return tmp

    def difference(self, *args):
        """
        Return the difference between this set with one or more sets
        """
        tmp = self
        for arg in args:
            tmp = _SetDifference(tmp, arg)
        return tmp

    def symmetric_difference(self, *args):
        """
        Return the symmetric difference of this set with one or more sets
        """
        tmp = self
        for arg in args:
            tmp = _SetSymmetricDifference(tmp, arg)
        return tmp

    def cross(self, *args):
        """
        Return the cross-product between this set and one or more sets
        """
        tmp = self
        for arg in args:
            tmp = _SetProduct(tmp, arg)
        return tmp

    # <= is equivalent to issubset
    # >= is equivalent to issuperset
    # |  is equivalent to union
    # &  is equivalent to intersection
    # -  is equivalent to difference
    # ^  is equivalent to symmetric_difference
    # *  is equivalent to cross

    __le__  = issubset
    __ge__  = issuperset
    __or__  = union
    __and__ = intersection
    __sub__ = difference
    __xor__ = symmetric_difference
    __mul__ = cross

    def __lt__(self,other):
        """
        Return True if the set is a strict subset of 'other'

        TODO: verify that this is more efficient than an explicit implimentation.
        """
        return self <= other and not self == other

    def __gt__(self,other):
        """
        Return True if the set is a strict superset of 'other'

        TODO: verify that this is more efficient than an explicit implimentation.
        """
        return self >= other and not self == other

    def construct(self, values=None):
        """
        Apply the rule to construct values in this set

        TODO: rework to avoid redundant code
        """
        if __debug__ and logger.isEnabledFor(logging.DEBUG):
                logger.debug("Constructing SimpleSet, name="+self.name+", from data="+repr(values))
        if self._constructed:
            return
        timer = ConstructionTimer(self)
        self._constructed=True

        if self.initialize is None:                             # TODO: deprecate this functionality
            self.initialize = getattr(self,'rule',None)
            if not self.initialize is None:
                logger.warning("DEPRECATED: The set 'rule' attribute cannot be used to initialize component "+self.name+". Use the 'initialize' attribute")
        #
        # Construct using the input values list
        #
        if values is not None:
            if type(self._bounds) is tuple:
                first=self._bounds[0]
                last=self._bounds[1]
            else:
                first=None
                last=None
            all_numeric=True
            #
            # TODO: verify that values is not a list
            #
            for val in values[None]:
                #
                # Skip the value if it is filtered
                #
                if not self.filter is None and not apply_indexed_rule(self, self.filter, self._parent(), val):
                    continue
                self.add(val)
                if type(val) in native_numeric_types:
                    if first is None or val<first:
                        first=val
                    if last is None or val>last:
                        last=val
                else:
                    all_numeric=False
            if all_numeric:
                self._bounds = (first, last)
        #
        # Construct using the initialize rule
        #
        elif type(self.initialize) is types.FunctionType:
            if self._parent is None:
                raise ValueError("Must pass the parent block in to initialize with a function")
            if self.initialize.__code__.co_argcount == 1:
                #
                # Using a rule of the form f(model) -> iterator
                #
                tmp = self.initialize(self._parent())
                for val in tmp:
                    if self.dimen == 0:
                        if type(val) in [tuple,list]:
                            self.dimen=len(val)
                        else:
                            self.dimen=1
                    if not self.filter is None and \
                       not apply_indexed_rule(self, self.filter, self._parent(), val):
                        continue
                    self.add(val)
            else:
                #
                # Using a rule of the form f(model, z) -> element
                #
                ctr=1
                val = apply_indexed_rule(self, self.initialize, self._parent(), ctr)
                if val is None:
                    raise ValueError("Set rule returned None instead of Set.Skip")
                if self.dimen == 0:
                    if type(val) in [tuple,list] and not val == Set.End:
                        self.dimen=len(val)
                    else:
                        self.dimen=1
                while not (val.__class__ is tuple and val == Set.End):
                    # Add the value if the filter is None or the filter return value is True
                    if self.filter is None or \
                       apply_indexed_rule(self, self.filter, self._parent(), val):
                       self.add(val)
                    ctr += 1
                    val = apply_indexed_rule(self, self.initialize, self._parent(), ctr)
                    if val is None:
                        raise ValueError("Set rule returned None instead of Set.Skip")

            # Update the bounds if after using the rule, the set is
            # a one dimensional list of all numeric values
            if self.dimen == 1:
                if type(self._bounds) is tuple:
                    first=self._bounds[0]
                    last=self._bounds[1]
                else:
                    first=None
                    last=None
                all_numeric=True
                for val in self.value:
                    if type(val) in native_numeric_types:
                        if first is None or val<first:
                            first=val
                        if last is None or val>last:
                            last=val
                    else:
                        all_numeric=False
                        break
                if all_numeric:
                    self._bounds = (first, last)

        #
        # Construct using the default values
        #
        elif self.initialize is not None:
            if type(self.initialize) is dict:
                raise ValueError("Cannot initialize set "+self.name+" with dictionary data")
            if type(self._bounds) is tuple:
                first=self._bounds[0]
                last=self._bounds[1]
            else:
                first=None
                last=None
            all_numeric=True
            for val in self.initialize:
                # Skip the value if it is filtered
                if not self.filter is None and \
                   not apply_indexed_rule(self, self.filter, self._parent(), val):
                    continue
                if type(val) in native_numeric_types:
                    if first is None or val<first:
                        first=val
                    if last is None or val>last:
                        last=val
                else:
                    all_numeric=False
                self.add(val)
            if all_numeric:
                self._bounds = (first,last)
        timer.report()


class SimpleSet(SimpleSetBase,_SetData):

    def __init__(self, *args, **kwds):
        self._bounds = kwds.pop('bounds', None)
        SimpleSetBase.__init__(self, *args, **kwds)
        _SetData.__init__(self, self, self._bounds)

    def __getitem__(self, key):
        """
        Return the specified member of the set.

        This method generates an exception because the set is unordered.
        """
        return _SetData.__getitem__(self, key)

    def _set_contains(self, element):
        """
        A wrapper function that tests if the element is in
        the data associated with a concrete set.
        """
        return element in self.value


class OrderedSimpleSet(SimpleSetBase,_OrderedSetData):

    def __init__(self, *args, **kwds):
        self._bounds = kwds.pop('bounds', None)
        SimpleSetBase.__init__(self, *args, **kwds)
        _OrderedSetData.__init__(self, self, self._bounds)

    def __getitem__(self, key):
        """
        Return the specified member of the set.
        """
        return _OrderedSetData.__getitem__(self, key)

    def _set_contains(self, element):
        """
        A wrapper function that tests if the element is in
        the data associated with a concrete set.
        """
        return element in self.order_dict


# REVIEW - START

@ModelComponentFactory.register("Define a Pyomo Set component using an iterable data object.")
class SetOf(SimpleSet):
    """
    A derived SimpleSet object that creates a set from external
    data without duplicating it.
    """

    def __init__(self, *args, **kwds):
        if len(args) > 1:
            raise TypeError("Only one set data argument can be specified")
        self.dimen = 0
        SimpleSet.__init__(self,**kwds)
        if len(args) == 1:
            self._elements = args[0]
        else:
            self._elements = self.initialize
        self.value = None
        self._constructed = True
        self._bounds = (None, None) # We cannot determine bounds, since the data may change
        self.virtual = False
        try:
            len(self._elements)
            self.concrete = True
        except:
            self.concrete = False
        #
        if self.dimen == 0:
            try:
                for i in self._elements:
                    if type(i) is tuple:
                        self.dimen = len(i)
                    else:
                        self.dimen = 1
                    break
            except TypeError:
                e = sys.exc_info()[1]
                raise TypeError("Cannot create a Pyomo set: "+e)

    def construct(self, values=None):
        """
        Disabled construction method
        """
        ConstructionTimer(self).report()

    def __len__(self):
        """
        The number of items in the set.
        """
        try:
            return len(self._elements)
        except:
            pass
        #
        # If self._elements cannot provide size information,
        # then we need to iterate through all set members.
        #
        ctr = 0
        for i in self:
            ctr += 1
        return ctr

    def __iter__(self):
        """
        Return an iterator for the underlying set
        """
        for i in self._elements:
            yield i

    def _set_contains(self, element):
        """
        A wrapper function that tests if the element is in
        the data associated with a concrete set.
        """
        return element in self._elements

    def data(self):
        """
        Return the underlying set data by constructing
        a python set() object explicitly.
        """
        return set(self)


class _SetOperator(SimpleSet):
    """A derived SimpleSet object that contains a concrete virtual single set."""

    def __init__(self, *args, **kwds):
        if len(args) != 2:
            raise TypeError("Two arguments required for a binary set operator")
        dimen_test = kwds.get('dimen_test',True)
        if 'dimen_test' in kwds:
            del kwds['dimen_test']
        SimpleSet.__init__(self,**kwds)
        self.value = None
        self._constructed = True
        self.virtual = True
        self.concrete = True
        #
        self._setA = args[0]
        if not self._setA.concrete:
            raise TypeError("Cannot perform set operations with non-concrete set '"+self._setA.name+"'")
        if isinstance(args[1],Set):
            self._setB = args[1]
        else:
            self._setB = SetOf(args[1])
        if not self._setB.concrete:
            raise TypeError("Cannot perform set operations with non-concrete set '"+self._setB.name+"'")
        if dimen_test and self._setA.dimen != self._setB.dimen:
            raise ValueError("Cannot perform set operation with sets "+self._setA.name+" and "+self._setB.name+" that have different element dimensions: "+str(self._setA.dimen)+" "+str(self._setB.dimen))
        self.dimen = self._setA.dimen
        #
        self.ordered = self._setA.ordered and self._setB.ordered

        #
        # This line is critical in order for nested set expressions to
        # properly clone (e.g., m.D = m.A | m.B | m.C). The intermediate
        # _SetOperation constructs must be added to the model, so we
        # highjack the hack in block.py for IndexedComponent to
        # deal with multiple indexing arguments.
        #
        self._implicit_subsets = [self._setA, self._setB]

    def construct(self, values=None):
        """ Disabled construction method """
        timer = ConstructionTimer(self).report()

    def __len__(self):
        """The number of items in the set."""
        ctr = 0
        for i in self:
            ctr += 1
        return ctr

    def __iter__(self):
        """Return an iterator for the underlying set"""
        raise IOError("Undefined set iterator")

    def _set_contains(self, element):
        raise IOError("Undefined set operation")

    def data(self):
        """The underlying set data."""
        return set(self)

class _SetUnion(_SetOperator):

    def __init__(self, *args, **kwds):
        _SetOperator.__init__(self, *args, **kwds)

    def __iter__(self):
        for elt in self._setA:
            yield elt
        for elt in self._setB:
            if not elt in self._setA:
                yield elt

    def _set_contains(self, elt):
        return elt in self._setA or elt in self._setB

class _SetIntersection(_SetOperator):

    def __init__(self, *args, **kwds):
        _SetOperator.__init__(self, *args, **kwds)

    def __iter__(self):
        for elt in self._setA:
            if elt in self._setB:
                yield elt

    def _set_contains(self, elt):
        return elt in self._setA and elt in self._setB

class _SetDifference(_SetOperator):

    def __init__(self, *args, **kwds):
        _SetOperator.__init__(self, *args, **kwds)

    def __iter__(self):
        for elt in self._setA:
            if not elt in self._setB:
                yield elt

    def _set_contains(self, elt):
        return elt in self._setA and not elt in self._setB

class _SetSymmetricDifference(_SetOperator):

    def __init__(self, *args, **kwds):
        _SetOperator.__init__(self, *args, **kwds)

    def __iter__(self):
        for elt in self._setA:
            if not elt in self._setB:
                yield elt
        for elt in self._setB:
            if not elt in self._setA:
                yield elt

    def _set_contains(self, elt):
        return (elt in self._setA) ^ (elt in self._setB)

class _SetProduct(_SetOperator):

    def __init__(self, *args, **kwd):
        kwd['dimen_test'] = False

        # every input argument in a set product must be iterable.
        for arg in args:
            # obviouslly, if the object has an '__iter__' method, then
            # it is iterable. Checking for this prevents us from trying
            # to iterate over unconstructed Sets (which would result in
            # an exception)
            if not hasattr(arg, '__iter__'):
                try:
                    iter(arg)
                except TypeError:
                    raise TypeError("Each input argument to a _SetProduct constructor must be iterable")

        _SetOperator.__init__(self, *args, **kwd)
        # the individual index sets definining the product set.
        if isinstance(self._setA,_SetProduct):
            self.set_tuple = self._setA.set_tuple
        else:
            self.set_tuple = [self._setA]
        if isinstance(self._setB,_SetProduct):
            self.set_tuple += self._setB.set_tuple
        else:
            self.set_tuple.append(self._setB)
        self._setA = self._setB = None
        # set the "dimen" instance attribute.
        self._compute_dimen()

    def __iter__(self):
        if self.is_flat_product():
            for i in itertools.product(*self.set_tuple):
                yield i
        else:
            for i in itertools.product(*self.set_tuple):
                yield pyutilib_misc_flatten_tuple(i)

    def _set_contains(self, element):
        # Do we really need to check if element is a tuple???
        # if type(element) is not tuple:
        #    return False
        try:
            ctr = 0
            for subset in self.set_tuple:
                d = subset.dimen
                if d == 1:
                    if not subset._set_contains(element[ctr]):
                        return False
                elif d is None:
                    for dlen in range(len(element), ctr, -1):
                        if subset._set_contains(element[ctr:dlen]):
                            d = dlen - ctr
                            break
                    if d is None:
                        if subset._set_contains(element[ctr]):
                            d = 1
                        else:
                            return False
                else:
                    # cast to tuple is not needed: slices of tuples
                    # return tuples!
                    if not subset._set_contains(element[ctr:ctr+d]):
                        return False
                ctr += d
            return ctr == len(element)
        except:
            return False

    def __len__(self):
        ans = 1
        for _set in self.set_tuple:
            ans *= len(_set)
        return ans

    def _compute_dimen(self):
        ans=0
        for _set in self.set_tuple:
            if _set.dimen is None:
                self.dimen=None
                return
            else:
                ans += _set.dimen
        self.dimen = ans

    def is_flat_product(self):
        """
        a simple utility to determine if each of the composite sets is
        of dimension one. Knowing this can significantly reduce the
        cost of iteration, as you don't have to call flatten_tuple.
        """

        for s in self.set_tuple:
            if s.dimen != 1:
                return False
        return True

    def _verify(self, element):
        """
        If this set is virtual, then an additional check is made
        to ensure that the element is in each of the underlying sets.
        """
        tmp = SimpleSet._verify(self, element)
        return tmp

        # WEH - when is this needed?
        if not tmp or not self.virtual:
            return tmp

        next_tuple_index = 0
        member_set_index = 0
        for member_set in self.set_tuple:
            tuple_slice = element[next_tuple_index:next_tuple_index + member_set.dimen]
            if member_set.dimen == 1:
                tuple_slice = tuple_slice[0]
            if tuple_slice not in member_set:
                return False
            member_set_index += 1
            next_tuple_index += member_set.dimen
        return True

# REVIEW - END

class IndexedSet(Set):
    """
    An array of sets, which are indexed by other sets
    """

    def __init__(self, *args, **kwds):      #pragma:nocover
        self._bounds = kwds.pop("bounds", None)
        Set.__init__(self, *args, **kwds)
        if 'virtual' in kwds:                                       #pragma:nocover
            raise TypeError("It doesn't make sense to create a virtual set array")
        if self.ordered:
            self._SetData = _IndexedOrderedSetData
        else:
            self._SetData = _IndexedSetData

    def size(self):
        """
        Return the number of elements in all of the indexed sets.
        """
        ans = 0
        for cdata in itervalues(self):
            ans += len(cdata)
        return ans

    def data(self):
        """
        Return the dictionary of sets
        """
        return self._data

    def clear(self):
        """
        Clear that data in this component.
        """
        if self.is_indexed():
            self._data = {}
        else:
            #
            # TODO: verify that this could happen
            #
            pass

    def _getitem_when_not_present(self, index):
        """
        Return the default component data value

        This returns an exception.
        """
        tmp = self._data[index] = self._SetData(self, self._bounds)
        return tmp

    def __setitem__(self, key, vals):
        """
        Add a set to the index.
        """
        if key not in self._index:
            raise KeyError("Cannot set index "+str(key)+" in array set "+self.name)
        #
        # Create a _SetData object if one doesn't already exist
        #
        if key in self._data:
            self._data[key].clear()
        else:
            self._data[key] = self._SetData(self, self._bounds)
        #
        # Add the elements in vals to the _SetData object
        #
        _set = self._data[key]
        for elt in vals:
            _set.add(elt)

    def check_values(self):
        """
        Verify the values of all indexed sets.

        TODO: document when unverified values could be set.
        """
        for cdata in itervalues(self):
            for val in cdata.value:
                self._verify(val)

    def _pprint(self):
        """
        Return data that will be printed for this component.
        """
        _ordered = self.ordered
        if type(_ordered) is bool:
            pass
        elif _ordered is Set.InsertionOrder:
            _ordered = 'Insertion'
        elif _ordered is Set.SortedOrder:
            _ordered = 'Sorted'
        else:
            _ordered = '{user}'
        return (
            [("Dim", self.dim()),
             ("Dimen", self.dimen),
             ("Size", self.size()),
             ("Domain", None if self.domain is None else self.domain.name),
             ("ArraySize", len(self._data)),
             ("Ordered", _ordered),
             ("Bounds", self._bounds)],
            iteritems(self._data),
            ("Members",),
            lambda k, v: [ _value_sorter(self, v) ]
            )

    def construct(self, values=None):
        """
        Apply the rule to construct values in each set
        """
        if __debug__ and logger.isEnabledFor(logging.DEBUG):
                logger.debug("Constructing IndexedSet, name="+self.name+", from data="+repr(values))
        if self._constructed:
            return
        timer = ConstructionTimer(self)
        self._constructed=True
        #
        if self.initialize is None:             # TODO: deprecate this functionality
            self.initialize = getattr(self,'rule',None)
            if not self.initialize is None:
                logger.warning("DEPRECATED: The set 'rule' attribute cannot be used to initialize component "+self.name+". Use the 'initialize' attribute")
        #
        # Construct using the values dictionary
        #
        if values is not None:
            for key in values:
                if type(key) is tuple and len(key)==1:
                    tmpkey=key[0]
                else:
                    tmpkey=key
                if tmpkey not in self._index:
                    raise KeyError("Cannot construct index "+str(tmpkey)+" in array set "+self.name)
                tmp = self._SetData(self, self._bounds)
                for val in values[key]:
                    tmp._add(val)
                self._data[tmpkey] = tmp
        #
        # Construct using the rule
        #
        elif type(self.initialize) is types.FunctionType:
            if self._parent is None:
                raise ValueError("Need parent block to construct a set array with a function")
            for key in self._index:
                tmp = self._SetData(self, self._bounds)
                self._data[key] = tmp
                #
                if isinstance(key,tuple):
                    tmpkey = key
                else:
                    tmpkey = (key,)
                #
                # self.initialize: model, index -> list
                #
                if self.initialize.__code__.co_argcount == len(tmpkey)+1:
                    rule_list = apply_indexed_rule(self, self.initialize, self._parent(), tmpkey)
                    for val in rule_list:
                        tmp._add( val )
                #
                # self.initialize: model, counter, index -> val
                #
                else:
                    ctr=1
                    val = apply_parameterized_indexed_rule(self, self.initialize, self._parent(), ctr, tmpkey)
                    if val is None:
                        raise ValueError("Set rule returned None instead of Set.Skip")
                    while not (val.__class__ is tuple and val == Set.End):
                        tmp._add( val )
                        ctr += 1
                        val = apply_parameterized_indexed_rule(self, self.initialize, self._parent(), ctr, tmpkey)
                        if val is None:
                            raise ValueError("Set rule returned None instead of Set.Skip")
        #
        # Treat self.initialize as an iterable
        #
        elif self.initialize is not None:
            if type(self.initialize) is not dict:
                for key in self._index:
                    tmp = self._SetData(self, self._bounds)
                    for val in self.initialize:
                        tmp._add(val)
                    self._data[key] = tmp
            else:
                for key in self.initialize:
                    tmp = self._SetData(self, self._bounds)
                    for val in self.initialize[key]:
                        tmp._add(val)
                    self._data[key] = tmp
        timer.report()



