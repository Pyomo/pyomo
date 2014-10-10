#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008 Sandia Corporation.
#  This software is distributed under the BSD License.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  For more information, see the Pyomo README.txt file.
#  _________________________________________________________________________

__all__ = ['SparseIndexedComponent', 'ActiveSparseIndexedComponent']

from six import iterkeys, itervalues, iteritems
import pyutilib.misc
from pyomo.core.base.component import Component

UnindexedComponent_set = set([None])


def normalize_index(index):
    """
    Flatten a component index.  If it has length 1, then 
    return just the element.  If it has length > 1, then
    return a tuple.
    """
    ndx = pyutilib.misc.flatten(index)
    if type(ndx) is list:
        if len(ndx) == 1:
            ndx = ndx[0]
        else:
            ndx = tuple(ndx)
    return ndx


class SparseIndexedComponent(Component):
    """
    This is the base class for all indexed modeling components.
    This class stores a dictionary, self._data, that maps indices
    to component data objects.  The object self._index defines valid
    keys for this dictionary, and the dictionary keys may be a
    strict subset.  

    The standard access and iteration methods iterate over the the
    keys of self._data.  This class supports a concept of a default
    component data value.  When enabled, the default does not
    change the access and iteration methods.

    Constructor arguments:
        ctype       The class type for the derived subclass
        doc         A text string describing this component

    Private class attributes:
        _data               A dictionary from the index set to 
                                component data objects
        _index              The set of valid indices
        _implicit_subsets   A temporary data element that stores
                                sets that are transfered to the model
    """

    #
    # If an index is supplied for which there is not a _data entry 
    # (specifically, in a get call), then this flag determines whether 
    # a check is performed to see if the input index is in the 
    # index set _index. This is extremely expensive, and so this flag
    # is provided to disable that feature globally.
    #
    _DEFAULT_INDEX_CHECKING_ENABLED = True

    def __init__(self, *args, **kwds):
        from pyomo.core.base.sets import process_setarg
        #
        Component.__init__(self, **kwds)
        #
        self._data = {}
        #
        if len(args) == 0:
            #
            # If no indexing sets are provided, generate a dummy index
            #
            self._implicit_subsets = None
            self._index = UnindexedComponent_set
        elif len(args) == 1:
            #
            # If a single indexing set is provided, just process it.
            #
            self._implicit_subsets = None
            self._index = process_setarg(args[0])
        else:
            #
            # If multiple indexing sets are provided, process them all,
            # and store the cross-product of these sets.  The individual
            # sets need to stored in the Pyomo model, so the
            # _implicit_subsets class data is used for this temporary
            # storage.
            # 
            # Example:  Pyomo allows things like 
            # "Param([1,2,3], range(100), initialize=0)".  This
            # needs to create *3* sets: two SetOf components and then
            # the SetProduct.  That means that the component needs to
            # hold on to the implicit SetOf objects until the component
            # is assigned to a model (where the implicit subsets can be
            # "transferred" to the model).
            #
            tmp = [process_setarg(x) for x in args]
            self._implicit_subsets = tmp
            self._index = tmp[0].cross(*tmp[1:])

    def to_dense_data(self):
        for ndx in self._index:
            if ndx not in self._data:
                self._default(ndx)

    def clear(self):
        """Clear the data in this component"""
        if UnindexedComponent_set != self._index:
            self._data = {}
        else:
            raise NotImplementedError(
                "Derived scalar component %s failed to define clear().\n"
                "\tPlease report this to the Pyomo developers"
                % (self.__class__.__name__,))

    def index_set(self):
        """Return the index set"""
        return self._index

    def is_indexed(self):
        """Return true if this component is indexed"""
        return UnindexedComponent_set != self._index

    def dim(self):
        """Return the dimension of the index"""
        if UnindexedComponent_set != self._index:
            return self._index.dimen
        else:
            return 0

    def __len__(self):
        """
        Return the number of component data objects stored by this
        component.  
        """
        return len(self._data)

    def __contains__(self, ndx):
        """Return true if the index is in the dictionary"""
        return ndx in self._data

    def __iter__(self):
        """Iterate over the keys in the dictionary"""
        if len(self._data) == len(self._index):
            #
            # If the data is dense then return the index iterator.
            #
            return self._index.__iter__()
        elif not hasattr(self._index, 'ordered') or not self._index.ordered:
            #
            # If the index set is not ordered, then return the 
            # data iterator.  This is in an arbitrary order, which is 
            # fine because the data is unordered.
            #
            return self._data.__iter__()
        else:
            #
            # Test each element of a sparse data with an ordered index set
            # in order.  This is potentially *slow*: if the component is in fact
            # very sparse, we could be iterating over a huge (dense)
            # index in order to sort a small number of indices.
            # However, this provides a consistent ordering that the user
            # expects. 
            #
            def _sparse_iter_gen(self):
                for idx in self._index.__iter__():
                    if idx in self._data:
                        yield idx
            return _sparse_iter_gen(self)

    def keys(self):
        """Return a list of keys in the dictionary"""
        return [ x for x in self ]

    def values(self):
        """Return a list of the component data objects in the dictionary"""
        return [ self[x] for x in self ]

    def items(self):
        """Return a list (index,data) tuples from the dictionary"""
        return [ (x, self[x]) for x in self ]

    def iterkeys(self):
        """Return an iterator of the keys in the dictionary"""
        return self.__iter__()

    def itervalues(self):
        """Return an iterator of the component data objects in the dictionary"""
        for key in self:
            yield self[key]
    
    def iteritems(self):
        """Return an iterator of (index,data) tuples from the dictionary"""
        for key in self:
            yield key, self[key]

    def __getitem__(self, ndx):
        """
        This method returns the data corresponding to the given index.
        """

        if ndx in self._data:
            # Return the data from the dictionary
            if ndx is None:
                return self
            else:
                return self._data[ndx]
        elif not self._constructed:
            # Generate an error because the component is not constructed
            if ndx is None:
                idx_str = ''
            elif ndx.__class__ is tuple:
                idx_str = "[" + ",".join(str(i) for i in ndx) + "]"
            else:
                idx_str = "[" + str(ndx) + "]"
            raise ValueError(
                "Error retrieving component %s%s: The component has "
                "not been constructed." % ( self.cname(True), idx_str,) )
        elif not SparseIndexedComponent._DEFAULT_INDEX_CHECKING_ENABLED:
            # Return the default value if the global flag dictates that
            return self._default(ndx)            
        elif ndx in self._index:
            # After checking that the index is value, return the default value
            # This check is expensive!
            return self._default(ndx)
        else:
            # Now we normalize the index and check again.  Usually,
            # indices will be normalized, so this operation is deferred.
            ndx = normalize_index(ndx)
            if ndx in self._data:
                # Note that ndx != None at this point
                return self._data[ndx]
            elif not SparseIndexedComponent._DEFAULT_INDEX_CHECKING_ENABLED:
                return self._default(ndx)            
            elif ndx in self._index:
                return self._default(ndx)
        #
        # Generate different errors, depending on the state of the index.
        #
        if not self.is_indexed():
            msg = "Error accessing indexed component: " \
                  "Cannot treat the scalar component '%s' as an array" \
                  % ( self.cname(True), )
        else:
            msg = "Error accessing indexed component: " \
                  "Index '%s' is not valid for array component '%s'" \
                  % ( ndx, self.cname(True), )
        raise KeyError(msg)

    def _default(self, index):
        """Returns the default component data value"""
        raise NotImplementedError(
            "Derived component %s failed to define _default().\n"
            "\tPlease report this to the Pyomo developers"
            % (self.__class__.__name__,))

    def set_value(self, value):
        """Set the value of a scalar component."""
        if UnindexedComponent_set != self._index:
            raise ValueError(
                "Cannot set the value for the indexed component '%s' "
                "without specifying an index value.\n"
                "\tFor example, model.%s[i] = value"
                % (self.name, self.name))
        else:
            raise NotImplementedError(
                "Derived component %s failed to define set_value() "
                "for scalar instances.\n"
                "\tPlease report this to the Pyomo developers"
                % (self.__class__.__name__,))

    def id_index_map(self):
        """
        Return an dictionary id->index for
        all ComponentData instances.
        """
        result = {}
        for index, component_data in iteritems(self):
            result[id(component_data)] = index
        return result


class ActiveSparseIndexedComponent(SparseIndexedComponent):
    """
    This is the base class for all indexed modeling components
    whose data members are subclasses of ActiveComponentData, e.g.,
    can be activated or deactivated.

    The activate and deactivate methods activate both the 
    component as well as all component data values.
    """

    def __init__(self, *args, **kwds):
        SparseIndexedComponent.__init__(self, *args, **kwds)

    def activate(self):
        """Set the active attribute to True"""
        Component.activate(self)
        for component_data in itervalues(self):
            component_data._active = True

    def deactivate(self):
        """Set the active attribute to False"""
        Component.deactivate(self)
        for component_data in itervalues(self):
            component_data._active = False

