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

from pyomo.core.base.set import Set
from pyomo.contrib.mpc.data.get_cuid import get_indexed_cuid


def _is_iterable(obj):
    if not hasattr(obj, "__iter__") and not hasattr(obj, "__getitem__"):
        # Here we know obj is not iterable.
        return False
    elif hasattr(obj, "is_indexed"):
        # Pyomo scalar components define __iter__ and __getitem__,
        # however for our purpose we would like to consider them
        # as not iterable.
        #
        # Note that sets implement is_indexed but are always iterable.
        return obj.is_indexed() or isinstance(obj, Set)
    else:
        try:
            iter(obj)
            return True
        except TypeError as err:
            if "not iterable" in str(err):
                # Hopefully this error message is not implementation
                # or version specific. Tested in CPython 3.7.8 and
                # PyPy 3.7.10.
                return False
            else:
                raise err


class _DynamicDataBase(object):
    """
    A base class for storing data associated with time-indexed variables.

    """

    def __init__(self, data, time_set=None, context=None):
        """
        Processes keys of the data dict.

        """
        # This is used if we ever need to process a VarData to get
        # a time-indexed CUID. We need to know what set to slice.
        self._orig_time_set = time_set

        self._data = {
            get_indexed_cuid(key, (self._orig_time_set,), context=context): val
            for key, val in data.items()
        }

    def __eq__(self, other):
        if isinstance(other, _DynamicDataBase):
            return self._data == other._data
        else:
            # Should this return False or raise TypeError?
            raise TypeError(
                "%s and %s are not comparable" % (self.__class__, other.__class__)
            )

    def get_data(self):
        """
        Return a dictionary mapping CUIDs to values

        """
        return self._data

    def get_cuid(self, key, context=None):
        """
        Get the time-indexed CUID corresponding to the provided key
        """
        return get_indexed_cuid(key, (self._orig_time_set,), context=context)

    def get_data_from_key(self, key, context=None):
        """
        Returns the value associated with the given key.

        """
        cuid = get_indexed_cuid(key, (self._orig_time_set,), context=context)
        return self._data[cuid]

    def contains_key(self, key, context=None):
        """
        Returns whether this object's dict contains the given key.

        """
        cuid = get_indexed_cuid(key, (self._orig_time_set,), context=context)
        return cuid in self._data

    def update_data(self, other, context=None):
        """
        Updates this object's data dict.

        """
        if isinstance(other, _DynamicDataBase):
            self._data.update(other.get_data())
        else:
            other = {
                get_indexed_cuid(key, (self._orig_time_set,), context=context): val
                for key, val in other.items()
            }
            self._data.update(other)

    def to_serializable(self):
        """
        Returns a json-serializable object.

        """
        # We have no idea whether the values in this object's dict are
        # json-serializable.
        raise NotImplementedError(
            "to_serializable has not been implemented by %s" % self.__class__
        )

    def extract_variables(self, variables, context=None, copy_values=False):
        """
        Return a new object that only keeps data values for the variables
        specified.

        """
        if copy_values:
            # We don't know what this object uses as values in its dict,
            # so we don't know how to copy them.
            raise NotImplementedError(
                "extract_variables with copy_values=True has not been"
                " implemented by %s" % self.__class__
            )
        data = {}
        for var in variables:
            cuid = get_indexed_cuid(var, (self._orig_time_set,), context=context)
            data[cuid] = self._data[cuid]
        MyClass = self.__class__
        # Subclasses likely have different construction signatures,
        # so this maybe shouldn't be implemented on the base class.
        return MyClass(data, time_set=self._orig_time_set)
