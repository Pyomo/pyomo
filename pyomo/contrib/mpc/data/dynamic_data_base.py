#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.contrib.mpc.data.get_cuid import (
    get_time_indexed_cuid,
)


class _DynamicDataBase(object):
    """
    A class for storing data associated with time-indexed variables.

    """
    
    def __init__(self, data, time_set=None):
        """
        Processes keys of the data dict and makes sure

        """
        self._orig_time_set = time_set
        self._data = {
            get_time_indexed_cuid(key, (self._orig_time_set,)): val
            for key, val in data.items()
        }

    def __eq__(self, other):
        if isinstance(other, _DynamicDataBase):
            return self._data == other._data
        else:
            # Should this return False or raise TypeError?
            raise TypeError(
                "%s and %s are not comparable"
                % (self.__class__, other.__class__)
            )

    def get_data(self):
        """
        Return a dictionary mapping CUIDs to values

        """
        return self._data

    def get_data_from_key(self, key):
        """
        Returns the value associated with the given key.

        """
        cuid = get_time_indexed_cuid(key, (self._orig_time_set,))
        return self._data[cuid]

    def contains_key(self, key):
        """
        Returns whether this object's dict contains the given key.

        """
        cuid = get_time_indexed_cuid(key, (self._orig_time_set,))
        return cuid in self._data

    def update_data(self, other):
        """
        Updates this object's data dict.

        """
        if isinstance(other, _DynamicDataBase):
            self._data.update(other.get_data())
        else:
            other = {
                get_time_indexed_cuid(key, (self._orig_time_set,)): val
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
            "to_serializable has not been implemented by %s"
            % self.__class__
        )

    def extract_variables(self, variables, copy_values=False):
        """
        Return a new object that only keep data values for the variables
        specified.

        """
        if copy_values:
            # We don't know what this object uses as values in its dict,
            # so we don't know how to copy them.
            raise NotImplementedError(
                "extract_variables with copy_values=True has not been"
                " implemented by %s"
                % self.__class__
            )
        data = {}
        for var in variables:        
            cuid = get_time_indexed_cuid(var, (self._orig_time_set,))
            data[cuid] = self._data[cuid]
        MyClass = self.__class__
        return MyClass(data, time_set=self._orig_time_set)
