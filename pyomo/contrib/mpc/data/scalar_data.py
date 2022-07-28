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

from collections import namedtuple

from pyomo.core.base.componentuid import ComponentUID
from pyomo.util.slices import slice_component_along_sets
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from pyomo.dae.flatten import get_slice_for_set

from pyomo.contrib.mpc.data.find_nearest_index import (
    find_nearest_index,
)
from pyomo.contrib.mpc.data.get_cuid import (
    get_time_indexed_cuid,
)


class ScalarData(object):
    """
    An object to store scalar data associated with time-indexed
    variables.
    """

    def __init__(self, data, time_set=None):
        """
        Arguments:
        ----------
        data: dict or ComponentMap
            Maps variables, names, or CUIDs to lists of values
        """
        # TODO: Do I have a good way to make sure that the values of
        # data are scalars?
        self._orig_time_set = time_set
        self._data = {
            get_time_indexed_cuid(key, (self._orig_time_set,)): val
            for key, val in data.items()
        }

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
        cuid = get_time_indexed_cuid(key, (self._orig_time_set,))
        return cuid in self._data

    def update_data(self, other):
        if isinstance(other, ScalarData):
            self._data.update(other.get_data())
        else:
            self._data.update(other)

    def to_serializable(self):
        """
        Convert to json-serializable object.
        """
        data = {str(cuid): values for cuid, values in self._data.items()}
        return data

    #
    # Unused
    #
    #def get_projection_onto_variables(self, variables):
    #    new = self.copy()
    #    new.project_onto_variables(variables)
    #    return new

    def extract_variables(self, variables):
        """
        Only keep variables specified by the user.
        """
        data = {}
        for var in variables:        
            cuid = get_time_indexed_cuid(var, (self._orig_time_set,))
            data[cuid] = self._data[cuid]
        return ScalarData(
            data,
            time_set=self._orig_time_set,
        )

    #
    # Unused
    #
    #def copy(self):
    #    data = {key: list(values) for key, values in self._data.items()}
    #    time = list(self._time)
    #    return TimeSeriesData(data, time)
