#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.core.base.component import ComponentData
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.common.errors import DeveloperError

def get_index_if_present(comp):
    try:
        return comp.index()
    except AttributeError:
        return None

def list_from_possible_scalar(source):
    try:
        return list(source)
    except TypeError:
        return [source]

def tuple_from_possible_scalar(source):
    try:
        return tuple(source)
    except TypeError:
        return (source,)

def slice_over_sets(comp, sets, context=None):
    """
    """
    component_stack = [(comp, get_index_if_present(comp))]

    parent_block = comp.parent_block()
    while (parent_block is not context and
            parent_block is not None):
        index = get_index_if_present(parent_block)
        parent_component = parent_block.parent_component()
        component_stack.append((parent_component, index))
        parent_block = parent_component.parent_block()

    sliced_comp, index = component_stack.pop()
    # TODO: Need to process this index.
    while len(component_stack) > 1:
        comp, index = component_stack.pop()
        index_set = comp.index_set()
        location_map = get_location_of_sets(index_set, sets)
        if not location_map:
            # Check if location map is nonempty so we don't
            # do all the work of replacing the index if there
            # is nothing to replace.
            #
            # Is there a better way to do this replacement?
            # Build the new index constructively?
            # Assemble a map: indices -> value?
            slice_index = replace_indices(index, location_map)
        else:
            slice_index = index
        sliced_comp = getattr(sliced_comp, comp.local_name)
        # No need to check if index is None. comp will not
        # be an indexed component unless it is the leaf
        # component.
        sliced_comp = sliced_comp[index]

    if component_stack:
        comp, index = component_stack.pop()
        sliced_comp = getattr(sliced_comp, comp.local_name)
        if index is not None:
            # Need to process index
            sliced_comp = sliced_comp[index]
        

    # Walk up component hierarchy.
    # When I encounter an index, get the index_set of the component.
    # Get the locations of sets within index_set
    location_map = get_location_of_sets(index_set, sets)
    slice_index = replace_indices(index, location_map)
    sliced_comp = sliced_comp[slice_index]
    # Construct tuple corresponding to index?
    # Or construct fixed, sliced, ellipsis dict?
    # Depends on how I want to construct the slice.
    # Will generate indices in reverse order.
    # Creating slice from "bottom up" seems difficult/impossible.
    # So have to create slice in forward order. Can do with
    # IndexedComponent_slice copy constructor or getitem/attr calls
    # Either way the right path seems to be to construct the tuple
    # of indices.

    # now map locations to sets
    # Really need to map locations to values/slices/ellipses
    # Don't need to know the numebr of indices because I have the index
    
    return sliced_comp

def replace_indices(index, location_map):
    if type(index) is not list:
        # For mutability
        index_list = list(index)
    for _set, locations in location_map.items():
        for loc in locations:
            start = loc
            dimen = _set.dimen
            if dimen is None:
                # if dimen is None, len(locations) == 1
                index_list[loc] = Ellipsis
            for i in range(dimen):
                # Some additional information will be necessary
                # here if start/stop/step slices are to be
                # supported.
                index_list[loc+i] = slice(None, None, None)
    return tuple(index_list)


def get_location_of_sets(setprod, sets):
    """
    Returns
    -------
    """
    if type(sets) is not ComponentSet:
        sets = ComponentSet(sets)
    location_map = ComponentMap((s, []) for s in sets)

    if hasattr(setprod, 'subsets'):
        subsets = setprod.subsets()
    else:
        subsets = [setprod]

    location = 0
    dimen_none_set = None
    dimen_none_set_coord = None
    for coord, _set in enumerate(subsets):
        if _set in sets:
            location_map[_set].append(location)
