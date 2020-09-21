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
from pyomo.core.base.indexed_component import normalize_index
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from pyomo.core.base.global_set import UnindexedComponent_set
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.common.errors import DeveloperError

def get_index_if_present(comp):
    try:
        return comp.index()
    except AttributeError:
        return None

def get_subsets_list(setprod):
    try:
        return list(setprod.subsets())
    except AttributeError:
        # Intended to catch the case where
        # setprod is not actually a product.
        return [setprod]

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

CALL_DICT = {
        IndexedComponent_slice.get_attribute: '__getattr__',
        IndexedComponent_slice.get_item: '__getitem__',
        }

def get_component_call_stack(comp, context=None):
    """Get the call stack necessary to locate a `Component`

    The call stack is a `list` of `tuple`s where the first entry is a 
    code for `__getattr__` or  `__getitem__`, using the same convention
    as `IndexedComponent_slice`. The second entry is the argument of
    the corresponding function. Following this sequence of calls from
    `context` (or the top-level model if `context is None`) will
    produce comp.

    Parameters:
    -----------
    comp : `pyomo.core.base.component.Component`
        The component to locate
    context : `pyomo.core.base.block.Block`
        The block within which to locate the component. If `None`, the
        top-level model will be used.

    Returns:
    --------
    `list` : Contains the necessary method calls and their arguments

    """
    # If comp is context, an empty call stack is returned.
    # A component is not said to exist in "the context of" itself.
    call_stack = []
    while comp.parent_block() is not None:
        # If parent_block is None, comp is the root 
        # of the model, so we don't add anything else
        # to the call stack.
        if comp is context:
            # We are done
            break

        # Add (get_item, index) to the call stack
        index = get_index_if_present(comp)
        if index is not None:
            call_stack.append((IndexedComponent_slice.get_item, index))
        parent_component = comp.parent_component()

        if parent_component is context:
            # We are done
            break

        # Add (get_attribute, name) to the call stack
        call_stack.append((
            IndexedComponent_slice.get_attribute,
            parent_component.local_name
            ))
        comp = comp.parent_block()
    return call_stack

def slice_component_data_along_sets(comp, sets, context=None):
    """
    """
    if context is None:
        context = comp.model()
    call_stack = get_component_call_stack(comp, context)

    # Maintain a pointer to the component so we can get
    # the index set and know which locations to slice.
    comp = context
    sliced_comp = context
    while call_stack:
        call, arg = call_stack.pop()
        if call is IndexedComponent_slice.get_attribute:
            print(arg)
            comp = getattr(comp, arg)
            sliced_comp = getattr(sliced_comp, arg)
        elif call is IndexedComponent_slice.get_item:
            print(arg)
            index_set = comp.index_set()
            comp = comp[arg]
            # Process arg to replace desired indices with slices.
            location_set_map = get_location_set_map(arg, index_set)
            arg = replace_indices(arg, location_set_map, sets)
            sliced_comp = sliced_comp[arg]

    return sliced_comp

def replace_indices(index, location_set_map, sets):
    """
    """
    index = tuple_from_possible_scalar(index)
    new_index = []
    loc = 0
    len_index = len(index)
    while loc < len_index:
        val = index[loc]
        _set = location_set_map[loc]
        dimen = _set.dimen
        if _set not in sets:
            new_index.append(val)
        elif dimen is not None:
            new_index.append(slice(None, None, None))
        else:
            dimen_none_set = _set
            new_index.append(Ellipsis)
            loc += 1
            while loc < len_index:
                # Skip all adjacent locations belonging to the same
                # set. These are covered by the Ellipsis.
                _set = location_set_map[loc]
                if _set is not dimen_none_set:
                    break
                loc += 1
            continue
        loc += 1
    return tuple(new_index)

#def replace_indices(index, location_map):
#    if type(index) is not list:
#        # For mutability
#        index_list = list(index)
#    for _set, locations in location_map.items():
#        for loc in locations:
#            start = loc
#            dimen = _set.dimen
#            if dimen is None:
#                # if dimen is None, len(locations) == 1
#                index_list[loc] = Ellipsis
#            for i in range(dimen):
#                # Some additional information will be necessary
#                # here if start/stop/step slices are to be
#                # supported.
#                index_list[loc+i] = slice(None, None, None)
#    return tuple(index_list)

def get_location_set_map(index, index_set):
    """
    """
    index = tuple_from_possible_scalar(index)
    len_index = len(index)
    locations_left = set(range(len_index))
    location_set_map = {}

    # Determine if we should exit early
    if index_set is UnindexedComponent_set:
        # index == (None,) and dimen is None in this case,
        # so without this catch, the function would return
        # {0: None}, which is not what we want for an
        # unindexed component.
        return {0: UnindexedComponent_set}
    elif not normalize_index.flatten:
        raise ValueError(
            'get_location_set_map does not support the case where '
            'normalize_index.flatten is False.'
            # Although in this case, the location of an index should
            # just be its position in the subsets list, so maybe
            # the info we need is actually more simple to obtain.
            )

    if hasattr(index_set, 'subsets'):
        subsets = list(index_set.subsets())
    else:
        subsets = [index_set]

    n_subsets = len(subsets)

    location = 0
    dimen_none_set = None
    dimen_none_set_coord = None
    # Step through subsets in forward order to assign as many
    # locations as possible.
    for sub_coord, sub in enumerate(subsets):
        dimen = sub.dimen
        if dimen is None:
            dimen_none_set = sub
            dimen_none_set_coord = sub_coord
            break
        for i in range(dimen):
            location_set_map[location+i] = sub
            locations_left.remove(location+i)
        location += dimen

    # We are either done or have encountered a set of dimen None
    if not locations_left:
        return location_set_map

    location = len_index - 1
    # Step through subsets in reverse order, assigning locations,
    # until a set of dimen==None is encountered.
    for sub_coord, sub in enumerate(reversed(subsets)):
        sub_coord = n_subsets - sub_coord - 1
        dimen = sub.dimen
        if dimen is None:
            if (dimen_none_set is not sub or
                    dimen_none_set_coord != sub_coord):
                # Make sure this set is the same one we encountered
                # earlier.
                raise RuntimeError(
                'Cannot get locations when multiple sets of dimen==None '
                'are present.'
                '\nFound %s at position %s and %s at position %s.'
                '\nLocation is ambiguous in this case.'
                % (dimen_none_set, dimen_none_set_coord, sub, sub_coord)
                )
            break
        for i in range(dimen):
            location_set_map[location-i] = sub
            locations_left.remove(location-i)
        location -= sub.dimen

    for loc in locations_left:
        # All remaining locations, that cannot be accessed from some 
        # constant offset from the beginning or end of the tuple,
        # must belong to the dimen-None set.
        location_set_map[loc] = dimen_none_set

    return location_set_map

def get_sets_of_locations(index, setprod, sets):
    # Make sure index is a tuple
    # get subsets of setprod
    # for each subset, incrementing pointer into index
    # if subset in sets, add pointer from location to that set
    #                    for next dimen locations
    # When a set of dimen None is encountered, point first location
    # to this set (if in sets). break. (still break if not in sets)
    # now iterate backwards over subsets,

    pass

def get_locations_of_sets(setprod, sets):
    """
    Returns
    -------
    """
    set_location_map = ComponentMap((s, []) for s in sets)

    if hasattr(setprod, 'subsets'):
        subsets = setprod.subsets()
    else:
        subsets = [setprod]

    n_subsets = len(subsets)

    location = 0
    dimen_none_set = None
    dimen_none_set_coord = None
    for coord, _set in enumerate(subsets):
        if _set in sets:
            set_location_map[_set].append(location)
        if _set.dimen is None:
            dimen_none_set = _set
            dimen_none_set_coord = coord
            break
        location += _set.dimen

    if coord == n_subsets - 1:
        return set_location_map

    location = 0
    for i, _set in reversed(enumerate(subsets)):
        coord = n_subsets - i
        if _set.dimen is None:
            if (dimen_none_set is not _set and
                    dimen_none_set_coord != coord):
                raise RuntimeError(
                'Cannot get locations when multiple sets of dimen==None '
                'are present. Found %s at position %s and %s at position %s.'
                '\nLocation is ambiguous in this case.'
                % (dimen_none_set, dimen_none_set_coord, _set, coord)
                )
            break
        location -= _set.dimen
        if _set in sets:
            set_location_map[_set].append(location)

    return set_location_map
