#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from collections import Counter


def is_explicitly_indexed_by(comp, s):
    """
    Returns True if component comp is directly indexed by set s.
    """
    if not comp.is_indexed():
        return False
    n = comp.index_set().dimen
    if n == 1:
        if comp.index_set() is s:
            return True
        else:
            return False
    elif n >= 2:
        if s in set(comp.index_set().set_tuple):
            # set_tuple must be converted to a python:set so a different
            # pyomo:Set with the same elements will not be conflated.
            # This works because pyomo:Set is hashable.
            return True
        else:
            return False


def is_implicitly_indexed_by(comp, s, stop_at=None):
    """
    Returns True if any of comp's parent blocks are indexed by s.
    Works by recursively checking parent blocks.

    If block stop_at (or its parent_component) is provided, function 
    will return False if stop_at is reached, regardless of whether 
    stop_at is indexed by s. Meant to be an "upper bound" for blocks 
    to check, like a flowsheet.
    """
    parent = comp.parent_block()

    # Stop when top-level block has been reached
    while parent is not None:
        # If we have reached our stopping point, quit.
        if parent is stop_at:
            return False

        # Look at the potentially-indexed block containing our component.
        parent = parent.parent_component()
        # Check again for stopping point in case IndexedBlock object was used.
        if parent is stop_at:
            return False

        # Check potentially-indexed block for index s.
        if is_explicitly_indexed_by(parent, s):
            return True
        # Continue up the tree, checking the parent block of our potentially-
        # indexed block, which I would like to assume contains the BlockData
        # we started from.
        else:
            parent = parent.parent_block()
    # Return False if top-level block was reached.
    return False


def get_index_set_except(comp, *sets):
    """
    Returns a dictionary:
      'set_except'   -> Pyomo Set or SetProduct indexing comp, with sets s 
                        omitted.
      'index_getter' -> Function to return an index for comp given an index
                        from set_except and a value from each set s.
                        Won't check if values are in s, so can be used to get
                        an index for a component that has different s sets.
    User should already have checked that comp is (directly) indexed
    by each set s.
    """
    n_set = len(sets)
    s_set = set(sets)
    info = {}

    if not comp.is_indexed():
        # This is not supported - should I return nothing or
        # raise exception. Probably latter.
        msg = 'Component must be indexed.'
        raise TypeError(msg)

    for s in sets:
        if not is_explicitly_indexed_by(comp, s):
            msg = comp.name + ' is not indexed by ' + s.name
            raise Exception(msg)

    if comp.dim() == 1:
        # In this case, assume that comp is indexed by *sets
        # Return the trivial set_except and index_getter
        info['set_except'] = [None]
        # index_getter here will only accept a single argument
        info['index_getter'] = (lambda incomplete_index, newval: newval)
        return info

    set_tuple = comp.index_set().set_tuple
    counter = Counter(set_tuple)

    for s in sets:
        if counter[s] != 1:
            msg = 'Cannot omit sets that appear multiple times'
            raise Exception(msg)

    # Need to know the location of each set within comp's index set
    # location will map:
    #     location_in_comp_index_set -> location_in_sets
    location = {}
    other_ind_sets = []
    for ind_loc, ind_set in enumerate(set_tuple):
        found_set = False
        for s_loc, s_set in enumerate(sets):
            if ind_set is s_set:
                location[ind_loc] = s_loc
                found_set = True
                break
        if not found_set:
            other_ind_sets.append(ind_set)

    # Trivial case where s contains every index set of comp:
    if comp.dim() == n_set:
        # I choose to return a list, as in other cases, so this object
        # can still be iterated over.
        info['set_except'] = [None]
        # The index_getter function simply returns an index corresponding
        # to the values passed into it, re-ordered according to the order
        # or indexing sets in the component - incomplete_index is 
        # inconsequential.
        info['index_getter'] = (lambda incomplete_index, *newvals:
                newvals[0] if len(newvals) <= 1 else 
                    tuple([newvals[location[i]] for i in location]))
        return info

    # Now may assume other_ind_sets is nonempty and has length 
    # comp.dim()-n_set

    # Create "indexing set" for sets not specified by this function's arguments
    if len(other_ind_sets) == 1:
        set_except = other_ind_sets[0]
    elif len(other_ind_sets) >= 2:
        set_except = other_ind_sets[0].cross(*other_ind_sets[1:])
    else:
        raise ValueError('Did not expect this to happen')

    index_getter = (lambda incomplete_index, *newvals:
            _complete_index(location, incomplete_index, *newvals))

    info['set_except'] = set_except
    info['index_getter'] = index_getter
    return info


def _complete_index(loc, index, *newvals):
    """
    index is a partial index, newvals are the values for the remaining 
    indexing sets
    loc maps location in the new index to location in newvals
    """
    if not isinstance(index, tuple):
        index = (index,)
    keys = sorted(loc.keys())
    if len(keys) != len(newvals):
        raise Exception('Wrong number of values to complete index')
    for i in sorted(loc.keys()):
        # Correctness relies on fact that indices i are visited in order 
        # from least to greatest.
        index = index[0:i] + (newvals[loc[i]],) + index[i:]
    return index
