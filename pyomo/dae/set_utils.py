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
from pyomo.common.collections import ComponentSet
from pyomo.core.base import Constraint, Block
from pyomo.core.base.set import SetProduct


def index_warning(name, index):
    return 'WARNING: %s has no index %s' % (name, index)


def is_explicitly_indexed_by(comp, *sets, **kwargs):
    """
    Function for determining whether a pyomo component is indexed by a 
    set or group of sets.

    Args:
        comp : Some Pyomo component, possibly indexed
        sets : Pyomo Sets to check indexing by
        expand_all_set_operators : Whether or not to expand all set operators
                                   in the subsets method

    Returns:
        A bool that is True if comp is directly indexed by every set in sets.
    """
    if not comp.is_indexed():
        return False
    for s in sets:
        if isinstance(s, SetProduct):
            msg = ('Checking for explicit indexing by a SetProduct '
                  'is not supported')
            raise TypeError(msg)

    expand_all_set_operators = kwargs.pop('expand_all_set_operators', False)
    if kwargs:
        keys = kwargs.keys()
        raise ValueError('Unrecognized keyword arguments: %s' % str(keys))

    projected_subsets = comp.index_set().subsets(expand_all_set_operators=
                                                 expand_all_set_operators)
    # Expanding all set operators here can be dangerous because it will not
    # distinguish between operators that contain their operands (e.g. union,
    # where you might consider the component to be considered indexed by 
    # the operands) and operators that don't.
    # Ideally would like to check for containment by inclusion and containment
    # by product in one search of the set operators.
    subset_set = ComponentSet(projected_subsets)

    return all([_ in subset_set for _ in sets])


def is_in_block_indexed_by(comp, s, stop_at=None):
    """
    Function for determining whether a component is contained in a 
    block that is indexed by a particular set.

    Args: 
        comp : Component whose parent blocks are checked
        s : Set for which indices are checked
        stop_at : Block at which to stop searching if reached, regardless
                  of whether or not it is indexed by s
                  
    Returns:
        Bool that is true if comp is contained in a block indexed by s
    """
    parent = comp.parent_block()

    # Stop when top-level block has been reached
    while parent is not None:
        # If we have reached our stopping point, quit.
        if parent is stop_at:
            return False

        # Look at the potentially-indexed block containing our component
        parent = parent.parent_component()
        # Check again for the stopping point in case an IndexedBlock was used
        if parent is stop_at:
            return False

        # Check potentially-indexed block for index s:
        if is_explicitly_indexed_by(parent, s):
            return True
        # Continue up the tree, checking the parent block of our
        # potentially-indexed block:
        else:
            parent = parent.parent_block()
    # Return False if top-level block was reached
    return False


def get_index_set_except(comp, *sets):
    """ 
    Function for getting indices of a component over a product of its
    indexing sets other than those specified. Indices for the specified 
    sets can be used to construct indices of the proper dimension for the 
    original component via the index_getter function.

    Args:
        comp : Component whose indexing sets are to be manipulated
        sets : Sets to omit from the set_except product

    Returns:
        A dictionary. Maps 'set_except' to a Pyomo Set or SetProduct
        of comp's index set, excluding those in sets. Maps
        'index_getter' to a function that returns an index of the
        proper dimension for comp, given an element of set_except
        and a value for each set excluded. These values must be provided
        in the same order their Sets were provided in the sets argument.
    """
    s_set = ComponentSet(sets)
    try:
        total_s_dim = sum([s.dimen for s in sets])
    except TypeError:
        msg = ('get_index_set_except does not support sets with '
              'dimen == None, including those with inconsistent dimen')
        raise TypeError(msg)

    info = {}

    if not is_explicitly_indexed_by(comp, *sets):
        msg = (comp.name + ' is not indexed by at least one of ' +
                str([s.name for s in sets]))
        raise ValueError(msg)

    index_set = comp.index_set()
    if isinstance(index_set, SetProduct):
        projection_sets = list(index_set.subsets())
        counter = Counter([id(_) for _ in projection_sets])
        for s in sets:
            if counter[id(s)] != 1:
                msg = 'Cannot omit sets that appear multiple times'
                raise ValueError(msg)
        # Need to know the location of each set within comp's index_set
        # location will map:
        #     location in comp's subsets() -> location in input sets
        location = {}
        # location should be well defined even for higher dimension sets
        # because this maps between lists of sets, not lists of indices
        other_ind_sets = []
        for ind_loc, ind_set in enumerate(projection_sets):
            found_set = False
            for s_loc, s_set in enumerate(sets):
                if ind_set is s_set:
                    location[ind_loc] = s_loc
                    found_set = True
                    break
            if not found_set:
                other_ind_sets.append(ind_set)
    else:
        # If index_set is not a SetProduct, only one set must have been
        # provided, so len(sets) == 1
        # Location in sets and in comp's indexing set are the same.
        location = {0: 0}
        other_ind_sets = []

    if comp.dim() == total_s_dim: 
        # comp indexed by all sets and having this dimension
        # is sufficient to know that comp is only indexed by 
        # Sets in *sets

        # In this case, return the trivial set_except and index_getter

        # Problem: cannot construct location without a set tuple
        #          is that a problem with this syntax?
        #          Here len(newvals) should == 1
        info['set_except'] = [None]
        # index_getter returns an index corresponding to the values passed to
        # it, re-ordered according to order of indexing sets in component.
        info['index_getter'] = (lambda incomplete_index, *newvals:
                newvals[0] if len(newvals) <= 1 else
                tuple([newvals[location[i]] for i in location]))
        return info

    # Now may assume other_ind_sets is nonempty.
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
    Function for inserting new values into a partial index.
    Used by get_index_set_except function to construct the 
    index_getter function for completing indices of a particular
    component with particular sets excluded.

    Args:
        loc : Dictionary mapping location in the new index to
              location in newvals
        index : Partial index
        newvals : New values to insert into index. Can be scalars
                  or tuples (for higher-dimension sets)

    Returns:
        An index (tuple) with values from newvals inserted in 
        locations specified by loc
    """
    if type(index) is not tuple:
        index = (index,)
    keys = sorted(loc.keys())
    if len(keys) != len(newvals):
        raise ValueError('Wrong number of values to complete index')
    for i in sorted(loc.keys()):
        newval = newvals[loc[i]]
        if type(newval) is not tuple:
            newval = (newval,)
        index = index[0:i] + newval + index[i:]
    return index


def deactivate_model_at(b, cset, pts, allow_skip=True, 
        suppress_warnings=False):
    """
    Finds any block or constraint in block b, indexed explicitly (and not 
    implicitly) by cset, and deactivates it at points specified. 
    Implicitly indexed components are excluded because one of their parent 
    blocks will be deactivated, so deactivating them too would be redundant.

    Args:
        b : Block to search
        cset : ContinuousSet of interest
        pts : Value or list of values, in ContinuousSet, to deactivate at

    Returns:
        A dictionary mapping points in pts to lists of
        component data that have been deactivated there
    """
    if type(pts) is not list:
        pts = [pts]
    for pt in pts:
        if pt not in cset:
            msg = str(pt) + ' is not in ContinuousSet ' + cset.name
            raise ValueError(msg)
    deactivated = {pt: [] for pt in pts}

    visited = set()
    for comp in b.component_objects([Block, Constraint], active=True):
        # Record components that have been visited in case component_objects
        # contains duplicates (due to references)
        if id(comp) in visited:
            continue
        visited.add(id(comp))

        if (is_explicitly_indexed_by(comp, cset) and
                not is_in_block_indexed_by(comp, cset)):
            info = get_index_set_except(comp, cset)
            non_cset_set = info['set_except']
            index_getter = info['index_getter']

            for non_cset_index in non_cset_set:
                for pt in pts:
                    index = index_getter(non_cset_index, pt)
                    try:
                        comp[index].deactivate()
                        deactivated[pt].append(comp[index])
                    except KeyError:
                        # except KeyError to allow Constraint/Block.Skip
                        if not suppress_warnings:
                            print(index_warning(comp.name, index))
                        if not allow_skip:
                            raise
                        continue

    return deactivated
