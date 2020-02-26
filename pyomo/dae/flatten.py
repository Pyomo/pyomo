#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
from pyomo.core.base import Block, Var, Reference
from pyomo.core.base.block import SubclassOf
from pyomo.core.base.sets import _SetProduct
from pyomo.core.base.indexed_component_slice import _IndexedComponent_slice

def identify_member_sets(index):
    """
    Identify all of the individual subsets in an indexing set. When the
    Set rewrite is finished this function should no longer be needed,
    the `subsets` method will provide this functionality.
    """

    if index is None:
        return []
    queue = [index]
    ans = []
    while queue:
        s = queue.pop(0)
        if not isinstance(s, _SetProduct):
            ans.append(s)
        else:
            queue.extend(s.set_tuple)
    return ans


def generate_time_only_slices(obj, time):
    o_sets = identify_member_sets(obj.index_set())
    # Given a potentially complex set, determine the index of the TIME
    # set, as well as all other "fixed" indices.  We will even support a
    # single Set with dimen==None (using ellipsis in the slice).
    ellipsis_idx = None
    time_idx = None
    regular_idx = []
    idx = 0
    for s in o_sets:
        if s is time:
            time_idx = idx
            idx += 1
        elif s.dimen is not None:
            for sub_idx in range(s.dimen):
                regular_idx.append(idx)
            idx += s.dimen
        elif ellipsis_idx is None:
            ellipsis_idx = idx
            idx += 1
        else:
            raise RuntimeError(
                "We can only handle a single Set with dimen=None")
    # To support Sets with dimen==None (using ellipsis), we need to have
    # all fixed/time indices be positive if they appear before the
    # ellipsis and negative (counting from the end of the list) if they
    # are after the ellipsis.
    if ellipsis_idx:
        if time_idx > ellipsis_idx:
            time_idx = time_idx - idx
        regular_idx = [ i - idx if i > ellipsis_idx else i
                      for i in fixed_idx ]
    # We now form a temporary slice that slices over all the regular
    # indices for a fixed value of the time index.
    tmp_sliced = {i: slice(None) for i in regular_idx}
    tmp_fixed = {time_idx: time.first()}
    tmp_ellipsis = ellipsis_idx
    _slice = _IndexedComponent_slice(
        obj, tmp_fixed, tmp_sliced, tmp_ellipsis
    )
    # For each combination of regular indices, we can generate a single
    # slice over the time index
    time_sliced = {time_idx: time.first()}
    for key in _slice.wildcard_keys():
        if type(key) is not tuple:
            key = (key,)
        time_fixed = dict(
            (i, val) if i<time_idx else (i+1, val)
            for i,val in enumerate(key)
        )
        yield _IndexedComponent_slice(obj, time_fixed, time_sliced, None)


def generate_time_indexed_block_slices(block, time):
    # TODO: We should probably do a sanity check that time does not
    # appear in any sub-block / var indices.
    queue = list( generate_time_only_slices(block, time) )
    while queue:
        _slice = queue.pop(0)
        # Pick a random block from this slice (i.e. TIME == TIME.first())
        #
        # TODO: we should probably check that the OTHER blocks
        # in the time set have the same variables.
        b = next(iter(_slice))
        # Any sub-blocks must be put on the queue to descend into and
        # process
        for sub_b in b.component_objects(Block, descend_into=False):
            _name = sub_b.local_name
            for idx in sub_b:
                queue.append(_slice.duplicate().component(_name)[idx])
        # Any Vars must be mapped to slices and returned
        for v in b.component_objects(Var, descend_into=False):
            _name = v.local_name
            for idx in v:
                yield _slice.duplicate().component(_name)[idx]
        

def flatten_dae_variables(model, time):
    """
    This function takes in a (hierarchical, block-structured) Pyomo
    model and a `ContinuousSet` and returns two lists of "flattened"
    variables. The first is a list of all `_VarData` that are not
    indexed by the `ContinuousSet` and the second is a list of
    `Reference` components such that each reference is indexed only by
    the specified `ContinuousSet`. This function is convenient for
    identifying variables that are implicitly indexed by the
    `ContinuousSet`, for example, a singleton `Var` living on a `Block`
    that is indexed by the `ContinuousSet`.

    Parameters
    ----------
    model : Concrete Pyomo model

    time : ``pyomo.dae.ContinuousSet``

    Returns
    -------
    Two lists
    """
    assert time.model() is model.model()

    block_queue = [model]
    regular_vars = []
    time_indexed_vars = []
    while block_queue:
        b = block_queue.pop(0)
        b_sets = identify_member_sets(b.index_set())
        if time in b_sets:
            for _slice in generate_time_indexed_block_slices(b, time):
                time_indexed_vars.append(Reference(_slice))
            continue
        block_queue.extend(
            list(b.component_objects(Block, descend_into=False))
        )
        for v in b.component_objects(SubclassOf(Var), descend_into=False):
            v_sets = identify_member_sets(v.index_set())
            if time in v_sets:
                for _slice in generate_time_only_slices(v, time):
                    time_indexed_vars.append(Reference(_slice))
            else:
                regular_vars.extend(list(v.values()))

    return regular_vars, time_indexed_vars
