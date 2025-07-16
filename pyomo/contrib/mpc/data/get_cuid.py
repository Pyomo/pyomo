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

from pyomo.core.base.componentuid import ComponentUID
from pyomo.util.slices import slice_component_along_sets
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from pyomo.dae.flatten import get_slice_for_set


def get_indexed_cuid(var, sets=None, dereference=None, context=None):
    """Attempt to convert the provided "var" object into a CUID with wildcards

    Arguments
    ---------
    var:
        Object to process. May be a VarData, IndexedVar (reference or otherwise),
        ComponentUID, slice, or string.
    sets: Tuple of sets
        Sets to use if slicing a vardata object
    dereference: None or int
        Number of times we may access referent attribute to recover a
        "base component" from a reference.
    context: Block
        Block with respect to which slices and CUIDs will be generated

    Returns
    -------
    ``ComponentUID``
        ComponentUID corresponding to the provided ``var`` and sets

    """
    # Should this function be generalized beyond a single indexing set?
    if isinstance(var, (str, IndexedComponent_slice, ComponentUID)):
        # TODO: Raise error if string and context is None
        return ComponentUID(var, context=context)
    # At this point we are assuming var is a Pyomo Var or VarData object.

    # Is allowing dereference to be an integer worth the confusion it might
    # add?
    if dereference is None:
        # Does this branch make sense? If given an unattached component,
        # we dereference, otherwise we don't dereference.
        remaining_dereferences = int(var.parent_block() is None)
    else:
        remaining_dereferences = int(dereference)
    if var.is_indexed():
        if var.is_reference() and remaining_dereferences:
            remaining_dereferences -= 1
            referent = var.referent
            if isinstance(referent, IndexedComponent_slice):
                return ComponentUID(referent, context=context)
            else:
                # If dereference is None, we propagate None, dereferencing
                # until we either reach a component attached to a block
                # or reach a non-reference component.
                dereference = (
                    dereference if dereference is None else remaining_dereferences
                )
                # NOTE: Calling this function recursively
                return get_indexed_cuid(referent, sets, dereference=dereference)
        else:
            # Assume that var is indexed only by time
            # TODO: Should we call slice_component_along_sets here as well?
            # To cover the case of b[t0].var, where var is indexed
            # by a set we care about, and we also care about time...
            # But then maybe we should slice only the sets we care about...
            # Don't want to do anything with these sets unless we're
            # presented with a vardata...
            #
            # Should we call flatten.slice_component_along_sets? Then we
            # might need to return/yield multiple components here...
            # I like making this a "simple" function. The caller can call
            # slice_component_along_set on their input data if they expect
            # to have components indexed by multiple sets.
            #
            # TODO: Assert that we're only indexed by the specified set(s)?
            # (If these sets are provided, of course...)
            index = tuple(get_slice_for_set(s) for s in var.index_set().subsets())
            return ComponentUID(var[index], context=context)
    else:
        if sets is None:
            raise ValueError(
                "A ComponentData %s was provided but no set. We need to know\n"
                "what set this component should be indexed by." % var.name
            )
        slice_ = slice_component_along_sets(var, sets)
        return ComponentUID(slice_, context=context)
