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

from pyomo.core.base.reference import Reference
from pyomo.common.collections import ComponentMap
from pyomo.common.modeling import unique_component_name


def rename_components(model, component_list, prefix):
    """Rename components in component_list using the prefix AND
    unique_component_name

    Parameters
    ----------
    model : Pyomo model (or Block)
       The variables, constraints and objective will be renamed on this model
    component_list : list
       List of components to rename
    prefix : str
       The prefix to use when building the new names

    Examples
    --------
    >>> model = pyo.ConcreteModel()
    >>> model.x = pyo.Var()
    >>> model.y = pyo.Var()
    >>> c_list = list(model.component_objects(ctype=pyo.Var, descend_into=True))
    >>> new = rename_components(model, component_list=c_list, prefix='special_')
    >>> str(new)
    "ComponentMap({'special_x (key=...)': 'x', 'special_y (key=...)': 'y'})"

    Returns
    -------
    ComponentMap : maps the renamed Component objects
       to a string that provides their old fully qualified names

    ToDo
    ----
    - need to add a check to see if someone accidentally passes a
      generator since this can lead to an infinite loop

    """
    # Need to collect any Reference first so that we can record the old mapping of data objects before renaming
    refs = {}
    for c in component_list:
        if c.is_reference():
            refs[c] = {}
            for k, v in c._data.items():
                refs[c][k] = (v.parent_block(), v.local_name)

    # Now rename all the non-Reference components
    name_map = ComponentMap()
    for c in component_list:
        # get the parent block - we will use this to ensure new names are
        # unique and to add the new "scaled" components
        if not c.is_reference():
            # Skip References for now
            parent = c.parent_block()
            old_name = c.name
            new_name = unique_component_name(parent, prefix + c.local_name)
            parent.del_component(c)
            parent.add_component(new_name, c)
            name_map[c] = old_name

    # Finally, replace all the References with remapped equivalents
    for c in refs:
        # Get mapping of new component names
        new_map = ComponentMap()
        for k, v in refs[c].items():
            # Assume the data was renamed
            new_data = v[0].find_component(prefix + v[1])

            if new_data is None:
                # If we couldn't find the new name, try the old one
                # This might happen if the reference pointed to something outside
                # the scope of the renaming
                new_data = v[0].find_component(v[1])

            if new_data is None:
                # If we still haven't found it, give up
                raise RuntimeError(
                    f"Unable to remap Reference {c.name} whilst renaming components."
                )

            new_map[k] = new_data

        parent = c.parent_block()
        old_name = c.name
        new_name = unique_component_name(parent, prefix + c.local_name)
        parent.del_component(c)
        cnew = Reference(new_map)
        parent.add_component(new_name, cnew)
        name_map[cnew] = old_name

    return name_map


def iter_component(obj):
    """Yield "child" objects from a component that is defined with either
    the `base` or `kernel` APIs.  If the component is not indexed, it
    returns itself.

    Parameters
    ----------
    obj : ComponentType
        eg. `TupleContainer`, `ListContainer`, `DictContainer`,
        `IndexedComponent`, or `Component`

    Returns
    -------
    Iterator[ComponentType] : Iterator of the component data objects.

    """
    try:
        # catches `IndexedComponent`, and kernel's `_dict`
        return iter(obj.values())
    except AttributeError:
        pass

    try:
        # catches list and kernel's `_list` and `_tuple`
        return iter(obj)
    except TypeError:
        return iter((obj,))
