#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.common.collections import ComponentMap
from pyomo.common.modeling import unique_component_name


def rename_components(model, component_list, prefix):
    """
    Rename components in component_list using the prefix AND
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
    >>> c_list = list(model.component_objects(ctype=Var, descend_into=True))
    >>> rename_components(model, component_list=c_list, prefix='special_')

    Returns
    -------
    ComponentMap : maps the renamed Component objects
       to a string that provides their old fully qualified names

    ToDo
    ----
    - need to add a check to see if someone accidentally passes a generator since this can lead to an infinite loop

    """
    name_map = ComponentMap()
    for c in component_list:
        # get the parent block - we will use this to ensure new names are
        # unique and to add the new "scaled" components
        parent = c.parent_block()
        old_name = c.name
        new_name = unique_component_name(parent, prefix + c.local_name)
        parent.del_component(c)
        parent.add_component(new_name, c)
        name_map[c] = old_name

    return name_map


def iter_component(obj):
    """
    Yield "child" objects from a component that is defined with either the `base` or `kernel` APIs.
    If the component is not indexed, it returns itself.

    Parameters
    ----------
    obj : ComponentType
        eg. `TupleContainer`, `ListContainer`, `DictContainer`, `IndexedComponent`, or `Component`

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
