#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import sys
import pprint as _pprint_

import pyomo.core
from pyomo.core.expr.numvalue import \
    NumericValue
from pyomo.core.kernel.component_map import ComponentMap
from pyomo.core.kernel.base import \
    (ICategorizedObject,
     _no_ctype)
from pyomo.core.kernel.block import block

import six

def generate_names(container,
                   ctype=_no_ctype,
                   active=None,
                   convert=str,
                   prefix=""):
    """
    Generate names relative to this container for all
    objects stored under it.

    This function is useful in situations where names
    are used often, but they do not need to be
    dynamically regenerated each time.

    Args:
        container: The container to generate names for.
        ctype: Indicate the category of objects to
            include. The default value indicates that all
            types should be included.
        active (:const:`True`/:const:`None`): Set to
            :const:`True` to indicate that only active
            components should be included. The default
            value of :const:`None` indicates that all
            components (including those that have been
            deactivated) should be included. *Note*:
            This flag is ignored for any objects that do
            not have an active flag.
        convert (function): A function that converts a
            storage key into a string
            representation. Default is str.
        prefix (str): A string to prefix names with.

    Returns:
        A component map that behaves as a dictionary
        mapping objects to names.
    """
    assert container._is_container
    assert active in (None, True)
    names = ComponentMap()

    # if not active, then nothing below is active
    if (active is not None) and \
       (not container.active):
        return names

    if not container._is_heterogeneous_container:
        orig_parent = container._parent
        orig_storage_key = container._storage_key
        container._parent = None
        container._storage_key = None
        tmp = block()
        tmp.container = container
        traversal = tmp.preorder_traversal(
            ctype=ctype,
            active=active)
        # skip the tmp root
        six.next(traversal)
        container._parent = orig_parent
        container._storage_key = orig_storage_key
    else:
        traversal = container.preorder_traversal(
            ctype=ctype,
            active=active)

    # skip this container
    six.next(traversal)

    for obj in traversal:
        parent = obj.parent
        name = (parent._child_storage_entry_string
                % convert(obj.storage_key))
        if parent is not container:
            names[obj] = (names[parent] +
                          parent._child_storage_delimiter_string +
                          name)
        else:
            names[obj] = prefix + name

    return names

def pprint(obj, indent=0, stream=sys.stdout):
    """pprint a kernel modeling object"""
    if not isinstance(obj, ICategorizedObject):
        if isinstance(obj, NumericValue):
            prefix = ""
            if indent > 0:
                prefix = (" "*indent)+" - "
            stream.write(prefix+str(obj)+"\n")
        else:
            assert indent == 0
            _pprint_.pprint(obj, indent=indent+1, stream=stream)
        return
    if not obj._is_container:
        prefix = ""
        if indent > 0:
            prefix = (" "*indent)+" - "
        # not a block
        clsname = obj.__class__.__name__
        if obj.ctype is pyomo.core.kernel.variable.IVariable:
            stream.write(prefix+"%s: %s(active=%s, value=%s, bounds=(%s,%s), domain_type=%s, fixed=%s, stale=%s)\n"
                  % (str(obj),
                     clsname,
                     obj.active,
                     obj.value,
                     obj.lb,
                     obj.ub,
                     obj.domain_type.__name__,
                     obj.fixed,
                     obj.stale))
        elif obj.ctype is pyomo.core.kernel.constraint.IConstraint:
              stream.write(prefix+"%s: %s(active=%s, expr=%s)\n"
                  % (str(obj),
                     clsname,
                     obj.active,
                     str(obj.expr)))
        elif obj.ctype is pyomo.core.kernel.objective.IObjective:
            stream.write(prefix+"%s: %s(active=%s, expr=%s)\n"
                  % (str(obj), clsname, obj.active, str(obj.expr)))
        elif obj.ctype is pyomo.core.kernel.expression.IExpression:
            stream.write(prefix+"%s: %s(active=%s, expr=%s)\n"
                  % (str(obj), clsname, obj.active, str(obj.expr)))
        elif obj.ctype is pyomo.core.kernel.parameter.IParameter:
            stream.write(prefix+"%s: %s(active=%s, value=%s)\n"
                  % (str(obj), clsname, obj.active, str(obj.value)))
        elif obj.ctype is pyomo.core.kernel.sos.ISOS:
            stream.write(prefix+"%s: %s(active=%s, level=%s, entries=%s)\n"
                  % (str(obj),
                     clsname,
                     obj.active,
                     obj.level,
                     str(["(%s,%s)" % (str(v), w)
                          for v,w in zip(obj.variables,
                                         obj.weights)])))
        else:
            assert obj.ctype is pyomo.core.kernel.suffix.ISuffix
            stream.write(prefix+"%s: %s(active=%s, size=%s)\n"
                  % (str(obj.name), clsname, obj.active, str(len(obj))))
    else:
        prefix = ""
        if indent > 0:
            prefix = (" "*indent)+" - "
        stream.write(prefix+"%s: %s(active=%s, ctype=%s)\n"
                     % (str(obj),
                        obj.__class__.__name__,
                        obj.active,
                        obj.ctype.__name__))
        for c in obj.children():
            pprint(c, indent=indent+1, stream=stream)
