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

from pyomo.common.collections import ComponentMap
import pyomo.core
from pyomo.core.expr.numvalue import \
    NumericValue
from pyomo.core.kernel.base import \
    (ICategorizedObject,
     _no_ctype,
     _convert_ctype,
     _convert_descend_into)

import six

def preorder_traversal(node,
                       ctype=_no_ctype,
                       active=True,
                       descend=True):
    """
    A generator that yields each object in the storage tree
    (including the root object) using a preorder traversal.

    Args:
        node: The root object.
        ctype: Indicates the category of components to
            include. The default value indicates that all
            categories should be included.
        active (:const:`True`/:const:`None`): Controls
            whether or not to filter the iteration to
            include only the active part of the storage
            tree. The default is :const:`True`. Setting this
            keyword to :const:`None` causes the active
            status of objects to be ignored.
        descend (bool, function): Controls if a container
            object should be descended into during the
            traversal. When a function is supplied, each
            container object will be passed into it and the
            return value will indicate if the traversal
            continues into children of the
            container. Default is True, which is equivalent
            to `lambda x: True`.

    Returns:
        iterator of objects in the storage tree, including
        the root object
    """
    assert active in (None, True)

    # if not active, then nothing below is active
    if (active is not None) and \
       (not node.active):
        return

    # convert AML types into Kernel types (hack for the
    # solver interfaces)
    ctype = _convert_ctype.get(ctype, ctype)

    # convert descend to a function
    descend = _convert_descend_into(descend)

    if (ctype is _no_ctype) or \
       (node.ctype is ctype) or \
       (node.ctype._is_heterogeneous_container):
        yield node

    if (not node._is_container) or \
        (not descend(node)):
        return

    for child in node.children():
        child_ctype = child.ctype
        if not child._is_container:
            # not a container
            if (active is None) or \
               child.active:
                if (ctype is _no_ctype) or \
                   (child_ctype is ctype):
                    yield child
        elif child._is_heterogeneous_container:
            # a heterogeneous container, so use
            # its traversal method
            for obj in preorder_traversal(
                    child,
                    ctype=ctype,
                    active=active,
                    descend=descend):
                yield obj
        else:
            # a homogeneous container
            if child_ctype._is_heterogeneous_container:
                # this function ensures that the user provided
                # descend function is not called twice
                # on heterogeneous containers
                def descend_(obj_):
                    if obj_._is_heterogeneous_container:
                        return False
                    else:
                        return descend(obj_)
                for obj in preorder_traversal(
                        child,
                        active=active,
                        descend=descend_):
                    if not obj._is_heterogeneous_container:
                        yield obj
                    else:
                        # a heterogeneous container, so use
                        # its traversal method and reapply the
                        # ctype filter
                        for item in preorder_traversal(
                                obj,
                                ctype=ctype,
                                active=active,
                                descend=descend):
                            yield item
            elif (ctype is _no_ctype) or \
                 (child_ctype is ctype):
                for obj in preorder_traversal(
                        child,
                        active=active,
                        descend=descend):
                    yield obj

def generate_names(node,
                   convert=str,
                   prefix="",
                   **kwds):
    """
    Generate names relative to this object for all
    objects stored under it.

    This function is useful in situations where names
    are used often, but they do not need to be
    dynamically regenerated each time.

    Args:
        node: The root object below which names are
            generated.
        convert (function): A function that converts a
            storage key into a string
            representation. Default is str.
        prefix (str): A string to prefix names with.
        **kwds: Additional keywords passed to the
            preorder_traversal function.

    Returns:
        A component map that behaves as a dictionary
        mapping objects to names.
    """
    traversal = preorder_traversal(node, **kwds)
    names = ComponentMap()

    # skip the root object
    try:
        six.next(traversal)
    except StopIteration:
        # might be an empty traversal
        return names

    for obj in traversal:
        parent = obj.parent
        name = (parent._child_storage_entry_string
                % convert(obj.storage_key))
        if parent is not node:
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
                  % (str(obj), clsname, obj.active, str(obj())))
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
