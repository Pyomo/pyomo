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

"""Custom Sphinx autodoc plugin for improved rendering of Enum types
that plays nicely with sphinx.ext.autosummary.

"""
import enum
import inspect
import re
import sphinx.locale

from sphinx.application import Sphinx
from sphinx.domains import ObjType
from sphinx.domains.python import PyClasslike, PyAttribute, PyXRefRole
from sphinx.ext import autodoc, autosummary
from sphinx.ext.autosummary import mangle_signature as _msig
from sphinx.ext.autosummary.generate import generate_autosummary_content as _gac
from sphinx.util.inspect import object_description
from sphinx_toolbox.more_autodoc.typehints import format_annotation
from typing import Type, Any, Dict, List, Tuple, Union

_pre_re = re.compile(r'^( = )(.*)')


def _mangle_signature(sig: str, max_chars: int = 30) -> str:
    """Override sphinx.ext.autosummary.mangle_signature() so we can exploit
    it to emit the enum member value using the sig field.  We overwrite
    mangle_signature to not return '({sig})' when the sig starts with
    ' = '

    """
    m = _pre_re.match(sig)
    if m:
        # Keep any initial whitespace; remove the parent that
        # mangle_signature adds
        return m.group(1) + _msig(m.group(2), max_chars)[1:-1]
    return _msig(sig, max_chars)


def _generate_autosummary_content(
    name: str,
    obj: Any,
    parent: Any,
    template: autosummary.generate.AutosummaryRenderer,
    *gac_args,
    **gac_kwargs,
) -> str:
    """Override sphinx.ext.autosummary.generate.generate_autosummary_content()
    to provide additional fields to the namespace dictionary.

    This allows us to create templates that itemize Enums separately
    from attributes.  Because we want to insert ourselves into the
    *middle* of the original function (between the point where the
    template namespace (``ns``) is set up and when it is rendered, we
    will actually overload the template.render() method and pass the
    modified template into the original generate_autosummary_content
    function.

    """
    if template.__class__.__name__ != '_pyomo_template_wrapper':

        class _pyomo_template_wrapper(template.__class__):
            """Wrap the provided template object so we can add fields to ns before
            calling the original render method.

            """

            def render(self, name, ns):
                # Overload render() so that we can intercept calls to it
                # and add additional fields to the NS.  Note that we
                # need variables from the generate_autosummary_content
                # context ... but we know that context is the calling
                # frame.  Seems like cheating, but it works.
                if ns['objtype'] not in ('module', 'enum'):
                    return super().render(name, ns)

                caller = inspect.currentframe().f_back
                l = caller.f_locals
                doc = l['doc']
                obj = l['obj']
                args = {'obj': obj}
                if '_get_members' in caller.f_globals:
                    _get_members = caller.f_globals['_get_members']
                    if 'config' in l:
                        # Sphinx >= 8.2.1
                        for field in ('config', 'doc', 'events', 'registry'):
                            args[field] = l[field]
                    else:
                        # Sphinx >= 7.2
                        args.update({'doc': doc, 'app': l['app']})
                else:
                    # Sphinx < 7.2
                    _get_members = caller.f_locals['get_members']

                if ns['objtype'] == 'module':
                    ns['enums'], ns['all_enums'] = _get_members(
                        types={'enum'}, imported=l['imported_members'], **args
                    )
                elif ns['objtype'] == 'enum':
                    ns['members'] = dir(obj)
                    ns['inherited_members'] = set(dir(obj)) - set(obj.__dict__.keys())
                    try:
                        # We need _get_members to eventually call
                        # _get_class_members, so we will (temporarily)
                        # set the doc.objtype back to "class"
                        doc.objtype = 'class'
                        ns['methods'], ns['all_methods'] = _get_members(
                            types={'method'}, include_public={'__init__'}, **args
                        )
                        ns['attributes'], ns['all_attributes'] = _get_members(
                            types={'attribute', 'property'}, **args
                        )
                        ns['enum_members'], ns['all_enum_members'] = _get_members(
                            types={'enum_member'}, **args
                        )
                    finally:
                        doc.objtype = 'enum'

                    mro = obj.__mro__
                    for _base in mro[: mro.index(enum.Enum)]:
                        if not isinstance(_base, enum.EnumMeta):
                            ns['member_type'] = (
                                f"Member type: {format_annotation(_base)}"
                            )
                            break
                return super().render(name, ns)

        template.__class__ = _pyomo_template_wrapper

    return _gac(name, obj, parent, template, *gac_args, **gac_kwargs)


class EnumDocumenter(autodoc.ClassDocumenter):
    objtype = "enum"

    # More than Class; less than Exception
    priority = autodoc.ClassDocumenter.priority + 1

    member_order = 15

    @classmethod
    def can_document_member(cls, member, membername, isattr, parent):
        return isinstance(member, enum.EnumMeta)

    def format_signature(self, **kwargs: Any) -> str:
        # hard-code the enum signature.  This is mostly to preserve the
        # behavior from enum-tools.autoenum.  We might want to revisit
        # this decision later.
        return "(value)"

    def sort_members(
        self, documenters: List[Tuple[autodoc.Documenter, bool]], order: str
    ) -> List[Tuple[autodoc.Documenter, bool]]:
        if order != 'groupwise':
            return super().sort_members(documenters, order)
        # If we are grouping the members, then we want the groups
        # alphabetical, *except* for the Enum members, which we want in
        # declaration order:
        mo = EnumDocumenter.member_order
        tmp = [
            (e, (e[0].member_order, (i if e[0].member_order == mo else e[0].name)))
            for i, e in enumerate(documenters)
        ]
        tmp.sort(key=lambda x: x[1])
        return [x[0] for x in tmp]


class EnumMemberDocumenter(autodoc.AttributeDocumenter):
    """Custom documenter for Enum members"""

    # Note that we want to flag these attributes as "special" (i.e., not
    # generic attributes, but we still want to emit regular py:attribute
    # directives (so that we still refer to them with :py:attr:
    # references).
    objtype = "enum_member"
    directivetype = autodoc.AttributeDocumenter.objtype

    # More than AttributeDocumenter
    priority = autodoc.AttributeDocumenter.priority + 3

    member_order = 15

    @classmethod
    def can_document_member(cls, member, membername, isattr, parent):
        return isinstance(member, enum.Enum)

    def format_signature(self, **kwargs: Any) -> str:
        """Custom format_signature() to return the value as the signature

        This has the effect of putting the value in the autosummary documentation.
        """
        return " = " + object_description(self.object)

    def add_directive_header(self, sig):
        """Custom add_directive_header to remove the enum value

        This undoes the effect of format_signature so that the entry
        renders correctly (and ``:py:enum:`` links will be generated and
        resolved correctly).

        """
        super().add_directive_header(sig.split(" = ", 1)[0].strip())


def setup(app: Sphinx) -> Dict[str, Any]:
    app.setup_extension('sphinx.ext.autodoc')
    app.setup_extension('sphinx.ext.autosummary')
    # Overwrite key parts of autosummary so that our version of autoenum
    # plays nicely with it.  We have tested this with Sphinx>7.2.
    # Notably, 7.1.2 does NOT work (and cannot be easily made to work)
    if 'generate_autosummary_content' not in dir(autosummary.generate):
        raise RuntimeError(
            "pyomo_autosummary_autoenum: Could not locate "
            "autosummary.generate.generate_autosummary_content() "
            "(possible incompatible Sphinx version)."
        )
    autosummary.generate.generate_autosummary_content = _generate_autosummary_content
    if 'mangle_signature' not in dir(autosummary):
        raise RuntimeError(
            "pyomo_autosummary_autoenum: Could not locate "
            "autosummary.mangle_signature() "
            "(possible incompatible Sphinx version)."
        )
    autosummary.mangle_signature = _mangle_signature

    app.add_autodocumenter(EnumMemberDocumenter)
    app.add_autodocumenter(EnumDocumenter)

    app.add_directive_to_domain("py", "enum", PyClasslike)
    app.add_role_to_domain("py", "enum", PyXRefRole())
    app.registry.domains["py"].object_types["enum"] = ObjType(
        sphinx.locale._("enum"), "enum", "class", "obj"
    )

    return {"version": '0.0.0', "parallel_read_safe": True, "parallel_write_safe": True}
