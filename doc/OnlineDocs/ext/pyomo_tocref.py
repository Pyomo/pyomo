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

"""Custom Sphinx autodoc plugin to generate TOC list-of-lists without
creating duplicate TOC entries

"""

from docutils import nodes
from sphinx.directives.other import TocTree
from sphinx.environment.adapters.toctree import TocTree as TocTree_Adapter


def setup(app):
    app.connect('doctree-resolved', expand_tocrefs)
    app.add_directive('tocref', TocRef)
    return {'version': '0.0.0'}


class tocref(nodes.Element):
    pass


class TocRef(TocTree):
    tagname = 'tocref'

    def run(self):
        wrapped_toc = super().run()[0]
        assert len(wrapped_toc.children) == 1
        toc = wrapped_toc.children[0]
        node = tocref()
        node.attributes = toc.attributes
        node.children = toc.children
        self.set_source_info(node)
        self.add_name(node)
        return [node]


def expand_tocrefs(app, doctree, docname):
    toctree_adapter = TocTree_Adapter(app.env)
    for node in doctree.findall(tocref):
        node.replace_self(
            toctree_adapter.resolve(
                docname,
                app.builder,
                node,
                maxdepth=node['maxdepth'],
                titles_only=node['titlesonly'],
            )
        )
