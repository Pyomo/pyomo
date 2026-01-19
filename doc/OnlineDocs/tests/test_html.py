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

import html.parser
import inspect
import os
import xml.etree.ElementTree as ET

import pyomo.common.enums as enums
import pyomo.common.unittest as unittest
from pyomo.common.fileutils import this_file_dir

_DEBUG = False
currdir = this_file_dir()
htmldir = os.path.abspath(os.path.join(currdir, '..', '_build', 'html'))


class SimplifiedElementTreeHTMLParser(html.parser.HTMLParser):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.root = ET.Element("__root__", {})
        self.context = [self.root]

    def handle_starttag(self, tag, attrs):
        if _DEBUG:
            print("  " * len(self.context), "start", tag)
        element = ET.Element(tag, dict(attrs))
        self.context[-1].append(element)
        self.context.append(element)

    def handle_endtag(self, tag):
        while self.context:
            closed_element = self.context.pop()
            txt = closed_element.text
            if _DEBUG and txt and txt.strip():
                print("  " * (len(self.context) + 1), txt.strip())
            if closed_element.tag == tag:
                break
        if not self.context:
            raise ValueError(f"Unbalanced element tags: unmatched closing tag {tag}")

        if _DEBUG:
            print("  " * (len(self.context)), "end", tag)

    def handle_data(self, data):
        el = self.context[-1]
        if el.text is None:
            el.text = data
        else:
            el.text += data

    @classmethod
    def parse(cls, html_string):
        parser = cls()
        parser.feed(html_string)
        parser.close()
        return parser.root


@unittest.skipUnless(os.path.exists(htmldir), "Doc HTML build not found")
class HtmlTester(unittest.TestCase):
    def test_enum_plugin_moduledoc(self):
        # Determine the Enums and Classes defined in the enums module
        _enums = []
        _classes = []
        for k in dir(enums):
            v = getattr(enums, k)
            if inspect.isclass(v):
                if v.__module__ != enums.__name__:
                    continue
                if issubclass(getattr(enums, k), enums.Enum):
                    _enums.append(v)
                else:
                    _classes.append(v)

        # Read the module documentation, find the main content
        with open(os.path.join(htmldir, 'api', 'pyomo.common.enums.html'), 'r') as F:
            root = SimplifiedElementTreeHTMLParser.parse(F.read())
        content = root.find('.//section[@id="enums"]')
        self.assertIsNotNone(content)
        section = None
        tables = {}
        for e in content:
            if e.tag == 'p':
                section = e.text
            if e.tag == 'table':
                tables[section] = list(e.findall('.//tr'))

        # Check that our plugin segregated the Enums into a separate table
        self.assertIn('Enums', tables)
        self.assertEqual(len(tables['Enums']), len(_enums))
        # Verify that the
        self.assertIn('Classes', tables)
        self.assertEqual(len(tables['Classes']), len(_classes))

    def test_enum_plugin_enumdoc(self):
        # Read the module documentation, find the main content
        with open(
            os.path.join(htmldir, 'api', 'pyomo.common.enums.ObjectiveSense.html'), 'r'
        ) as F:
            root = SimplifiedElementTreeHTMLParser.parse(F.read())
        content = root.find('.//section[@id="objectivesense"]')
        self.assertIsNotNone(content)
        self.assertEqual(content.find('./p').text, '(enum from )')
        classdef = content.find('.//dl[@class="py enum"]/dd')
        self.assertIsNotNone(classdef)
        section = None
        tables = {}
        for e in classdef:
            if e.tag == 'p' and e.attrib.get('class', '') == 'rubric':
                section = e.text
            if e.tag == 'table':
                tables[section] = list(e.findall('.//tr'))

        table_col1 = {
            k: [' '.join(e.text for e in tr.findall('.//span')) for tr in v]
            for k, v in tables.items()
        }

        # Check that our plugin segregated the Enum members into a separate table
        self.assertIn('Enum Members', tables)
        self.assertEqual(len(tables['Enum Members']), len(enums.ObjectiveSense))
        self.assertIn('Attributes', tables)
        # Verify that the enum members are not in the class attributes.
        for field in table_col1['Enum Members']:
            self.assertNotIn(field, table_col1['Attributes'])
        # ...and vice versa
        for field in table_col1['Attributes']:
            self.assertNotIn(field, table_col1['Enum Members'])

    def test_sphinx_issue_14223(self):
        # Sphinx 9.1 fails to correctly generate links (confusing
        # classes and class attributes.  See
        # https://github.com/sphinx-doc/sphinx/issues/14223
        with open(
            os.path.join(htmldir, 'api', 'pyomo.common.config.IsInstance.html'), 'r'
        ) as F:
            root = SimplifiedElementTreeHTMLParser.parse(F.read())

        content = root.find('.//section[@id="isinstance"]')
        fields = list(content.findall('.//dl[@class="field-list simple"]//li'))
        bases = fields[0]
        self.assertEqual(bases.find('./p/strong').text, '*bases')
        links = list(bases.findall('.//a'))
        self.assertEqual(len(links), 2)
        ref = 'https://docs.python.org/3/'
        for a in links:
            href = a.get('href')
            self.assertTrue(
                href.startswith(ref), f"{a} href='{href}' does not start with '{ref}'"
            )
