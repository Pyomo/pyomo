#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import os.path
try:
    import lxml.etree.ElementTree as ET
except:
    import xml.etree.ElementTree as ET

from pyomo.dataportal.factory import DataManagerFactory
from pyomo.dataportal import TableData


@DataManagerFactory.register("xml", "XML file interface")
class XMLTable(TableData):

    def __init__(self):
        TableData.__init__(self)

    def open(self):
        if self.filename is None:
            raise IOError("No filename specified")

    def close(self):
        pass

    def read(self):
        from pyomo.core.base.param import Param

        if not os.path.exists(self.filename):
            raise IOError("Cannot find file '%s'" % self.filename)
        #
        tree = ET.parse(self.filename)
        if not self.options.query is None:
            if self.options.query[0] == '"' or self.options.query[0] == "'":
                self.options.query = self.options.query[1:-1]
            parents = [parent for parent in tree.findall(self.options.query)]
        else:
            parents = [parent for parent in tree.getroot()]
        tmp=[]
        labels = []
        for parent in parents:
            if len(tmp) == 0:
                for child in parent:
                    labels.append(child.tag)
                tmp.append(labels)
            row = {}
            for child in parent:
                if child.text is None:
                    row[child.tag] = child.get('value')
                else:
                    row[child.tag] = child.text
            tmp.append( [row.get(label,'.') for label in labels] )
        #
        if len(tmp) == 0:
            raise IOError("Empty *.xml file")
        elif len(tmp) == 1:
            if not self.options.param is None:
                if type(self.options.param) in (list, tuple):
                    p = self.options.param[0]
                else:
                    p = self.options.param
                if isinstance(p, Param):
                    self.options.model = p._model()
                    p = p.local_name
                self._info = ["param",p,":=",tmp[0][0]]
            elif len(self.options.symbol_map) == 1:
                self._info = ["param",self.options.symbol_map[self.options.symbol_map.keys()[0]],":=",tmp[0][0]]
            else:
                raise IOError("Data looks like a parameter, but multiple parameter names have been specified: %s" % str(self.options.symbol_map))
        else:
            self._set_data(tmp[0], tmp[1:])

    def write(self, data):
        if self.options.set is None and self.options.param is None:
            raise IOError("Unspecified model component")
        root = ET.Element('table')
        table = self._get_table()
        labels = table[0]
        for i in range(len(labels)):
            labels[i] = str(labels[i])
        for trow in table[1:]:
            row = ET.SubElement(root, 'row')
            for i in range(len(labels)):
                data = ET.SubElement(row, labels[i])
                data.set('value', str(trow[i]))
        #
        tree = ET.ElementTree(root)
        tree.write(self.filename)

