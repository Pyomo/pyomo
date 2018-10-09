#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""
Define the plugin for COLIN XML IO
"""

from pyomo.opt.blackbox.problem_io import BlackBoxOptProblemIOFactory
import xml.dom.minidom
from pyutilib.misc import tostr

try:
    unicode
    intlist = [int, long, float]
except:
    basestring = str
    intlist = [int, float]


@BlackBoxOptProblemIOFactory.register('colin')
class ColinXmlIO(object):
    """The reader/writer for the COLIN XML IO Formats"""

    def read(self, filename, point):
        """
        Read a point and request information.
        This method returns a tuple: point, requests
        """
        input_doc = xml.dom.minidom.parse(filename)
        point = point.process(input_doc.getElementsByTagName("Parameters")[0])
        requests = self._handleRequests(input_doc.getElementsByTagName("Requests")[0])
        return point, requests

    def _handleRequests(self, node):
        """
        A function that processes the requests
        """
        requests = {}
        for child in node.childNodes:
            if child.nodeType == node.ELEMENT_NODE:
                tmp = {}
                for (name,value) in child.attributes.items():
                    tmp[name]=value
                if not 'index' in tmp:
                    tmp['index'] = []
                else:
                    tmp['index'] = map(re.split('[ \t]+', tmp['index'].strip()), int)
                requests[str(child.nodeName)] = tmp
        return requests

    def write(self, filename, response):
        """
        Write response information to a file.
        """
        output_doc = self._process(response)
        OUTPUT = open(filename,"w")
        output_doc.writexml(OUTPUT," "," ","\n","UTF-8")
        OUTPUT.close()

    def _process(self, response):
        """
        Process the XML document
        """
        doc = xml.dom.minidom.Document()
        root = doc.createElement("ColinResponse")
        doc.appendChild(root)
        for key in response:
            elt = doc.createElement(str(key))
            root.appendChild(elt)
            if isinstance(response[key], basestring):
                text_elt = doc.createTextNode( response[key] )
            elif type(response[key]) in intlist:
                text_elt = doc.createTextNode( str(response[key]) )
            else:
                text_elt = doc.createTextNode( tostr(response[key]) )
            elt.appendChild(text_elt)
        return doc
