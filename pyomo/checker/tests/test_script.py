#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import os
import tempfile

import pyutilib.th as unittest

from pyomo.checker import ModelScript

currdir = os.path.dirname(os.path.abspath(__file__))

class ScriptTest(unittest.TestCase):
    """
    Test the ModelScript class. Checks both raw text and file-based
    interfaces (using the tempfile module).
    """

    testScripts = [
        "print('Hello, world!')\n",
        "import sys\nsys.stdout.write('Hello, world!\\n')\n"
        "for i in range(10):\n\tprint(i)\n"
    ]

    def testScriptText(self):
        "Check ModelScript handling of raw text scripts"

        for text in self.testScripts:
            script = ModelScript(text = text)
            self.assertEqual(text, script.read())
            self.assertEqual("<unknown>", script.filename())

    def testScriptFile(self):
        "Check ModelScript handling of file-based scripts"

        for text in self.testScripts:
            file, filename = tempfile.mkstemp()

            with os.fdopen(file, 'w') as f:
                f.write(text)

            script = ModelScript(filename = filename)

            self.assertEqual(text, script.read())
            self.assertEqual(filename, script.filename())

            os.unlink(filename)


if __name__ == "__main__":
    unittest.main()

