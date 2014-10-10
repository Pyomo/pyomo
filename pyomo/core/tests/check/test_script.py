import os
import tempfile

import pyutilib.th as unittest
from nose.tools import nottest
from pyomo.core.check import *

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

