"""Testing for deprecated function."""
import pyutilib.th as unittest
from pyomo.util.deprecation import deprecated

__author__ = "Qi Chen <https://github.com/qtothec>"


class TestDeprecated(unittest.TestCase):
    """Tests for deprecated function decorator."""

    def test_deprecated_decorator(self):
        """Test for deprecated function decorator."""
        @deprecated()
        def foo(bar='yeah'):
            pass

        @deprecated('This is a custom message, too.')
        def foo_with_docstring(bar='yeah'):
            """Show that I am a good person.

            Because I document my public functions.

            """
            pass

        self.assertIn('DEPRECATION WARNING', foo.__doc__)
        self.assertIn('DEPRECATION WARNING', foo_with_docstring.__doc__)
        self.assertIn('I am a good person.', foo_with_docstring.__doc__)


if __name__ == '__main__':
    unittest.main()
