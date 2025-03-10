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

from pyomo.common.autoslots import AutoSlots


# Note: in an ideal world, PyomoObject would use the AutoSlots
# metaclass.  However, declaring a custom (non-type) metaclass has
# measurable performance implications.  It is faster to just look for
# the __auto_slots__ attribute and generate it if it is not present than
# to slow down the entire class hierarchy by declaring a metaclass.
class PyomoObject(AutoSlots.Mixin):
    __slots__ = ()

    def is_component_type(self):
        """Return True if this class is a Pyomo component"""
        return False

    def is_numeric_type(self):
        """Return True if this class is a Pyomo numeric object"""
        return False

    def is_parameter_type(self):
        """Return False unless this class is a parameter object"""
        return False

    def is_variable_type(self):
        """Return False unless this class is a variable object"""
        return False

    def is_expression_type(self, expression_system=None):
        """Return True if this numeric value is an expression"""
        return False

    def is_named_expression_type(self):
        """Return True if this numeric value is a named expression"""
        return False

    def is_logical_type(self):
        """Return True if this class is a Pyomo Boolean object.

        Boolean objects include constants, variables, or logical expressions.
        """
        return False

    def is_reference(self):
        """Return True if this object is a reference."""
        return False
