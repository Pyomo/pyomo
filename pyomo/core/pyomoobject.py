#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________


class PyomoObject(object):
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

    def is_expression_type(self):
        """Return True if this numeric value is an expression"""
        return False

    def is_named_expression_type(self):
        """Return True if this numeric value is a named expression"""
        return False

    def is_logical_type(self):
        """Return True if this class is a Pyomo Boolean value, variable, or expression."""
        return False

    def is_reference(self):
        """Return True if this object is a reference."""
        return False
