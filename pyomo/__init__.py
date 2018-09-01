#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.common.ext import _ImportRedirect

#
# A dictionary that defines pyomo extension packages with nonstandard names.
# When a user imports
#
#    import pyomo.ext.foo
#
# the default behavior is to import the package 'pyomoext_foo'.  However, this 
# dictionary can specify an arbitrary package mapping.  Thus, 
#
#    import pyomo.ext.foo
#
# can map to the 'bar' package by having foo->bar in the dictionary.
#
_nonstandard_extension_packages = {
    'bilevel': 'pyomo.bilevel',
    'dae': 'pyomo.dae',
    'gdp': 'pyomo.gdp',
    'mpec': 'pyomo.mpec',
    'pysp': 'pyomo.pysp',
    'simplemodel': 'pyomocontrib_simplemodel'}


ext = _ImportRedirect('pyomo.ext' if __name__ == '__main__' else __name__ + ".ext", 
            'pyomoext_%s', 
            __file__, 
            _nonstandard_extension_packages).module

