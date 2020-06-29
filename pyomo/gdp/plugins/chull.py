from pyomo.common.deprecation import deprecation_warning
deprecation_warning(
    'The pyomo.gdp.plugins.chull module is deprecated.  '
    'Import the Hull reformulation objects from pyomo.gdp.plugins.hull.',
    version='5.7')

from .hull import _Deprecated_Name_Hull as ConvexHull_Transformation
