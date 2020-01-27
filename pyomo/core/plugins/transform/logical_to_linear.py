"""Transformation from BooleanVar and LogicalStatement to Binary and Constraints."""
from pyomo.core import TransformationFactory
from pyomo.core.plugins.transform.hierarchy import IsomorphicTransformation


@TransformationFactory.register("core.logical_to_linear", doc="Convert logic to linear constraints")
class LogicalToLinear(IsomorphicTransformation):
    """
    Re-encode logical statements as linear constraints,
    converting Boolean variables to binary.
    """

    def _apply_to(self, model, **kwds):
        # TODO create corresponding binary variables
        # TODO create corresponding linear constraints
        pass
