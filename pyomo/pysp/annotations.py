#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

from pyomo.core.base.suffix import ComponentMap

def locate_annotations(model, annotation_type, max_allowed=None):
    annotations = [(name, obj) for name, obj in vars(model).items()
                   if isinstance(obj, annotation_type)]
    if (max_allowed is not None) and (len(annotations) > max_allowed):
        raise ValueError("Too many annotations of type %s found on "
                         "model %s. The maximum allowed is %s."
                         % (annotation_type.__name__,
                            model.cname(True),
                            max_allowed))
    return annotations

class PySP_Annotation(object):
    def __init__(self):
        self.data = ComponentMap()
        try:
            self.declare(None)
            assert None in self.data
            self.default = self.data[None]
            del self.data[None]
        except TypeError:
            self.default = None
    def declare(self, component, *args, **kwds):
        raise NotImplementedError("This method is abstract")
    def pprint(self, *args, **kwds): self.data.pprint(*args, **kwds)

class PySP_ConstraintStageAnnotation(PySP_Annotation):
    """
    This annotation is used to identify what time-stage a
    constraint belongs to.

    The declare method should be called with an additional
    argument that is an integer greater than or equal to
    one, which signifies the time-stage of the initial
    component argument.
    """
    def declare(self, component, stage):
        assert int(stage) == stage
        assert stage >= 1
        self.data[component] = stage

class PySP_StochasticRHSAnnotation(PySP_Annotation):
    """
    This annotation is used to identify constraints that have
    stochastic bound data.

    When calling declare, at most one of the keywords 'lb' or
    'ub' can be set to False to disable the annotation on
    one side of any range type constraints.
    """
    def declare(self, component, lb=True, ub=True):
        assert lb or ub
        assert (lb is True) or (lb is False)
        assert (ub is True) or (ub is False)
        if lb and ub:
            self.data[component] = True
        else:
            self.data[component] = (lb, ub)

class PySP_StochasticMatrixAnnotation(PySP_Annotation):
    """
    This annotation is used to identify variable
    coefficients within constraints that are stochastic.

    When calling declare, the 'variables' keyword can be set to
    a list of variables whose coefficients should be treated
    as stochastic. Leaving 'variables' at its default of
    None signifies that the coefficients of all variables
    appearing in the expression should be considered
    stochastic.
    """
    def declare(self, component, variables=None):
        self.data[component] = variables

class PySP_StochasticObjectiveAnnotation(PySP_Annotation):
    """
    This annotation is used to identify variable
    cost-coefficients that are stochastic.

    When calling declare, the 'variables' keyword can be set to
    a list of variables whose coefficients should be treated
    as stochastic. Leaving 'variables' at its default of
    None signifies that the coefficients of all variables
    appearing in the expression should be considered
    stochastic. The 'include_constant' keyword signifies
    whether or not the constant term in the cost expression
    should be treated as stochastic (default is True).
    """
    def declare(self, component, variables=None, include_constant=True):
        self.data[component] = (variables, include_constant)
