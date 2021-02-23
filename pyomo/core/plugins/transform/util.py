#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

# """
# Useful functions for transformations.
# """

from inspect import isroutine
from pyomo.core import Var, Objective, Constraint, Set, Param


def collectAbstractComponents(model):
    """
    Returns all abstract constraints, objectives, variables,
    parameters, and sets. Does not query instance values, if
    present. Returns nested dictionaries describing the model,
    including all rules, domains, bounds, and index sets.
    """

    # Clone the model for sanity
    cp = model.clone()

    #
    # Component Name
    #   - Field in map
    #
    # Constraint
    #   - Index set
    #   - Rule
    #
    # Var
    #   - Index set
    #   - Domain
    #   - Bounds
    #
    # Objective
    #   - Index set
    #   - Rule
    #
    # Set
    #   - Index set
    #   - Initialize
    #   - Domain
    #
    # Param
    #   - Index set
    #   - initialize
    #   - domain/bounds
    #
    # All components are primarily indexed by their attribute name (string)
    #

    # Model components (will include all subclasses of the base class)
    constraints = {}
    variables = {}
    objectives = {}
    parameters = {}
    sets = {}

    # Mapping naming scheme
    conName = "Constraint"
    varName = "Var"
    objName = "Objective"
    paramName = "Param"
    setName = "Set"
    index = "index"
    bounds = "bounds"
    domain = "domain"
    initialize = "initialize"
    rule = "rule"

    # Iterate over all model components
    for comp in cp._ctypes:

        # Collect all Constraint objects
        if issubclass(comp, Constraint):
            comps = cp.component_map(comp, active=True)
            for (name, obj) in [(name, comps[name]) for name in comps]:
                # Query this constraint's attributes
                data = {}

                # Get indices
                data[index] = _getAbstractIndices(obj)

                # Get the rule
                data[rule] = _getAbstractRule(obj)

                # Add this constraint
                constraints[name] = data

        # Collect all Objective objects
        if issubclass(comp, Objective):
            comps = cp.component_map(comp, active=True)
            for (name, obj) in [(name, comps[name]) for name in comps]:
                # Query this objective's attributes
                data = {}

                # Get indices
                data[index] = _getAbstractIndices(obj)

                # Get the rule
                data[rule] = _getAbstractRule(obj)

                # Add this constraint
                objectives[name] = data

        # Collect all Var objects
        if issubclass(comp, Var):
            comps = cp.component_map(comp, active=True)
            for (name, obj) in [(name, comps[name]) for name in comps]:
                # Query this variable's attributes
                data = {}

                # Get indices
                data[index] = _getAbstractIndices(obj)

                # Get the domain
                data[domain] = _getAbstractDomain(obj)

                # Get the bounds
                data[bounds] = _getAbstractBounds(obj)

                # Add this constraint
                variables[name] = data

        # Collect all Set objects
        if issubclass(comp, Set):
            comps = cp.component_map(comp, active=True)
            for (name, obj) in [(name, comps[name]) for name in comps]:
                # Query this variable's attributes
                data = {}

                # Get indices
                data[index] = _getAbstractIndices(obj)

                # Get the domain
                data[domain] = _getAbstractDomain(obj)

                # Add this constraint
                sets[name] = data

        # Collect all Param objects
        if issubclass(comp, Param):
            comps = cp.component_map(comp, active=True)
            for (name, obj) in [(name, comps[name]) for name in comps]:
                # Query this variable's attributes
                data = {}

                # Get indices
                data[index] = _getAbstractIndices(obj)

                # Get the domain
                data[domain] = _getAbstractDomain(obj)

                # Add this constraint
                parameters[name] = data

    # Construct master data set
    master = {}
    master[conName] = constraints
    master[objName] = objectives
    master[varName] = variables
    master[paramName] = parameters
    master[setName] = sets

    return master

def _getAbstractIndices(comp):
    """
    Returns the index or index set of this component
    """
    if type(comp._index) != type({}):
        # Singly indexed component
        return comp._index
    else:
        # Unindexed constraint
        return {None: None}

def _getAbstractRule(comp):
    """
    Returns the rule defining this component
    """
    return comp.rule

def _getAbstractDomain(comp):
    """
    Returns the domain of this component
    """
    return getattr(comp,'domain', None)

def _getAbstractBounds(comp):
    """
    Returns the bounds of this component
    """
    if getattr(comp,'bounds',None) is None:
        return (None, None)
    else:
        return comp.bounds

def _getAbstractInitialize(comp):
    """
    Returns the initialization rule. If initialize is a container; return None;
    that information will be collected during construction.
    """
    if isroutine(comp.initialize):
        return comp.initialize
    else:
        return None

try:
    from functools import partial as _partial
except ImportError:
    # functools doesn't exist in Python 2.4
    def _partial(f, *args, **kwds):
        """
        Returns a new function with positional and keyword arguments
        partially applied

        """
        def closure(*cargs, **ckwds):
            # Collect positional arguments
            tmp_args = list(args)
            tmp_args.extend(list(cargs))

            tmp_kwds = dict(kwds)
            tmp_kwds.update(ckwds)

            # Call the original function
            return f(*tmp_args, **tmp_kwds)
        return closure

def partial(*args, **kwargs):
    """
    copy.deepcopy balks at copying anonymous functions. This overrides
    the default behavior of functools.partial to make deepcopy return
    the function itself, rather than attempting to copy it.
    """
    func = _partial(*args, **kwargs)

    def _partial_deepcopy(memo={}):
        return func

    func.__deepcopy__ = _partial_deepcopy
    return func

def process_canonical_repn(expr):
    """
    Returns a dictionary of {var_name_or_None: coef} values. None
    indicates a numeric constant.
    """

    terms = {}

    # Get the variables from the canonical representation
    vars = expr.pop(-1, {})

    # Find the linear terms
    linear = expr.pop(1, {})
    for k in linear:
        # FrozenDicts don't support (k, v)-style iteration
        v = linear[k]

        # There's exactly 1 variable in each term
        terms[vars[k.keys()[0]].label] = v

    # Get the constant term, if present
    const = expr.pop(0, {})
    if None in const:
        terms[None] = const[None]

    if len(expr) != 0:
        raise TypeError("Nonlinear terms in expression")

    return terms
