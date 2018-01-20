#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import logging

from pyomo.core import *
from pyomo.core.base.indexed_component import IndexedComponent
from pyomo.core.base.misc import apply_indexed_rule
from pyomo.core.base.block import _BlockData, IndexedBlock
from pyomo.dae import *

from six import iterkeys, itervalues, iteritems

logger = logging.getLogger('pyomo.core')


def generate_finite_elements(ds, nfe):
    """
    This function first checks to see if the number of finite elements
    in the differential set is equal to nfe. If the number of finite
    elements is less than nfe, additional points will be generated. If
    the number of finite elements is greater than or equal to nfe the
    differential set will not be modified
    """
    if (len(ds) - 1) >= nfe:
        # In this case the differentialset already contains the
        # desired number or more than the desired number of finite
        # elements so no additional points are needed.
        return
    elif len(ds) == 2:
        # If only bounds have been specified on the differentialset we
        # generate the desired number of finite elements by
        # spreading them evenly over the interval
        step = (max(ds) - min(ds)) / float(nfe)
        tmp = min(ds) + step
        while round(tmp, 6) <= round((max(ds) - step), 6):
            ds.add(round(tmp, 6))
            tmp += step
        ds.set_changed(True)
        ds._sort()
        ds._fe = list(ds)
        return
    else:
        # This is the case where some points have been specified
        # inside of the bounds however the desired number of finite
        # elements has not been met. We first look at the step sizes
        # between the existing points. Then an additional point
        # is placed at the midpoint of the largest step. This
        # process is repeated until we have achieved the desired
        # number of finite elements. If there are multiple "largest steps"
        # the point will be placed at the first occurance of the
        # largest step

        addpts = nfe - (len(ds) - 1)
        while addpts > 0:
            _add_point(ds)
            addpts -= 1
        ds.set_changed(True)
        ds._sort()
        ds._fe = list(ds)
        return


def _add_point(ds):
    sortds = sorted(ds)
    maxstep = sortds[1] - sortds[0]
    maxloc = 0
    for i in range(2, len(sortds)):
        if (sortds[i] - sortds[i - 1]) > maxstep:
            maxstep = sortds[i] - sortds[i - 1]
            maxloc = i - 1

    ds.add(round((sortds[maxloc] + maxstep / 2.0), 6))


def generate_colloc_points(ds, tau):
    """
    This function adds collocation points between the finite elements
    in the differential set
    """
    fes = sorted(ds)
    for i in range(1, len(fes)):
        h = fes[i] - fes[i - 1]
        for j in range(len(tau)):
            if tau[j] == 1 or tau[j] == 0:
                continue
            pt = fes[i - 1] + h * tau[j]
            pt = round(pt, 6)
            if pt not in ds:
                ds.add(pt)
                ds.set_changed(True)
    ds._sort()


def update_contset_indexed_component(comp):
    """
    Update any model components which are indexed by a ContinuousSet
    that has changed
    """

    # This implemenation will *NOT* check for or update
    # components which use a ContinuousSet implicitly. ex) an
    # objective function which iterates through a ContinuousSet and
    # sums the squared error.  If you use a ContinuousSet implicitly
    # you must initialize it with every index you would like to have
    # access to!

    if comp.type() is Suffix:
        return
    
    # Params indexed by a ContinuousSet should include an initialize
    # and/or default rule which will be called automatically when the
    # parameter value at a new point in the ContinuousSet is
    # requested. Therefore, no special processing is required for
    # Params.
    if comp.type() is Param:
        return

    # Components indexed by a ContinuousSet must have a dimension of at
    # least 1
    if comp.dim() == 0:
        return

    # Extract the indexing sets. Must treat components with a single
    # index separately from components with multiple indexing sets.
    if comp._implicit_subsets is None:
        indexset = [comp._index]
    else:
        indexset = comp._implicit_subsets

    for s in indexset:
        if s.type() == ContinuousSet and s.get_changed():
            if isinstance(comp, Var):  # Don't use the type() method here
                # because we want to catch DerivativeVar components as well
                # as Var components
                _update_var(comp)
            elif comp.type() == Constraint:
                _update_constraint(comp)
            elif comp.type() == Expression:
                _update_expression(comp)
            elif isinstance(comp, Piecewise):
                _update_piecewise(comp)
            elif comp.type() == Block: 
                _update_block(comp)    
            else:
                raise TypeError(
                    "Found component %s of type %s indexed "
                    "by a ContinuousSet. Components of this type are "
                    "not currently supported by the automatic "
                    "discretization transformation in pyomo.dae. "
                    "Try adding the component to the model "
                    "after discretizing. Alert the pyomo developers "
                    "for more assistance." % (str(comp), comp.type()))


def _update_var(v):
    """
    This method will construct any additional indices in a variable
    resulting from the discretization of a ContinuousSet.
    """

    # Note: This is not required it is handled by the _default method on
    #       Var (which is now a IndexedComponent). However, it
    #       would be much slower to rely on that method to generate new
    #       _VarData for a large number of new indices.
    new_indices = set(v._index) - set(iterkeys(v._data))
    for index in new_indices:
        v.add(index)


def _update_constraint(con):
    """
    This method will construct any additional indices in a constraint
    resulting from the discretization of a ContinuousSet.
    """

    _rule = con.rule
    _parent = con._parent()
    for i in con.index_set():
        if i not in con:
            # Code taken from the construct() method of Constraint
            con.add(i, apply_indexed_rule(con, _rule, _parent, i))


def _update_expression(expre):
    """
    This method will construct any additional indices in an expression
    resulting from the discretization of a ContinuousSet.
    """
    _rule = expre._init_rule
    _parent = expre._parent()
    for i in expre.index_set():
        if i not in expre:
            # Code taken from the construct() method of Expression
            expre.add(i, apply_indexed_rule(expre, _rule, _parent, i))


def _update_block(blk):
    """
    This method will construct any additional indices in a block
    resulting from the discretization of a ContinuousSet. For
    Block-derived components we check if the Block construct method has
    been overridden. If not then we update it like a regular block. If
    construct has been overridden then we try to call the component's
    update_after_discretization method. If the component hasn't
    implemented this method then we throw a warning and try to update it
    like a normal block. The issue, when construct is overridden, is that
    anything could be happening and we can't automatically assume that
    treating the block-derived component like a normal block will be
    sufficient to update it correctly.

    """
    
    # Check if Block construct method is overridden
    # getattr needed below for Python 2, 3 compatibility
    if blk.construct.__func__ is not getattr(IndexedBlock.construct,
                                             '__func__',
                                             IndexedBlock.construct):
        # check for custom update function
        try:
            blk.update_after_discretization()
            return
        except AttributeError:
            logger.warning(
                'DAE(misc): Attempting to apply a discretization '
                'transformation to the Block-derived component "%s". The '
                'component overrides the Block construct method but no '
                'update_after_discretization() function was found. Will '
                'attempt to update as a standard Block but user should verify '
                'that the component was expanded correctly. To suppress this '
                'warning, please provide an update_after_discretization() '
                'function on Block-derived components that override '
                'construct()' % blk.name)

    # Code taken from the construct() method of Block
    missing_idx = set(blk._index) - set(iterkeys(blk._data))
    for idx in list(missing_idx):
        _block = blk[idx]
        obj = apply_indexed_rule(
            blk, blk._rule, _block, idx, blk._options)
 
        if isinstance(obj, _BlockData) and obj is not _block:
            # If the user returns a block, use their block instead
            # of the empty one we just created.
            for c in list(obj.component_objects(descend_into=False)):
                obj.del_component(c)
                _block.add_component(c.local_name, c)
                # transfer over any other attributes that are not components
            for name, val in iteritems(obj.__dict__):
                if not hasattr(_block, name) and not hasattr(blk, name):
                    super(_BlockData, _block).__setattr__(name, val)


def _update_piecewise(pw):
    """
    This method will construct any additional indices in a Piecewise
    object resulting from the discretization of a ContinuousSet.
    """
    pw._constructed = False
    pw.construct()


def create_access_function(var):
    """
    This method returns a function that returns a component by calling
    it rather than indexing it
    """
    def _fun(*args):
        return var[args]
    return _fun


def create_partial_expression(scheme, expr, ind, loc):
    """
    This method returns a function which applies a discretization scheme
    to an expression along a particular indexind set. This is admittedly a
    convoluted looking implementation. The idea is that we only apply a
    discretization scheme to one indexing set at a time but we also want
    the function to be expanded over any other indexing sets.
    """
    def _fun(*args):
        return scheme(lambda i: expr(*(args[0:loc] + (i,) + args[loc + 1:])),
                      ind)
    return lambda *args: _fun(*args)(args[loc])


def add_discretization_equations(block, d):
    """
    Adds the discretization equations for DerivativeVar d to the Block block.
    Because certain indices will be valid for some discretization schemes and
    not others, we skip any constraints which raise an IndexError.
    """

    def _disc_eq(m, *args):
        try:
            return d[args] == d._expr(*args)
        except IndexError:
            return Constraint.Skip

    if d.dim() == 1:
        block.add_component(d.local_name + '_disc_eq',
                            Constraint(d._index, rule=_disc_eq))
    else:
        block.add_component(d.local_name + '_disc_eq',
                            Constraint(*d._implicit_subsets, rule=_disc_eq))


def add_continuity_equations(block, d, i, loc):
    """
    Adds continuity equations in the case that the polynomial basis function
    does not have a root at the finite element boundary
    """
    svar = d.get_state_var()
    nme = svar.local_name + '_' + i.local_name + '_cont_eq'
    if block.find_component(nme) is not None:
        return

    def _cont_exp(v, s):
        ncp = s.get_discretization_info()['ncp']
        afinal = s.get_discretization_info()['afinal']

        def _fun(i):
            tmp = sorted(s)
            idx = tmp.index(i)
            low = s.get_lower_element_boundary(i)
            if i != low or idx == 0:
                raise IndexError("list index out of range")
            low = s.get_lower_element_boundary(tmp[idx - 1])
            lowidx = tmp.index(low)
            return sum(v(tmp[lowidx + j]) * afinal[j] for j in range(ncp + 1))
        return _fun
    expr = create_partial_expression(_cont_exp, create_access_function(svar),
                                     i, loc)

    def _cont_eq(m, *args):
        try:
            return svar[args] == expr(*args)
        except IndexError:
            return Constraint.Skip

    if d.dim() == 1:
        block.add_component(nme, Constraint(d._index, rule=_cont_eq))
    else:
        block.add_component(nme, Constraint(*d._implicit_subsets,
                                            rule=_cont_eq))


def block_fully_discretized(b):
    """
    Checks to see if all ContinuousSets in a block have been discretized
    """

    for i in itervalues(b.component_map(ContinuousSet)):
        if 'scheme' not in i.get_discretization_info():
            return False
    return True


def get_index_information(var, ds):
    """
    This method will find the index location of the set ds in the var,
    return a list of the non_ds indices and return a function that can be
    used to access specific indices in var indexed by a ContinuousSet by
    specifying the finite element and collocation point. Users of this
    method should have already confirmed that ds is an indexing set of var
    and that it's a ContinuousSet
    """

    # Find index order of ContinuousSet in the variable
    indargs = []
    dsindex = 0
    tmpds2 = None

    if var.dim() != 1:
        indCount = 0
        for index in var._implicit_subsets:
            if isinstance(index, ContinuousSet):
                if index == ds:
                    dsindex = indCount
                else:
                    # If var is indexed by multiple ContinuousSets treat
                    # other ContinuousSets like a normal indexing set
                    indargs.append(index)
                indCount += 1     # A ContinuousSet must be one dimensional
            else:
                indargs.append(index)
                indCount += index.dimen

    if indargs == []:
        non_ds = (None,)
    elif len(indargs) > 1:
        non_ds = tuple(a for a in indargs)
    else:
        non_ds = (indargs[0],)

    if None in non_ds:
        tmpidx = (None,)
    elif len(non_ds) == 1:
        tmpidx = non_ds[0]
    else:
        tmpidx = non_ds[0].cross(*non_ds[1:])

    # Lambda function used to generate the desired index
    # more concisely
    idx = lambda n, i, k: _get_idx(dsindex, ds, n, i, k)

    info = dict()
    info['non_ds'] = tmpidx
    info['index function'] = idx
    return info


def _get_idx(l, ds, n, i, k):
    """
    This function returns the appropriate index for a variable
    indexed by a differential set. It's needed because the collocation
    constraints are indexed by finite element and collocation point
    however a ContinuousSet contains a list of all the discretization
    points and is not separated into finite elements and collocation
    points.
    """
    t = sorted(ds)
    tmp = t.index(ds._fe[i])
    tik = t[tmp + k]
    if n is None:
        return tik
    else:
        tmpn = n
        if not isinstance(n, tuple):
            tmpn = (n,)
    return tmpn[0:l] + (tik,) + tmpn[l:]
