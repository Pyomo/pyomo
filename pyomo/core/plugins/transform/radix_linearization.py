#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

from pyomo.util.plugin import alias
from pyomo.core import Binary, value, as_numeric
from pyomo.core.base import Transformation, Var, Constraint, ConstraintList, Block, RangeSet
from pyomo.core.base.expr import _ProductExpression, _PowExpression
from pyomo.core.base.var import _VarData

from six import iteritems

import logging
logger = logging.getLogger(__name__)

class RadixLinearization(Transformation):
    """
    This plugin generates linear relaxations of bilinear problems using
    the multiparametric disaggregation technique of Kolodziej, Castro,
    and Grossmann.  See:

    Scott Kolodziej, Pedro M. Castro, and Ignacio E. Grossmann. "Global
       optimization of bilinear programs with a multiparametric
       disaggregation technique."  J.Glob.Optim 57 pp.1039-1063. 2013.
       (DOI 10.1007/s10898-012-0022-1)
    """

    alias("core.radix_linearization",
           doc="Linearize bilinear and quadratic terms through "
           "radix discretization (multiparametric disaggregation)" )

    def _create_using(self, model, **kwds):
        precision = kwds.pop('precision',8)
        user_discretize = kwds.pop('discretize',set())
        verbose = kwds.pop('verbose',False)

        M = model.clone()

        # TODO: if discretize is not empty, we must translate those
        # components over to the components on the cloned instance
        _discretize = {}
        if user_discretize:
            for _var in user_discretize:
                _v = M.find_component(_var.name)
                if _v.component() is _v:
                    for _vv in _v.itervalues():
                        _discretize.setdefault(id(_vv), len(_discretize))
                else:
                    _discretize.setdefault(id(_v), len(_discretize))

        # Iterate over all Constraints and identify the bilinear and
        # quadratic terms
        bilinear_terms = []
        quadratic_terms = []
        for constraint in M.component_map(Constraint, active=True).itervalues():
            for cname, c in constraint._data.iteritems():
                if c.body.polynomial_degree() != 2:
                    continue
                self._collect_bilinear(c.body, bilinear_terms, quadratic_terms)

        # We want to find the (minimum?) number of variables to
        # discretize so that we cover all the bilinearities -- without
        # discretizing both sides of any single bilinear expression.
        # First step: figure out how many expressions each term appears
        # in
        _counts = {}
        for q in quadratic_terms:
            if not q[1].is_continuous():
                continue
            _id = id(q[1])
            if _id not in _counts:
                _counts[_id] = (q[1], set())
            _counts[_id][1].add(_id)
        for bi in bilinear_terms:
            for i in (0,1):
                if not bi[i+1].is_continuous():
                    continue
                _id = id(bi[i+1])
                if _id not in _counts:
                    _counts[_id] = (bi[i+1], set())
                _counts[_id][1].add(id(bi[2-i]))

        _tmp_counts = dict(_counts)
        # First, remove the variables that the user wants to have discretized
        for _id in _discretize:
            for _i in _tmp_counts[_id][1]:
                if _i == _id:
                    continue
                _tmp_counts[_i][1].remove(_id)
            del _tmp_counts[_id]
        # All quadratic terms must be discretized (?)
        #for q in quadratic_terms:
        #    _id = id(q[1])
        #    if _id not in _tmp_counts:
        #        continue
        #    _discretize.setdefault(_id, len(_discretize))
        #    for _i in _tmp_counts[_id][1]:
        #        if _i == _id:
        #            continue
        #        _tmp_counts[_i][1].remove(_id)
        #    del _tmp_counts[_id]

        # Now pick a (minimal) subset of the terms in bilinear expressions
        while _tmp_counts:
            _ct, _id = max( (len(_tmp_counts[i][1]), i) for i in _tmp_counts )
            if not _ct:
                break
            _discretize.setdefault(_id, len(_discretize))
            for _i in list(_tmp_counts[_id][1]):
                if _i == _id:
                    continue
                _tmp_counts[_i][1].remove(_id)
            del _tmp_counts[_id]

        #
        # Discretize things
        #

        # Define a block (namespace) for holding the disaggregated
        # variables and new constraints
        if False: # Set to true when the LP writer is fixed
            M._radix_linearization = Block()
            _block = M._radix_linearization
        else:
            _block = M
        _block.DISCRETIZATION = RangeSet(precision)
        _block.DISCRETIZED_VARIABLES = RangeSet(0, len(_discretize)-1)
        _block.z = Var( _block.DISCRETIZED_VARIABLES, _block.DISCRETIZATION,
                         within=Binary )
        _block.dv = Var( _block.DISCRETIZED_VARIABLES,
                         bounds=(0,2**-precision) )

        # Actually discretize the terms we have marked for discretization
        for _id, _idx in iteritems(_discretize):
            if verbose:
                logger.info("Discretizing variable %s as %s" %
                            (_counts[_id][0].name, _idx))
            self._discretize_variable(_block, _counts[_id][0], _idx)

        _known_bilinear = {}
        # For each quadratic term, if it hasn't been discretized /
        # generated, do so, and remember the resulting W term for later
        # use...
        #for _expr, _x1 in quadratic_terms:
        #    self._discretize_term( _expr, _x1, _x1,
        #                           _block, _discretize, _known_bilinear )
        # For each bilinear term, if it hasn't been discretized /
        # generated, do so, and remember the resulting W term for later
        # use...
        for _expr, _x1, _x2 in bilinear_terms:
            self._discretize_term( _expr, _x1, _x2,
                                   _block, _discretize, _known_bilinear )

        # Return the discretized instance!
        return M


    def _discretize_variable(self, b, v, idx):
        _lb, _ub = v.bounds
        if _lb is None or _ub is None:
            raise RuntimeError("Couldn't discretize variable %s: missing "
                               "finite lower/upper bounds." % (v.name))
        _c = Constraint(
            expr= v == _lb + (_ub-_lb) * ( b.dv[idx] +
                sum(b.z[idx,k] * 2**-k for k in b.DISCRETIZATION) ) )
        b.add_component("c_discr_v%s" % idx, _c)


    def _discretize_term(self, _expr, _x1, _x2, _block, _discretize, _known_bilinear):
        if id(_x1) in _discretize:
            _v = _x1
            _u = _x2
        elif id(_x2) in _discretize:
            _u = _x1
            _v = _x2
        else:
            raise RuntimeError("Couldn't identify discretized variable "
                               "for expression '%s'!" % _expr)
        _id = (id(_v), id(_u))
        if _id not in _known_bilinear:
            _known_bilinear[_id] = self._discretize_bilinear(
                _block, _v, _discretize[id(_v)], _u, len(_known_bilinear))
        # _expr should be a "simple" product expression; substitute
        # in the bilinear "W" term for the raw bilinear terms
        _expr._numerator = [ _known_bilinear[_id] ]


    def _discretize_bilinear(self, b, v, v_idx, u, u_idx):
        _z = b.z
        _dv = b.dv[v_idx]
        _u = Var(b.DISCRETIZATION, within=u.domain, bounds=u.bounds)
        logger.info("Discretizing (v=%s)*(u=%s) as u%s_v%s"
                    % (v.name, u.name, u_idx, v_idx ))
        b.add_component( "u%s_v%s" % (u_idx, v_idx), _u)
        _lb, _ub = u.bounds
        if _lb is None or _ub is None:
             raise RuntimeError("Couldn't relax variable %s: missing "
                               "finite lower/upper bounds." % (u.name))
        _c = ConstraintList()
        b.add_component( "c_disaggregate_u%s_v%s" % (u_idx, v_idx), _c )
        for k in b.DISCRETIZATION:
            # _lb * z[v_idx,k] <= _u[k] <= _ub * z[v_idx,k]
            _c.add(expr= _lb*_z[v_idx,k] <= _u[k] )
            _c.add(expr= _u[k] <= _ub*_z[v_idx,k] )
            # _lb * (1-z[v_idx,k]) <= u - _u[k] <= _ub * (1-z[v_idx,k])
            _c.add(expr= _lb * (1-_z[v_idx,k]) <= u - _u[k] )
            _c.add(expr= u - _u[k] <= _ub * (1-_z[v_idx,k]))

        _v_lb, _v_ub = v.bounds
        _bnd_rng = (_v_lb*_lb, _v_lb*_ub, _v_ub*_lb, _v_ub*_ub)
        _w = Var(bounds=(min(_bnd_rng), max(_bnd_rng)))
        b.add_component( "w%s_v%s" % (u_idx, v_idx), _w)

        K = max(b.DISCRETIZATION)

        _dw = Var(bounds=( min(0, _lb*2**-K, _ub*2**-K),
                           max(0, _lb*2**-K, _ub*2**-K) ))
        b.add_component( "dw%s_v%s" % (u_idx, v_idx), _dw)

        _c = Constraint(expr= _w == _v_lb*u + (_v_ub-_v_lb) * (
                sum(2**-k * _u[k] for k in b.DISCRETIZATION) + _dw ) )
        b.add_component( "c_bilinear_u%s_v%s" % (u_idx, v_idx), _c )

        _c = ConstraintList()
        b.add_component( "c_mccormick_u%s_v%s" % (u_idx, v_idx), _c )
        # u_lb * dv <= dw <= u_ub * dv
        _c.add(expr= _lb*_dv <= _dw )
        _c.add(expr= _dw <= _ub*_dv )
        # (u-u_ub)*2^-K + u_ub*dv <= dw <= (u-u_lb)*2^-K + u_lb*dv
        _c.add(expr= (u - _ub)*2**-K + _ub*_dv <= _dw )
        _c.add(expr= _dw <= (u - _lb)*2**-K + _lb*_dv )

        return _w

    def _collect_bilinear(self, expr, bilin, quad):
        if not expr.is_expression():
            return
        if type(expr) is _ProductExpression:
            if len(expr._numerator) != 2:
                for e in expr._numerator:
                    self._collect_bilinear(e, bilin, quad)
                # No need to check denominator, as this is poly_degree==2
                return
            if not isinstance(expr._numerator[0], _VarData) or \
                    not isinstance(expr._numerator[1], _VarData):
                raise RuntimeError("Cannot yet handle complex subexpressions")
            if expr._numerator[0] is expr._numerator[1]:
                quad.append( (expr, expr._numerator[0]) )
            else:
                bilin.append( (expr, expr._numerator[0], expr._numerator[1]) )
            return
        if type(expr) is _PowExpression and value(expr._args[1]) == 2:
            # Note: directly testing the value of the exponent above is
            # safe: we have already verified that this expression is
            # polynominal, so the exponent must be constant.
            tmp = _ProductExpression()
            tmp._numerator = [ expr._args[0], expr._args[0] ]
            tmp._denominator = []
            expr._args = (tmp, as_numeric(1))
            #quad.append( (tmp, tmp._args[0]) )
            self._collect_bilinear(tmp, bilin, quad)
            return
        # All other expression types
        for e in expr._args:
            self._collect_bilinear(e, bilin, quad)

