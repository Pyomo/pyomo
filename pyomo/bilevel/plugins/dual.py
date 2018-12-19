#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.core.base import Constraint, Objective, Block
from pyomo.repn import generate_standard_repn
from pyomo.core.base.plugin import TransformationFactory
from pyomo.core.base import Var, Set
from pyomo.bilevel.plugins.transform import Base_BilevelTransformation

import logging
logger = logging.getLogger('pyomo.core')


@TransformationFactory.register('bilevel.linear_dual', doc="Dualize a SubModel block")
class LinearDual_BilevelTransformation(Base_BilevelTransformation):

    def __init__(self):
        super(LinearDual_BilevelTransformation, self).__init__()

    def _apply_to(self, instance, **kwds):
        #
        # Process options
        #
        submodel = self._preprocess('bilevel.linear_dual', instance, **kwds)
        self._fix_all()
        #
        # Generate the dual
        #
        setattr(instance, self._submodel+'_dual', self._dualize(submodel, self._unfixed_upper_vars))
        instance.reclassify_component_type(self._submodel+'_dual', Block)
        #
        # Deactivate the original subproblem and upper-level objective
        #
        for (oname, odata) in submodel._parent().component_map(Objective, active=True).items():
            odata.deactivate()
        submodel.deactivate()
        #
        # Unfix the upper variables
        #
        self._unfix_all()
        #
        # Disable the original submodel
        #
        sub = getattr(instance,self._submodel)
        # TODO: Cache the list of components that were deactivated
        for (name, data) in sub.component_map(active=True).items():
            if not isinstance(data,Var) and not isinstance(data, Set):
                data.deactivate()


    def _dualize(self, submodel, unfixed):
        """
        Generate the dual of a submodel
        """ 
        transform = TransformationFactory('duality.linear_dual')
        return transform._dualize(submodel, unfixed)

    def _xfrm_bilinearities(self, dual):
        """
        Replace bilinear terms in constraints with disjunctions
        """ 
        for (name, data) in dual.component_map(Constraint, active=True).items():
            for ndx in data:
                con = data[ndx]
                degree = con.body.polynomial_degree()
                if degree > 2:
                    raise "RuntimeError: Cannot transform a model with polynomial degree %d" % degree
                if degree == 2:
                    terms = generate_standard_repn(con.body)
                    for i, var in enumerate(terms.quadratic_vars):
                        print("%s %s %s" % (i, str(var), str(terms.quadratic_coefs[i])))

