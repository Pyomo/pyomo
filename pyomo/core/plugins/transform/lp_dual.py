#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2024
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.common.autoslots import AutoSlots
from pyomo.common.collections import ComponentMap
from pyomo.common.errors import MouseTrap
from pyomo.common.dependencies import scipy
from pyomo.core import (
    ConcreteModel, Var, Constraint, Objective, TransformationFactory,
    NonPositiveReals, maximize
)
from pyomo.opt import WriterFactory

@TransformationFactory.register(
    'core.lp_dual', 'Generate the linear programming dual of the given model')
class LinearProgrammingDual(object):
    def apply_to(self, model, **options):
        raise MouseTrap(
            "The 'core.lp_dual' transformation does not currently implement "
            "apply_to since it is a bit ambiguous what it means to take a dual "
            "in place. Please use 'create_using' and do what you wish with the "
            "returned model."
        )

    def create_using(self, model, ostream=None, **options):
        """Take linear programming dual of a model

        Returns
        -------
        ConcreteModel containing linear programming dual

        Parameters
        ----------
        model: ConcreteModel
            The concrete Pyomo model to take the dual of

        ostream: None
            This is provided for API compatibility with other writers
            and is ignored here.

        """
        std_form = WriterFactory('compile_standard_form').write(model,
                                                                nonnegative_vars=True)
        dual = ConcreteModel(name="%s dual" % model.name)
        A_transpose = scipy.sparse.csc_matrix.transpose(std_form.A)
        rows = range(A_transpose.shape[0])
        cols = range(A_transpose.shape[1])
        dual.x = Var(cols, domain=NonPositiveReals)
        dual.constraints = Constraint(rows)
        for i in rows:
            dual.constraints[i] = sum(A_transpose[i, j]*dual.x[j] for j in cols) <= \
                                  std_form.c[0, i]
        
        dual.obj = Objective(expr=sum(std_form.rhs[j]*dual.x[j] for j in cols),
                             sense=maximize)

        return dual
