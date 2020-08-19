#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyomo.kernel as pmo
from pyomo.core import ConcreteModel, Var, Objective, Constraint, RangeSet, ConstraintList
from pyomo.solvers.tests.models.base import _BaseTestModel, register_model
from pyomo.repn.beta.matrix import compile_block_linear_constraints

has_numpy = False
try:
    import numpy
    has_numpy = True
except:
    pass

has_scipy = False
try:
    import scipy
    import scipy.sparse
    has_scipy = True
except:
    pass

@register_model
class LP_compiled(_BaseTestModel):
    """
    A continuous linear model that is compiled into a
    MatrixConstraint object.
    """

    description = "LP_compiled"
    capabilities = set(['linear'])
    test_pickling = False
    size = (13, 22, None)

    def __init__(self):
        _BaseTestModel.__init__(self)
        self.add_results(self.description+".json")
        self.disable_suffix_tests = True

    def _generate_model(self):
        self.model = ConcreteModel()
        model = self.model
        model._name = self.description

        model.s = RangeSet(1,12)
        model.x = Var(model.s)
        model.x[1].setlb(-1)
        model.x[1].setub(1)
        model.x[2].setlb(-1)
        model.x[2].setub(1)
        model.obj = Objective(expr=sum(model.x[i]*((-1)**(i+1))
                                       for i in model.x.index_set()))
        model.c = ConstraintList()
        # to make the variable used in the constraint match the name
        model.c.add(Constraint.Skip)
        model.c.add(Constraint.Skip)
        model.c.add(model.x[3]>=-1.)
        model.c.add(model.x[4]<=1.)
        model.c.add(model.x[5]==-1.)
        model.c.add(model.x[6]==-1.)
        model.c.add(model.x[7]==1.)
        model.c.add(model.x[8]==1.)
        model.c.add((-1.,model.x[9],-1.))
        model.c.add((-1.,model.x[10],-1.))
        model.c.add((1.,model.x[11],1.))
        model.c.add((1.,model.x[12],1.))
        cdata = model.c.add((0, 1, 3))
        assert cdata.lower == 0
        assert cdata.upper == 3
        assert cdata.body() == 1
        assert not cdata.equality
        cdata = model.c.add((0, 2, 3))
        assert cdata.lower == 0
        assert cdata.upper == 3
        assert cdata.body() == 2
        assert not cdata.equality
        cdata = model.c.add((0, 1, None))
        assert cdata.lower == 0
        assert cdata.upper is None
        assert cdata.body() == 1
        assert not cdata.equality
        cdata = model.c.add((None, 0, 1))
        assert cdata.lower is None
        assert cdata.upper == 1
        assert cdata.body() == 0
        assert not cdata.equality
        cdata = model.c.add((1,1))
        assert cdata.lower == 1
        assert cdata.upper == 1
        assert cdata.body() == 1
        assert cdata.equality

        model.fixed_var = Var()
        model.fixed_var.fix(1.0)
        cdata = model.c.add((0, 1+model.fixed_var, 3))
        cdata = model.c.add((0, 2 + model.fixed_var, 3))
        cdata = model.c.add((0, model.fixed_var, None))
        cdata = model.c.add((None, model.fixed_var, 1))
        cdata = model.c.add((model.fixed_var, 1))

        model.c_inactive = ConstraintList()
        # to make the variable used in the constraint match the name
        model.c_inactive.add(Constraint.Skip)
        model.c_inactive.add(Constraint.Skip)
        model.c_inactive.add(model.x[3]>=-2.)
        model.c_inactive.add(model.x[4]<=2.)

        compile_block_linear_constraints(model, 'Amatrix')

    def warmstart_model(self):
        assert self.model is not None
        model = self.model
        for i in model.s:
            model.x[i].value = None

if has_numpy and has_scipy:
    # TODO: we need to somehow label this as a skip rather
    #       than not defining the test class

    @register_model
    class LP_compiled_dense_kernel(LP_compiled):

        def _get_dense_data(self):
            assert has_numpy and has_scipy
            A = numpy.array(
                [[1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                 [0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                 [0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                 [0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
                 [0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.],
                 [0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.],
                 [0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.],
                 [0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.],
                 [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.],
                 [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.],
                 [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                 [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                 [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                 [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                 [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                 [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                 [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                 [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                 [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                 [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                 [1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                 [0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]],
                dtype=float)
            lb = numpy.array(
                [-1.0, -numpy.inf, -1.0, -1.0,
                 1.0, 1.0, -1.0, -1.0,
                 1.0, 1.0, -1.0, -2.0,
                 -1.0, -numpy.inf, 0.0, -2.0,
                 -3.0, -1.0, -numpy.inf, 0.0,
                 -2.0, -numpy.inf])
            ub = numpy.array([
                numpy.inf, 1.0, -1.0, -1.0,
                1.0, 1.0, -1.0, -1.0,
                1.0, 1.0, 2.0, 1.0,
                numpy.inf, 1.0, 0.0, 1.0,
                0.0, numpy.inf, 0.0, 0.0,
                numpy.inf, 2.0])
            eq_index = [2,3,4,5,14,19]

            return A, lb, ub, eq_index

        def _generate_base_model(self):

            self.model = pmo.block()
            model = self.model
            model._name = self.description

            model.s = list(range(1,13))
            model.x = pmo.variable_dict(
                ((i, pmo.variable()) for i in model.s))
            model.x[1].lb = -1
            model.x[1].ub = 1
            model.x[2].lb = -1
            model.x[2].ub = 1
            model.obj = pmo.objective(expr=sum(model.x[i]*((-1)**(i+1))
                                               for i in model.s))
            variable_order = [
                model.x[3],
                model.x[4],
                model.x[5],
                model.x[6],
                model.x[7],
                model.x[8],
                model.x[9],
                model.x[10],
                model.x[11],
                model.x[12]]

            return variable_order

        def _generate_model(self):
            x = self._generate_base_model()
            model = self.model
            A, lb, ub, eq_index = self._get_dense_data()
            model.Amatrix = pmo.matrix_constraint(
                A, lb=lb, ub=ub, x=x, sparse=False)
            for i in eq_index:
                assert model.Amatrix[i].lb == \
                    model.Amatrix[i].ub
                model.Amatrix[i].rhs = \
                    model.Amatrix[i].lb

    @register_model
    class LP_compiled_sparse_kernel(
            LP_compiled_dense_kernel):

        def _generate_model(self):
            x = self._generate_base_model()
            model = self.model
            A, lb, ub, eq_index = self._get_dense_data()
            model.Amatrix = pmo.matrix_constraint(
                A, lb=lb, ub=ub, x=x, sparse=True)
            for i in eq_index:
                assert model.Amatrix[i].lb == \
                    model.Amatrix[i].ub
                model.Amatrix[i].rhs = \
                    model.Amatrix[i].lb
