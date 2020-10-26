#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

#
# NOTE: deprecated code
#
from pyomo.common.deprecation import deprecated
from pyomo.core import TransformationFactory, Constraint, Set, Var, Objective, AbstractModel, maximize
from pyomo.repn import generate_standard_repn
from pyomo.core.plugins.transform.hierarchy import IsomorphicTransformation
from pyomo.core.plugins.transform.standard_form import StandardForm
from pyomo.core.plugins.transform.util import partial, process_canonical_repn


@TransformationFactory.register("core.lagrangian_dual", doc="Create the LP dual model.")
class DualTransformation(IsomorphicTransformation):
    """
    Creates a standard form Pyomo model that is equivalent to another model

    Options
        dual_constraint_suffix      Defaults to _constraint
        dual_variable_prefix        Defaults to p_
        slack_names                 Defaults to auxiliary_slack
        excess_names                Defaults to auxiliary_excess
        lb_names                    Defaults to _lower_bound
        ub_names                    Defaults to _upper_bound
        pos_suffix                  Defaults to _plus
        neg_suffix                  Defaults to _minus
    """

    @deprecated(
        "Use of the pyomo.duality package is deprecated. There are known bugs "
        "in pyomo.duality, and we do not recommend the use of this code. "
        "Development of dualization capabilities has been shifted to "
        "the Pyomo Adversarial Optimization (PAO) library. Please contact "
        "William Hart for further details (wehart@sandia.gov).",
        version='5.6.2')
    def __init__(self, **kwds):
        kwds['name'] = "linear_dual"
        super(DualTransformation, self).__init__(**kwds)

    def _create_using(self, model, **kwds):
        """
        Tranform a model to its Lagrangian dual.
        """

        # Optional naming schemes for dual variables and constraints
        constraint_suffix = kwds.pop("dual_constraint_suffix", "_constraint")
        variable_prefix = kwds.pop("dual_variable_prefix", "p_")

        # Optional naming schemes to pass to StandardForm
        sf_kwds = {}
        sf_kwds["slack_names"] = kwds.pop("slack_names", "auxiliary_slack")
        sf_kwds["excess_names"] = kwds.pop("excess_names", "auxiliary_excess")
        sf_kwds["lb_names"] = kwds.pop("lb_names", "_lower_bound")
        sf_kwds["ub_names"] = kwds.pop("ub_names", "_upper_bound")
        sf_kwds["pos_suffix"] = kwds.pop("pos_suffix", "_plus")
        sf_kwds["neg_suffix"] = kwds.pop("neg_suffix", "_minus")

        # Get the standard form model
        sf_transform = StandardForm()
        sf = sf_transform(model, **sf_kwds)

        # Roughly, parse the objectives and constraints to form A, b, and c of
        #
        # min  c'x
        # s.t. Ax  = b
        #       x >= 0
        #
        # and create a new model from them.

        # We use sparse matrix representations

        # {constraint_name: {variable_name: coefficient}}
        A = _sparse(lambda: _sparse(0))

        # {constraint_name: coefficient}
        b = _sparse(0)

        # {variable_name: coefficient}
        c = _sparse(0)

        # Walk constaints
        for (con_name, con_array) in sf.component_map(Constraint, active=True).items():
            for con in (con_array[ndx] for ndx in con_array._index):
                # The qualified constraint name
                cname = "%s%s" % (variable_prefix, con.local_name)

                # Process the body of the constraint
                body_terms = process_canonical_repn(
                    generate_standard_repn(con.body))

                # Add a numeric constant to the 'b' vector, if present
                b[cname] -= body_terms.pop(None, 0)

                # Add variable coefficients to the 'A' matrix
                row = _sparse(0)
                for (vname, coef) in body_terms.items():
                    row["%s%s" % (vname, constraint_suffix)] += coef

                # Process the upper bound of the constraint. We rely on
                # StandardForm to produce equality constraints, thus
                # requiring us only to check the lower bounds.
                lower_terms = process_canonical_repn(
                    generate_standard_repn(con.lower))

                # Add a numeric constant to the 'b' matrix, if present
                b[cname] += lower_terms.pop(None, 0)

                # Add any variables to the 'A' matrix, if present
                for (vname, coef) in lower_terms.items():
                    row["%s%s" % (vname, constraint_suffix)] -= coef

                A[cname] = row

        # Walk objectives. Multiply all coefficients by the objective's 'sense'
        # to convert maximizing objectives to minimizing ones.
        for (obj_name, obj_array) in sf.component_map(Objective, active=True).items():
            for obj in (obj_array[ndx] for ndx in obj_array._index):
                # The qualified objective name

                # Process the objective
                terms = process_canonical_repn(
                    generate_standard_repn(obj.expr))

                # Add coefficients
                for (name, coef) in terms.items():
                    c["%s%s" % (name, constraint_suffix)] += coef*obj_array.sense

        # Form the dual
        dual = AbstractModel()

        # Make constraint index set
        constraint_set_init = []
        for (var_name, var_array) in sf.component_map(Var, active=True).items():
            for var in (var_array[ndx] for ndx in var_array._index):
                constraint_set_init.append("%s%s" %
                                           (var.local_name, constraint_suffix))

        # Make variable index set
        variable_set_init = []
        dual_variable_roots = []
        for (con_name, con_array) in sf.component_map(Constraint, active=True).items():
            for con in (con_array[ndx] for ndx in con_array._index):
                dual_variable_roots.append(con.local_name)
                variable_set_init.append("%s%s" % (variable_prefix, con.local_name))

        # Create the dual Set and Var objects
        dual.var_set = Set(initialize=variable_set_init)
        dual.con_set = Set(initialize=constraint_set_init)
        dual.vars = Var(dual.var_set)

        # Make the dual constraints
        def constraintRule(A, c, ndx, model):
            return sum(A[v][ndx] * model.vars[v] for v in model.var_set) <= \
                   c[ndx]
        dual.cons = Constraint(dual.con_set,
                               rule=partial(constraintRule, A, c))

        # Make the dual objective (maximizing)
        def objectiveRule(b, model):
            return sum(b[v] * model.vars[v] for v in model.var_set)
        dual.obj = Objective(rule=partial(objectiveRule, b), sense=maximize)

        return dual.create()


class _sparse(dict):
    """
    Represents a sparse map. Uses a user-provided value to initialize
    entries. If the default value is a callable object, it is called
    with no arguments.

    Examples

      # Sparse vector
      v = _sparse(0)

      # 2-dimensional sparse matrix
      A = _sparse(lambda: _sparse(0))

    """

    def __init__(self, default, *args, **kwds):
        dict.__init__(self, *args, **kwds)

        if hasattr(default, "__call__"):
            self._default_value = None
            self._default_func = default
        else:
            self._default_value = default
            self._default_func = None

    def __getitem__(self, ndx):
        if ndx in self:
            return dict.__getitem__(self, ndx)
        else:
            if self._default_func is not None:
                return self._default_func()
            else:
                return self._default_value

