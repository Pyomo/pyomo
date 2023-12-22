import pybnb
from pyomo.core.base.block import _BlockData
from pyomo.common.modeling import unique_component_name
import pyomo.environ as pe
from pyomo.core.expr.visitor import identify_variables
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.contrib import appsi
from pyomo.common.config import ConfigDict, ConfigValue, PositiveFloat
from pyomo.contrib.coramin.clone import clone_active_flat
from pyomo.contrib.coramin.relaxations.auto_relax import relax
from pyomo.repn.standard_repn import generate_standard_repn, StandardRepn
from pyomo.contrib.coramin.relaxations.split_expr import split_expr
from pyomo.core.expr.numeric_expr import LinearExpression


def _get_clone_and_var_map(m1: _BlockData):
    orig_vars = list()
    for c in cm.nonrelaxation_component_data_objects(
        m1, pe.Constraint, active=True, descend_into=True
    ):
        for v in identify_variables(c.body, include_fixed=False):
            orig_vars.append(v)
    obj = cm.get_objective(m1)
    if obj is not None:
        for v in identify_variables(obj.expr, include_fixed=False):
            orig_vars.append(v)
    for r in cm.relaxation_data_objects(m1, descend_into=True, active=True):
        orig_vars.extend(r.get_rhs_vars())
        orig_vars.append(r.get_aux_var())
    orig_vars = list(ComponentSet(orig_vars))
    tmp_name = unique_component_name(m1, "active_vars")
    setattr(m1, tmp_name, orig_vars)
    m2 = m1.clone()
    new_vars = getattr(m2, tmp_name)
    var_map = ComponentMap(zip(new_vars, orig_vars))
    delattr(m1, tmp_name)
    delattr(m1, tmp_name)
    return m2, var_map


class BnBConfig(ConfigDict):
    def __init__(
        self,
        description=None,
        doc=None,
        implicit=False,
        implicit_domain=None,
        visibility=0,
    ):
        super().__init__(description, doc, implicit, implicit_domain, visibility)
        self.feasibility_tol = self.declare(
            "feasibility_tol", ConfigValue(domain=PositiveFloat, default=1e-8)
        )


def impose_structure(m):
    m.aux = pe.VarList()

    for key, c in list(m.nonlinear.cons.items()):
        repn: StandardRepn = generate_standard_repn(c.body, quadratic=False, compute_values=True)
        expr_list = split_expr(repn.nonlinear_expr)
        if len(expr_list) == 1:
            continue

        linear_coefs = list(repn.linear_coefs)
        linear_vars = list(repn.linear_vars)
        for term in expr_list:
            v = m.aux.add()
            linear_coefs.append(1)
            linear_vars.append(v)
            m.vars.append(v)
            m.nonlinear.cons.add(v == term)
        new_expr = LinearExpression(constant=repn.constant, linear_coefs=linear_coefs, linear_vars=linear_vars)
        m.linear.cons.add((c.lb, new_expr, c.ub))
        del m.nonlinear.cons[key]


def _fix_vars_with_close_bounds(m, tol=1e-12):
    for v in m.vars:
        lb, ub = v.bounds
        if abs(ub - lb) <= tol * min(abs(lb), abs(ub)):
            v.fix(0.5 * (lb + ub))
        if v.is_fixed():
            v.setlb(v.value)
            v.setub(v.value)


def find_cut_generators(m):
    


class _BnB(pybnb.Problem):
    def __init__(self, model: _BlockData, config: BnBConfig):
        # remove all parameters, fixed variables, etc.
        nlp, relaxation = clone_active_flat(model, 2)
        self.nlp: _BlockData = nlp
        relaxation: _BlockData = relaxation
        self.config = config

        # perform fbbt before constructing relaxations in case
        # we can identify things like x**3 is convex because
        # x >= 0
        self.interval_tightener = it = appsi.fbbt.IntervalTightener()
        it.config.deactivate_satisfied_constraints = True
        it.config.feasibility_tol = config.feasibility_tol
        it.perform_fbbt(self.nlp)
        _fix_vars_with_close_bounds(self.nlp)

        impose_structure(relaxation)
        find_cut_generators(relaxation)
        self.relaxation = relaxation = relax(relaxation)


def solve_with_bnb(model: _BlockData, config: BnBConfig):
    # we don't want to modify the original model
    model, orig_var_map = _get_clone_and_var_map(model)
