import pyomo.environ as pyo
from pyomo.core.base.set import UnindexedComponent_set
from farmer_example_block import build_block_model
from pyomo.dae.flatten import flatten_components_along_sets

def only_scenario_indexed():
    yields = [2.5, 3.0, 20.0]
    m = build_block_model(yields)

    sets = (m.scenarios,)
    # This partitions model components according to how they are indexed
    sets_list, vars_list = flatten_components_along_sets(
        m,
        sets,
        pyo.Var,
    )
    assert len(sets_list) <= 2
    assert len(vars_list) <= 2

    for sets, vars in zip(sets_list, vars_list):
        if len(sets) == 1 and sets[0] is UnindexedComponent_set:
            scalar_vars = vars
        elif len(sets) == 1 and sets[0] is m.scenarios:
            scenario_vars = vars
        else:
            # We only expect two cases here:
            # (a) unindexed
            # (b) indexed by scenario
            raise RuntimeError()

    sets_list, cons_list = flatten_components_along_sets(
        m,
        sets,
        pyo.Constraint,
    )
    assert len(sets_list) <= 2
    assert len(sets_list) <= 2
    scenario_cons = []
    for sets, cons in zip(sets_list, cons_list):
        if len(sets) == 1 and sets[0] is UnindexedComponent_set:
            scalar_cons = cons
        elif len(sets) == 1 and sets[0] is m.scenarios:
            scenario_cons = cons
        else:
            # We only expect two cases here:
            # (a) unindexed
            # (b) indexed by scenario
            raise RuntimeError()

    # The block hierarchy has been "flattened."
    # Not to be confused with "flattening" a high-dimension
    # index set into single-dimension index set.
    flattened_model = pyo.ConcreteModel()
    flattened_model.unindexed_vars = pyo.Reference(scalar_vars)
    for i, var in enumerate(scenario_vars):
        # var is already a reference.
        flattened_model.add_component("scenario_var_%s" % i, var)

    flattened_model.unindexed_constraints = pyo.Reference(scalar_cons)
    for i, con in enumerate(scenario_cons):
        flattened_model.add_component("scenario_con_%s" % i, con)

    flattened_model.obj = pyo.Reference(m.OBJ)

    solver = pyo.SolverFactory("ipopt")
    solver.solve(flattened_model, tee=True)


def all_sets():
    yields = [2.5, 3.0, 20.0]
    m = build_block_model(yields)

    # It makes sense to build the flattened model ahead of time
    # in this case, so we don't need to know what "set combinations"
    # we're looking for a priori
    flattened_model = pyo.ConcreteModel()

    sets = tuple(m.component_data_objects(pyo.Set))
    # This partitions model components according to how they are indexed
    sets_list, vars_list = flatten_components_along_sets(
        m,
        sets,
        pyo.Var,
    )
    for i, (sets, vars) in enumerate(zip(sets_list, vars_list)):
        if len(sets) == 1 and sets[0] is UnindexedComponent_set:
            flattened_model.unindexed_vars = pyo.Reference(vars)
        else:
            for j, var in enumerate(vars):
                flattened_model.add_component("var_%s_%s" % (i, j), var)

    sets_list, cons_list = flatten_components_along_sets(
        m,
        sets,
        pyo.Constraint,
    )
    for i, (sets, cons) in enumerate(zip(sets_list, cons_list)):
        if len(sets) == 1 and sets[0] is UnindexedComponent_set:
            flattened_model.unindexed_constraints = pyo.Reference(cons)
        else:
            for j, con in enumerate(cons):
                flattened_model.add_component("con_%s_%s" % (i, j), con)

    flattened_model.obj = pyo.Reference(m.OBJ)

    solver = pyo.SolverFactory("ipopt")
    solver.solve(flattened_model, tee=True)


def main():
    only_scenario_indexed()
    all_sets()


if __name__ == "__main__":
    main()
