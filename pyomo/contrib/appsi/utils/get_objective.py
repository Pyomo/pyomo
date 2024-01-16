from pyomo.core.base.objective import Objective


def get_objective(block):
    obj = None
    for o in block.component_data_objects(
        Objective, descend_into=True, active=True, sort=True
    ):
        if obj is not None:
            raise ValueError('Multiple active objectives found')
        obj = o
    return obj
