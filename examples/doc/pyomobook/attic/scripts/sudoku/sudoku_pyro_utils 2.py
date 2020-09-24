import pyutilib.pyro
from pyomo.environ import Var

def tighten_variable_bounds(model):

    # connect to a local pyro dispatcher through which we
    # will transmit tasks to workers
    client = pyutilib.pyro.Client(host='127.0.0.1')

    # for each variable not already fixed, transmit a task to
    # check for tighter bounds based on feasibility
    task_count = 0
    for r in model.ROWS:
        for c in model.COLS:
            for v in model.VALUES:
                if not model.y[r,c,v].fixed:
                    task_count += 1
                    task_data = {'board': model.board,
                                 'index': (r,c,v)}
                    client.add_task(
                        pyutilib.pyro.Task(data=task_data))

    # collect the worker responses and fix any binary
    # variables whose feasible upper and lower bounds are
    # the same
    fixed_count = 0
    while task_count > 0:
        print("Waiting on %s tasks" % (task_count))
        task = client.get_result(block=True, timeout=None)
        lb, ub = task['result']
        data = task['data']
        # make sure that the worker has rounded to integers
        assert lb in [0,1]
        assert ub in [0,1]
        if lb == ub:
            model.y[data['index']].fix(lb)
            fixed_count += 1
        task_count -= 1

    print("Bounds Tightening Fixed %s Variables"
          % (fixed_count))

    # compute and return counts for the total number of
    # fixed and free variables
    total_fixed, total_free = 0, 0
    for var in model.component_data_objects(Var):
        if var.fixed:
            total_fixed += 1
        else:
            total_free += 1

    return total_free, total_fixed
