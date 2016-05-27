from pyutilib.pyro import (TaskWorker,
                           TaskWorkerServer)
from pyomo.opt import (SolverFactory,
                       TerminationCondition)
from sudoku import create_sudoku_model

# this class defines a custom pyro worker that accepts
# request to run a feasibility-based bounds tightening
# procedure given a Sudoku board and a variable index
class CustomPyroWorker(TaskWorker):

    def __init__(self, host=None, port=None):
        super(CustomPyroWorker, self).__init__(host=host,
                                               port=port)

    # this is an abstract method that must be defined for
    # TaskWorker implementations, where the 'data' argument
    # is defined by the client who generates the request.
    def process(self, data):

        instance = create_sudoku_model(data['board'])

        var = instance.y[data['index']]
        original_lb, original_ub = var.bounds
        original_objective = instance.obj.expr

        # check for tighter lower and upper bounds based on
        # feasibility, and round to zero or one to eliminate
        # the need for the client to handle solver
        # tolerances
        with SolverFactory('glpk') as solver:
            instance.obj.expr = var
            results = solver.solve(instance)
            if results.solver.termination_condition == \
               TerminationCondition.optimal:
                var.setlb(int(round(var())))
            instance.obj.expr = -var
            results = solver.solve(instance)
            if results.solver.termination_condition == \
               TerminationCondition.optimal:
                var.setub(int(round(var())))

        instance.obj.expr = original_objective

        if (var.lb > original_lb) or \
           (var.ub < original_ub):
            print("%s: new bounds %s" % (var, var.bounds))
        else:
            print("%s: no bound reduction" % (var))

        return var.bounds

if __name__ == "__main__":

    # spawn the task worker daemon, which will attempt to
    # connect to a 'dispatch_srvr' process that has
    # registered itself with a 'pyomo_ns' process
    TaskWorkerServer(CustomPyroWorker, host="127.0.0.1")
