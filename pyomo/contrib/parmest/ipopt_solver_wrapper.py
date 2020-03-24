import pyutilib.services
from pyomo.opt import TerminationCondition

def ipopt_solve_with_stats(model, solver, max_iter=500, max_cpu_time=120):
    """
    Run the solver (must be ipopt) and return the convergence statistics

    Parameters
    ----------
    model : Pyomo model
       The pyomo model to be solved

    solver : Pyomo solver
       The pyomo solver to use - it must be ipopt, but with whichever options are preferred

    max_iter : int
       The maximum number of iterations to allow for ipopt

    max_cpu_time : int
       The maximum cpu time to allow for ipopt (in seconds)

    Returns
    -------
       Returns a tuple with (solve status object, bool (solve successful or not), number of iters, solve time, regularization value at solution)
    """
    # ToDo: Check that the "solver" is, in fact, IPOPT

    pyutilib.services.TempfileManager.push()
    tempfile = pyutilib.services.TempfileManager.create_tempfile(suffix='ipopt_out', text=True)
    opts = {'output_file': tempfile,
            'max_iter': max_iter,
            'max_cpu_time': max_cpu_time}

    status_obj = solver.solve(model, options=opts, tee=True)
    solved = True
    if status_obj.solver.termination_condition != TerminationCondition.optimal:
        solved = False

    iters = 0
    time = 0
    line_m_2 = None
    line_m_1 = None
    # parse the output file to get the iteration count, solver times, etc.
    with open(tempfile, 'r') as f:
        for line in f:
            if line.startswith('Number of Iterations....:'):
                tokens = line.split()
                iters = int(tokens[3])
                tokens_m_2 = line_m_2.split()
                regu = str(tokens_m_2[6])
            elif line.startswith('Total CPU secs in IPOPT (w/o function evaluations)   ='):
                tokens = line.split()
                time += float(tokens[9])
            elif line.startswith('Total CPU secs in NLP function evaluations           ='):
                tokens = line.split()
                time += float(tokens[8])
            line_m_2 = line_m_1
            line_m_1 = line

    pyutilib.services.TempfileManager.pop(remove=True)
    return status_obj, solved, iters, time, regu
