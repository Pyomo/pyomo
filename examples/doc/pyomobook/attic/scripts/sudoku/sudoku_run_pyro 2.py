from pyomo.opt import (SolverFactory,
                       TerminationCondition)
from sudoku import (create_sudoku_model,
                    print_solution,
                    add_integer_cut)
from sudoku_pyro_utils import tighten_variable_bounds

# define the board
board = [(1,1,5),(1,2,3),(1,5,7), \
         (2,1,6),(2,4,1),(2,5,9),(2,6,5), \
         (3,2,9),(3,3,8),(3,8,6), \
         (4,1,8),(4,5,6),(4,9,3), \
         (5,1,4),(5,4,8),(5,6,3),(5,9,1), \
         (6,1,7),(6,5,2),(6,9,6), \
         (7,2,6),(7,7,2),(7,8,8), \
         (8,4,4),(8,5,1),(8,6,9),(8,9,5), \
         (9,5,8),(9,8,7),(9,9,9)]

model = create_sudoku_model(board)

free, fixed = tighten_variable_bounds(model)
print("Model has %s fixed variables and %s free variables"
            % (fixed, free))
if free == 0:
    print("All variables have been fixed")
    print("Solution #1:")
    print_solution(model)
    exit(0)

solution_count = 0
while 1:

    with SolverFactory("glpk") as opt:
        results = opt.solve(model)
        if results.solver.termination_condition != \
           TerminationCondition.optimal:
            print("All board solutions have been found")
            break

    solution_count += 1

    add_integer_cut(model)

    print("Solution #%d" % (solution_count))
    print_solution(model)
