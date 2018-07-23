import time
import Pyro4

#
# We need to customize the knapsack object because
# it uses slots.  :( :( :(
#
from knapsack import Knapsack
#
# Configure to use Pyro4
#
Pyro4.config.SERIALIZER = 'pickle'


def run(datafile, counter_uri):
    begin = time.time()
    problem = Knapsack(filename=datafile)
    with Pyro4.Proxy(counter_uri) as solver:
        value, solution = solver.solve(problem=problem)
    if value is None:
        print("No workers available")
    else:
        print(value)
        problem.print_solution(solution)
    time_taken = round(time.time()-begin, 2)
    print("Time taken:", time_taken, "sec.")


if __name__ == "__main__":
    print("\nCalling the server")
    run("scor-500-1.txt", "PYRONAME:pbb1")
