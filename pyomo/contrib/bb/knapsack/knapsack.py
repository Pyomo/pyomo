
from bb import BranchAndBound
import re
import copy
import math

###
### KNAPSACK EXAMPLE
###


class Bunch(dict):
    """
    A class that can be used to store a bunch of data dynamically

    foo = Bunch(data=y, sq=y*y, val=2)
    print foo.data
    print foo.sq
    print foo.val

    Adapted from code developed by Alex Martelli and submitted to
    the ActiveState Programmer Network http://aspn.activestate.com
    """

    def __init__(self, **kw):
        dict.__init__(self, kw)
        self.__dict__.update(kw)



class Knapsack(BranchAndBound):

    __slots__ = ('locked_in', 'locked_out', 'last')

    def __init__(self, filename=None, names=None, values=None, weights=None, capacity=None, context=None):
        if context is None:
            #
            # Create context
            #
            if not filename is None:
                capacity, names, weights, values = self.read_file(filename)
            context = Bunch(names=names, values=values, weights=weights, capacity=capacity)
            tmp = [values[i]/(1.0*weights[i]) for i in range(len(values))]
            context.order = [i[0] for i in sorted(enumerate(tmp), key=lambda x:x[1], reverse=True)]
            self.locked_in = set()   # indices into order
            self.locked_out = set()   # indices into order
        BranchAndBound.__init__(self, context, -1)
        
    def debug(self):
        print("LOCKED IN")
        print(self.locked_in)
        print("LOCKED OUT")
        print(self.locked_out)
        print("SOLUTION")
        print(self.solution)
        print("SOLUTION VALUE")
        print(self.solution_value)
        print("BOUND")
        print(self.bound)
        print("LAST")
        print(getattr(self,'last',None))

    def read_file(self, filename):
        names=[]
        values=[]
        weights=[]
        capacity=0.0
        with open(filename, 'r') as INPUT:
            tmp = INPUT.readline()
            capacity = float(tmp.strip())
            for line in INPUT.readlines():
                tokens = re.split('[ ]+', line.strip())
                names.append(tokens[0])
                weights.append(float(tokens[1]))
                values.append(float(tokens[2]))
        return (capacity, names, weights, values)

    def compute_bound(self):
        cvalue = 0.0
        cweight = 0.0
        #
        soln = copy.copy(self.locked_in)
        for i in self.locked_in:
            cvalue += self.context.values[self.context.order[i]]
            cweight += self.context.weights[self.context.order[i]]
        if cweight > self.context.capacity:
            self.bound = float("-Inf")
            return self.bound
        #
        self.last = None
        for i in range(len(self.context.values)):
            if not i in self.locked_out and not i in self.locked_in:
                if cweight + self.context.weights[self.context.order[i]] > self.context.capacity: #TOLERANCE
                    self.last = i
                    break
                soln.add(i)
                cvalue += self.context.values[self.context.order[i]]
                cweight += self.context.weights[self.context.order[i]]
        #
        self.solution_value = cvalue
        self.solution = soln
        if self.last is None:
            self.bound = cvalue
        else:
            self.bound = cvalue + \
                    self.context.values[self.context.order[self.last]] * \
                        (self.context.capacity-cweight)/self.context.weights[self.context.order[self.last]]
        return self.bound

    def make_child(self, which_child):
        child = Knapsack(context=self.context)
        child.bound = self.bound
        if which_child == 0:    # Down
            child.locked_in = copy.copy(self.locked_in)
            child.locked_in.add(self.last)
            child.locked_out = self.locked_out
        elif which_child == 1:
            child.locked_in = self.locked_in
            child.locked_out = copy.copy(self.locked_out)
            child.locked_out.add(self.last)
        else:
            raise RuntimeError("Unknown child %d" % which_child)
        return child

    def separate(self):
        return 2

    def terminal(self):
        """
        Return True if this is a terminal.
        """
        #return math.isclose(self.solution_value, self.bound)    # TOLERANCE
        return math.fabs(self.solution_value - self.bound) < 1e-7    # TOLERANCE
   
    def get_solution(self): 
        """
        Return a solution, if we can find one easily.
        """
        if self.solution is None:
            return (None, None)
        return (self.solution_value, self.solution    )

    def print_solution(self, solution):
        for i in sorted(self.context.order[j] for j in solution):
            print((self.context.names[i], self.context.values[i], self.context.weights[i]))

    def get_abs_tol(self):
        num_items = len(self.context.values)
        if num_items < 1 :
            return 0
        tol = self.context.values[0]
        from fractions import gcd
        for i in range(1,num_items) :
            tol = gcd(tol,self.context.values[i])
        return tol



if __name__ == '__main__':
    from bb import SerialBBSolver
    #problem = Knapsack(filename='animal1.txt')
    problem = Knapsack(filename='scor-500-1.txt')
    solver = SerialBBSolver()
    value, solution = solver.solve(problem=problem)
    print(value)
    problem.print_solution(solution)

    from pbb import ParallelBBSolver_serial
    solver = ParallelBBSolver_serial()
    value, solution = solver.solve(problem=problem)
    print(value)
    problem.print_solution(solution)
