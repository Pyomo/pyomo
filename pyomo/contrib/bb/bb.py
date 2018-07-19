
__all__ = ['BBSolver', 'BranchAndBound']

import itertools
from time import clock
from heapq import heappush, heappop, heapify


class PriorityQueue(object):

    def __init__(self, sense=1):
        self.sense = sense
        self.pq = []
        self.counter = itertools.count()

    def __len__(self):
        return len(self.pq)

    def add(self, task, priority=None):
        'Add a new task.  This does not update the priority of an existing task.'
        if priority is None:
            priority = self.sense*task.bound
        else:
            priority *= self.sense
        count = next(self.counter)
        entry = [priority, count, task]
        heappush(self.pq, entry)

    def pop(self):
        'Remove and return the lowest priority task. Raise KeyError if empty.'
        if len(self.pq) == 0:
            raise RuntimeError('pop from an empty priority queue')
        return heappop(self.pq)[2]

    def prune(self, cutoff):
        val = self.sense*cutoff
        i = len(self.pq) - 1
        if i<0:
            return
        k = i//2
        work = []
        #
        # Check all leaves
        #
        while k <= i:
            if self.pq[i][0] >= val:
                self._remove(i)
                # Add parents to the work list
                work.append( (i + 1)//2 - 1 )
            i -= 1
        #
        # Check all parents
        #
        while len(work) > 0:
            i = work.pop(0)
            if self.pq[i][0] >= val:
                self._remove(i)
                # Add parents to the work list
                work.append( (i + 1)//2 - 1 )
            i -= 1
            
    def _remove(self, k):
        'Mark an existing task as None.  Raise KeyError if not found.'
        self.pq[k] = self.pq.pop()
        heapify(self.pq)


class BBSolver(object):

    def __init__(self):
        """
        Constructor
        """
        pass

    def solve(self, **kwds):
        """
        Perform BB search using keyword options.
        """
        self.root = kwds['problem']
        return self._solve()

    def _solve(self):
        """
        Private interface for solver logic
        """
        raise RuntimeError("No branch-and-bound solver specified")


class SerialBBSolver(BBSolver):

    def _solve(self):
        """
        Private interface for solver logic
        """
        start_time = clock()
        #
        # Initialize priority queue
        #
        sense = self.root.sense
        incumbent_value = sense*float('Inf')
        queue = PriorityQueue(sense=sense)
        queue.add(self.root)
        abs_tol = self.root.get_abs_tol()
        nbounded = 0
        #
        # Search
        #
        while len(queue) > 0:
            subproblem = queue.pop()

            if nbounded % 1000 == 0 :
                print("#" + str(nbounded) + " pool=" + str(len(queue)) + " inc=" \
                          + str(incumbent_value) + " bnd=" + str(subproblem.bound))

            bound = subproblem.compute_bound()
            nbounded += 1
            if sense*bound <= sense*incumbent_value:   # TOLERANCE
                #
                # Find new incumbent
                #
                (value, solution) = subproblem.get_solution()
                if (not value is None) and (sense*value < sense*incumbent_value):
                    incumbent_value = value
                    incumbent_solution = solution
                    queue.prune(incumbent_value - sense*abs_tol)

                if sense*bound < sense*incumbent_value - abs_tol and not subproblem.terminal():
                    #
                    # Generate children
                    #
                    numchildren = subproblem.separate()
                    for i in range(numchildren):
                        child = subproblem.make_child(i)
                        assert(not child is None)
                        queue.add( child )
        #
        # Save information and return
        #
        run_time = clock() - start_time
        print(str(nbounded) + " subproblems bounded")
        print("Run time " + str(run_time) + " seconds")
        self._incumbent_value = incumbent_value
        self._incumbent_solution = incumbent_solution
        return (incumbent_value, incumbent_solution)


class BranchAndBound(object):

    __slots__ = ('sense', 'context', 'bound', 
                 'solution', 'solution_value')

    def __init__(self, context, sense):
        self.sense = sense
        self.context = context
        self.bound = float('-Inf')
        self.solution = None
        self.solution_value = None

    def compute_bound(self):
        """
        Compute the bound on the current subproblem.

        Set the values:
            self.bound

        Optionally set the values:
            self.solution
            self.solution_value

        Return
            self.bound
        """
        raise RuntimeError("compute_bound method is undefined.")

    def make_child(self, which_child):
        """
        Create a BranchAndBound child, set the bound, compute
        a child subproblem, and then return the child.

            child = Knapsack(context=self.context)
            child.bound = self.bound
            ...
            return child
        """
        raise RuntimeError("make_child method is undefined.")

    def separate(self):
        """
        Perform computations needed to setup the 
        construction of child subproblems.

        Return:
            Number of children
        """
        raise RuntimeError("separate method is undefined.")

    def terminal(self):
        """
        Return True if this is a terminal subproblem.
        """
        raise RuntimeError("terminal method is undefined.")
   
    def get_solution(self): 
        """
        Return a solution, if we can find one easily.
       
        If no solution is available, then return (None, None).
        Otherwise, return (value, solution).
        """
        return (None, None)

    def get_abs_tol(self):
        """
        Return an absolute tolerance suitable for this problem instance.
        Default is 0.
        """
        return 0

    def print_solution(self, solution):
        """
        An auxilliary function that prints a solution.
        """
        print(solution)

