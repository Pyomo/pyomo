
from bb import PriorityQueue, BBSolver
from time import clock, sleep
import Pyro4

#
# Synchronous execution of subproblems "in parallel"
#
# Work is pushed to workers
#
class ParallelBBSolver_serial(BBSolver):

    def _solve(self):
        """
        Private interface for solver logic
        """
        start_time = clock()
        #
        # Setup parallelization
        #
        nworkers = 100
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
            #
            # Send subproblems to workers
            #
            # We are sending the incumbent value, because the worker may conditionally
            # decide to generate an incumbent solution.
            
            sent = []
            i=0
            while (i < nworkers) and len(queue) > 0:
                i += 1
                subproblem = queue.pop()
                sent.append( self.send_subproblem(subproblem, incumbent_value) )
            #
            # Get results from workers
            #
            # result = (bound, value, solution)
            # If value is None, then no solution is returned
            #
            results = []
            for subproblem_id in sent:
                results.append( self.get_results(subproblem_id) )
            #
            # Process results
            #
            for i, result in enumerate(results):
                bound, value, solution = result
                #
                # Update nbounded and print diagnostics
                #
                nbounded += 1
                if nbounded % 1000 == 0 :
                    print("#" + str(nbounded) + " pool=" + str(len(queue)) + " inc=" \
                              + str(incumbent_value) + " bnd=" + str(bound))
                #
                # Update incumbent and prune the queue if an improving incumbent has been found
                #
                if (value is not None) and (sense*value < sense*incumbent_value):
                    incumbent_value = value
                    incumbent_solution = solution
                    queue.prune(incumbent_value - sense*abs_tol)
                #
                # Generate children for non-terminal nodes
                #
                if sense*bound < sense*incumbent_value - abs_tol and not sent[i].terminal():
                    numchildren = sent[i].separate()
                    for j in range(numchildren):
                        child = sent[i].make_child(j)
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

    def send_subproblem(self, subproblem, incumbent_value):
        self._incumbent_value = incumbent_value
        return subproblem

    def get_results(self, subproblem):
        bound = subproblem.compute_bound()
        if self.root.sense*bound <= self.root.sense*self._incumbent_value:
            value, solution = subproblem.get_solution()
            return bound, value, solution
        else:
            return bound, None, None

#
# Synchronous execution of subproblems with Pyro4
#
@Pyro4.expose
@Pyro4.behavior(instance_mode="single")
class ParallelBBSolver_pyro_synchronize(BBSolver):

    @Pyro4.expose
    def solve(self, **kwds):
       return BBSolver.solve(self, **kwds)

    def _solve(self):
        """
        Private interface for solver logic
        """
        start_time = clock()
        #
        # Setup parallelization
        #
        with Pyro4.locateNS() as ns:
            all_workers = ns.list(prefix="worker.")
        workers = [Pyro4.Proxy(uri) for uri in all_workers.values()]
        for w in workers:
            w._pyroAsync()   # set proxy in asynchronous mode
        nworkers = len(workers)
        if nworkers == 0:
            return None, None
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
            #
            # Send subproblems to workers
            #
            # We are sending the incumbent value, because the worker may conditionally
            # decide to generate an incumbent solution.
            #
            handles = []
            waiting = set()
            results = {}
            i=0
            while (i < nworkers) and len(queue) > 0:
                subproblem = queue.pop()
                #handles.append( workers[i].async_compute_results(subproblem, incumbent_value, sense) )
                #waiting.add(i)
                results[i] = workers[i].async_compute_results(subproblem, incumbent_value, sense)
                while True:
                    if results[i].ready:
                        results[i] = results[i].value
                        break
                print("HERE")
                i += 1
            #
            # Get results from workers
            #
            # result = (bound, value, solution, terminal_flag)
            # If value is None, then no solution is returned
            #
            while True:
                for i in list(waiting):
                    if handles[i].ready:
                        results[i] = handles[i].value
                        waiting.discard(i)
                if len(waiting) == 0:
                    break
                sleep(0.001)
            #print("HERE")
            #
            # Process results
            #
            for i, result in results.items():
                bound, value, solution, subproblem = result
                #
                # Update nbounded and print diagnostics
                #
                nbounded += 1
                if True or nbounded % 1000 == 0 :
                    print("#" + str(nbounded) + " pool=" + str(len(queue)) + " inc=" \
                              + str(incumbent_value) + " bnd=" + str(bound))
                #
                # Update incumbent and prune the queue if an improving incumbent has been found
                #
                if (value is not None) and (sense*value < sense*incumbent_value):
                    incumbent_value = value
                    incumbent_solution = solution
                    queue.prune(incumbent_value - sense*abs_tol)
                #
                # Generate children for non-terminal nodes
                #
                if sense*bound < sense*incumbent_value - abs_tol and not subproblem.terminal():
                    numchildren = subproblem.separate()
                    for j in range(numchildren):
                        child = subproblem.make_child(j)
                        assert(not child is None)
                        queue.add( child )
        #
        # Terminate parallelization
        #
        for w in workers:
            w._pyroRelease()
        #
        # Save information and return
        #
        run_time = clock() - start_time
        print(str(nbounded) + " subproblems bounded")
        print("Run time " + str(run_time) + " seconds")
        self._incumbent_value = incumbent_value
        self._incumbent_solution = incumbent_solution
        return (incumbent_value, incumbent_solution)


@Pyro4.expose
class ParallelBBWorker_pyro_synchronize(object):

    def __init__(self):
        self._results = (None, None, None)

    @Pyro4.expose
    def async_compute_results(self, subproblem, incumbent_value, sense):
        bound = subproblem.compute_bound()
        if sense*bound <= sense*incumbent_value:
            value, solution = subproblem.get_solution()
            return (bound, value, solution, subproblem)
        return (bound, None, None, subproblem)

