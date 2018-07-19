import Pyro4
from pbb import ParallelBBSolver_pyro_synchronize, ParallelBBWorker_pyro_synchronize


if __name__ == '__main__':
    print("Spinning up 5 workers and 1 master")
    solver = ParallelBBSolver_pyro_synchronize()
    worker = [ParallelBBWorker_pyro_synchronize() for i in range(5)]

    Pyro4.config.SERIALIZER = 'pickle'
    Pyro4.config.SERVERTYPE = "thread"
    Pyro4.Daemon.serveSimple(
        {
            worker[0]:    "worker.1",
            #worker[1]:    "worker.2",
            #worker[2]:    "worker.3",
            #worker[3]:    "worker.4",
            #worker[4]:    "worker.5",
            solver:       "pbb1",
        }, verbose=True
    )
