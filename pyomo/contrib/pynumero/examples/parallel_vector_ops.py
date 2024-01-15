import numpy as np
from pyomo.common.dependencies import mpi4py
from pyomo.contrib.pynumero.sparse.mpi_block_vector import MPIBlockVector


def main():
    comm = mpi4py.MPI.COMM_WORLD
    rank = comm.Get_rank()

    owners = [2, 0, 1, -1]
    x = MPIBlockVector(4, rank_owner=owners, mpi_comm=comm)
    x.set_block(owners.index(rank), np.ones(3) * (rank + 1))
    x.set_block(3, np.array([1, 2, 3]))

    y = MPIBlockVector(4, rank_owner=owners, mpi_comm=comm)
    y.set_block(owners.index(rank), np.ones(3) * (rank + 1))
    y.set_block(3, np.array([1, 2, 3]))

    z1: MPIBlockVector = x + y  # add x and y
    z2 = x.dot(y)  # dot product
    z3 = np.abs(x).max()  # infinity norm

    z1_local = z1.make_local_copy()
    if rank == 0:
        print(z1_local.flatten())
        print(z2)
        print(z3)

    return z1_local, z2, z3


if __name__ == '__main__':
    main()
