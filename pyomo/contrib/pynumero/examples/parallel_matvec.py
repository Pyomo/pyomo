import numpy as np
from pyomo.common.dependencies import mpi4py
from pyomo.contrib.pynumero.sparse.mpi_block_vector import MPIBlockVector
from pyomo.contrib.pynumero.sparse.mpi_block_matrix import MPIBlockMatrix
from scipy.sparse import random


def main():
    comm = mpi4py.MPI.COMM_WORLD
    rank = comm.Get_rank()

    owners = [0, 1, 2, -1]
    x = MPIBlockVector(4, rank_owner=owners, mpi_comm=comm)

    owners = np.array([[0, -1, -1, 0], [-1, 1, -1, 1], [-1, -1, 2, 2]])
    a = MPIBlockMatrix(3, 4, rank_ownership=owners, mpi_comm=comm)

    np.random.seed(0)
    x.set_block(3, np.random.uniform(-10, 10, size=10))

    np.random.seed(rank)
    x.set_block(rank, np.random.uniform(-10, 10, size=10))
    a.set_block(rank, rank, random(10, 10, density=0.1))
    a.set_block(rank, 3, random(10, 10, density=0.1))

    b = a * x  # parallel matrix-vector dot product

    # check the answer
    local_x = x.make_local_copy().flatten()
    local_a = a.to_local_array()
    local_b = b.make_local_copy().flatten()

    err = np.abs(local_a.dot(local_x) - local_b).max()

    if rank == 0:
        print('error: ', err)

    return err


if __name__ == '__main__':
    main()
