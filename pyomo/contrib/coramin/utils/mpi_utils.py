from mpi4py import MPI
import numpy as np
import sys
import os


class MPISyncError(Exception):
    pass


class MPIInterface:
    def __init__(self):
        self._comm = MPI.COMM_WORLD
        self._size = self._comm.Get_size()
        self._rank = self._comm.Get_rank()

    @property
    def comm(self):
        return self._comm
    
    @property
    def rank(self):
        return self._rank
    
    @property
    def size(self):
        return self._size


class MPIAllocationMap:
    def __init__(self, mpi_interface, global_N):
        self._mpi_interface = mpi_interface
        self._global_N = global_N

        rank = self._mpi_interface.rank
        size = self._mpi_interface.size

        # there must be a better way to do this
        # find which entries in global correspond
        # to this process (want them to be contiguous
        # for the MPI Allgather calls later
        local_N = [0 for i in range(self._mpi_interface.size)]
        for i in range(global_N):
            process_i = i % size
            local_N[process_i] += 1

        start = 0
        end = None
        for i,v in enumerate(local_N):
            if i == self._mpi_interface.rank:
                end = start + v
                break
            else:
                start += v

        self._local_map = list(range(start, end))

    def local_allocation_map(self):
        return list(self._local_map)
    
    def local_list(self, global_data):
        local_data = list()
        assert(len(global_data) == self._global_N)
        for i in self._local_map:
            local_data.append(global_data[i])
        return local_data

    def global_list_float64(self, local_data_float64):
        assert(len(local_data_float64) == len(self._local_map))
        global_data_numpy = np.zeros(self._global_N, dtype='d')*np.nan
        local_data_numpy = np.asarray(local_data_float64, dtype='d')
        comm = self._mpi_interface.comm
        comm.Allgatherv([local_data_numpy, MPI.DOUBLE],
                        [global_data_numpy, MPI.DOUBLE])

        return global_data_numpy.tolist()


def activate_mpi_printing(style='rank-0-console', rank_0_filename='output_rank_0.txt'):
    """
    Redirect standard output based on process rank.
    
    Parameters
    ----------
    style: str
        Can be set to one of:
           * 'ignore-all': ignore all printing (actually, redirect all printing to os.devnull)
           * 'rank-0-console': printing from rank 0 will go to the console, 
             printing from other processes will be ignored
           * 'rank-0-console-x-files': printing from rank 0 will go to the console, 
             printing from other processes will go to a separate file ('output_rank_x.txt')
           * 'rank-0-file': printing from rank 0 will go to 'output_rank_0.txt'
           * 'separate-files': printing from each processor will be redirected to a separate
             file for each process ('output_rank_x.txt')
    """
    rank = MPIInterface().rank
    if style == 'ignore-all':
        sys.stdout = open(os.devnull, 'w')
    elif style == 'rank-0-console':
        if rank != 0:
            sys.stdout = open(os.devnull, 'w')
    elif style == 'rank-0-file':
        if rank == 0:
            sys.stdout = open(rank_0_filename, 'w')
        else:
            sys.stdout = open(os.devnull, 'w')
    elif style == 'rank-0-console-x-files':
        if rank != 0:
            sys.stdout = open('output_rank_{0}.txt'.format(str(MPIInterface().rank)), 'w')
    elif style == 'separate-files':
        sys.stdout = open('output_rank_{0}.txt'.format(str(MPIInterface().rank)), 'w')
