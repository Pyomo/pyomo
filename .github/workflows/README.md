GitHub Actions Drivers
======================

This directory contains the driver scripts for the Pyomo CI through
GitHub Actions.  There are two main driver files:

- `unix_python_matrix_test.yml` (PR/master testing on Linux/OSX)
- `win_python_matrix_test.yml`  (PR/master testing on Windows)

There are three other drivers that are derived from these two base
drivers:

- `mpi_matrix_test.yml` (PR/master testing with MPI on Linux)
- `push_branch_unix_test.yml` (branch testing on Linux/OSX)
- `push_branch_win_test.yml` (branch testing on Windows)

These workflows should not be directly edited.  Instead, we maintain
patch files that can be applied to the base workflows to regenerate the
three derived workflows.  The `validate.sh` script automates this
process to help developers validate that the derived workflows have not
drifted.  

If it becomes necessary to update the derived files, the easiest
process is probably to edit the derived workflow(s) and then regenerate
the respective patch file(s).
