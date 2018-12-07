# README

This is a first pass at creating an AMPL executable for the PETSc solver library, allowing PETSc to be used with Pyomo.  This version supports the TS (time-stepping) and SNES (nonlinear) solvers; however in the future TAO (optimization) solvers and additional features may also be supported.

## Building
This will probably be reasonably accurate for Linux, Mac, and Unix, but Windows may be more difficult.  The petsc AMPL executable is built with the PETSc build system, so if you can use the PETSc documentation to build PETSc, building the solver wrapper should be easy.

1. Build PETSc according the documentation: https://www.mcs.anl.gov/petsc/.
   * The default linear solver is currently set to MUMPS so, be sure to build PETSc with MUMPS (there is a configuration option to download MUMPS).
   * A good approach to building PETSc is to include everything that is easy to include in case you want to use it.  Here is an example configuration command.
   ```
   ./configure --download-cmake --download-fblaspack --download-mpich
      --download-mumps --download-sundials --download-superlu
      --download-scalapack --download-metis --download-parmetis
      --download-ptscotch --download-ml --download-suitesparse
      --download-strumpack
   ```

2. Get the ASL library from: https://ampl.com/netlib/ampl/solvers.tgz.
  1. Extract the files.
  2. In the directory where the source code was extracted run    ```./configure```
  3. Go to sys._arch_ and run ```make``` (_arch_ depends on your machine).

3. Set some environment variables (may want to add to .bashrc or a script if you do this regularly).
  1. PETSC_DIR = location of PETSc source
  2. PETSC_ARCH = subdirectory where PETSc was compiled
  3. ASL_BUILD = directory where ASL was compiled
4. Run ```make``` the directory where building the petsc executable. This mostly uses the PETSc build system, so this part is pretty simple.
5. Copy the executable to a location in your execution path.

## Usage

The petsc executable solver will run like any other AMPL executable and can take the standard PETSc options.  The options can be set from within Pyomo.  For use with Pyomo it is easiest if the petsc executable is in your executable path.

<TODO> Will fill out later, for now see examples and PETSc docs.  PETSc command line options work with the PETSc solvers. The are some additional command line arguments (see ```petsc -help```).  Command line arguments can be passed through Pyomo's solver interface.

## Testing

The included tl.nl file can be used to test the petsc solver. The problem is an old version of the IDAES MEA model, but that's not important. The initial values of variables in the file are the solution.  To test the solver the initial values can be perturbed before solving the problem the new solution can be compared to the old initial values to ensure the problem solved.

To test the petsc executable run:

petsc -s t1 -snes_monitor -perturb_test 1.1

This multiplies the initial values by 1.1 and resolves.  The results shows any differences between the new and old solutions greater than 1e-6, and the values of those variables.  Depending on the value of the variables, difference of more than 1e-6 are not necessarily bad.
