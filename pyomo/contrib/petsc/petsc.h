/*
AMPL solver interface for PETSc
John Eslick
*/
#ifndef PETSC_H
#define PETSC_H

#include<asl.h>
#undef filename
#include<petscsnes.h>
#include<petscts.h>

#define MSG_BUF_SIZE 2000

/* Define some default setting, all changable through options */
#define DEFAULT_LINEAR_PACK MATSOLVERMUMPS   //be sure to build petsc with mumps
#define DEFAULT_SNES SNESNEWTONLS // newton line search
#define DEFAULT_PC PCLU        //LU decomposition direct solve
#define DEFAULT_KSP KSPPREONLY //default preconditioner solves this so preonly
#define DEFAULT_SNES_ATOL 1e-9
#define DEFAULT_SNES_RTOL 1e-9
#define DEFAULT_KSP_ATOL 1e-10
#define DEFAULT_KSP_RTOL 1e-15
#define DEFAULT_SNES_MAX_IT 2000
#define DEFAULT_SNES_MAX_FUNC 50000

#define COLOR_NORMAL  "\x1b[0m"
#define COLOR_RED     "\x1b[31m"
#define COLOR_GREEN   "\x1b[32m"

/* Variable scaling methods */
typedef enum{
    VAR_SCALE_NONE=0,
    VAR_SCALE_GRAD=1,
    VAR_SCALE_MULTIGRAD=2 //not ready yet
}VARSCALE_TYPE;

/* Equation scaling methods */
typedef enum{
    EQ_SCALE_NONE=0,
    EQ_SCALE_MAX_GRAD=1,
    EQ_SCALE_MAX_MULTIGRAD=2 //not ready yet
}EQSCALE_TYPE;

typedef enum{  //keep these under 50 and shouldn't confilict with PETSc codes
   P_EXIT_NORMAL = 0, //Finished okay (solved is another matter)
   P_EXIT_INTEGER = 1, //Exited due to integer variables
   P_EXIT_DOF = 2, //Exited on DOF != 0
   P_EXIT_INEQ = 3, //Exited on inequalities
   P_EXIT_NO_NL_FILE_ERROR = 5, //Exited due to file not specified
   P_EXIT_NL_FILE_ERROR = 6, //Exited due to nl file didn't read right
   P_EXIT_DOF_DAE = 7, //DOF wrong for DAE problem
   P_EXIT_VAR_DAE_MIS = 8, //number of derivatives mismatch
   P_EXIT_MULTIPLE_TIME = 9 //more than on time variable
}P_EXIT_CODES;

typedef struct{
  PetscMPIInt    mpi_size; // Number of processors (should be 1 for now)
  PetscBool      show_cl; //show the command line, and transformed CL
  char           stub[PETSC_MAX_PATH_LEN]; // File name (with or without ext)
  fint           stublen; // Stub string length
  PetscBool      got_stub;  // file stub was specified with -s
  PetscBool      show_con;  // Option to show initial constraint values
  PetscBool      show_init; // show initial values for x vec
  PetscBool      show_jac;  // show jacobian at intial value
  PetscBool      ampl_opt;  // -AMPL specified I catch it but ignore
  PetscBool      per_test;  // perturb inital solved value and resolve  to test
  PetscBool      use_bounds; // give solver variable bounds
  PetscBool      scale_var; // scale the variables based on jacobian at init
  PetscBool      scale_eq;  // scale the equations based on jacobian at init
  PetscBool      dae_solve; //use dae solver (requires suffix information)
  PetscBool      jac_explicit_diag; // explicitly include jacobian diagonal
  EQSCALE_TYPE   eq_scale_method; // method to scale equations
  VARSCALE_TYPE  var_scale_method; // method to scale variables
  PetscScalar    ptest; // factor for perturb test
}Solver_options;

typedef struct{
  ASL *asl; // ASL context
  Solver_options opt; // command-line options
  SufDesc *dae_suffix_var; // DAE suffixes on variables
  SufDesc *dae_link_var; // DAE link derivatives to vars
  int dae_map_t; // ASL index of time variable (-1 for none)
  int *dae_map_x; // PETSc index in x vec -> ASL index
  int *dae_map_xdot; // PETSc index in xdot vec -> ASL index
  int *dae_map_back; // ASL var index -> PETSc index (in x or xdot)
  int *dae_link; // ASL index -> ASL index of linked derivative or differential var
  int n_ineq; // Number of inequality constraints
  int n_var_diff; //DAE number of differential vars
  int n_var_deriv; //DAE number of derivative vars
  int n_var_state; //DAE number of state vars
  int n_var_alg;  //DAE number of algebraic variables
  int explicit_time; //DAE includes time variable? 1=yes 0=no
  int dof; // Degrees of freedom
  FILE *nl; // nl-file pointer
}Solver_ctx;

/* Function prototypes (just stuffing everything in one file for now) */
PetscErrorCode FormJacobian(SNES,Vec,Mat,Mat,void*);
PetscErrorCode FormFunction(SNES,Vec,Vec,void*);
PetscErrorCode FormDAEFunction(TS, PetscReal, Vec, Vec, Vec, void*);
PetscErrorCode FormDAEJacobian(TS, PetscReal, Vec, Vec, PetscReal, Mat, Mat, void*);

void get_dae_info(Solver_ctx *sol_ctx);
void dae_var_map(Solver_ctx *sol_ctx);
void sol_ctx_init(Solver_ctx *ctx);
int get_snes_sol_message(char *msg, SNESConvergedReason term_reason, ASL *asl);
int ScaleVars(VARSCALE_TYPE method, ASL *asl);
int ScaleVarsGrad(ASL *asl);
int ScaleEqs(EQSCALE_TYPE method, ASL *asl);
int ScaleEqs_Largest_Grad(ASL *asl);
char **transform_args(int argc, char** argv, int *size);
void print_commandline(const char* msg, int argc, char **argv);
void print_x_asl(ASL *asl);
void print_jac_asl(ASL *asl, real u, real l);

#endif
