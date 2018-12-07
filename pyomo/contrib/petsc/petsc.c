/*
AMPL solver interface for PETSc
John Eslick
*/

//TODO<jce> Need to return asl solver status code for DAE solver
//TODO<jce> Add TAO optimization solvers

#include"help.h"
#include"petsc.h"
#include<stdio.h>
#include<math.h>

int main(int argc, char **argv){
  PetscErrorCode ierr;         // Error code from PETSc functions
  PetscInt       its;          // Number of solver iterations
  ASL            *asl;         // ASL context
  Solver_ctx     sol_ctx;      // solver context
  int            err;          // Error code  from ASL fulctions
  int            i=0;         // Loop counters
  real           *R;           // ASL constraint body test
  PetscInt       temp_int;    // a temporary variable to store a option
  int            argc_new;   // new number of arguments reformated for PETSc
  char           **argv_new; // argv transformed to PETSc's format
  static SufDecl suftab[] = { // suffixes to read in
    //doc for this at https://ampl.com/netlib/ampl/solvers/README.suf
    {"dae_suffix", NULL, ASL_Sufkind_var, 1}, //var kinds for DAE solver
    {"dae_suffix", NULL, ASL_Sufkind_con, 1},
    {"dae_link",   NULL, ASL_Sufkind_var, 1}}; //link derivatives to vars
  Vec            x,r,xl,xu; // solution, residual, bound vectors
  Mat            J;            // Jacobian matrix
  TS             ts;           // DAE solver
  SNES           snes;         // nonlinear solver context
  KSP            ksp;          // linear solver context
  PC             pc;           // linear preconditioner context
  SNESLineSearch linesearch;   // line search context
  SNESConvergedReason cr; // reason for convergence (or lack of convergence)
  char           msg[MSG_BUF_SIZE]; // just a string buffer
  PetscScalar    *xx, *xxl, *xxu; // for accessing x, xlb, and xub vectors
  real t; //time for DAE solution
  TSConvergedReason tscr;
  real *x_asl;

  // Set some initial values in sol_ctx
  sol_ctx_init(&sol_ctx);
  // Change the AMPL style args into what PETSc would expect
  argv_new = transform_args(argc, argv, &argc_new);
  // Initialize up PETSc stuff
  PetscInitialize(&argc_new, &argv_new, (char*)0, help);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD, &sol_ctx.opt.mpi_size);CHKERRQ(ierr);
  // Get added options
  PetscOptionsGetString(NULL, NULL, "-s", sol_ctx.opt.stub, PETSC_MAX_PATH_LEN-1, &sol_ctx.opt.got_stub);
  PetscOptionsHasName(NULL, NULL, "-show_constraints", &sol_ctx.opt.show_con);
  PetscOptionsHasName(NULL, NULL, "-show_jac", &sol_ctx.opt.show_jac);
  PetscOptionsHasName(NULL, NULL, "-show_initial", &sol_ctx.opt.show_init);
  PetscOptionsHasName(NULL, NULL, "-use_bounds", &sol_ctx.opt.use_bounds);
  PetscOptionsGetScalar(NULL, NULL, "-perturb_test",&sol_ctx.opt.ptest, &sol_ctx.opt.per_test);
  if(!sol_ctx.opt.per_test) sol_ctx.opt.ptest = 1.0;
  PetscOptionsGetInt(NULL, NULL, "-scale_eqs",&temp_int, &sol_ctx.opt.scale_eq);
  sol_ctx.opt.eq_scale_method = (EQSCALE_TYPE)temp_int;
  PetscOptionsGetInt(NULL, NULL, "-scale_vars", &temp_int, &sol_ctx.opt.scale_var);
  sol_ctx.opt.var_scale_method = (VARSCALE_TYPE)temp_int;
  PetscOptionsHasName(NULL, NULL, "-AMPL", &sol_ctx.opt.ampl_opt); // I don't use this
  PetscOptionsHasName(NULL, NULL, "-jac_explicit_diag", &sol_ctx.opt.jac_explicit_diag);
  PetscOptionsHasName(NULL, NULL, "-dae_solve", &sol_ctx.opt.dae_solve);
  PetscOptionsHasName(NULL, NULL, "-show_cl", &sol_ctx.opt.show_cl);
  // If show_cl otion, show the original and transformed command line
  if(sol_ctx.opt.show_cl){
    PetscPrintf(PETSC_COMM_SELF, "-----------------------------------------------------------------\n");
    print_commandline("Original Exec:\n  ", argc, argv);
    print_commandline("Transformed Exec:\n  ", argc_new, argv_new);
    PetscPrintf(PETSC_COMM_SELF, "-----------------------------------------------------------------\n");
  }
  // Make sure that a file was specified, and get the string length
  if(!sol_ctx.opt.got_stub) strcpy(sol_ctx.opt.stub, argv[1]); //assume first arg is file if no -s
  if(sol_ctx.opt.stub[0]=='-') exit(P_EXIT_NL_FILE_ERROR); // is an option name not filename
  sol_ctx.opt.stublen = strlen(sol_ctx.opt.stub);
  // Create ASL context and read nl file
  sol_ctx.asl = ASL_alloc(ASL_read_fg); // asl context
  asl = sol_ctx.asl;
  // set the suffix data structure
  suf_declare(suftab, sizeof(suftab)/sizeof(SufDecl));
  // get file pointer and basic problem info from nl file
  sol_ctx.nl = jac0dim(sol_ctx.opt.stub, sol_ctx.opt.stublen);
  if(sol_ctx.nl==NULL){
      PetscPrintf(PETSC_COMM_SELF, "Could not read nl file %s\n", sol_ctx.opt.stub);
      exit(P_EXIT_NL_FILE_ERROR);
  }
  // Allocated space for some ASL items and test calculations
  X0 = (real*)Malloc(n_var*sizeof(real));  /* Initial X values */
  R = (real*)Malloc(n_con*sizeof(real));   /* Constraint body values */
  LUv = (real*)Malloc(n_var*sizeof(real)); /* Variable lower bounds */
  Uvx = (real*)Malloc(n_var*sizeof(real)); /* Variable upper bounds */
  LUrhs = (real*)Malloc(n_con*sizeof(real)); /* Lower constraint right side */
  Urhsx = (real*)Malloc(n_con*sizeof(real)); /* Upper constraint right side */
  // count inequalities
  for(i=0; i<n_con; ++i) if(LUrhs[i] - Urhsx[i] > 1e-10) ++sol_ctx.n_ineq;
  // count degrees of freedom (n_var and n_con are macros from asl.h)
  sol_ctx.dof = n_var - n_con + sol_ctx.n_ineq;
  // Print basic problem information
  PetscPrintf(PETSC_COMM_SELF, "---------------------------------------------------\n");
  PetscPrintf(PETSC_COMM_SELF, "DAE: %d\n", sol_ctx.opt.dae_solve);
  PetscPrintf(PETSC_COMM_SELF, "Reading nl file: %s\n", sol_ctx.opt.stub);
  PetscPrintf(PETSC_COMM_SELF, "Number of constraints: %d\n", n_con);
  PetscPrintf(PETSC_COMM_SELF, "Number of nonlinear constraints: %d\n", nlc);
  PetscPrintf(PETSC_COMM_SELF, "Number of linear constraints: %d\n", n_con-nlc);
  PetscPrintf(PETSC_COMM_SELF, "Number of inequalities: %d\n", sol_ctx.n_ineq);
  PetscPrintf(PETSC_COMM_SELF, "Number of variables: %d\n", n_var);
  PetscPrintf(PETSC_COMM_SELF, "Number of integers: %d\n", niv);
  PetscPrintf(PETSC_COMM_SELF, "Number of binary: %d\n", nbv);
  PetscPrintf(PETSC_COMM_SELF, "Number of objectives: %d (Ignoring)\n", n_obj);
  PetscPrintf(PETSC_COMM_SELF, "Number of non-zeros in Jacobian: %d \n", nzc);
  PetscPrintf(PETSC_COMM_SELF, "Number of degrees of freedom: %d\n", sol_ctx.dof);
  // There are some restrictions (at least for now) to check
  if(nbv + niv > 0){ // no integer vars (nbv and niv are ASL macros)
    PetscPrintf(PETSC_COMM_SELF, "ERROR: Contains integer or binary variables.");
    ASL_free(&(sol_ctx.asl));
    exit(P_EXIT_INTEGER);}
  else if(sol_ctx.dof != 0 && !sol_ctx.opt.dae_solve){ //dof must == 0 for nonlinear solve
    PetscPrintf(PETSC_COMM_SELF, "ERROR: Degrees of freedom not equal to 0\n");
    ASL_free(&(sol_ctx.asl));
    exit(P_EXIT_DOF);}
  else if(sol_ctx.n_ineq > 0){ // no inequalities for nonlinear sys or DAE
    PetscPrintf(PETSC_COMM_SELF, "ERROR: contains inequalities");
    ASL_free(&(sol_ctx.asl));
    exit(P_EXIT_INEQ);}
  // Read nl file and make function for jacobian
  err = fg_read(sol_ctx.nl, 0);
  PetscPrintf(PETSC_COMM_SELF, "Called fg_read, err: %d (0 is good)\n", err);
  // If DAES, get DAE var types and map vars between ASL and PETSc
  if(sol_ctx.opt.dae_solve){
    get_dae_info(&sol_ctx);
    dae_var_map(&sol_ctx);
    PetscPrintf(PETSC_COMM_SELF, "Explicit time variable: %d\n", sol_ctx.explicit_time);
    PetscPrintf(PETSC_COMM_SELF, "Number of derivatives: %d\n", sol_ctx.n_var_deriv);
    PetscPrintf(PETSC_COMM_SELF, "Number of differential vars: %d\n", sol_ctx.n_var_diff);
    PetscPrintf(PETSC_COMM_SELF, "Number of algebraic vars: %d\n", sol_ctx.n_var_alg);
    PetscPrintf(PETSC_COMM_SELF, "Number of state vars: %d\n", sol_ctx.n_var_state);
  }
  PetscPrintf(PETSC_COMM_SELF, "---------------------------------------------------\n");

  // Equation/variable scaling
  ScaleEqs(sol_ctx.opt.eq_scale_method, sol_ctx.asl);
  if(!sol_ctx.opt.dae_solve) ScaleVars(sol_ctx.opt.var_scale_method, sol_ctx.asl); //for now not compatable with dae
  if(sol_ctx.opt.var_scale_method && !sol_ctx.opt.eq_scale_method){
    PetscPrintf(PETSC_COMM_SELF, "Warning: scale equations if scaling vars\n");}
  // Optionally print some stuff
  if(sol_ctx.opt.show_init) print_x_asl(sol_ctx.asl);
  if(sol_ctx.opt.show_jac) print_jac_asl(sol_ctx.asl, 101, 1e-3);
  if(sol_ctx.opt.show_con) for(i=0; i<n_con; ++i){
    PetscPrintf(PETSC_COMM_SELF, "c%d: %e <= %e <= %e\n", i, LUrhs[i], R[i], Urhsx[i]);}

  if(sol_ctx.opt.dae_solve){  //This block sets up DAE solve and solves
    ierr = TSCreate(PETSC_COMM_WORLD, &ts); CHKERRQ(ierr);
    ierr = VecCreate(PETSC_COMM_WORLD,&x); CHKERRQ(ierr); //create an x vector
    ierr = VecSetSizes(x, PETSC_DECIDE, sol_ctx.n_var_state); CHKERRQ(ierr);
    ierr = VecSetFromOptions(x); CHKERRQ(ierr); //command line options for vec
    ierr = VecDuplicate(x, &r);CHKERRQ(ierr);  // duplicate x for resuiduals
    /* Make x vec set initial guess from nl file, also get lb and ub */
    ierr = VecGetArray(x,&xx);CHKERRQ(ierr);
    for(i=0;i<n_var;++i){
      if(sol_ctx.dae_suffix_var->u.i[i]!=2 && sol_ctx.dae_suffix_var->u.i[i]!=3){
        xx[sol_ctx.dae_map_back[i]] = X0[i];
      }
    }
    ierr = VecRestoreArray(x, &xx);CHKERRQ(ierr);
    // Make Jacobian matrix (by default sparse AIJ)
    ierr = MatCreate(PETSC_COMM_WORLD,&J); CHKERRQ(ierr);
    ierr = MatSetSizes(J, PETSC_DECIDE, PETSC_DECIDE, n_con, sol_ctx.n_var_state); CHKERRQ(ierr);
    ierr = MatSetFromOptions(J); CHKERRQ(ierr); //command line options override defaults
    ierr = MatSetUp(J); CHKERRQ(ierr);
    /* Explicitly add diagonal elements if needed (some preconditioners need)*/
    for(i=n_conjac[0];i<n_conjac[1]; ++i){
      ierr = MatSetValue(J, i, i, 0.0, INSERT_VALUES);CHKERRQ(ierr);
    }
    /* finish up jacobian */
    ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    /* Set residual and jacobian functions */
    ierr = TSSetIFunction(ts, r, FormDAEFunction, &sol_ctx);CHKERRQ(ierr);
    ierr = TSSetIJacobian(ts, J, J, FormDAEJacobian, &sol_ctx);CHKERRQ(ierr);
    /* First set a bunch of default options, then read CL to override */
    ierr = TSSetProblemType(ts, TS_NONLINEAR);
    ierr = TSSetEquationType(ts, TS_EQ_IMPLICIT);
    ierr = TSGetSNES(ts, &snes);CHKERRQ(ierr);
    ierr = SNESGetKSP(snes, &ksp);CHKERRQ(ierr);
    ierr = SNESGetLineSearch(snes, &linesearch);CHKERRQ(ierr);
    ierr = SNESLineSearchSetType(linesearch, SNESLINESEARCHBT);
    ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
    ierr = PCSetType(pc,DEFAULT_PC);CHKERRQ(ierr);
    ierr = PCFactorSetMatSolverType(pc, DEFAULT_LINEAR_PACK);CHKERRQ(ierr);
    ierr = PCFactorReorderForNonzeroDiagonal(pc, 1e-10);CHKERRQ(ierr);
    ierr = KSPSetType(ksp,DEFAULT_KSP);CHKERRQ(ierr);
    ierr = TSSetTimeStep(ts, 1);
    ierr = TSSetMaxTime(ts, 10);
    ierr = TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP);
    // Set up solver from CL options
    ierr = TSSetFromOptions(ts);CHKERRQ(ierr);
    // Solve
    ierr = TSSolve(ts, x);
    ierr = TSGetTime(ts, &t);
    ierr = SNESGetConvergedReason(snes, &cr); CHKERRQ(ierr);
    ierr = TSGetConvergedReason(ts, &tscr); CHKERRQ(ierr);
    /* Get the results */
    x_asl = (real*)malloc((n_var)*sizeof(real));
    ierr = VecGetArray(x, &xx);CHKERRQ(ierr);
    for(i=0;i<sol_ctx.n_var_state;++i) x_asl[sol_ctx.dae_map_x[i]] = xx[i];
    ierr = VecRestoreArray(x,&xx);CHKERRQ(ierr);
    if(sol_ctx.explicit_time) x_asl[sol_ctx.dae_map_t] = t;
    /* write the AMPL solution file */
    sprintf(msg, "TSConvergedReason = %d", tscr);  //Reason it stopped
    write_sol(msg, x_asl, NULL, NULL); // write ASL sol file
    ierr = TSDestroy(&ts);
  } //end ts solve
  else{ // nonlinear solver setup and solve
    /*Create nonlinear solver context*/
    ierr = SNESCreate(PETSC_COMM_WORLD, &snes); CHKERRQ(ierr);
    /*Create vectors for solution and nonlinear function*/
    ierr = VecCreate(PETSC_COMM_WORLD,&x); CHKERRQ(ierr); //create an x vector
    ierr = VecSetSizes(x,PETSC_DECIDE,n_var);CHKERRQ(ierr); //vecs are n_vars long
    ierr = VecSetFromOptions(x);CHKERRQ(ierr); //command line options for vec
    ierr = VecDuplicate(x,&r);CHKERRQ(ierr);  // duplicate x for resuiduals
    ierr = VecDuplicate(x,&xl);CHKERRQ(ierr); // duplicate x for lower bounds
    ierr = VecDuplicate(x,&xu);CHKERRQ(ierr); // duplicate x for upper bounds
    //Make Jacobian matrix (by default sparse AIJ)
    ierr = MatCreate(PETSC_COMM_WORLD,&J);CHKERRQ(ierr);
    ierr = MatSetSizes(J,PETSC_DECIDE,PETSC_DECIDE,n_con,n_var);CHKERRQ(ierr);
    ierr = MatSetFromOptions(J);CHKERRQ(ierr); //command line options override defaults
    ierr = MatSetUp(J);CHKERRQ(ierr);
    /* Explicitly add diagonal elements if needed (some preconditioners need)*/
    if(sol_ctx.opt.jac_explicit_diag) for(i=n_conjac[0];i<n_conjac[1]; ++i){
        ierr = MatSetValue(J, i, i, 0.0, INSERT_VALUES);CHKERRQ(ierr);
    }
    /* finish up jacobian */
    ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    /* Set residual and jacobian functions */
    ierr = SNESSetFunction(snes, r, FormFunction, &sol_ctx);CHKERRQ(ierr);
    ierr = SNESSetJacobian(snes, J, J, FormJacobian, &sol_ctx);CHKERRQ(ierr);
    /* Default solver setup override from CL later*/
    ierr = SNESSetType(snes, DEFAULT_SNES);CHKERRQ(ierr);
    ierr = SNESGetKSP(snes, &ksp);CHKERRQ(ierr);
    ierr = SNESGetLineSearch(snes, &linesearch);CHKERRQ(ierr);
    ierr = SNESLineSearchSetType(linesearch, SNESLINESEARCHBT);
    ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
    ierr = PCSetType(pc,DEFAULT_PC);CHKERRQ(ierr);
    ierr = PCFactorSetMatSolverType(pc, DEFAULT_LINEAR_PACK);CHKERRQ(ierr);
    ierr = PCFactorReorderForNonzeroDiagonal(pc, 1e-10);CHKERRQ(ierr);
    ierr = KSPSetType(ksp,DEFAULT_KSP);CHKERRQ(ierr);
    ierr = SNESSetTolerances(snes, DEFAULT_SNES_ATOL, DEFAULT_SNES_RTOL,
          0, DEFAULT_SNES_MAX_IT, DEFAULT_SNES_MAX_FUNC);CHKERRQ(ierr);
    ierr = KSPSetTolerances(
          ksp,DEFAULT_KSP_ATOL,DEFAULT_KSP_RTOL,PETSC_DEFAULT,200);CHKERRQ(ierr);
    /* Read command line options for solver */
    ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);
    /* Make x vec set initial guess from nl file, also get lb and ub */
    ierr = VecGetArray(x,&xx);CHKERRQ(ierr);
    ierr = VecGetArray(xl,&xxl);CHKERRQ(ierr);
    ierr = VecGetArray(xu,&xxu);CHKERRQ(ierr);
    for(i=0;i<n_var;++i){
        xx[i] = X0[i]*sol_ctx.opt.ptest; //ptest is usually 1.0 could be differnt to test solver
        xxl[i] = LUv[i]; //lower bound
        xxu[i] = Uvx[i]; //upper bound
    }
    ierr = VecRestoreArray(x,&xx);CHKERRQ(ierr);
    ierr = VecRestoreArray(x,&xxl);CHKERRQ(ierr);
    ierr = VecRestoreArray(x,&xxu);CHKERRQ(ierr);
    /* Set upper and lower bound most solver don't like but a few can use
       if you include bounds and solver can't use will cause failure */
    if(sol_ctx.opt.use_bounds) ierr = SNESVISetVariableBounds(snes, xl, xu);CHKERRQ(ierr);
    /* Solve it */
    ierr = SNESSolve(snes, NULL, x);CHKERRQ(ierr);
    ierr = SNESGetConvergedReason(snes, &cr); CHKERRQ(ierr);
    ierr = SNESGetIterationNumber(snes, &its);CHKERRQ(ierr);
    /* Get the results */
    ierr = VecGetArray(x, &xx);CHKERRQ(ierr);
    /* If duing the perturb test compare to intial values from nl file */
    if(sol_ctx.opt.per_test){
        PetscPrintf(PETSC_COMM_SELF, "Perturb test result, showing differences from initial > 1e-6\n");
        for(i=0;i<n_var;++i) if(xx[i] - X0[i] > 1e-6){
            PetscPrintf(PETSC_COMM_SELF, "v%d - %e, %e, %e\n",i, xx[i], X0[i], xx[i] - X0[i]);
        }
    }
    /* write the AMPL solution file */
    PetscPrintf(PETSC_COMM_SELF, "SNESConvergedReason = %d, in %d iterations\n", cr, its);
    get_snes_sol_message(msg, cr, sol_ctx.asl);
    write_sol(msg, (real*)xx, NULL, NULL);
    ierr = VecRestoreArray(x,&xx);CHKERRQ(ierr);
    ierr = VecDestroy(&xl);CHKERRQ(ierr);
    ierr = VecDestroy(&xu);CHKERRQ(ierr);
    ierr = SNESDestroy(&snes);CHKERRQ(ierr); // ksp and pc are part of this
  } //end snes solve

  /* Should free stuff, but program ending anyway, so what's the point? */
  ASL_free(&asl);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&r);CHKERRQ(ierr);
  ierr = MatDestroy(&J);CHKERRQ(ierr);
  ierr = PetscFinalize(); //All done with PETSc
  return P_EXIT_NORMAL;
}

char **transform_args(int argc, char** argv, int *size){
  /* change the format of the command line arguments from ASL to PETSc*/
  const int max_args=200, max_arg_len=200;
  char argv2[max_args][max_arg_len]; // buffer for reformatted args
  char **argv3;
  int argc2 = argc; //reformatted number of args
  int i=0, j=0, k=0, h=0; // another counter

  for(i=0; i<argc; ++i){
    for(j=0;argv[i][j]!='\0';++j){
      if(argv[i][j]==' '||argv[i][j]=='\t'||argv[i][j]=='\n'){} //strip white spc
      else if(argv[i][j]=='='){ // split on '=''
        h=0;
        ++k;
      }
      else{
        argv2[k][h] = argv[i][j]; //keep anything else
        ++h;
      }
    }
    ++k; //on to next arg
    h=0;
  }
  argc2 = k;
  argv3 = (char**)malloc(sizeof(char*)*argc2);
  for(i=0; i<argc2; ++i){
    argv3[i] = (char*)malloc(sizeof(char)*(strlen(argv2[i])+2));
    memcpy(argv3[i], argv2[i], sizeof(char)*(strlen(argv2[i])+2));
  }
  *size = argc2;
  return argv3;
}

void sol_ctx_init(Solver_ctx *ctx){
  //Initialize some values in the solver context struct
  ctx->dae_map_t=-1;  //ASL index of time variable (-1 for none)
  ctx->n_var_diff=0; //DAE number of differential vars
  ctx->n_var_deriv=0; //DAE number of derivative vars
  ctx->n_var_state=0; //DAE number of state vars
  ctx->n_var_alg=0;  //DAE number of algebraic variables
  ctx->n_ineq=0;  // Number of inequality constraints
  ctx->explicit_time=0; //DAE includes time variable? 1=yes 0=no
  ctx->dof=0; //degrees of freedom
  ctx->opt.ptest=1.0; // method to scale equations
  ctx->opt.eq_scale_method=EQ_SCALE_MAX_GRAD; // method to scale equations
  ctx->opt.var_scale_method=VAR_SCALE_GRAD; // method to scale variables
}
