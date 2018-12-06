#include"petsc.h"

PetscErrorCode FormFunction(SNES snes,Vec x,Vec f, void *ctx){
  Solver_ctx        *sol_ctx = (Solver_ctx*)ctx;
  ASL               *asl=(ASL*)(sol_ctx->asl);
  PetscErrorCode    ierr;
  const PetscScalar *xx;
  PetscScalar       *ff;
  int               err=0;
  int               i;

  /* Get pointers to residual and variable data */
  ierr = VecGetArrayRead(x,&xx);CHKERRQ(ierr);
  ierr = VecGetArray(f,&ff);CHKERRQ(ierr);
  /* Compute function */
  conval((real*)xx, (real*)ff, &err);
  for(i=n_conjac[0];i<n_conjac[1];++i){
    ff[i] -= LUrhs[i];
  }
  /* Restore vectors */
  ierr = VecRestoreArrayRead(x,&xx);CHKERRQ(ierr);
  ierr = VecRestoreArray(f,&ff);CHKERRQ(ierr);
  return 0;
}

PetscErrorCode FormJacobian(SNES snes,Vec x,Mat jac,Mat B,void *ctx){
  Solver_ctx        *sol_ctx = (Solver_ctx*)ctx;
  ASL               *asl=(ASL*)(sol_ctx->asl);
  const PetscScalar *xx;     /* Variable vector*/
  PetscScalar       A[nzc];  /* Temporary storage for Jacobian Calc. */
  PetscErrorCode    ierr;    /* PETSc Error code */
  int               err;     /* ASL Error code */
  unsigned long int i;       /* Constraint index */
  cgrad             *cg;     /* Constraint gradient information */

  /* Compute Jacobian entries and insert into matrix. */
  ierr = VecGetArrayRead(x,&xx);CHKERRQ(ierr);
  jacval((real*)xx,(real*)A,&err);
  ierr = VecRestoreArrayRead(x,&xx);CHKERRQ(ierr);

  for(i=n_conjac[0];i<n_conjac[1]; ++i){ /*i is constraint index */
    cg = Cgrad[i];
    while(cg!=NULL){
      MatSetValue(B, i, cg->varno, A[cg->goff],INSERT_VALUES);
      cg=cg->next;
    }
  }
  /* Assemble matrix */
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  return 0;
}

int get_snes_sol_message(char *msg, SNESConvergedReason term_reason, ASL *asl){
  //solve_result_num is an ASL macro.  I stole the code numbers from ipopt
  //so at least I know pyomo will recognize them.  I havent found documentation
  //yet.
  if(term_reason==SNES_CONVERGED_ITERATING){
    strcpy(msg, "SNES_CONVERGED_ITERATING");
    solve_result_num = 2;}
  else if(term_reason==SNES_CONVERGED_FNORM_ABS){
    strcpy(msg, "SNES_CONVERGED_FNORM_ABS");
    solve_result_num = 2;}
  else if(term_reason==SNES_CONVERGED_FNORM_RELATIVE){
    strcpy(msg, "SNES_CONVERGED_FNORM_RELATIVE");
    solve_result_num = 2;}
  else if(term_reason==SNES_CONVERGED_SNORM_RELATIVE){
    strcpy(msg, "SNES_CONVERGED_SNORM_RELATIVE");
    solve_result_num = 2;}
  else if(term_reason==SNES_CONVERGED_ITS){
    strcpy(msg, "SNES_CONVERGED_ITS");
    solve_result_num = 400;}
  else if(term_reason==SNES_CONVERGED_TR_DELTA){
    strcpy(msg, "SNES_CONVERGED_TR_DELTA");
    solve_result_num = 2;}
  else if(term_reason==SNES_DIVERGED_FUNCTION_DOMAIN){
    strcpy(msg, "SNES_DIVERGED_FUNCTION_DOMAIN");
    solve_result_num = 300;}
  else if(term_reason==SNES_DIVERGED_FUNCTION_COUNT){
    strcpy(msg, "SNES_DIVERGED_FUNCTION_COUNT");
    solve_result_num = 300;}
  else if(term_reason==SNES_DIVERGED_LINEAR_SOLVE){
    strcpy(msg, "SNES_DIVERGED_LINEAR_SOLVE");
    solve_result_num = 300;}
  else if(term_reason==SNES_DIVERGED_FNORM_NAN){
    strcpy(msg, "SNES_DIVERGED_FNORM_NAN");
    solve_result_num = 300;}
  else if(term_reason==SNES_DIVERGED_MAX_IT){
    strcpy(msg, "SNES_DIVERGED_MAX_IT");
    solve_result_num = 400;}
  else if(term_reason==SNES_DIVERGED_LINE_SEARCH){
    strcpy(msg, "SNES_DIVERGED_LINE_SEARCH");
    solve_result_num = 300;}
  else if(term_reason==SNES_DIVERGED_INNER){
    strcpy(msg, "SNES_DIVERGED_INNER");
    solve_result_num = 300;}
  else if(term_reason==SNES_DIVERGED_LOCAL_MIN){
    strcpy(msg, "SNES_DIVERGED_LOCAL_MIN");
    solve_result_num = 200;}
  else if(term_reason==SNES_DIVERGED_DTOL){
    strcpy(msg, "SNES_DIVERGED_DTOL");
    solve_result_num = 300;}
  else{
    strcpy(msg, "Unknown Error");
    solve_result_num = 500;}
  return solve_result_num;
}
