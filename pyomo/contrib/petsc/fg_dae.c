#include"petsc.h"

PetscErrorCode FormDAEFunction(TS ts, PetscReal t, Vec x, Vec xdot, Vec f, void *ctx){
  Solver_ctx        *sol_ctx = (Solver_ctx*)ctx;
  ASL               *asl=(ASL*)(sol_ctx->asl);
  PetscErrorCode    ierr;
  const PetscScalar *xx, *xxdot;     /* Variable vector*/
  PetscScalar       *ff;
  int               err=0;
  int               i;
  real              x_asl[n_var];

  /* Take values from petsc vectors and put into ASL vector */
  if(sol_ctx->dae_map_t > 0) x_asl[sol_ctx->dae_map_t] = t;
  ierr = VecGetArrayRead(x, &xx); CHKERRQ(ierr);
  ierr = VecGetArrayRead(xdot, &xxdot); CHKERRQ(ierr);
  for(i=0;i<sol_ctx->n_var_state;++i){
    x_asl[sol_ctx->dae_map_x[i]] = xx[i];
    if(sol_ctx->dae_map_xdot[i] >= 0) x_asl[sol_ctx->dae_map_xdot[i]] = xxdot[i];
  }
  ierr = VecRestoreArrayRead(x,&xx);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xdot,&xxdot);CHKERRQ(ierr);
  /* Compute */
  ierr = VecGetArray(f,&ff);CHKERRQ(ierr);
  conval(x_asl, (real*)ff, &err);
  for(i=0;i<sol_ctx->n_var_state;++i) ff[i] -= LUrhs[i];
  ierr = VecRestoreArray(f,&ff);CHKERRQ(ierr);
  return 0;
}

PetscErrorCode FormDAEJacobian(
  TS ts, PetscReal t, Vec x, Vec xdot, PetscReal sig, Mat jac, Mat B,void *ctx){
  Solver_ctx        *sol_ctx = (Solver_ctx*)ctx;
  ASL               *asl=(ASL*)(sol_ctx->asl);
  const PetscScalar *xx, *xxdot;     /* Variable vector*/
  PetscScalar       A[nzc];  /* Temporary storage for Jacobian Calc. */
  PetscErrorCode    ierr;    /* PETSc Error code */
  int               err;     /* ASL Error code */
  unsigned long int i, j;    /* Constraint index, var */
  cgrad             *cg;     /* Constraint gradient information */
  real              x_asl[n_var];

  /* Compute Jacobian entries */
  /* Take values from petsc vectors and put into ASL vector */
  if(sol_ctx->dae_map_t > 0) x_asl[sol_ctx->dae_map_t] = t;
  ierr = VecGetArrayRead(x, &xx); CHKERRQ(ierr);
  ierr = VecGetArrayRead(xdot, &xxdot); CHKERRQ(ierr);
  for(i=0;i<sol_ctx->n_var_state;++i){
    x_asl[sol_ctx->dae_map_x[i]] = xx[i];
    if(sol_ctx->dae_map_xdot[i] >= 0) x_asl[sol_ctx->dae_map_xdot[i]] = xx[i];
  }
  ierr = VecRestoreArrayRead(x,&xx);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xdot,&xxdot);CHKERRQ(ierr);
  jacval((real*)x_asl, (real*)A, &err);
  /* The ASL isn't setup for DAEs, but not too hard to make the right Jacobian
    From asl we have a Jacobian where t, x, and xdot are all treated the same,
    PETSc want J = sigma*F_xdot(t, x, xdot) + F_x(t, x, xdot), so we need to
    eliminate the t column and combine the x and xdot columns appropriatly */
  ierr = MatZeroEntries(B); CHKERRQ(ierr); //since I'm going to add things
  for(i=n_conjac[0];i<n_conjac[1]; ++i){ /*i is constraint index */
    cg = Cgrad[i];
    while(cg!=NULL){
      j = cg->varno;
      if(sol_ctx->dae_suffix_var->u.i[j]==2){
        MatSetValue(B, i, sol_ctx->dae_map_back[j], sig*A[cg->goff], ADD_VALUES);}
      else if(sol_ctx->dae_suffix_var->u.i[j]==3); //drop t
      else MatSetValue(B, i, sol_ctx->dae_map_back[j], A[cg->goff], ADD_VALUES);
      cg=cg->next;
    }
  }
  /* Assemble matrix */
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  return 0;
}

void dae_var_map(Solver_ctx *sol_ctx){
  int i=0, j=0, cx;
  ASL *asl=(ASL*)(sol_ctx->asl); //need this for ASL macros

  /* create a mapping PETSc <-> asl */
  sol_ctx->dae_link = (int*)malloc(n_var*sizeof(int));
  for(i=0;i<n_var;++i) sol_ctx->dae_link[i] = -1;
  for(i=0;i<n_var;++i){
    if(sol_ctx->dae_suffix_var->u.i[i]==2){ //a derivative var
      for(j=0;j<n_var;++j){
        if(sol_ctx->dae_link_var->u.i[i] == sol_ctx->dae_link_var->u.i[j]){
          sol_ctx->dae_link[i] = j;
          sol_ctx->dae_link[j] = i;
        }
      } //end for j
    } //end if suffix == 2
  } //end for i
  sol_ctx->dae_map_x = (int*)malloc(sol_ctx->n_var_state*sizeof(int));
  sol_ctx->dae_map_xdot = (int*)malloc(sol_ctx->n_var_state*sizeof(int));
  sol_ctx->dae_map_back = (int*)malloc(n_var*sizeof(int));
  for(i=0;i<sol_ctx->n_var_state;++i) sol_ctx->dae_map_xdot[i] = -1;
  cx = 0;
  for(i=0;i<n_var;++i){
    if (sol_ctx->dae_suffix_var->u.i[i]==3){ //time variable (0 or 1 of these)
      sol_ctx->dae_map_t = i;
    }
    else if (sol_ctx->dae_suffix_var->u.i[i]==2);
    else{ //agebraic and differential variables
      sol_ctx->dae_map_x[cx] = i;
      sol_ctx->dae_map_back[i] = cx;
      ++cx;
    }
  } //end for i
  for(i=0;i<n_var;++i){
    // there is a derivative in xdot for all state vars, but in a dae some are
    // just going to be zero in the function and jacobian calcualtions.
    if (sol_ctx->dae_suffix_var->u.i[i] == 1){
      j = sol_ctx->dae_link[i]; // index of associated derivative
      sol_ctx->dae_map_xdot[sol_ctx->dae_map_back[i]] = j;
      sol_ctx->dae_map_back[j] = sol_ctx->dae_map_back[i];
    }//end if suffix == 1
  }// end for i
}

void get_dae_info(Solver_ctx *sol_ctx){
  int i=0;
  ASL *asl=(ASL*)(sol_ctx->asl); //need this for ASL macros

  sol_ctx->dae_suffix_var = suf_get("dae_suffix", ASL_Sufkind_var);
  sol_ctx->dae_link_var = suf_get("dae_link", ASL_Sufkind_var);
  for(i=0; i<n_var; ++i){
    if(sol_ctx->dae_suffix_var->u.i[i]==1) ++sol_ctx->n_var_diff;
    else if(sol_ctx->dae_suffix_var->u.i[i]==2) ++sol_ctx->n_var_deriv;
    else if(sol_ctx->dae_suffix_var->u.i[i]==3) ++sol_ctx->explicit_time;
  }
  sol_ctx->n_var_state = n_var - sol_ctx->n_var_deriv - sol_ctx->explicit_time;
  sol_ctx->n_var_alg = sol_ctx->n_var_state - sol_ctx->n_var_diff;
  if(sol_ctx->explicit_time>1){
    PetscPrintf(PETSC_COMM_SELF, "ERROR: DAE: Multiple time variable (allowed 1 at most)");
    ASL_free(&(sol_ctx->asl));
    exit(P_EXIT_MULTIPLE_TIME);
  }
  if(sol_ctx->dof != sol_ctx->n_var_deriv + sol_ctx->explicit_time){
    PetscPrintf(PETSC_COMM_SELF, "ERROR: DAE: DOF != number of derivative vars");
    ASL_free(&(sol_ctx->asl));
    exit(P_EXIT_DOF_DAE);
  }
  if(sol_ctx->n_var_diff != sol_ctx->n_var_deriv){
    PetscPrintf(PETSC_COMM_SELF, "ERROR: DAE: number of differential vars != number of derivatives");
    ASL_free(&(sol_ctx->asl));
    exit(P_EXIT_VAR_DAE_MIS);
  }
}
