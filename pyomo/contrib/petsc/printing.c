#include"petsc.h"

void print_commandline(const char* msg, int argc, char **argv){
  /* print command line arguments */
  int i=0;
  PetscPrintf(PETSC_COMM_SELF, msg);
  for(i=0; i<argc; ++i) PetscPrintf(PETSC_COMM_SELF, "%s ", argv[i]);
  PetscPrintf(PETSC_COMM_SELF, "\n");
}

void print_init_diagnostic(Solver_ctx *sol_ctx){
  if(sol_ctx->opt.show_jac){
    print_jac_asl(sol_ctx->asl, 100, 1e-6);
  }
  if(sol_ctx->opt.show_init){
    print_x_asl(sol_ctx->asl);
  }
  if(sol_ctx->opt.show_scale_factors){
    print_var_scale_factors_asl(sol_ctx->asl);
    print_con_scale_factors_asl(sol_ctx->asl);
  }
}

void print_x_asl(ASL *asl){
  int i=0;
  char color_code[20];
  PetscPrintf(PETSC_COMM_SELF, "Initial Values (scaled)\n");
  for (i=0;i<n_var;++i){
     if(X0[i] > Uvx[i] || X0[i] < LUv[i]){
       memcpy(color_code, COLOR_RED, 15*sizeof(char));
     }
     else memcpy(color_code, COLOR_NORMAL, 15*sizeof(char));
     PetscPrintf(PETSC_COMM_SELF, "%sv%d: %e <= %e <= %e%s\n",
     color_code, i, LUv[i], X0[i], Uvx[i], COLOR_NORMAL);
  }
}

void print_var_scale_factors_asl(ASL *asl){
  int i;
  if(asl->i.vscale!=NULL){
    for(i=0;i<n_var;++i) printf("scale Factor v%d: %f\n", i, asl->i.vscale[i]);}
}

void print_con_scale_factors_asl(ASL *asl){
  int i;
  if(asl->i.cscale!=NULL){
    for(i=0;i<n_con;++i) printf("scale factor c%d: %f\n", i, asl->i.cscale[i]);}
}

void print_jac_asl(ASL *asl, real u, real l){
  /* Print sparse Jacobian  highlight elements over u or under l*/
  cgrad          *cg;  /* sparse jacobian elements*/
  real           *Jac; /* ASL test Jacobian */
  char           color_code[20] = COLOR_NORMAL;
  int            err;  /* Error code  from ASL fulctions */
  int            i=0;

  Jac = (real *)Malloc(nzc*sizeof(real)); /* Jacobian space */
  jacval(X0, Jac, &err); /*calculate jacobian */
  PetscPrintf(PETSC_COMM_SELF, "Computed Jacobian, err = %d\n", err);
  PetscPrintf(PETSC_COMM_SELF, "Computed Jacobian values (scaled):\n");
  for(i=n_conjac[0];i<n_conjac[1]; ++i){ /*i is constraint index */
    cg = Cgrad[i];
    PetscPrintf(PETSC_COMM_SELF, "c%d", i);
    while(cg!=NULL){
      if(fabs(Jac[cg->goff]) > u || fabs(Jac[cg->goff]) < l){
        memcpy(color_code, COLOR_RED, 15*sizeof(char));}
      else memcpy(color_code, COLOR_NORMAL, 15*sizeof(char));
      PetscPrintf(PETSC_COMM_SELF, " %sv%d(%e)%s", color_code,cg->varno, Jac[cg->goff], COLOR_NORMAL);
      cg=cg->next;
    }
    PetscPrintf(PETSC_COMM_SELF, "\n");
  }
  free(Jac);
}
