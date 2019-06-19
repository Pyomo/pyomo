#include"petsc.h"

int ScaleEqs(Solver_ctx *sol_ctx){
    EQSCALE_TYPE method = sol_ctx->opt.eq_scale_method;
    ASL *asl = sol_ctx->asl;
    if(method==EQ_SCALE_NONE) return 0;
    else if(method==EQ_SCALE_MAX_GRAD) return ScaleEqs_Largest_Grad(sol_ctx);
    else if(method==EQ_SCALE_USER) return ScaleEqsUser(sol_ctx);
    return 1;
}

int ScaleVars(Solver_ctx *sol_ctx){
    VARSCALE_TYPE method = sol_ctx->opt.var_scale_method;
    ASL *asl = sol_ctx->asl;
    if(method==VAR_SCALE_NONE) return 0;
    else if(method==VAR_SCALE_USER) return ScaleVarsUser(sol_ctx);
    return 1;
}

int ScaleEqs_Largest_Grad(Solver_ctx *sol_ctx){
    ASL *asl = sol_ctx->asl;
    real s;
    int i;
    int err;
    real A[nzc];
    real jv, jv_max, jv_min;
    cgrad *cg;

    //Calculate jacobian values at intial point
    jacval(X0,A,&err);
    //Check the max partial derivatives of residuals
    for(i=n_conjac[0];i<n_conjac[1]; ++i){ /*i is constraint index */
        cg = Cgrad[i];
        jv_max = 0.0;
        while(cg!=NULL){
            jv = fabs(A[cg->goff]);
            if(jv > jv_max) jv_max = jv;
            cg=cg->next;
        }
        if(jv_max > sol_ctx->opt.scale_eq_jac_max)
          s = sol_ctx->opt.scale_eq_jac_max/jv_max;
        else s = 1.0;
        if(s < sol_ctx->opt.scale_eq_fac_min) s = sol_ctx->opt.scale_eq_fac_min;
        conscale(i, s, &err);
    }
    return 0;
}

int ScaleEqsUser(Solver_ctx *sol_ctx){
    ASL *asl = sol_ctx->asl;
    int i = 0, err=0;
    real s;

    if (sol_ctx->scaling_factor_con->u.r == NULL) return 0; //no scaling factors provided
    for(i=0;i<n_con;++i){ //n_con is asl vodoo incase you wonder where it came from
      s = sol_ctx->scaling_factor_con->u.r[i];
      conscale(i, s, &err); //if s is not provided, s is 0, conscale sets those to 1.0
    }
    return err;
}

int ScaleVarsUser(Solver_ctx *sol_ctx){
    //Use scalling factors set in the scaling_factor suffix, for DAEs ignore
    //scaling on the derivatives and use scaling from the differntial vars
    //instead varaibles and there derivatives should be scaled the same
    int i = 0, err=0;
    real s;
    ASL *asl = sol_ctx->asl;
    if (sol_ctx->scaling_factor_var->u.r == NULL) return 0; //no scaling factors provided
    if(sol_ctx->opt.dae_solve){ //dae so match scaling on derivatives
      for(i=0;i<n_var;++i){ //n_var is asl vodoo incase you wonder where it came from
        s = sol_ctx->scaling_factor_var->u.r[i];
        if(sol_ctx->dae_suffix_var->u.i[i] == 2){
          if(s == 0.0) s = 1.0;
          else s = sol_ctx->scaling_factor_var->u.r[sol_ctx->dae_link[i]];
        }
        else if(sol_ctx->dae_suffix_var->u.i[i] == 3) s = 0.0; //can't scale time
        varscale(i, s, &err); //if s is not provided, s is 0, varscale sets those to 1.0
      }
    }
    else{ // no dae so use scale factors given
      for(i=0;i<n_var;++i){ //n_var is asl vodoo incase you wonder where it came from
        s = sol_ctx->scaling_factor_var->u.r[i];
        varscale(i, s, &err); //if s is not provided, s is 0, varscale sets those to 1.0
      }
    }
    return err;
}
