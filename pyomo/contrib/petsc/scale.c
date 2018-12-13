#include"petsc.h"

int ScaleEqs(EQSCALE_TYPE method, ASL *asl){
    if(method==EQ_SCALE_NONE) return 0;
    else if(method==EQ_SCALE_MAX_GRAD) return ScaleEqs_Largest_Grad(asl);
    return 1;
}

int ScaleVars(VARSCALE_TYPE method, ASL *asl){
    if(method==VAR_SCALE_NONE) return 0;
    else if(method==VAR_SCALE_GRAD) return ScaleVarsGrad(asl);
    return 1;
}

int ScaleEqs_Largest_Grad(ASL *asl){
    real s[n_con];
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
        jv_min = fabs(A[cg->goff]);
        while(cg!=NULL){
            jv = fabs(A[cg->goff]);
            if(jv > jv_max) jv_max = jv;
            if(jv < jv_min) jv_min = jv;
            cg=cg->next;
        }
        if(jv_max > 100.0) s[i] = 100.0/jv_max;
        else s[i] = 1.0;
        if(s[i] < 1e-6) s[i] = 1e-6;
        conscale(i, s[i], &err);
    }
    return 0;
}

int ScaleVarsGrad(ASL *asl){
    real s[n_var];
    int i;
    int err;
    real A[nzc];
    real jv, jv_min[n_var], jv_max[n_var];
    cgrad *cg;

    jacval(X0,A,&err);
    for(i=0;i<n_var;++i){
        s[i] = 1.0;
        jv_min[i] = 10000.0;
        jv_max[i] = 0;
    }
    for(i=n_conjac[0];i<n_conjac[1]; ++i){ /*i is constraint index */
        cg = Cgrad[i];
        while(cg!=NULL){
            jv = fabs(A[cg->goff]);
            if(jv > jv_max[cg->varno]) jv_max[cg->varno] = jv;
            if(jv < jv_min[cg->varno]) jv_min[cg->varno] = jv;
            cg=cg->next;
        }
    }
    /* Now want to scale so that jv_max <= 100 and jv_min is as close
     * to 0.01 as possible without pushing jv_max too high.*/
    for(i=0;i<n_var;++i) if(jv_max[i] > 100){
      // Smallest jv is too small attempt to scale
      s[i] = 100.0/jv_min[i];
      if(s[i] < 1e-8) s[i] = 1e-8;
      varscale(i, s[i], &err);
    }
    return 0;
}
