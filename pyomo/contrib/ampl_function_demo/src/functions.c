#include "functions.h"
#include <math.h>
#include <string.h>

#undef printf

void funcadd(AmplExports *ae){
    /* Arguments for addfunc (this is not fully detailed see funcadd.h)
     * 1) Name of function in AMPL
     * 2) Function pointer to C function
     * 3) see FUNCADD_TYPE enum in funcadd.h
     * 4) Number of arguments (the -1 is variable arg list length)
     * 5) Void pointer to function info */
    addfunc("cbrt", (rfunc)scbrt, FUNCADD_REAL_VALUED, 1, NULL);
    addfunc("testing_only", (rfunc)testing_only, FUNCADD_REAL_VALUED|FUNCADD_STRING_ARGS, -1, NULL);
}


extern real testing_only(arglist *al){
    const char* sarg = al->sa[-(al->at[0]+1)];
    int i, j;
    real s=0;
    char inv=0;

    if(!strcmp("inv", sarg)){
      inv = 1;
    }
    if(!inv){
      for(i=0;i<al->nr;++i){
        s += al->ra[i];
      }
    }
    else{
      for(i=0;i<al->nr;++i){
        s += 1.0/al->ra[i];
      }
    }

    if(al->derivs!=NULL){
      for(i=0;i<al->nr;++i){
        if(!inv){
          al->derivs[i] = 1.0;
        }
        else{
          al->derivs[i] = -1.0/(al->ra[i]*al->ra[i]);
        }
      }
      if(al->hes!=NULL){
        for(i=0;i<(al->nr*al->nr - al->nr)/2 + al->nr;++i){
          al->hes[i] = 0;
        }
        if(inv){
          j = -1;
          for(i=0;i<al->nr;++i){
            j = j + (i+1);
              al->hes[j] = 2.0/(al->ra[i]*al->ra[i]*al->ra[i]);
          }
        }
      }
    }
    return s;
}


extern real scbrt(arglist *al){
    // al is the argument list data structure
    // al->ra is an array of real arguments
    // al->at is an array of of argument positions
    // The reason for using al->at for the position is
    // that there could also be integer or string
    // arguments
    real x = al->ra[al->at[0]];

    //al->derivs is a pointer to an array of first derivatives
    //of the function with respect to each real arg if
    //derivs is NULL solver isn't requesting derivatives.
    if(al->derivs!=NULL){
      if(fabs(x) < 6e-9) al->derivs[0] = 1e5;
      else al->derivs[0] = pow(cbrt(x), -2.0)/3.0;
      //al->hes is a pointer the Hessian matrix if NULL second
      //derivatives are not requested.  This function takes a
      //single argument, but the Hessian form in general is
      //the upper triangle in column-major form like
      // Args Index  0 1 2 3 ...
      //          0  0 1 3 6
      //          1    2 4 7
      //          2      5 8
      //          3        9
      //        ...
      if(al->hes!=NULL){
        al->hes[0] = -2.0*pow(cbrt(x), -5.0)/9.0;
      }
    }
    return cbrt(x);
}
