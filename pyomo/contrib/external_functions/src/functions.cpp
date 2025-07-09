/* ___________________________________________________________________________
 * Pyomo: Python Optimization Modeling Objects
 * Copyright (c) 2008-2025
 *  National Technology and Engineering Solutions of Sandia, LLC
 * Under the terms of Contract DE-NA0003525 with National Technology and
 * Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
 * rights in this software.
 * This software is distributed under the 3-clause BSD License.
 * ___________________________________________________________________________
*/
#include <math.h>
#include "funcadd.h"

extern real sinc(arglist *al) {
   real x = al->ra[al->at[0]];
   real y = 0;
   real r = 0.1;
   if(fabs(x) < r){ // use taylor series near 0
      y = 1 
      - x * x / 3.0 / 2.0
      + pow(x, 4) / 5.0 / 4.0 / 3.0 / 2.0
      - pow(x, 6) / 7.0 / 6.0 / 5.0 / 4.0 / 3.0 / 2.0
      + pow(x, 8) / 9.0 / 8.0 / 7.0 / 6.0 / 5.0 / 4.0 / 3.0 / 2.0;
      if (al->derivs!=NULL) {
         al->derivs[0] = - x / 3.0
         + pow(x, 3) / 5.0 / 3.0 / 2.0
         - pow(x, 5) / 7.0 / 5.0 / 4.0 / 3.0 / 2.0
         + pow(x, 7) / 9.0 / 7.0 / 6.0 / 5.0 / 4.0 / 3.0 / 2.0
         - pow(x, 9) /11.0 / 9.0 / 8.0 / 7.0 / 6.0 / 5.0 / 4.0 / 3.0 / 2.0;
      }
      if (al->hes!=NULL) {
         al->hes[0] = - 1 / 3.0
         + pow(x, 2) / 5.0 / 2.0
         - pow(x, 4) / 7.0 / 4.0 / 3.0 / 2.0
         + pow(x, 6) / 9.0 / 6.0 / 5.0 / 4.0 / 3.0 / 2.0
         - pow(x, 8) /11.0 / 8.0 / 7.0 / 6.0 / 5.0 / 4.0 / 3.0 / 2.0;
      }
   }
   else{  // away from 0 use sin(x) / x
      y = sin(x) / x;
      if (al->derivs!=NULL) {
         al->derivs[0] = cos(x) / x - sin(x) / x / x;
      }
      if (al->hes!=NULL) {
         al->hes[0] = -sin(x) / x - 2 * cos(x) / x / x
         + 2 * sin(x) / x / x / x;
      }
   }
   return y;
}

extern real sgnsqr(arglist *al) {
   real x = al->ra[al->at[0]];
   real y = copysign(x * x, x);

   // Compute the first derivative, if requested.
   if (al->derivs!=NULL) {
      al->derivs[0] =  2 * fabs(x);
   }

   // Compute the second derivative, if requested.
   if (al->hes!=NULL) {
      al->hes[0] = copysign(2, x);
   }
   return y;
}

extern real sgnsqr_c4(arglist *al) {
   real x = al->ra[al->at[0]];
   const real c[] = {
      0.0,
      0.0273437500000006,
      3.1086244689504383e-15,
      10.937499999999813,
      6.821210263296962e-13,
      -546.8749999999808,
      -1.2114878257930237e-10,
      21874.99999999906,
      7.251377026937331e-09,
      -390624.99999998155,
      -1.582935171828823e-07,
   };
   const real r = 0.1;
   real y = 0;
   unsigned char i = 0;

   if(fabs(x) < r){
      for(i=0; i<11; ++i){
         y += c[i] * pow(x, i);
      }
      if (al->derivs!=NULL) {
         al->derivs[0] = 0;
         for(i=1; i<11; ++i){
            al->derivs[0] += c[i] * i * pow(x, i - 1);
         }
      }
      if (al->hes!=NULL) {
         al->hes[0] = 0;
         for(i=2; i<11; ++i){
            al->hes[0] += c[i] * i * (i - 1) * pow(x, i - 2);
         }
      }
   }
   else{
      y = copysign(x * x, x);
      if (al->derivs!=NULL) {
         al->derivs[0] =  2 * fabs(x);
      }
      if (al->hes!=NULL) {
         al->hes[0] = copysign(2, x);
      }
   }

   return y;
}


extern real sgnsqrt_c4(arglist *al) {
   real x = al->ra[al->at[0]];
   const real c[] = {
      0.0,
      5.1186281462197964,
      -5.115907697472721e-13,
      -409.490251697574,
      -2.4253192047278085e-12,
      34124.18764146428,
      3.827043323720301e-09,
      -1574962.506529117,
      -2.7353306479626024e-07,
      30109577.330703676,
      6.383782391594595e-06,
   };
   const real r = 0.1;
   real y = 0;
   unsigned char i = 0;

   if(fabs(x) < r){
      for(i=0; i<11; ++i){
         y += c[i] * pow(x, i);
      }
      if (al->derivs!=NULL) {
         al->derivs[0] = 0;
         for(i=1; i<11; ++i){
            al->derivs[0] += c[i] * i * pow(x, i - 1);
         }
      }
      if (al->hes!=NULL) {
         al->hes[0] = 0;
         for(i=2; i<11; ++i){
            al->hes[0] += c[i] * i * (i - 1) * pow(x, i - 2);
         }
      }
   }
   else{
      y = copysign(sqrt(fabs(x)), x);
      if (al->derivs!=NULL) {
         al->derivs[0] =  1.0 / 2.0 / sqrt(fabs(x));
      }
      if (al->hes!=NULL) {
         al->hes[0] = -copysign(1.0/4.0*pow(fabs(x), -1.5), x);
      }
   }

   return y;
}


// Register external functions defined in this library with the ASL
void funcadd(AmplExports *ae){
   // Arguments for addfunc (this is not fully detailed; see funcadd.h)
   // 1) Name of function in AMPL
   // 2) Function pointer to C function
   // 3) see FUNCADD_TYPE enum in funcadd.h
   // 4) Number of arguments
   //    >=0 indicates a fixed number of arguments
   //    < 0 indicates a variable length list (requiring at least -(n+1)
   //        arguments)
   // 5) Void pointer to function info
   addfunc(
      "sinc", 
      (rfunc)sinc,
      FUNCADD_REAL_VALUED, 
      1, 
      NULL
   );
   addfunc(
      "sgnsqr", 
      (rfunc)sgnsqr,
      FUNCADD_REAL_VALUED, 
      1, 
      NULL
   );
   addfunc(
      "sgnsqr_c4", 
      (rfunc)sgnsqr_c4,
      FUNCADD_REAL_VALUED, 
      1, 
      NULL
   );
   addfunc(
      "sgnsqrt_c4", 
      (rfunc)sgnsqrt_c4,
      FUNCADD_REAL_VALUED, 
      1, 
      NULL
   );
}
