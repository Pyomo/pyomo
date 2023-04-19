/* ___________________________________________________________________________
 * Pyomo: Python Optimization Modeling Objects
 * Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
 * Under the terms of Contract DE-NA0003525 with National Technology and
 * Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
 * rights in this software.
 * This software is distributed under the 3-clause BSD License.
 * ___________________________________________________________________________
 *
 * Portions of this module were originally developed as part of the
 * IDAES/idaes-ext package. The Institute for the Design of Advanced
 * Energy Systems Integrated Platform Framework (IDAES IP) was produced
 * under the DOE Institute for the Design of Advanced Energy Systems
 * (IDAES), and is copyright (c) 2018-2021 by the software owners: The
 * Regents of the University of California, through Lawrence Berkeley
 * National Laboratory, National Technology & Engineering Solutions of
 * Sandia, LLC, Carnegie Mellon University, West Virginia University
 * Research Corporation, et al.  All rights reserved.
 * ___________________________________________________________________________
 */

/* This module provides an example for developing an ASL-compatible
 * external (compiled) function for Pyomo.  For more information, see
 * the "User-defined functions" section of the "Hooking your solver to
 * AMPL" report (https://ampl.com/REFS/HOOKING/index.html).
 *
 * The module defines two functions:
 *   - a demo function demonstrating the use of string arguments and
 *     variable-length argument lists
 *   - a "safe" cube root function that avoids infinite derivatives at 0
 *
 * The functaions are registered with the ASL using the funcadd()
 * function at the end of this module.
 */

// include the ASL interface header
#include "funcadd.h"

#include <math.h>
#include <string.h>

/* A "demo" function, demonstrating the use of variable-length
 * argument lists and passing string arguments to external functions
 *
 * real demo_only(<mode>, [val, ...])
 *    mode (string): if 'inv' compute the sum of the inverses of the
 *        values; otherwise compute the sum of the values.
 *    val (real): 0 or more numeric values
 */
extern real demo_function(arglist *al) {
   /* AMPL passes the al->n arguments from the user (al->nr of which are
    * numeric) through two vectors:
    *     al->ra holds the real arguments
    *     al->sa holds the string arguments
    *
    * The mapping from the user's original arguments to these two
    * vectors is specified by al->at:
    *    al->at[i] >=0 specifies that argument i is at al->ra[al->at[i]]
    *    al->at[i] < 0 specifies that argument i is at al->sa[-(1 + al->at[i])]
    */

   // the first argument is the string "mode"
   const char* mode = al->sa[-(al->at[0]+1)];
   int i, j;
   real s = 0;
   char inv = 0;

   if (!strcmp("inv", mode)) {
      inv = 1;
   }

   // Compute the function return value
   if (!inv) {
      for(i=0; i < al->nr; ++i) {
         s += al->ra[i];
      }
   } else {
      for(i=0; i < al->nr; ++i) {
         s += 1.0/al->ra[i];
      }
   }

   /* If al->derivs is non-NULL, then the ASL is requesting the first
    * partial derivative with respect to the input arguments, al->ra
    */
   if (al->derivs != NULL) {
      for (i=0; i<al->nr; ++i) {
         if (!inv) {
            al->derivs[i] = 1.0;
         } else {
            al->derivs[i] = -1.0/(al->ra[i]*al->ra[i]);
         }
      }
   }

   /* If al->hes is non-NULL, then the ASL is also requesting the upper
    * triangle of the Hessian matrix (second partial derivative).
    *
    * The Hessian is stored column-wise in a linear vector:
    *    Args Index  0 1 2 3 ...
    *             0  0 1 3 6
    *             1    2 4 7
    *             2      5 8
    *             3        9
    *           ...
    *
    * That is, al->hes[i + j*(j+1)/2] holds the the second partial
    * with respect to al->ra[i] and al->ra[j].
    */
   if (al->hes != NULL) {
      for (i=0; i < (al->nr*al->nr - al->nr)/2 + al->nr; ++i) {
         al->hes[i] = 0;
      }
      if (inv) {
         j = -1;
         for (i=0; i<al->nr; ++i) {
            j = j + (i+1);
            al->hes[j] = 2.0/(al->ra[i]*al->ra[i]*al->ra[i]);
         }
      }
   }
   return s;
}


/* A "safe" cube root function that avoids infinite derivatives at 0.
 */
extern real scbrt(arglist *al){
   /* This function is registered as only taking a single numeric
    * argument, so the sole argument is stored at al->ra[0].  However,
    * for consistency with more general external functions, we will
    * still show the indirection through the al->at mapping:
    */
   real x = al->ra[al->at[0]];

   real safe_x = x;
   if (fabs(x) < 6e-9) {
      safe_x = copysign(6e-9, x);
   }

   // Note that the reported first and second derivatives are not
   // technically correct near 0; that is:
   //    - al->derivs[0] != derivative(scbrt(x))
   //    - al->hes[0] != derivative(al->derivs[0])
   // when x is near 0

   // Compute the first derivative, if requested.
   if (al->derivs!=NULL) {
      al->derivs[0] = pow(cbrt(safe_x), -2.0)/3.0;
   }

   // Compute the second derivative, if requested.
   if (al->hes!=NULL) {
      al->hes[0] = -2.0*pow(cbrt(safe_x), -5.0)/9.0;
   }

   return cbrt(x);
}


// Register external functions defined in this library with the ASL
void funcadd(AmplExports *ae){
    /* Arguments for addfunc (this is not fully detailed; see funcadd.h)
     * 1) Name of function in AMPL
     * 2) Function pointer to C function
     * 3) see FUNCADD_TYPE enum in funcadd.h
     * 4) Number of arguments
     *    >=0 indicates a fixed number of arguments
     *    < 0 indicates a variable length list (requiring at least -(n+1)
     *        arguments)
     * 5) Void pointer to function info
     */
    addfunc("safe_cbrt", (rfunc)scbrt,
            FUNCADD_REAL_VALUED, 1, NULL);
    addfunc("demo_function", (rfunc)demo_function,
            FUNCADD_REAL_VALUED|FUNCADD_STRING_ARGS, -2, NULL);
}
