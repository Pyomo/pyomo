/**___________________________________________________________________________
 *
 * Pyomo: Python Optimization Modeling Objects
 * Copyright (c) 2008-2025
 * National Technology and Engineering Solutions of Sandia, LLC
 * Under the terms of Contract DE-NA0003525 with National Technology and
 * Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
 * rights in this software.
 * This software is distributed under the 3-clause BSD License.
 * ___________________________________________________________________________
**/

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>

// This would normally be in a header file, but as we do not need one,
// we will explicitly include it here.
#if defined(_WIN32) || defined(_WIN64)
#  if defined(BUILDING_PYNUMERO_MA27)
#    define PYNUMERO_HSL_EXPORT __declspec(dllexport)
#  else
#    define PYNUMERO_HSL_EXPORT __declspec(dllimport)
#  endif
#else
#  define PYNUMERO_HSL_EXPORT
#endif

// Forward declaration of MA27 fortran routines
extern "C" {
   void ma27id_(int* ICNTL, double* CNTL);
   void ma27ad_(int *N, int *NZ, int *IRN, int* ICN,
                int *IW, int* LIW, int* IKEEP, int *IW1,
                int* NSTEPS, int* IFLAG, int* ICNTL,
                double* CNTL, int *INFO, double* OPS);
   void ma27bd_(int *N, int *NZ, int *IRN, int* ICN,
                double* A, int* LA, int* IW, int* LIW,
                int* IKEEP, int* NSTEPS, int* MAXFRT,
                int* IW1, int* ICNTL, double* CNTL,
                int* INFO);
   void ma27cd_(int *N, double* A, int* LA, int* IW,
                int* LIW, double* W, int* MAXFRT,
                double* RHS, int* IW1, int* NSTEPS,
                int* ICNTL, int* INFO);
} // extern "C"

void abort_bad_memory(int status) {
   printf("Bad memory allocation in MA27 C interface. Aborting.");
   exit(status);
}


struct MA27_struct {
   // Constructor: set defaults, initialize cached arrays to NULL
   MA27_struct():
      LA(0),
      LIW_a(0),
      LIW_b(0),
      NSTEPS(0),
      IFLAG(0),
      MAXFRT(0),
      IW_factor(1.2),
      A_factor(2.0),
      OPS(0),
      IW_a(NULL),
      IW_b(NULL),
      IKEEP(NULL),
      A(NULL)
   {
      ma27id_(this->ICNTL, this->CNTL);
   }
   // Destructor: delete all cached arrays
   virtual ~MA27_struct() {
      if ( this->A ) {
         delete[] this->A;
      }
      if ( this->IW_a ) {
         delete[] this->IW_a;
      }
      if ( this->IW_b ) {
         delete[] this->IW_b;
      }
      if ( this->IKEEP ) {
         delete[] this->IKEEP;
      }
   }

   int LA, LIW_a, LIW_b, NSTEPS, IFLAG, MAXFRT;
   double IW_factor, A_factor, OPS;
   int* IW_a;
   int* IW_b;
   // Use different arrays for IW that is sent to MA27A and that sent to
   // MA27B because IW must be discarded after MA27A but kept after MA27B.
   // If these arrays are the same, and a symbolic factorization is performed
   // after a numeric factorization (e.g. on a new matrix), user-defined
   // and MA27B-defined allocations of IW can be conflated.
   int* IKEEP;
   double* A;
   int ICNTL[30], INFO[20];
   double CNTL[5];
};

extern "C" {

   PYNUMERO_HSL_EXPORT
   MA27_struct* new_MA27_struct(void) {
      MA27_struct* ma27 = new MA27_struct;
      if (ma27 == NULL) { abort_bad_memory(1); }
      // Return pointer to ma27 that Python program can pass to other
      // functions in this code
      return ma27;
   }


   PYNUMERO_HSL_EXPORT
   void free_MA27_struct(MA27_struct* ma27) {
      delete ma27;
   }

   // Functions for setting/accessing INFO/CNTL arrays:
   PYNUMERO_HSL_EXPORT
   void set_icntl(MA27_struct* ma27, int i, int val) {
      ma27->ICNTL[i] = val;
   }

   PYNUMERO_HSL_EXPORT
   int get_icntl(MA27_struct* ma27, int i) {
      return ma27->ICNTL[i];
   }

   PYNUMERO_HSL_EXPORT
   void set_cntl(MA27_struct* ma27, int i, double val) {
      ma27->CNTL[i] = val;
   }

   PYNUMERO_HSL_EXPORT
   double get_cntl(MA27_struct* ma27, int i) {
      return ma27->CNTL[i];
   }

   PYNUMERO_HSL_EXPORT
   int get_info(MA27_struct* ma27, int i) {
      return ma27->INFO[i];
   }

   // Functions for allocating WORK/FACT arrays:
   PYNUMERO_HSL_EXPORT
   void alloc_iw_a(MA27_struct* ma27, int l) {
      if ( ma27->IW_a ) {
         delete[] ma27->IW_a;
      }
      ma27->LIW_a = l;
      ma27->IW_a = new int[l];
      if (ma27->IW_a == NULL) { abort_bad_memory(1); }
   }

   PYNUMERO_HSL_EXPORT
   void alloc_iw_b(MA27_struct* ma27, int l) {
      if ( ma27->IW_b ) {
         delete[] ma27->IW_b;
      }
      ma27->LIW_b = l;
      ma27->IW_b = new int[l];
      if (ma27->IW_b == NULL) { abort_bad_memory(1); }
   }

   PYNUMERO_HSL_EXPORT
   void alloc_a(MA27_struct* ma27, int l) {
      if ( ma27->A ) {
         delete[] ma27->A;
      }
      ma27->LA = l;
      ma27->A = new double[l];
      if (ma27->A == NULL) { abort_bad_memory(1); }
   }

   PYNUMERO_HSL_EXPORT
   void do_symbolic_factorization(MA27_struct* ma27, int N, int NZ,
                                  int* IRN, int* ICN) {
      // Arrays, presumably supplied from Python, are assumed to have base-
      // zero indices. Convert to base-one before sending to Fortran.
      for (int i=0; i<NZ; i++) {
         IRN[i] = IRN[i] + 1;
         ICN[i] = ICN[i] + 1;
      }

      if ( ! ma27->IW_a ) {
         int min_size = 2*NZ + 3*N + 1;
         int size = (int)(ma27->IW_factor*min_size);
         alloc_iw_a(ma27, size);
      }

      if ( ma27->IKEEP ) {
         delete[] ma27->IKEEP;
      }
      ma27->IKEEP = new int[3*N];
      if (ma27->IKEEP == NULL) { abort_bad_memory(1); }
      int* IW1 = new int[2*N];
      if (IW1 == NULL) { abort_bad_memory(1); }

      ma27ad_(&N,
              &NZ,
              IRN,
              ICN,
              ma27->IW_a,
              &(ma27->LIW_a),
              ma27->IKEEP,
              IW1,
              &(ma27->NSTEPS),
              &(ma27->IFLAG),
              ma27->ICNTL,
              ma27->CNTL,
              ma27->INFO,
              &(ma27->OPS));

      delete[] IW1;
      delete[] ma27->IW_a;
      ma27->IW_a = NULL;
   }

   PYNUMERO_HSL_EXPORT
   void do_numeric_factorization(MA27_struct* ma27, int N, int NZ,
                                 int* IRN, int* ICN, double* A) {

      // Convert indices to base-one for Fortran
      for (int i=0; i<NZ; i++) {
         IRN[i] = IRN[i] + 1;
         ICN[i] = ICN[i] + 1;
      }

      // Get memory estimates from INFO, allocate A and IW
      if ( ! ma27->A ) {
         int info5 = ma27->INFO[5-1];
         int size = (int)(ma27->A_factor*info5);
         alloc_a(ma27, size);
         // A is now allocated
      }
      // Regardless of ma27->A's previous allocation status, copy values from A.
      memcpy(ma27->A, A, NZ*sizeof(double));

      if ( ! ma27->IW_b ) {
         int info6 = ma27->INFO[6-1];
         int size = (int)(ma27->IW_factor*info6);
         alloc_iw_b(ma27, size);
      }

      int* IW1 = new int[N];
      if (IW1 == NULL) { abort_bad_memory(1); }

      ma27bd_(&N,
              &NZ,
              IRN,
              ICN,
              ma27->A,
              &(ma27->LA),
              ma27->IW_b,
              &(ma27->LIW_b),
              ma27->IKEEP,
              &(ma27->NSTEPS),
              &(ma27->MAXFRT),
              IW1,
              ma27->ICNTL,
              ma27->CNTL,
              ma27->INFO);

      delete[] IW1;
   }

   PYNUMERO_HSL_EXPORT
   void do_backsolve(MA27_struct* ma27, int N, double* RHS) {

      double* W = new double[ma27->MAXFRT];
      if (W == NULL) { abort_bad_memory(1); }
      int* IW1 = new int[ma27->NSTEPS];
      if (IW1 == NULL) { abort_bad_memory(1); }

      ma27cd_(
              &N,
              ma27->A,
              &(ma27->LA),
              ma27->IW_b,
              &(ma27->LIW_b),
              W,
              &(ma27->MAXFRT),
              RHS,
              IW1,
              &(ma27->NSTEPS),
              ma27->ICNTL,
              ma27->INFO
              );

      delete[] IW1;
      delete[] W;
   }

} // extern "C"
