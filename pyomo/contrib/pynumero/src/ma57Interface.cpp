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

// This would normally be in a header file, but as we do not need one,
// we will explicitly include it here.
#if defined(_WIN32) || defined(_WIN64)
#  if defined(BUILDING_PYNUMERO_MA57)
#    define PYNUMERO_HSL_EXPORT __declspec(dllexport)
#  else
#    define PYNUMERO_HSL_EXPORT __declspec(dllimport)
#  endif
#else
#  define PYNUMERO_HSL_EXPORT
#endif

// Forward declaration of MA57 fortran routines
extern "C" {
   void ma57id_(double* CNTL, int* ICNTL);
   void ma57ad_(int *N, int *NE, const int *IRN, const int* JCN,
                int *LKEEP, int* KEEP, int* IWORK, int *ICNTL,
                int* INFO, double* RINFO);
   void ma57bd_(int *N, int *NE, double* A, double* FACT, int* LFACT,
                int* IFACT, int* LIFACT, int* LKEEP, int* KEEP, int* IWORK,
                int* ICNTL, double* CNTL, int* INFO, double* RINFO);
   void ma57cd_(int* JOB, int *N, double* FACT, int* LFACT,
                int* IFACT, int* LIFACT, int* NRHS, double* RHS,
                int* LRHS, double* WORK, int* LWORK, int* IWORK,
                int* ICNTL, int* INFO);
   void ma57dd_(int* JOB, int *N, int *NE, int *IRN, int *JCN,
                double *FACT, int *LFACT, int *IFACT, int *LIFACT,
                double *RHS, double *X, double *RESID, double *WORK,
                int *IWORK, int *ICNTL, double *CNTL, int *INFO,
                double *RINFO);
   void ma57ed_(int *N, int* IC, int* KEEP, double* FACT, int* LFACT,
                double* NEWFAC, int* LNEW, int* IFACT, int* LIFACT,
                int* NEWIFC, int* LINEW, int* INFO);
} // extern "C"

void abort_bad_memory(int status){
   printf("Bad memory allocation in MA57 C interface. Aborting.");
   exit(status);
}


struct MA57_struct {
   MA57_struct():
      LKEEP(0), LIFACT(0), LWORK(0), LFACT(0),
      LRHS(0), NRHS(0), JOB(0),
      NRHS_set(false),
      LRHS_set(false),
      JOB_set(false),
      WORK_factor(1.2),
      FACT_factor(2.0),
      IFACT_factor(2.0),
      KEEP(NULL),
      IFACT(NULL),
      WORK(NULL),
      FACT(NULL)
   {
      ma57id_(this->CNTL, this->ICNTL);
   }
   virtual ~MA57_struct() {
      if ( this->WORK ) {
         delete[] this->WORK;
      }
      if ( this->FACT ) {
         delete[] this->FACT;
      }
      if ( this->IFACT ) {
         delete[] this->IFACT;
      }
      if ( this->KEEP ) {
         delete[] this->KEEP;
      }
   }

   int LKEEP, LIFACT, LWORK, LFACT, LRHS, NRHS, JOB;
   bool NRHS_set, LRHS_set, JOB_set;
   double WORK_factor, FACT_factor, IFACT_factor;
   int* KEEP;
   int* IFACT;
   double* WORK;
   double* FACT;
   int ICNTL[20], INFO[40];
   double CNTL[5], RINFO[20];
};

extern "C" {

   PYNUMERO_HSL_EXPORT
   MA57_struct* new_MA57_struct(void){

      MA57_struct* ma57 = new MA57_struct;
      if (ma57 == NULL) { abort_bad_memory(1); }
      // Return pointer to ma57 that Python program can pass to other
      // functions in this code
      return ma57;
   }

   PYNUMERO_HSL_EXPORT
   void free_MA57_struct(MA57_struct* ma57) {
      delete ma57;
   }

   // Functions for setting/accessing INFO/CNTL arrays:
   PYNUMERO_HSL_EXPORT
   void set_icntl(MA57_struct* ma57, int i, int val) {
      ma57->ICNTL[i] = val;
   }

   PYNUMERO_HSL_EXPORT
   int get_icntl(MA57_struct* ma57, int i) {
      return ma57->ICNTL[i];
   }

   PYNUMERO_HSL_EXPORT
   void set_cntl(MA57_struct* ma57, int i, double val) {
      ma57->CNTL[i] = val;
   }

   PYNUMERO_HSL_EXPORT
   double get_cntl(MA57_struct* ma57, int i) {
      return ma57->CNTL[i];
   }

   PYNUMERO_HSL_EXPORT
   int get_info(MA57_struct* ma57, int i) {
      return ma57->INFO[i];
   }

   PYNUMERO_HSL_EXPORT
   double get_rinfo(MA57_struct* ma57, int i) {
      return ma57->RINFO[i];
   }

   // Functions for allocating WORK/FACT arrays:
   PYNUMERO_HSL_EXPORT
   void alloc_keep(MA57_struct* ma57, int l) {
      if ( ma57->KEEP ) {
         delete[] ma57->KEEP;
      }
      ma57->LKEEP = l;
      ma57->KEEP = new int[l];
      if (ma57->KEEP == NULL) { abort_bad_memory(1); }
   }

   PYNUMERO_HSL_EXPORT
   void alloc_work(MA57_struct* ma57, int l) {
      if ( ma57->WORK ) {
         delete[] ma57->WORK;
      }
      ma57->LWORK = l;
      ma57->WORK = new double[l];
      if (ma57->WORK == NULL) { abort_bad_memory(1); }
   }

   PYNUMERO_HSL_EXPORT
   void alloc_fact(MA57_struct* ma57, int l) {
      if ( ma57->FACT ) {
         delete[] ma57->FACT;
      }
      ma57->LFACT = l;
      ma57->FACT = new double[l];
      if (ma57->FACT == NULL) { abort_bad_memory(1); }
   }

   PYNUMERO_HSL_EXPORT
   void alloc_ifact(MA57_struct* ma57, int l) {
      if ( ma57->IFACT ) {
         delete[] ma57->IFACT;
      }
      ma57->LIFACT = l;
      ma57->IFACT = new int[l];
      if (ma57->IFACT == NULL) { abort_bad_memory(1); }
   }

   // Functions for specifying dimensions of RHS:
   PYNUMERO_HSL_EXPORT
   void set_nrhs(MA57_struct* ma57, int n) {
      ma57->NRHS = n;
      ma57->NRHS_set = true;
   }

   PYNUMERO_HSL_EXPORT
   void set_lrhs(MA57_struct* ma57, int l) {
      ma57->LRHS = l;
      ma57->LRHS_set = true;
   }

   // Specify what job to be performed - maybe make an arg to functions
   PYNUMERO_HSL_EXPORT
   void set_job(MA57_struct* ma57, int j) {
      ma57->JOB = j;
      ma57->JOB_set = true;
   }


   PYNUMERO_HSL_EXPORT
   void do_symbolic_factorization(MA57_struct* ma57, int N, int NE,
                                  int* IRN, int* JCN) {

      // Arrays, presumably supplied from Python, are assumed to have base-
      // zero indices. Convert to base-one before sending to Fortran.
      for (int i=0; i<NE; i++) {
         IRN[i] = IRN[i] + 1;
         JCN[i] = JCN[i] + 1;
      }

      if ( ! ma57->KEEP ) {
         // KEEP must be >= 5*N+NE+MAX(N,NE)+42
         int size = 5*N + NE + (NE + N) + 42;
         alloc_keep(ma57, size);
      }

      // This is a hard requirement, no need to give the user the option
      // to change
      int* IWORK = new int[5*N];
      if (IWORK == NULL) { abort_bad_memory(1); }
	
      ma57ad_(&N, &NE, IRN, JCN,
              &(ma57->LKEEP), ma57->KEEP,
              IWORK, ma57->ICNTL,
              ma57->INFO, ma57->RINFO);

      delete[] IWORK;
   }


   PYNUMERO_HSL_EXPORT
   void do_numeric_factorization(MA57_struct* ma57, int N, int NE,
                                 double* A) {

      // Get memory estimates from INFO, allocate FACT and IFACT
      if ( ! ma57->FACT ) {
         int info9 = ma57->INFO[9-1];
         int size = (int)(ma57->FACT_factor*info9);
         alloc_fact(ma57, size);
      }
      if ( ! ma57->IFACT ) {
         int info10 = ma57->INFO[10-1];
         int size = (int)(ma57->IFACT_factor*info10);
         alloc_ifact(ma57, size);
      }

      // Again, length of IWORK is a hard requirement
      int* IWORK = new int[N];
      if (IWORK == NULL) { abort_bad_memory(1); }

      ma57bd_(&N, &NE, A,
              ma57->FACT, &(ma57->LFACT),
              ma57->IFACT, &(ma57->LIFACT),
              &(ma57->LKEEP), ma57->KEEP,
              IWORK, ma57->ICNTL,
              ma57->CNTL, ma57->INFO,
              ma57->RINFO);

      delete[] IWORK;
   }


   PYNUMERO_HSL_EXPORT
   void do_backsolve(MA57_struct* ma57, int N, double* RHS) {

      // Set number and length (principal axis) of RHS if not already set
      if (!ma57->NRHS_set) {
         set_nrhs(ma57, 1);
      }
      if (!ma57->LRHS_set) {
         set_lrhs(ma57, N);
      }

      // Set JOB. Default is to perform full factorization
      if (!ma57->JOB_set) {
         set_job(ma57, 1);
      }

      // Allocate WORK if not done. Should be >= N
      if ( ! ma57->WORK ) {
         int size = (int)(ma57->WORK_factor*ma57->NRHS*N);
         alloc_work(ma57, size);
      }

      // IWORK should always be length N
      int* IWORK = new int[N];
      if (IWORK == NULL) { abort_bad_memory(1); }

      ma57cd_(
              &(ma57->JOB),
              &N,
              ma57->FACT,
              &(ma57->LFACT),
              ma57->IFACT,
              &(ma57->LIFACT),
              &(ma57->NRHS),
              RHS,
              &(ma57->LRHS),
              ma57->WORK,
              &(ma57->LWORK),
              IWORK,
              ma57->ICNTL,
              ma57->INFO
              );

      delete[] IWORK;
      delete[] ma57->WORK;
      ma57->WORK = NULL;
   }


   PYNUMERO_HSL_EXPORT
   void do_iterative_refinement(MA57_struct* ma57, int N, int NE,
                                double* A, int* IRN, int* JCN, double* RHS, double* X, double* RESID) {
      // Number of steps of iterative refinement can be controlled with ICNTL[9-1]

      // Set JOB if not set. Controls how (whether) X and RESID will be used
      if (!ma57->JOB_set) {
         set_job(ma57, 1);
      }

      // Need to allocate WORK differently depending on ICNTL options
      if ( ! ma57->WORK ) {
         int icntl9 = ma57->ICNTL[9-1];
         int icntl10 = ma57->ICNTL[10-1];
         int size;
         if (icntl9 == 1) {
            size = (int)(ma57->WORK_factor*N);
         } else if (icntl9 > 1 && icntl10 == 0) {
            size = (int)(ma57->WORK_factor*3*N);
         } else if (icntl9 > 1 && icntl10 > 0) {
            size = (int)(ma57->WORK_factor*4*N);
         }
         alloc_work(ma57, size);
      }

      int* IWORK = new int[N];
      if (IWORK == NULL) { abort_bad_memory(1); }

      ma57dd_(
              &(ma57->JOB),
              &N,
              &NE,
              IRN,
              JCN,
              ma57->FACT,
              &(ma57->LFACT),
              ma57->IFACT,
              &(ma57->LIFACT),
              RHS,
              X,
              RESID,
              ma57->WORK,
              IWORK,
              ma57->ICNTL,
              ma57->CNTL,
              ma57->INFO,
              ma57->RINFO
              );

      delete[] IWORK;
      delete[] ma57->WORK;
      ma57->WORK = NULL;
   }


   PYNUMERO_HSL_EXPORT
   void do_reallocation(MA57_struct* ma57, int N, double realloc_factor, int IC) {
      // Need realloc_factor > 1 here

      // MA57 seems to require that both LNEW and LINEW are larger than the old
      // values, regardless of which is being reallocated (set by IC)
      int LNEW = (int)(realloc_factor*ma57->LFACT);
      double* NEWFAC = new double[LNEW];
      if (NEWFAC == NULL) { abort_bad_memory(1); }

      int LINEW = (int)(realloc_factor*ma57->LIFACT);
      int* NEWIFC = new int[LINEW];
      if (NEWIFC == NULL) { abort_bad_memory(1); }

      ma57ed_(
              &N,
              &IC,
              ma57->KEEP,
              ma57->FACT,
              &(ma57->LFACT),
              NEWFAC,
              &LNEW,
              ma57->IFACT,
              &(ma57->LIFACT),
              NEWIFC,
              &LINEW,
              ma57->INFO
              );

      if (IC <= 0) {
         // Copied real array; new int array is garbage
         delete[] ma57->FACT;
         ma57->LFACT = LNEW;
         ma57->FACT = NEWFAC;
         delete[] NEWIFC;
      } else if (IC >= 1) {
         // Copied int array; new real array is garbage
         delete[] ma57->IFACT;
         ma57->LIFACT = LINEW;
         ma57->IFACT = NEWIFC;
         delete[] NEWFAC;
      } // Now either FACT or IFACT, whichever was specified by IC, can be used
      // as normal in MA57B/C/D
   }

} // extern "C"
