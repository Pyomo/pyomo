#include <stdio.h>
//#include <string.h>
#include <stdlib.h>
#include <stdbool.h>

void abort_bad_memory(int status){
	printf("Bad memory allocation in MA57 C interface. Aborting.");
	exit(status);
}

struct MA57_struct {
	int LRHS, LFACT, LKEEP, LIFACT, LWORK, NRHS;
	bool KEEP_allocated, WORK_allocated, FACT_allocated, IFACT_allocated;
	bool NRHS_set, LRHS_set, JOB_set;
	double WORK_factor, FACT_factor, IFACT_factor;
	int* IWORK;
	int* KEEP; 
	int* IFACT;
	int ICNTL[20], INFO[40];
	int JOB;
	double* WORK; 
	double* FACT;
	double CNTL[5], RINFO[20];
};

struct MA57_struct* new_MA57_struct(void){

	struct MA57_struct* ma57 = malloc(sizeof(struct MA57_struct));
	if (ma57 == NULL) { abort_bad_memory(1); }

	ma57id_(ma57->CNTL, ma57->ICNTL);

	// Set default values of parameters
	ma57->KEEP_allocated = ma57->WORK_allocated = false;
	ma57->FACT_allocated = ma57->IFACT_allocated = false;
	ma57->NRHS_set = ma57->LRHS_set = ma57->JOB_set = false;
	ma57->WORK_factor = 1.2;
	ma57->FACT_factor = 2.0;
	ma57->IFACT_factor = 2.0;

	// Return pointer to ma57 that Python program can pass to other functions
	// in this code
	return ma57;
}

// Functions for setting/accessing INFO/CNTL arrays:
void set_icntl(struct MA57_struct* ma57, int i, int val) {
  ma57->ICNTL[i] = val;
}
int get_icntl(struct MA57_struct* ma57, int i) {
	return ma57->ICNTL[i];
}
void set_cntl(struct MA57_struct* ma57, int i, double val) {
	ma57->CNTL[i] = val;
}
double get_cntl(struct MA57_struct* ma57, int i) {
	return ma57->CNTL[i];
}
int get_info(struct MA57_struct* ma57, int i) {
	return ma57->INFO[i];
}
double get_rinfo(struct MA57_struct* ma57, int i) {
	return ma57->RINFO[i];
}

// Functions for allocating WORK/FACT arrays:
void alloc_keep(struct MA57_struct* ma57, int l) {
	ma57->LKEEP = l;
	ma57->KEEP = malloc(l*sizeof(int));
	if (ma57->KEEP == NULL) { abort_bad_memory(1); }
	ma57->KEEP_allocated = true;
}
void alloc_work(struct MA57_struct* ma57, int l) {
	ma57->LWORK = l;
	ma57->WORK = malloc(l*sizeof(double));
	if (ma57->WORK == NULL) { abort_bad_memory(1); }
	ma57->WORK_allocated = true;
}
void alloc_fact(struct MA57_struct* ma57, int l) {
	ma57->LFACT = l;
	ma57->FACT = malloc(l*sizeof(double));
	if (ma57->FACT == NULL) { abort_bad_memory(1); }
	ma57->FACT_allocated = true;
}
void alloc_ifact(struct MA57_struct* ma57, int l) {
	ma57->LIFACT = l;
	ma57->IFACT = malloc(l*sizeof(int));
	if (ma57->IFACT == NULL) { abort_bad_memory(1); }
	ma57->IFACT_allocated = true;
}

// Functions for specifying dimensions of RHS:
void set_nrhs(struct MA57_struct* ma57, int n) {
	ma57->NRHS = n;
	ma57->NRHS_set = true;
}
void set_lrhs(struct MA57_struct* ma57, int l) {
	ma57->LRHS = l;
	ma57->LRHS_set = true;
}

// Specify what job to be performed - maybe make an arg to functions
void set_job(struct MA57_struct* ma57, int j) {
	ma57->JOB = j;
	ma57->JOB_set = true;
}

void do_symbolic_factorization(struct MA57_struct* ma57, int N, int NE, 
		int* IRN, int* JCN) {

	if (!ma57->KEEP_allocated) {
		// KEEP must be >= 5*N+NE+MAX(N,NE)+42
		int size = 5*N + NE + (NE + N) + 42;
		alloc_keep(ma57, size);
	}

	// This is a hard requirement, no need to give the user the option to change
	ma57->IWORK = malloc(5*N*sizeof(int));
	if (ma57->IWORK == NULL) { abort_bad_memory(1); }
	
	ma57ad_(&N, &NE, IRN, JCN, 
			&(ma57->LKEEP), ma57->KEEP, 
			ma57->IWORK, ma57->ICNTL, 
			ma57->INFO, ma57->RINFO);

	free(ma57->IWORK);
}

void do_numeric_factorization(struct MA57_struct* ma57, int N, int NE, 
		double* A) {

	// Get memory estimates from INFO, allocate FACT and IFACT
	if (!ma57->FACT_allocated) {
		int info9 = ma57->INFO[9-1];
		int size = (int)(ma57->FACT_factor*info9);
		alloc_fact(ma57, size);
	}
	if (!ma57->IFACT_allocated) {
		int info10 = ma57->INFO[10-1];
    int size = (int)(ma57->IFACT_factor*info10);
		alloc_ifact(ma57, size);
	}

	// Again, length of IWORK is a hard requirement
	ma57->IWORK = malloc(N*sizeof(int));
	if (ma57->IWORK == NULL) { abort_bad_memory(1); }

	ma57bd_(&N, &NE, A, 
			ma57->FACT, &(ma57->LFACT),
			ma57->IFACT, &(ma57->LIFACT),
			&(ma57->LKEEP), ma57->KEEP,
			ma57->IWORK, ma57->ICNTL,
			ma57->CNTL, ma57->INFO,
			ma57->RINFO);

	free(ma57->IWORK);
}

void do_backsolve(struct MA57_struct* ma57, int N, double* RHS) {

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
	if (!ma57->WORK_allocated) {
		int size = (int)(ma57->WORK_factor*ma57->NRHS*N);
		alloc_work(ma57, size);
	}

	// IWORK should always be length N
	ma57->IWORK = malloc(N*sizeof(int));
	if (ma57->IWORK == NULL) { abort_bad_memory(1); }
  
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
			ma57->IWORK,
			ma57->ICNTL, 
			ma57->INFO
			);

	free(ma57->IWORK);
	free(ma57->WORK);
	ma57->WORK_allocated = false;
}

void do_iterative_refinement(struct MA57_struct* ma57, int N, int NE,
		double* A, int* IRN, int* JCN, double* RHS, double* X, double* RESID) {
	// Number of steps of iterative refinement can be controlled with ICNTL[9-1]

	// Set JOB if not set. Controls how (whether) X and RESID will be used
	if (!ma57->JOB_set) {
		set_job(ma57, 1);
	}

	// Need to allocate WORK differently depending on ICNTL options
	if (!ma57->WORK_allocated) {
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

	ma57->IWORK = malloc(N*sizeof(int));
	if (ma57->IWORK == NULL) { abort_bad_memory(1); }

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
			ma57->IWORK,
			ma57->ICNTL,
			ma57->CNTL,
			ma57->INFO,
			ma57->RINFO
			);

	free(ma57->IWORK);
	free(ma57->WORK);
	ma57->WORK_allocated = false;
}

void do_reallocation(struct MA57_struct* ma57, int N, double realloc_factor, int IC) {
	// Need realloc_factor > 1 here

	// MA57 seems to require that both LNEW and LINEW are larger than the old
	// values, regardless of which is being reallocated (set by IC)
	int LNEW = (int)(realloc_factor*ma57->LFACT);
	double* NEWFAC = malloc(LNEW*sizeof(double));
	if (NEWFAC == NULL) { abort_bad_memory(1); }

	int LINEW = (int)(realloc_factor*ma57->LIFACT);
	int* NEWIFC = malloc(LINEW*sizeof(int));
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
		free(ma57->FACT);
		ma57->LFACT = LNEW;
		ma57->FACT = NEWFAC;
		free(NEWIFC);
	} else if (IC >= 1) {
		// Copied int array; new real array is garbage
		free(ma57->IFACT);
		ma57->LIFACT = LINEW;
		ma57->IFACT = NEWIFC;
		free(NEWFAC);
	} // Now either FACT or IFACT, whichever was specified by IC, can be used
	  // as normal in MA57B/C/D
}

void free_memory(struct MA57_struct* ma57) {
	if (ma57->WORK_allocated) {
		free(ma57->WORK);
	}
	if (ma57->FACT_allocated) {
		free(ma57->FACT);
	}
	if (ma57->IFACT_allocated) {
		free(ma57->IFACT);
	}
	if (ma57->KEEP_allocated) {
		free(ma57->KEEP);
	}
	free(ma57);
}
