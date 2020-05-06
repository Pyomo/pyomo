#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>

void abort_bad_memory(int status){
	printf("Bad memory allocation in MA27 C interface. Aborting.");
	exit(status);
}

struct MA27_struct {
	int LIW_a, LIW_b, NSTEPS, IFLAG, LA, MAXFRT;
	double IW_factor, A_factor;
	bool A_allocated, IKEEP_allocated;
	bool IW_a_allocated, IW_b_allocated;
	int* IW_a;
	int* IW_b;
	// Use different arrays for IW that is sent to MA27A and that sent to
	// MA27B because IW must be discarded after MA27A but kept after MA27B.
	// If these arrays are the same, and a symbolic factorization is performed
	// after a numeric factorization (e.g. on a new matrix), user-defined
	// and MA27B-defined allocations of IW can be conflated.
	int* IW1;
	int* IKEEP;
	int ICNTL[30], INFO[20];
	double OPS;
	double* W;
	double* A;
	double CNTL[5];
};

struct MA27_struct* new_MA27_struct(void){

	struct MA27_struct* ma27 = malloc(sizeof(struct MA27_struct));
	if (ma27 == NULL) { abort_bad_memory(1); }

	ma27id_(ma27->ICNTL, ma27->CNTL);

	// Set default values of parameters
	ma27->A_allocated = ma27->IKEEP_allocated = false;
	ma27->IW_a_allocated = ma27->IW_b_allocated = false;
	ma27->IFLAG = 0;
	ma27->IW_factor = 1.2;
	ma27->A_factor = 2.0;

	// Return pointer to ma27 that Python program can pass to other functions
	// in this code
	return ma27;
}

// Functions for setting/accessing INFO/CNTL arrays:
void set_icntl(struct MA27_struct* ma27, int i, int val) {
  ma27->ICNTL[i] = val;
}
int get_icntl(struct MA27_struct* ma27, int i) {
	return ma27->ICNTL[i];
}
void set_cntl(struct MA27_struct* ma27, int i, double val) {
	ma27->CNTL[i] = val;
}
double get_cntl(struct MA27_struct* ma27, int i) {
	return ma27->CNTL[i];
}
int get_info(struct MA27_struct* ma27, int i) {
	return ma27->INFO[i];
}

// Functions for allocating WORK/FACT arrays:
void alloc_iw_a(struct MA27_struct* ma27, int l) {
	ma27->LIW_a = l;
	ma27->IW_a = malloc(l*sizeof(int));
	if (ma27->IW_a == NULL) { abort_bad_memory(1); }
	ma27->IW_a_allocated = true;
}
void alloc_iw_b(struct MA27_struct* ma27, int l) {
	ma27->LIW_b = l;
	ma27->IW_b = malloc(l*sizeof(int));
	if (ma27->IW_b == NULL) { abort_bad_memory(1); }
	ma27->IW_b_allocated = true;
}
void alloc_a(struct MA27_struct* ma27, int l) {
	ma27->LA = l;
	//ma27->A = realloc(A, l*sizeof(double));
	ma27->A = malloc(l*sizeof(double));
	if (ma27->A == NULL) { abort_bad_memory(1); }
	//memcpy(ma27->A, A, NZ*sizeof(double));
	ma27->A_allocated = true;
}

void do_symbolic_factorization(struct MA27_struct* ma27, int N, int NZ, 
		int* IRN, int* ICN) {

	if (!ma27->IW_a_allocated) {
		int min_size = 2*NZ + 3*N + 1;
		int size = (int)(ma27->IW_factor*min_size);
		alloc_iw_a(ma27, size);
	}

	ma27->IKEEP = malloc(3*N*sizeof(int));
	if (ma27->IKEEP == NULL) { abort_bad_memory(1); }
	ma27->IKEEP_allocated = true;
	ma27->IW1 = malloc(2*N*sizeof(int));
	if (ma27->IW1 == NULL) { abort_bad_memory(1); }

	ma27ad_(&N, 
			&NZ, 
			IRN, 
			ICN, 
			ma27->IW_a,
			&(ma27->LIW_a),
			ma27->IKEEP, 
			ma27->IW1, 
			&(ma27->NSTEPS),
			&(ma27->IFLAG),
			ma27->ICNTL, 
			ma27->CNTL,
			ma27->INFO, 
			&(ma27->OPS));

	free(ma27->IW1);
	free(ma27->IW_a);
	ma27->IW_a_allocated = false;
}

void do_numeric_factorization(struct MA27_struct* ma27, int N, int NZ, 
		int* IRN, int* ICN, double* A) {

	// Get memory estimates from INFO, allocate A and IW
	if (!ma27->A_allocated) {
		int info5 = ma27->INFO[5-1];
		int size = (int)(ma27->A_factor*info5);
		alloc_a(ma27, size);
		// A is now allocated
	}
	// Regardless of ma27->A's previous allocation status, copy values from A.
	memcpy(ma27->A, A, NZ*sizeof(double));

	if (!ma27->IW_b_allocated) {
		int info6 = ma27->INFO[6-1];
		int size = (int)(ma27->IW_factor*info6);
		alloc_iw_b(ma27, size);
	}

	ma27->IW1 = malloc(N*sizeof(int));
	if (ma27->IW1 == NULL) { abort_bad_memory(1); }

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
			ma27->IW1,
			ma27->ICNTL,
			ma27->CNTL,
			ma27->INFO);

	free(ma27->IW1);
}

void do_backsolve(struct MA27_struct* ma27, int N, double* RHS) {

	ma27->W = malloc(ma27->MAXFRT*sizeof(double));
	if (ma27->W == NULL) { abort_bad_memory(1); }
	ma27->IW1 = malloc(ma27->NSTEPS*sizeof(int));
	if (ma27->IW1 == NULL) { abort_bad_memory(1); }
  
	ma27cd_(
			&N, 
			ma27->A, 
			&(ma27->LA),
			ma27->IW_b, 
			&(ma27->LIW_b),
			ma27->W,
			&(ma27->MAXFRT),
			RHS,
			ma27->IW1,
			&(ma27->NSTEPS), 
			ma27->ICNTL, 
			ma27->INFO
			);

	free(ma27->IW1);
	free(ma27->W);
}

void free_memory(struct MA27_struct* ma27) {
	if (ma27->A_allocated) {
		free(ma27->A);
	}
	if (ma27->IW_a_allocated) {
		free(ma27->IW_a);
	}
	if (ma27->IW_a_allocated) {
		free(ma27->IW_a);
	}
	if (ma27->IKEEP_allocated) {
		free(ma27->IKEEP);
	}
	free(ma27);
}
