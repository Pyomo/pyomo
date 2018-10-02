#include "ma27Interface.hpp"
#include <algorithm>

#define F77_FUNC(name,NAME) name ## _

extern "C"
{
  void F77_FUNC(ma27id,MA27ID)(int* ICNTL, double* CNTL);
  void F77_FUNC(ma27ad,MA27AD)(int *N, int *NZ, const int *IRN, const int* ICN,
                               int *IW, int* LIW, int* IKEEP, int *IW1,
                               int* NSTEPS, int* IFLAG, int* ICNTL,
                               double* CNTL, int *INFO, double* OPS);
  void F77_FUNC(ma27bd,MA27BD)(int *N, int *NZ, const int *IRN, const int* ICN,
                               double* A, int* LA, int* IW, int* LIW,
                               int* IKEEP, int* NSTEPS, int* MAXFRT,
                               int* IW1, int* ICNTL, double* CNTL,
                               int* INFO);
  void F77_FUNC(ma27cd,MA27CD)(int *N, double* A, int* LA, int* IW,
                               int* LIW, double* W, int* MAXFRT,
                               double* RHS, int* IW1, int* NSTEPS,
                               int* ICNTL, double* CNTL);
}

MA27_LinearSolver::MA27_LinearSolver(double pivtol /* = 1e-4*/)
  :
  nnz_(0),
  dim_(0),
  n_iw_(0),
  iw_(NULL),
  irows_(NULL),
  jcols_(NULL),
  n_a_(0),
  a_(NULL),
  ikeep_(NULL),
  nsteps_(0),
  maxfrt_(0),
  num_neg_evals_(-1)
{
  // initialize the options to their defaults
  F77_FUNC(ma27id,MA27ID)(icntl_, cntl_);
  cntl_[0] = pivtol;
}

MA27_LinearSolver::~MA27_LinearSolver()
{
  delete [] iw_;
  delete [] irows_;
  delete [] jcols_;
  delete [] a_;
  delete [] ikeep_;

  //  numeric_fac_timer_.PrintTotal(std::cout, "numeric factor time");
  //  backsolve_timer_.PrintTotal(std::cout, "backsolve time");
}

void MA27_LinearSolver::DoSymbolicFactorization(int nrowcols, int* irow, int* jcol, int nnonzeros)
{

  delete [] iw_;
  iw_ = NULL;
  n_iw_ = 0;

  delete [] irows_;
  irows_ = NULL;

  delete [] jcols_;
  jcols_ = NULL;
  nnz_ = 0;

  delete [] a_;
  a_ = NULL;
  n_a_ = 0;

  delete [] ikeep_;
  ikeep_ = NULL;

  dim_ = nrowcols;

  nnz_ = nnonzeros;

  irows_ = new int[nnz_];
  jcols_ = new int[nnz_];

  // copy data
  std::copy(irow, irow + nnonzeros, irows_);
  std::copy(jcol, jcol + nnonzeros, jcols_);

  // overestimate size by 2
  n_iw_ = 4*(2*nnz_+3*dim_+1);
  iw_ = new int[n_iw_];

  ikeep_ = new int[3*dim_];

  int N = dim_;
  int NZ = nnz_;
  int IFLAG = 0;
  double OPS;
  int INFO[20];
  int* IW1 = new int[2*dim_];

  F77_FUNC(ma27ad,MA27AD)(&N, &NZ, irows_, jcols_, iw_, &n_iw_, ikeep_, IW1, &nsteps_,
	  &IFLAG, icntl_, cntl_, INFO, &OPS);
  delete [] IW1;

  int retflag = INFO[0];
  if (retflag != 0) {
    if (retflag == 1) {
      std::cerr << "An index of the matrix is out of range" << std::endl;
      exit(1);
    }
    std::cerr << "Unknown error in ma27ad_" << std::endl;
    exit(1);
  }

  // retflag == 0

  int recommended_n_iw = INFO[5];
  n_iw_ = 20*recommended_n_iw;
  delete [] iw_;
  iw_ = NULL;
  iw_ = new int[n_iw_];

  int recommended_n_a = INFO[4];

  n_a_ = (nnz_ > 20*recommended_n_a) ? nnz_ : 60*recommended_n_a;
  delete [] a_;
  a_ = NULL;
  a_ = new double[n_a_];
}

MA27_LinearSolver::MA27_FACT_STATUS MA27_LinearSolver::DoNumericFactorization(int nrowcols, int nnonzeros, double* values, int desired_num_neg_evals)
{

  _ASSERT_(nrowcols == dim_);
  _ASSERT_(nnonzeros == nnz_);

  MA27_LinearSolver::MA27_FACT_STATUS status = MA27_SUCCESS;
  int N = dim_;
  int NZ = nnz_;
  int* IW1 = new int[2*dim_];
  int INFO[20];

  // reset number of negative eigenvalues
  num_neg_evals_ = - 1;

  // copy data
  std::copy(values, values + nnonzeros, a_);

  F77_FUNC(ma27bd,MA27BD)(&N, &NZ, irows_, jcols_, a_, &n_a_, iw_, &n_iw_, ikeep_, &nsteps_, 
	  &maxfrt_, IW1, icntl_, cntl_, INFO);
  
  delete [] IW1;

  int retflag = INFO[0];
  if (retflag == -3) {
    std::cerr << "Error in MA27, n_iw_ is too small" << std::endl;
    exit(1);
  }
  else if (retflag == -4) {
    std::cerr << "Error in MA27, n_a_ is too small" << std::endl;
    exit(1);
  }
  else if (retflag == -5) {
//    std::cerr << "Error in MA27, matrix is singular" << std::endl;
//    exit(1);
    status = MA27_MATRIX_SINGULAR;

  }
  else if (retflag == 3) {
    //    std::cout << "Warning in MA27, rank deficiency detected" << std::endl;
    //    std::cerr << "Error in MA27, rank deficiency detected" << std::endl;
    //    exit(1);
    status = MA27_MATRIX_SINGULAR;
  }
  else if (retflag != 0) {
    std::cerr << "Error in MA27." << std::endl;
    exit(1);
  }

  int num_int_compressions = INFO[12];
  int num_double_compressions = INFO[11];
  if (num_int_compressions >= 10) {
    std::cerr << "MA27: Number of integer compressions is high - increase n_iw_" << std::endl;
    exit(1);
  }
  
  if (num_double_compressions >= 10) {
    std::cerr << "MA27: Number of double compressions is high - increase n_a_" << std::endl;
    exit(1);
  }

  int num_neg_evals = INFO[14];
  if (desired_num_neg_evals != -1 && desired_num_neg_evals != num_neg_evals) {
//    std::cerr << "MA27: Number of negative eigenvalues is not correct" << std::endl;
//    exit(1);
    if (status != MA27_MATRIX_SINGULAR)
    {
        status = MA27_INCORRECT_INERTIA;
        num_neg_evals_ = num_neg_evals;
    }

  }
  return status;

}

void MA27_LinearSolver::DoBacksolve(double* rhs, int nrhs, double* sol, int nsol)
{

  _ASSERT_(nrhs == dim_);
  _ASSERT_(nsol == dim_);

  int N = dim_;
  double* W = new double[maxfrt_];
  int* IW1 = new int[nsteps_];

  double* soln_vals = new double[N];
  std::copy(rhs, rhs + N, soln_vals);

  F77_FUNC(ma27cd,MA27CD)(&N, a_, &n_a_, iw_, &n_iw_, W, &maxfrt_, soln_vals, IW1, &nsteps_, icntl_, cntl_);

  std::copy(soln_vals, soln_vals + N, sol);

  delete [] soln_vals;
  delete [] W;
  delete [] IW1;

}

void MA27_LinearSolver::get_values(int nnz, double* valuesfact)
{
  _ASSERT_(nnz == nnz_);
  //std::copy(rhs, rhs + N, soln_vals);
}

extern "C"
{

MA27_LinearSolver* EXTERNAL_MA27Interface_new(double pivot)
{ return new MA27_LinearSolver(pivot);}

int EXTERNAL_MA27Interface_get_nnz(MA27_LinearSolver* p_hi)
{ return p_hi->get_nnz();}

int EXTERNAL_MA27Interface_get_dim(MA27_LinearSolver* p_hi)
{ return p_hi->get_dim();}

int EXTERNAL_MA27Interface_get_num_neg_evals(MA27_LinearSolver* p_hi)
{ return p_hi->get_num_neg_evals();}

void EXTERNAL_MA27Interface_get_values(MA27_LinearSolver* p_hi, int nnz, double* valuesfact)
{ p_hi->get_values(nnz, valuesfact);}

void EXTERNAL_MA27Interface_do_symbolic_factorization(MA27_LinearSolver* p_hi,
                                                      int nrowcols,
                                                      int* irow,
                                                      int* jcol,
                                                      int nnonzeros)
{p_hi->DoSymbolicFactorization(nrowcols, irow, jcol, nnonzeros);}

int EXTERNAL_MA27Interface_do_numeric_factorization(MA27_LinearSolver* p_hi,
                                                     int nrowcols,
                                                     int nnonzeros,
                                                     double* values,
                                                     int desired_num_neg_evals) {
  int status = p_hi->DoNumericFactorization(nrowcols, nnonzeros, values, desired_num_neg_evals);
  return status;
}

void EXTERNAL_MA27Interface_do_backsolve(MA27_LinearSolver* p_hi, double* rhs, int nrhs, double* sol, int nsol)
{p_hi->DoBacksolve(rhs, nrhs, sol, nsol);}

void EXTERNAL_MA27Interface_free_memory(MA27_LinearSolver* p_hi)
{p_hi->~MA27_LinearSolver();}

}

