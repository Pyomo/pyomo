#ifndef __SPARSEUTIL_HPP__
#define __SPARSEUTIL_HPP__

#include <algorithm>
#include <functional>
#include <vector>

#include <iostream>

// index start from 0
void sym_coo_matvec(int* irow,
                    int* jcol,
                    double* values,
                    int nnz,
                    double* x,
                    int nx,
                    double* result,
                    int nres);

void sym_csr_matvec(int nrows,
                    int* row_pointer,
                    int nrowp,
                    int* col_indices,
                    double* values,
                    int nnz,
                    double* x,
                    int nx,
                    double* result,
                    int nres);

void sym_csc_matvec(int ncols,
                    int* col_pointer,
                    int ncolp,
                    int* row_indices,
                    double* values,
                    int nnz,
                    double* x,
                    int nx,
                    double* result,
                    int nres);

void csr_matvec_no_diag(int nrows,
                        int* row_pointer,
                        int nrowp,
                        int* col_indices,
                        double* values,
                        int nnz,
                        double* x,
                        int nx,
                        double* result,
                        int nres);

void csc_matvec_no_diag(int ncols,
                        int* col_pointer,
                        int ncolp,
                        int* row_indices,
                        double* values,
                        int nnz,
                        double* x,
                        int nx,
                        double* result,
                        int nres);

void sym_csr_diagonal(int* row_pointer,
                      int nrowp,
                      int* col_indices,
                      double* values,
                      int nnz,
                      double* diag,
                      int nrows);

void sym_csc_diagonal(int* col_pointer,
                      int ncolp,
                      int* row_indices,
                      double* values,
                      int nnz,
                      double* diag,
                      int ncols);

#endif