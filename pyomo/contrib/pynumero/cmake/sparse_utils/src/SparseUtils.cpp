#include "SparseUtils.hpp"

// index start from 0
void sym_coo_matvec(int* irow,
                    int* jcol,
                    double* values,
                    int nnz,
                    double* x,
                    int nx,
                    double* result,
                    int nres)
{
    int row, col;
    double value;
    for (int i=0; i<nnz; ++i)
    {
        row = irow[i];
        col = jcol[i];
        value = values[i];
        result[row] += value*x[col];
        if (row != col) {  //calculate for the upper triangular except the diagonal elements
            result[col] += value*x[row];
        }
    }
}

void sym_csr_matvec(int nrows,
                    int* row_pointer,
                    int nrowp,
                    int* col_indices,
                    double* values,
                    int nnz,
                    double* x,
                    int nx,
                    double* result,
                    int nres)
{
    int row, col;
    double sum, value;

    int* irows = new int[nnz];
    int* jcols = new int[nnz];
    double* data = new double[nnz];

    int counter=0;
    for (int i=0; i<nrows; ++i)
    {
        row = i;
        sum = 0.0;
        for(int jj = row_pointer[row]; jj < row_pointer[row+1]; ++jj)
        {
            value = values[jj];
            col = col_indices[jj];
            sum += value * x[col];

            if(row>col)
            {
                irows[counter] = col;
                jcols[counter] = row;
                data[counter] = value;
                ++counter;
            }
        }
        result[row] = sum;
    }

    for (int i=0; i<counter; ++i)
    {
        row = irows[i];
        col = jcols[i];
        value = data[i];
        result[row] += value*x[col];
    }

    delete [] irows;
    delete [] jcols;
    delete [] data;
}

void csr_matvec_no_diag(int nrows,
                        int* row_pointer,
                        int nrowp,
                        int* col_indices,
                        double* values,
                        int nnz,
                        double* x,
                        int nx,
                        double* result,
                        int nres)
{
    int row, col;
    double sum, value;

    for (int i=0; i<nrows; ++i)
    {
        row = i;
        sum = 0.0;
        for(int jj = row_pointer[row]; jj < row_pointer[row+1]; ++jj)
        {
            value = values[jj];
            col = col_indices[jj];
            if(row!=col)
            {
                sum += value * x[col];
            }
        }
        result[row] = sum;
    }

}

void csc_matvec_no_diag(int ncols,
                        int* col_pointer,
                        int ncolp,
                        int* row_indices,
                        double* values,
                        int nnz,
                        double* x,
                        int nx,
                        double* result,
                        int nres)
{
    int row, col;
    double value;

    for (int j=0; j<ncols; ++j)
    {
        col = j;
        int col_start = col_pointer[col];
        int col_end = col_pointer[col+1];
        for(int ii = col_start; ii < col_end; ++ii)
        {
            value = values[ii];
            row = row_indices[ii];
            if(row != col)
            {
                result[row] += value * x[col];
            }
        }

    }
}

void sym_csc_matvec(int ncols,
                    int* col_pointer,
                    int ncolp,
                    int* row_indices,
                    double* values,
                    int nnz,
                    double* x,
                    int nx,
                    double* result,
                    int nres)
{
    int row, col;
    double value;

    int* irows = new int[nnz];
    int* jcols = new int[nnz];
    double* data = new double[nnz];

    int counter=0;
    for (int j=0; j<ncols; ++j)
    {
        col = j;
        int col_start = col_pointer[col];
        int col_end = col_pointer[col+1];
        for(int ii = col_start; ii < col_end; ++ii)
        {
            value = values[ii];
            row = row_indices[ii];
            result[row] += value * x[col];
            if(row>col)
            {
                irows[counter] = col;
                jcols[counter] = row;
                data[counter] = value;
                ++counter;
            }
        }

    }

    for (int i=0; i<counter; ++i)
    {
        row = irows[i];
        col = jcols[i];
        value = data[i];
        result[row] += value*x[col];
    }

    delete [] irows;
    delete [] jcols;
    delete [] data;
}

void sym_csr_diagonal(int* row_pointer,
                      int nrowp,
                      int* col_indices,
                      double* values,
                      int nnz,
                      double* diag,
                      int nrows)
{
    int row, col;
    double value;
    for (int i=0; i<nrows; ++i)
    {
        row = i;
        int last_element_row = row_pointer[row+1] - 1;
        if(last_element_row > row_pointer[row])
        {
            value = values[last_element_row];
            col = col_indices[last_element_row];
            if(col==row)
            {
                diag[col] = value;
            }
        }

    }
}

void sym_csc_diagonal(int* col_pointer,
                      int ncolp,
                      int* row_indices,
                      double* values,
                      int nnz,
                      double* diag,
                      int ncols)
{

    int col;
    double value;

    for (int j=0; j<ncols; ++j)
    {
        col = j;
        int col_start = col_pointer[col];
        int col_end = col_pointer[col+1];
        if(col_end > col_start)
        {
            value = values[col_start];
            diag[col] = value;
        }

    }
}

extern "C"
{
void EXTERNAL_SPARSE_sym_coo_matvec(int* irow,
                                    int* jcol,
                                    double* values,
                                    int nnz,
                                    double* x,
                                    int nx,
                                    double* result,
                                    int nres)
{ sym_coo_matvec(irow, jcol, values, nnz, x, nx, result, nres);}

void EXTERNAL_SPARSE_sym_csr_matvec(int nrows,
                                    int* row_pointer,
                                    int nrowp,
                                    int* col_indices,
                                    double* values,
                                    int nnz,
                                    double* x,
                                    int nx,
                                    double* result,
                                    int nres)
{ sym_csr_matvec(nrows, row_pointer, nrowp, col_indices, values, nnz, x, nx, result, nres);}

void EXTERNAL_SPARSE_sym_csc_matvec(int ncols,
                                    int* col_pointer,
                                    int ncolp,
                                    int* row_indices,
                                    double* values,
                                    int nnz,
                                    double* x,
                                    int nx,
                                    double* result,
                                    int nres)
{ sym_csc_matvec(ncols, col_pointer, ncolp, row_indices, values, nnz, x, nx, result, nres);}

void EXTERNAL_SPARSE_csr_matvec_no_diag(int nrows,
                                        int* row_pointer,
                                        int nrowp,
                                        int* col_indices,
                                        double* values,
                                        int nnz,
                                        double* x,
                                        int nx,
                                        double* result,
                                        int nres)
{ csr_matvec_no_diag(nrows, row_pointer, nrowp, col_indices, values, nnz, x, nx, result, nres);}

void EXTERNAL_SPARSE_csc_matvec_no_diag(int ncols,
                                        int* col_pointer,
                                        int ncolp,
                                        int* row_indices,
                                        double* values,
                                        int nnz,
                                        double* x,
                                        int nx,
                                        double* result,
                                        int nres)
{ csc_matvec_no_diag(ncols, col_pointer, ncolp, row_indices, values, nnz, x, nx, result, nres);}

void EXTERNAL_SPARSE_sym_csr_diagonal(int* row_pointer,
                                      int nrowp,
                                      int* col_indices,
                                      double* values,
                                      int nnz,
                                      double* diag,
                                      int nrows)
{ sym_csr_diagonal(row_pointer, nrowp, col_indices, values, nnz, diag, nrows);}

void EXTERNAL_SPARSE_sym_csc_diagonal(int* col_pointer,
                                      int ncolp,
                                      int* row_indices,
                                      double* values,
                                      int nnz,
                                      double* diag,
                                      int ncols)
{ sym_csc_diagonal(col_pointer, ncolp, row_indices, values, nnz, diag, ncols);}

}
