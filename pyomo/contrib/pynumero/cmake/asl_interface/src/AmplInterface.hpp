#ifndef __AMPLINTERFACE_HPP__
#define __AMPLINTERFACE_HPP__

#include <cstddef>
#include <vector>
#include <string>
#include <iostream>
#include <cstdlib>
#include <stdio.h>
#include "AsserUtils.hpp"

/* forward declaration */
struct ASL_pfgh;
struct SufDecl;
struct SufDesc;


class AmplInterface {
public:

    AmplInterface();

    virtual ~AmplInterface();

    void initialize(const char *nlfilename); 

    virtual FILE* open_nl(ASL_pfgh *asl, char *stub) = 0;

    void get_nlp_dimensions(int &n_x, int &n_g, int &nnz_jac_g, int &nnz_hes_lag) const;

    int get_n_vars() const;

    // find out if this is only equality
    int get_n_constraints() const;

    int get_nnz_jac_g() const;

    int get_nnz_hessian_lag() const;

    void get_lower_bounds_x(double *invec, int n);

    void get_upper_bounds_x(double *invec, int n);

    void get_lower_bounds_g(double *invec, int m);

    void get_upper_bounds_g(double *invec, int m);

    void get_bounds_info(double *xl, double *xu, int n, double *gl, double *gu, int m);

    void get_init_x(double *invec, int n);

    void get_init_multipliers(double *invec, int n);

    void get_starting_point(double *x, int nx, double *lam, int ng);

    bool eval_f(double *const_x, int nx, double& f);

    // the second integer is just so that we can use the typemaps from numpy
    bool eval_deriv_f(double *const_x, double *deriv_f, int nx);

    // the second integer is just so that we can use the typemaps from numpy
    void struct_jac_g(int *irow, int *jcol, int nnz_jac_g);

    bool eval_jac_g(double *const_x, int nx, double *jac_g_values, int nnz_jac_g);

    void struct_hes_lag(int *irow, int *jcol, int nnz_hes_lag);

    bool eval_g(double *const_x, int nx, double *g, int ng);

    // NOTE: This must be called AFTER a call to objval and conval
    // (i.e. You must call eval_f and eval_c with the same x before calling this)
    bool eval_hes_lag(double *const_x, int nx, double *const_lam, int nc, double *hes_lag, int nnz_hes_lag);

    void finalize_solution(int status, double *const_x, int nx, double *const_lam, int nc);

/*
    //void report_solution(int n_x, const double* x) const;
*/
private:

    AmplInterface(const AmplInterface &);

    void operator=(const AmplInterface &);

protected:
    // ASL pointer
    ASL_pfgh *asl_;

    // obj. sense ... -1 = maximize, 1 = minimize
    double obj_sense_;

    // initial values cached
    double *xinit_;
    char *xinit_present_;

    double *laminit_;
    char *laminit_present_;

    int nnz_hes_lag_;

};

class AmplInterface_file : public AmplInterface {
public:
   AmplInterface_file();

   virtual FILE* open_nl(ASL_pfgh *asl, char *stub);
};

class AmplInterface_str : public AmplInterface {
public:
    AmplInterface_str(char* nl, size_t size);

    virtual FILE* open_nl(ASL_pfgh *asl, char *stub);

private:
    char *nl_content;
    size_t nl_size;
};

#endif
