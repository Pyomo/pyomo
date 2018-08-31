#ifndef __AMPLINTERFACE_HPP__
#define __AMPLINTERFACE_HPP__

#include <iostream>

// Forward declaration for ASL structure
struct ASL_pfgh;

/**
This class provides the C++ side of the PyNumero interface to AMPL
**/
class AmplInterface {
public:
   AmplInterface();
   virtual ~AmplInterface();

   void initialize(const char *nlfilename); 

   virtual FILE* open_nl(ASL_pfgh *asl, char *stub) = 0;

   // number of variables (x)
   int get_n_vars() const;
   
   // number of constraints (g)
   int get_n_constraints() const;

   // number of nonzeros in the jacobian of g wrt x
   int get_nnz_jac_g() const;

   // number of nonzeros in the Hessian of the Lagrangian
   int get_nnz_hessian_lag() const;
   
   // get the lower bounds on x (full dimension)
   void get_lower_bounds_x(double *invec, int n);
   
   // get the upper bounds on x (full dimension)
   void get_upper_bounds_x(double *invec, int n);

   // get the lower bounds on g (full dimension)
   void get_lower_bounds_g(double *invec, int m);

   // get the upper bounds on g (full dimension)
   void get_upper_bounds_g(double *invec, int m);

   // get the initial values for x
   void get_init_x(double *invec, int n);

   // get the initia values for the multipliers lambda
   void get_init_multipliers(double *invec, int n);

   // evaluate the objective function
   bool eval_f(double *const_x, int nx, double& f);

   // evaluate the gradient of f
   bool eval_deriv_f(double *const_x, double *deriv_f, int nx);

   // evaluate the body of the constraints g
   bool eval_g(double *const_x, int nx, double *g, int ng);

   // get the irow, jcol sparse structure of the Jacobian of g wrt x
   void struct_jac_g(int *irow, int *jcol, int nnz_jac_g);

   // get the Jacobian of g wrt x
   bool eval_jac_g(double *const_x, int nx, double *jac_g_values, int nnz_jac_g);

   // get the irow, jcol sparse structure of the Hessian of the Lagrangian
   void struct_hes_lag(int *irow, int *jcol, int nnz_hes_lag);

   // evaluate the Hessian of the Lagrangian
   // Because of a requirement in AMPL, any time "x" changes
   // you must make a call to eval_f and eval_g BEFORE calling this method
   bool eval_hes_lag(double *const_x, int nx, double *const_lam, int nc, double *hes_lag, int nnz_hes_lag);

   // write the solution to the .sol file
   void finalize_solution(int status, double *const_x, int nx, double *const_lam, int nc);

private:

   // Make these private so the compiler does not give default implementations for them
   AmplInterface(const AmplInterface &);
   void operator=(const AmplInterface &);

protected:
   // ASL pointer
   ASL_pfgh *_p_asl;

   // obj. sense ... -1 = maximize, 1 = minimize
   double obj_sense_;

   // initial values cached
   double *xinit_;
   char *xinit_present_;

   double *laminit_;
   char *laminit_present_;

   int nnz_hes_lag_;

};

class AmplInterfaceFile : public AmplInterface {
public:
   AmplInterfaceFile();

   virtual FILE* open_nl(ASL_pfgh *asl, char *stub);
};

class AmplInterfaceStr : public AmplInterface {
public:
    AmplInterfaceStr(char* nl, size_t size);

    virtual FILE* open_nl(ASL_pfgh *asl, char *stub);

private:
    char *nl_content;
    size_t nl_size;
};

#endif
