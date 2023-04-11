/**___________________________________________________________________________
 *
 * Pyomo: Python Optimization Modeling Objects
 * Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
 * Under the terms of Contract DE-NA0003525 with National Technology and
 * Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
 * rights in this software.
 * This software is distributed under the 3-clause BSD License.
 * ___________________________________________________________________________
**/
#ifndef __AMPLINTERFACE_HPP__
#define __AMPLINTERFACE_HPP__

#include <iostream>

#if defined(_WIN32) || defined(_WIN64)
#  if defined(BUILDING_PYNUMERO_ASL)
#    define PYNUMERO_ASL_EXPORT __declspec(dllexport)
#  else
#    define PYNUMERO_ASL_EXPORT __declspec(dllimport)
#  endif
#else
#  define PYNUMERO_ASL_EXPORT
#endif

// Forward declaration for ASL structure
struct ASL_pfgh;
struct Option_Info;

// This class provides the C++ side of the 
// PyNumero interface to AMPL
class PYNUMERO_ASL_EXPORT AmplInterface {
public:
   AmplInterface();
   virtual ~AmplInterface();

   void initialize(const char *nlfilename, const char *amplfunc="");

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
   bool eval_hes_lag(double *const_x,
                     int nx,
                     double *const_lam,
                     int nc,
                     double *hes_lag,
                     int nnz_hes_lag,
                     double objective_factor);

   // write the solution to the .sol file
   // pass in the ampl_solve_status_num (this is the "solve_status_num" from
   // the AMPL documentation. It should be interpreted as follows:
   //
   //   number   string       interpretation
   //   0 -  99   solved       optimal solution found
   //   100 - 199   solved?      optimal solution indicated, but error likely
   //   200 - 299   infeasible   constraints cannot be satisfied
   //   300 - 399   unbounded    objective can be improved without limit
   //   400 - 499   limit        stopped by a limit that you set (such as on iterations)
   //   500 - 599   failure      stopped by an error condition in the solver routines
   //
   void finalize_solution(int ampl_solve_status_num, char* msg, 
                          double *const_x, int nx, double *const_lam, int nc);

   // get the ASLdate that the interface was compiled against
   long get_asl_date();
private:

   // Make these private so the compiler does not give default implementations for them
   AmplInterface(const AmplInterface &);
   void operator=(const AmplInterface &);

protected:
   // ASL pointer
   ASL_pfgh *_p_asl;
   // ASL option info pointer
   Option_Info *oi;

   // maximize (-1) or minimize (1)
   double _obj_direction;

   // caching this (set with single call to sphsetup)
   int nnz_hes_lag_;

};

// File-based specialization of AmplInterface
class PYNUMERO_ASL_EXPORT AmplInterfaceFile : public AmplInterface {
public:
   AmplInterfaceFile();

   virtual FILE* open_nl(ASL_pfgh *asl, char *stub);
};

// String-based specialization of AmplInterface
class PYNUMERO_ASL_EXPORT AmplInterfaceStr : public AmplInterface {
public:
    AmplInterfaceStr(char* nl, size_t size);

    virtual FILE* open_nl(ASL_pfgh *asl, char *stub);

private:
    char *nl_content;
    size_t nl_size;
};

#endif
