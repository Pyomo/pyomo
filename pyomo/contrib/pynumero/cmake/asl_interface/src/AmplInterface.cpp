/**___________________________________________________________________________
 *
 * Pyomo: Python Optimization Modeling Objects
 * Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
 * Under the terms of Contract DE-NA0003525 with National Technology and
 * Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
 * rights in this software.
 * This software is distributed under the 3-clause BSD License.
 * ___________________________________________________________________________
**/

#include "AmplInterface.hpp"
#include "AssertUtils.hpp"
#include "asl_pfgh.h"
#include "getstub.h"

#include <vector>

AmplInterface::AmplInterface()
   :
   _p_asl(NULL), // pointer to the ASL struct
   _obj_direction(1), // minimize by default
   nnz_hes_lag_(-1) // cache this since sphsetup called only once
{
}

char* new_char_p_from_std_str(std::string str)
{
   char* ret = new char[str.length() + 1];
   strcpy(ret, str.c_str());
   return ret;
}

void AmplInterface::initialize(const char *nlfilename)
{
   // The includes from the Ampl Solver Library
   // have a number of macros that expand to include
   // the local variable "asl".
   // For example:
   // #define X0		asl->i.X0_
   // Therefore, in many of these methods, you will 
   // often see the assignment the asl pointer followed
   // by calls to the macros from the ASL.

   // TODO: add possible options later
   std::vector <std::string> options;

   std::string cp_nlfilename(nlfilename);

   // translate options to command input
   std::vector <std::string> arguments;
   arguments.push_back("pynumero");
   arguments.push_back(cp_nlfilename);
   for (const auto &opt : options) {
      arguments.push_back(opt);
   }

   std::vector<char *> argv;

   for (const auto &arg : arguments)
      argv.push_back((char *) arg.data());
   argv.push_back(nullptr);

   // Allocate memory for the asl structure
   ASL_pfgh *asl = (ASL_pfgh *) ASL_alloc(ASL_read_pfgh);
   _p_asl = asl; // store this pointer to write back to "asl" when necessary
   _ASSERT_(_p_asl);

   // Create the Option_Info structure - see getstub.h (more entries than in hooking.pdf)
   // ToDo: should allow many of these to be passed in to initialize (so different solvers
   // can set them appropriately).
   oi = new Option_Info;
   oi->sname = new_char_p_from_std_str("solver_exe_name_not_set");
   oi->bsname = new_char_p_from_std_str("Solver_name_not_set");
   oi->opname = new_char_p_from_std_str("solver_options_env_var_not_set");
   oi->keywds = NULL;
   oi->n_keywds = 0;
   oi->flags = 0;
   oi->version = NULL;
   oi->usage = NULL;
   oi->kwf = NULL;
   oi->feq = NULL;
   oi->options = NULL;
   oi->n_options = 0;
   oi->driver_date = 0;
   oi->wantsol = 0;
   oi->nS = 0;
   oi->S = NULL;
   oi->uinfo = NULL;
   oi->asl = NULL;
   oi->eqsign = NULL;
   oi->n_badopts = 0;
   oi->option_echo = 0;
   oi->nnl = 0;

   // read the options and get the name of the .nl file (stub)
   char *stub = getstops(argv.data(), oi);
   
   delete[] oi->sname;
   oi->sname = NULL;
   delete[] oi->bsname;
   oi->bsname = NULL;
   delete[] oi->opname;
   oi->opname = NULL;   
   // this pointer may need to be stored for the call to write_sol
   //delete oi;

   FILE *nl = this->open_nl(asl, stub);
   _ASSERT_(nl != NULL);

   // want initial values for the variables and the 
   // multipliers
   want_xpi0 = 1 | 2;
   // allocate space in the ASL structure for the initial values
   X0 = new double[n_var];
   havex0 = new char[n_var];
   pi0 = new double[n_con];
   havepi0 = new char[n_con];

   _ASSERT_EXIT_(n_var > 0, "Problem does not have any continuous variables");
   _ASSERT_EXIT_(nbv == 0 && niv == 0, "PyNumero does not support discrete variables");
   _ASSERT_EXIT_(nwv == 0 && nlnc == 0 && lnc == 0, 
                 "PyNumero does not support network constraints");
   _ASSERT_EXIT_(n_cc == 0, "PyNumero does not support complementarities");

   // call ASL to parse the nl file
   int retcode = pfgh_read(nl, ASL_findgroups);
   _ASSERT_EXIT_(retcode == ASL_readerr_none,
                 "Error reading the ASL .nl file");

   // determine maximization or minimization
   _ASSERT_EXIT_(n_obj == 1, "PyNumero supports single objective problems only");
   _obj_direction = 1;
   if (objtype[0] != 0) {
      _obj_direction = -1;
   }

   // see comments in https://github.com/ampl/mp/blob/master/src/asl/solvers/changes
   // void hesset(int flags, int obj, int nnobj, int con, int nncon)
   // tells AMPL which objectives and constraints to include when building the
   // Hessian structure. Seems like:
   // obj is the obj. number to start,
   // nnobj is the number past that to include
   // con is the constraint number to start
   // nncon is the number past that to include
   // we only support single objective problems
   hesset(1, 0, 1, 0, nlc);

   // setup the structure for the Hessian of the Lagrangian
   nnz_hes_lag_ = sphsetup(-1, 1, 1, 1); // num obj, factor on obj, flag to indicate if multipliers supplied, and flag for upper triangular
}

AmplInterface::~AmplInterface() {
    ASL_pfgh *asl = _p_asl;
    delete[] X0;
    X0 = NULL;
    delete[] havex0;
    havex0 = NULL;
    delete[] pi0;
    pi0 = NULL;
    delete[] havepi0;
    havepi0 = NULL;
    delete oi;

    if (asl) {
        ASL *p_asl_to_free = (ASL *) _p_asl;
        ASL_free(&p_asl_to_free);
        _p_asl = NULL;
    }
}

int AmplInterface::get_n_vars() const {
    ASL_pfgh *asl = _p_asl;
    _ASSERT_(_p_asl);
    int n_x;
    n_x = n_var;
    return n_x;
}

int AmplInterface::get_n_constraints() const {
    ASL_pfgh *asl = _p_asl;
    _ASSERT_(_p_asl);
    int n_c;
    n_c = n_con;
    return n_c;
}

int AmplInterface::get_nnz_jac_g() const {
    ASL_pfgh *asl = _p_asl;
    _ASSERT_(_p_asl);
    int nnz_jac_g;
    nnz_jac_g = nzc;
    return nnz_jac_g;
}

int AmplInterface::get_nnz_hessian_lag() const {
    ASL_pfgh *asl = _p_asl;
    _ASSERT_(asl);
    int nnz_hes_lag;
    nnz_hes_lag = nnz_hes_lag_;
    return nnz_hes_lag;
}

void AmplInterface::get_lower_bounds_x(double *invec, int n) {
    ASL_pfgh *asl = _p_asl;
    _ASSERT_(_p_asl);
    _ASSERT_(n == n_var);
    for (int i = 0; i < n; i++) {
        invec[i] = LUv[2 * i];
    }
}

void AmplInterface::get_upper_bounds_x(double *invec, int n) {
    ASL_pfgh *asl = _p_asl;
    _ASSERT_(_p_asl);
    _ASSERT_(n == n_var);

    for (int i = 0; i < n; i++) {
        invec[i] = LUv[2 * i + 1];
    }
}

void AmplInterface::get_lower_bounds_g(double *invec, int m) {
    ASL_pfgh *asl = _p_asl;
    _ASSERT_(_p_asl);
    _ASSERT_(m == n_con);
    for (int i = 0; i < m; i++) {
        invec[i] = LUrhs[2 * i];
    }
}

void AmplInterface::get_upper_bounds_g(double *invec, int m) {
    ASL_pfgh *asl = _p_asl;
    _ASSERT_(_p_asl);
    _ASSERT_(m == n_con);

    for (int i = 0; i < m; i++) {
        invec[i] = LUrhs[2 * i + 1];
    }
}

void AmplInterface::get_init_x(double *invec, int n) {
    ASL_pfgh *asl = _p_asl;
    _ASSERT_(_p_asl);
    _ASSERT_(n == n_var);

    for (int i = 0; i < n; i++) {
        if (havex0[i]) {
            invec[i] = X0[i];
        } else {
            invec[i] = 0.0;
        }
    }
}

void AmplInterface::get_init_multipliers(double *invec, int n) {
    ASL_pfgh *asl = _p_asl;
    _ASSERT_(_p_asl);

    // get dual starting point
    if (n_con == 0) { return; } // unconstrained problem or do not want to use the exist dual values
    _ASSERT_(n == n_con);

    for (int i = 0; i < n; i++) {
        if (havepi0[i]) {
            invec[i] = pi0[i];
        } else {
            invec[i] = 0.0;
        }
    }
}

bool AmplInterface::eval_f(double *const_x, int nx, double& f) {
    ASL_pfgh *asl = _p_asl;
    _ASSERT_(_p_asl);
    _ASSERT_(n_obj == 1 && "AMPL problem must have a single objective function");

    int nerror = 1;
    double retval = objval(obj_no, (double *) const_x, &nerror);

    if (nerror != 0) {
        return false;
    }
    f = _obj_direction * retval;
    return true;

}

bool AmplInterface::eval_deriv_f(double *const_x, double *deriv_f, int nx) {
    ASL_pfgh *asl = _p_asl;
    _ASSERT_(_p_asl);
    _ASSERT_(n_obj == 1 && "AMPL problem must have a single objective function");

    int nerror = 1;
    objgrd(obj_no, (double *) const_x, deriv_f, &nerror);

    if (nerror != 0) {
        return false;
    }

    if (_obj_direction == -1) {
        for (int i = 0; i < nx; i++) {
            deriv_f[i] *= -1.0;
        }
    }
    return true;
}

bool AmplInterface::eval_g(double *const_x, int nx, double *g, int ng) {
    ASL_pfgh *asl = _p_asl;
    _ASSERT_(nx == n_var);
    _ASSERT_(ng == n_con);

    int nerror = 1;
    conval((double *) const_x, g, &nerror);
    if (nerror != 0) {
        return false;
    }
    return true;
}

void AmplInterface::struct_jac_g(int *irow, int *jcol, int nnz_jac_g) {
    ASL_pfgh *asl = _p_asl;
    _ASSERT_(_p_asl);
    _ASSERT_(nnz_jac_g == nzc);
    _ASSERT_(irow && jcol);

    // get the non zero structure of the Jacobian of g wrt x
    for (int i = 0; i < n_con; i++) {
        for (cgrad *cg = Cgrad[i]; cg; cg = cg->next) {
            irow[cg->goff] = i + 1;
            jcol[cg->goff] = cg->varno + 1;
        }
    }
}

bool AmplInterface::eval_jac_g(double *const_x, int nx, double *jac_g_values, int nnz_jac_g) {
    ASL_pfgh *asl = _p_asl;
    _ASSERT_(_p_asl);
    _ASSERT_(nx == n_var);
    _ASSERT_(nnz_jac_g == nzc);
    _ASSERT_(jac_g_values);

    int nerror = 1;
    jacval((double *) const_x, jac_g_values, &nerror);
    if (nerror != 0) {
        return false;
    }
    return true;
}

void AmplInterface::struct_hes_lag(int *irow, int *jcol, int nnz_hes_lag) {
    ASL_pfgh *asl = _p_asl;
    _ASSERT_(_p_asl);
    _ASSERT_(nnz_hes_lag_ == nnz_hes_lag);

    int idx = 0;
    for (int i = 0; i < n_var; i++) {
        for (int j = sputinfo->hcolstarts[i]; j < sputinfo->hcolstarts[i + 1]; j++) {
            irow[idx] = i + 1;
            jcol[idx] = sputinfo->hrownos[j] + 1;
            idx++;
        }
    }
}

bool AmplInterface::eval_hes_lag(double *const_x,
                                 int nx,
                                 double *const_lam,
                                 int nc,
                                 double *hes_lag,
                                 int nnz_hes_lag,
                                 double obj_factor) {
    ASL_pfgh *asl = _p_asl;
    _ASSERT_(_p_asl);
    _ASSERT_(nx == n_var);
    _ASSERT_(nc == n_con);
    _ASSERT_(n_obj == 1);
    _ASSERT_(nnz_hes_lag_ == nnz_hes_lag);

    double OW = _obj_direction * obj_factor;
    sphes(hes_lag, -1, &OW, (double *) const_lam);
    return true;
}

void AmplInterface::finalize_solution(int ampl_solve_result_num, char* msg, double *const_x, int nx, double *const_lam, int nc) {
    ASL_pfgh *asl = _p_asl;
    _ASSERT_(asl);
    _ASSERT_(const_x && const_lam);
    
    // set the AMPL solver status'
    _ASSERT_MSG_(ampl_solve_result_num >= 0 && ampl_solve_result_num < 600,
                 "ampl_solve_result_num must be between 0 and 599 in AmplInterface::finalize_solution");

    write_sol(msg, const_cast<double *>(const_x), const_cast<double *>(const_lam), 0);
}

AmplInterfaceFile::AmplInterfaceFile()
   : AmplInterface()
{}

FILE* AmplInterfaceFile::open_nl(ASL_pfgh *asl, char* stub)
{
   _ASSERT_EXIT_(stub, "No .nl file was specified.");
   return jac0dim(stub, (int) strlen(stub));
}

AmplInterfaceStr::AmplInterfaceStr(char* nl, size_t size)
   : AmplInterface(),
     nl_content(nl),
     nl_size(size)
{}

// THIS METHOD IS DIABLED FOR NOW
FILE* AmplInterfaceStr::open_nl(ASL_pfgh *asl, char* stub)
{
   // Ignore the stub and use the cached NL file content
   //#if defined(__APPLE__) && defined(__MACH__)
   //FILE* nl = fmemopen(this->nl_content, this->nl_size, "rb");
   //return jac0dim_FILE(nl);
   return NULL;
   //   #elif defined(_WIN32)
   //return NULL;
   //#else
   //FILE* nl = fmemopen(this->nl_content, this->nl_size, "rb");
   //return jac0dim_FILE(nl);
   //return NULL;
   //#endif

}


extern "C"
{
   AmplInterface *EXTERNAL_AmplInterface_new_file(char *nlfilename) {
      AmplInterface* ans = new AmplInterfaceFile();
      ans->initialize(nlfilename);
      return ans;
   }

   AmplInterface *EXTERNAL_AmplInterface_new_str(char *nl, size_t size) {
      AmplInterface* ans = new AmplInterfaceStr(nl, size);
      ans->initialize("membuf.nl");
      return ans;
   }

   AmplInterface *EXTERNAL_AmplInterface_new(char *nlfilename) {
      return EXTERNAL_AmplInterface_new_file(nlfilename);
   }

   int EXTERNAL_AmplInterface_n_vars(AmplInterface *p_ai) {
      return p_ai->get_n_vars();
   }

   int EXTERNAL_AmplInterface_n_constraints(AmplInterface *p_ai) {
      return p_ai->get_n_constraints();
   }

   int EXTERNAL_AmplInterface_nnz_jac_g(AmplInterface *p_ai) {
      return p_ai->get_nnz_jac_g();
   }

   int EXTERNAL_AmplInterface_nnz_hessian_lag(AmplInterface *p_ai) {
      return p_ai->get_nnz_hessian_lag();
   }

   void EXTERNAL_AmplInterface_x_lower_bounds(AmplInterface *p_ai, double *invec, int n) {
      p_ai->get_lower_bounds_x(invec, n);
   }

   void EXTERNAL_AmplInterface_x_upper_bounds(AmplInterface *p_ai, double *invec, int n) {
      p_ai->get_upper_bounds_x(invec, n);
   }

   void EXTERNAL_AmplInterface_g_lower_bounds(AmplInterface *p_ai, double *invec, int m) {
      p_ai->get_lower_bounds_g(invec, m);
   }

   void EXTERNAL_AmplInterface_g_upper_bounds(AmplInterface *p_ai, double *invec, int m) {
      p_ai->get_upper_bounds_g(invec, m);
   }

   void EXTERNAL_AmplInterface_get_init_x(AmplInterface *p_ai, double *invec, int n) {
      p_ai->get_init_x(invec, n);
   }
   
   void EXTERNAL_AmplInterface_get_init_multipliers(AmplInterface *p_ai, double *invec, int n) {
      p_ai->get_init_multipliers(invec, n);
   }
   
   bool EXTERNAL_AmplInterface_eval_f(AmplInterface *p_ai, double *invec, int n, double& f) {
      return p_ai->eval_f(invec, n, f);
   }

   bool EXTERNAL_AmplInterface_eval_deriv_f(AmplInterface *p_ai, double *const_x, double *deriv_f, int nx) {
      return p_ai->eval_deriv_f(const_x, deriv_f, nx);
   }

   bool EXTERNAL_AmplInterface_eval_g(AmplInterface *p_ai, double *const_x, int nx, double *g, int ng) {
      return p_ai->eval_g(const_x, nx, g, ng);
   }

   void EXTERNAL_AmplInterface_struct_jac_g(AmplInterface *p_ai, int *irow, int *jcol, int nnz_jac_g) {
      p_ai->struct_jac_g(irow, jcol, nnz_jac_g);
   }

   bool EXTERNAL_AmplInterface_eval_jac_g(AmplInterface *p_ai, double *const_x, int nx, double *jac_g_values,
                                          int nnz_jac_g) {
      return p_ai->eval_jac_g(const_x, nx, jac_g_values, nnz_jac_g);
   }

   void EXTERNAL_AmplInterface_struct_hes_lag(AmplInterface *p_ai, int *irow, int *jcol,
                                              int nnz_hes_lag) {
      p_ai->struct_hes_lag(irow, jcol, nnz_hes_lag);
   }

   bool EXTERNAL_AmplInterface_eval_hes_lag(AmplInterface *p_ai, double *const_x, int nx,
                                            double *const_lam, int nc, double *hes_lag,
                                            int nnz_hes_lag, double obj_factor) {
      return p_ai->eval_hes_lag(const_x, nx, const_lam, nc, hes_lag, nnz_hes_lag, obj_factor);
   }

   void EXTERNAL_AmplInterface_finalize_solution(AmplInterface *p_ai,
                                                 int ampl_solve_result_num,
                                                 char* msg,
                                                 double *const_x, int nx, 
                                                 double *const_lam, int nc) {
      p_ai->finalize_solution(ampl_solve_result_num, msg,
                              const_x, nx, const_lam, nc);
   }

   void EXTERNAL_AmplInterface_free_memory(AmplInterface *p_ai) {
      p_ai->~AmplInterface();
   }

   void EXTERNAL_AmplInterface_dummy(AmplInterface *p_ai) {
       std::cout<<"hola\n";
   }

}

