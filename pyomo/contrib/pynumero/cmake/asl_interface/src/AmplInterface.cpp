#include "AmplInterface.hpp"

// AMPL includes
#include "asl.h"
#include "asl_pfgh.h"
#include "getstub.h"

#include <cstring>
#include <string>

#include <algorithm>    // std::transform
#include <functional>   // std::plus
#include <vector>
#include <map>

#include <iostream>

#define MAX_LENGTH 100
#define MAX_ARGS 100

//AmplInterface::AmplInterface(int argc, char**& argv)
AmplInterface::AmplInterface()
        :
        asl_(NULL),
        obj_sense_(1),
        xinit_(NULL),
        xinit_present_(NULL),
        laminit_(NULL),
        laminit_present_(NULL),
        nnz_hes_lag_(-1)
{}

void AmplInterface::initialize(const char *nlfilename)
{
    // The ASL include files #define certain
    // variables that they expect you to work with.
    // These variables then appear as though they are
    // global variables when, in fact, they are not
    // Most of them are data members of an asl object

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

    // Create the ASL structure
    ASL_pfgh *asl = (ASL_pfgh *) ASL_alloc(ASL_read_pfgh);
    _ASSERT_(asl);
    asl_ = asl; // keep the pointer for ourselves to use later...

    // Read the options and stub
    Option_Info *Oinfo = new Option_Info;
    char sname[] = "Prototype Algorithm";
    Oinfo->sname = new char[strlen(sname) + 1];
    strcpy(Oinfo->sname, sname);

    char bsname[] = "Prototype Algorithm: No Version Assigned";
    Oinfo->bsname = new char[strlen(bsname) + 1];
    strcpy(Oinfo->bsname, bsname);

    char opname[] = "alg_options";
    Oinfo->opname = new char[strlen(opname) + 1];
    strcpy(Oinfo->opname, opname);

    Oinfo->keywds = NULL;
    Oinfo->n_keywds = 0;
    Oinfo->flags = 0;
    Oinfo->version = NULL;
    Oinfo->usage = NULL;
    Oinfo->kwf = NULL;
    Oinfo->feq = NULL;
    Oinfo->options = NULL;
    Oinfo->n_options = 0;
    Oinfo->driver_date = 0;
    Oinfo->wantsol = 0;
    Oinfo->nS = 0;
    Oinfo->S = NULL;
    Oinfo->uinfo = NULL;
    Oinfo->asl = NULL;
    Oinfo->eqsign = NULL;
    Oinfo->n_badopts = 0;
    Oinfo->option_echo = 0;
    Oinfo->nnl = 0;

    // read the options and get the name of the .nl file (stub)
    char *stub = getstops(argv.data(), Oinfo);

    delete[] Oinfo->sname;
    Oinfo->sname = NULL;
    delete[] Oinfo->bsname;
    Oinfo->bsname = NULL;
    delete[] Oinfo->opname;
    Oinfo->opname = NULL;

    // this pointer may need to be stored for the call to write_solution
    delete Oinfo;
    FILE *nl = this->open_nl(asl, stub);
    _ASSERT_(nl != NULL);

    // check that this is the right problem class
    _ASSERT_(n_var > 0); // make sure we have continuous variables
    _ASSERT_(nbv == 0 && niv == 0); // can't handle binaries
    _ASSERT_(nwv == 0 && nlnc == 0 && lnc == 0); // can't handle these special variables
    assert_exit(n_obj == 1, "This code only handles problems with a exactly one objective function");


    // get initial values for the variables
    want_xpi0 = 1;

    X0 = new double[n_var];
    havex0 = new char[n_var];
    xinit_ = X0;
    xinit_present_ = havex0;

    pi0 = new double[n_con];
    havepi0 = new char[n_con];
    laminit_ = pi0;
    laminit_present_ = havepi0;


    // read the rest of the nl file
    int retcode = pfgh_read(nl, ASL_return_read_err | ASL_findgroups);

    switch (retcode) {
        case ASL_readerr_none: {
        }
            break;
        case ASL_readerr_nofile: {
            printf("Cannot open .nl file\n");
            exit(1);
        }
            break;
        case ASL_readerr_corrupt: {
            printf("Corrupt .nl file\n");
            exit(1);
        }
            break;
        case ASL_readerr_bug: {
            printf("Bug in .nl reader\n");
            exit(1);
        }
            break;
        default: {
            printf("Unknown error in stub file read. retcode = %d\n", retcode);
            exit(1);
        }
    }


    // get the sense of the obj function (1 for min or -1 for max)
    obj_sense_ = 1;
    if (objtype[0] != 0) {
        obj_sense_ = -1;
    }

    // tell AMPL info about which obj etc.
    hesset(1, 0, 1, 0, nlc);

    int coeff_obj = 1;
    int mult_supplied = 1; // multipliers will be supplied
    int uptri = 1; // only need the upper triangular part
    nnz_hes_lag_ = sphsetup(-1, coeff_obj, mult_supplied, uptri);

}

AmplInterface::~AmplInterface() {
    ASL_pfgh *asl = asl_;
    delete[] xinit_;
    xinit_ = NULL;
    delete[] xinit_present_;
    xinit_present_ = NULL;
    delete[] laminit_;
    laminit_ = NULL;
    delete[] laminit_present_;
    laminit_present_ = NULL;

    if (asl) {
        ASL *asl_to_free = (ASL *) asl_;
        ASL_free(&asl_to_free);
        asl_ = NULL;
    }

}


void AmplInterface::get_nlp_dimensions(int &n_x, int &n_g, int &nnz_jac_g, int &nnz_hes_lag) const {
    ASL_pfgh *asl = asl_;
    _ASSERT_(asl_);

    n_x = n_var; // # of variables (variable types have been asserted in the constructor
    n_g = n_con; // # of constraints
    nnz_jac_g = nzc; // # of non-zeros in the jacobian
    nnz_hes_lag = nnz_hes_lag_; // # of non-zeros in the hessian
}

int AmplInterface::get_n_vars() const {
    ASL_pfgh *asl = asl_;
    _ASSERT_(asl_);
    int n_x;
    n_x = n_var;
    return n_x;
}

int AmplInterface::get_n_constraints() const {
    ASL_pfgh *asl = asl_;
    _ASSERT_(asl_);
    int n_c;
    n_c = n_con;
    return n_c;
}

int AmplInterface::get_nnz_jac_g() const {
    ASL_pfgh *asl = asl_;
    _ASSERT_(asl_);
    int nnz_jac_g;
    nnz_jac_g = nzc;
    return nnz_jac_g;
}

int AmplInterface::get_nnz_hessian_lag() const {
    ASL_pfgh *asl = asl_;
    _ASSERT_(asl);
    int nnz_hes_lag;
    nnz_hes_lag = nnz_hes_lag_;
    return nnz_hes_lag;
}

void AmplInterface::get_lower_bounds_x(double *invec, int n) {
    ASL_pfgh *asl = asl_;
    _ASSERT_(asl_);
    _ASSERT_(n == n_var);
    for (int i = 0; i < n; i++) {
        invec[i] = LUv[2 * i];
    }
}

void AmplInterface::get_upper_bounds_x(double *invec, int n) {
    ASL_pfgh *asl = asl_;
    _ASSERT_(asl_);
    _ASSERT_(n == n_var);

    for (int i = 0; i < n; i++) {
        invec[i] = LUv[2 * i + 1];
    }
}

void AmplInterface::get_lower_bounds_g(double *invec, int m) {
    ASL_pfgh *asl = asl_;
    _ASSERT_(asl_);
    _ASSERT_(m == n_con);
    for (int i = 0; i < m; i++) {
        invec[i] = LUrhs[2 * i];
    }
}

void AmplInterface::get_upper_bounds_g(double *invec, int m) {
    ASL_pfgh *asl = asl_;
    _ASSERT_(asl_);
    _ASSERT_(m == n_con);

    for (int i = 0; i < m; i++) {
        invec[i] = LUrhs[2 * i + 1];
    }
}

void AmplInterface::get_bounds_info(double *xl, double *xu, int n, double *gl, double *gu, int m) {
    ASL_pfgh *asl = asl_;
    _ASSERT_(asl_);
    _ASSERT_(n == n_var);
    _ASSERT_(m == n_con);

    for (int i = 0; i < n; ++i) {
        xl[i] = LUv[2 * i];
        xu[i] = LUv[2 * i + 1];
    }

    for (int i = 0; i < m; ++i) {
        gl[i] = LUrhs[2 * i];
        gu[i] = LUrhs[2 * i + 1];
    }
}

// maybe update this to keep a separate one that does not get overwritten
void AmplInterface::get_init_x(double *invec, int n) {
    ASL_pfgh *asl = asl_;
    _ASSERT_(asl_);
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
    ASL_pfgh *asl = asl_;
    _ASSERT_(asl_);

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


void AmplInterface::get_starting_point(double *x, int nx, double *lam, int ng) {
    ASL_pfgh *asl = asl_;
    _ASSERT_(asl_);
    _ASSERT_(nx == n_var);

    for (int i = 0; i < nx; i++) {
        if (havex0[i]) {
            x[i] = X0[i];
        } else {
            x[i] = 0.0;
        }
    }

    // get dual starting point
    if (n_con == 0) { return; } // unconstrained problem or do not want to use the exist dual values
    _ASSERT_(ng == n_con);

    for (int i = 0; i < ng; i++) {
        if (havepi0[i]) {
            lam[i] = pi0[i];
        } else {
            lam[i] = 0.0;
        }
    }
}

bool AmplInterface::eval_f(double *const_x, int nx, double& f) {
    ASL_pfgh *asl = asl_;
    _ASSERT_(asl_);
    _ASSERT_(n_obj == 1 && "AMPL problem must have a single objective function");

    int nerror = 1;
    double retval = objval(obj_no, (double *) const_x, &nerror);

    if(nerror!=0){
        return false;
    }
    f = obj_sense_ * retval;
    return true;

}

bool AmplInterface::eval_deriv_f(double *const_x, double *deriv_f, int nx) {
    ASL_pfgh *asl = asl_;
    _ASSERT_(asl_);
    _ASSERT_(n_obj == 1 && "AMPL problem must have a single objective function");

    int nerror = 1;
    objgrd(obj_no, (double *) const_x, deriv_f, &nerror);

    if(nerror!=0){
        return false;
    }

    if (obj_sense_ == -1) {
        for (int i = 0; i < nx; i++) {
            deriv_f[i] *= -1.0;
        }
    }
    return true;
}

void AmplInterface::struct_jac_g(int *irow, int *jcol, int nnz_jac_g) {
    ASL_pfgh *asl = asl_;
    _ASSERT_(asl_);
    _ASSERT_(nnz_jac_g == nzc);
    _ASSERT_(irow && jcol);

    // get the non zero structure of the jacobian of c
    int current_nz = 0;
    for (int i = 0; i < n_con; i++) {
        for (cgrad *cg = Cgrad[i]; cg; cg = cg->next) {
            irow[cg->goff] = i + 1;
            jcol[cg->goff] = cg->varno + 1;
            current_nz++;
        }
    }
    _ASSERT_(current_nz == nnz_jac_g);
}

bool AmplInterface::eval_jac_g(double *const_x, int nx, double *jac_g_values, int nnz_jac_g) {
    ASL_pfgh *asl = asl_;
    _ASSERT_(asl_);
    _ASSERT_(nx == n_var);
    _ASSERT_(nnz_jac_g == nzc);
    _ASSERT_(jac_g_values);

    int nerror = 1;
    jacval((double *) const_x, jac_g_values, &nerror);
    if(nerror!=0){
        return false;
    }
    return true;
}

void AmplInterface::struct_hes_lag(int *irow, int *jcol, int nnz_hes_lag) {
    ASL_pfgh *asl = asl_;
    _ASSERT_(asl_);
    _ASSERT_(nnz_hes_lag_ == nnz_hes_lag);

    // setup the structure
    int k = 0;
    for (int i = 0; i < n_var; i++) {
        for (int j = sputinfo->hcolstarts[i]; j < sputinfo->hcolstarts[i + 1]; j++) {
            irow[k] = i + 1;
            jcol[k] = sputinfo->hrownos[j] + 1;
            k++;
        }
    }
}

bool AmplInterface::eval_g(double *const_x, int nx, double *g, int ng) {
    ASL_pfgh *asl = asl_;
    _ASSERT_(nx == n_var);
    _ASSERT_(ng == n_con);

    // call AMPL to evaluate the constraints
    int nerror = 1;
    conval((double *) const_x, g, &nerror);
    _ASSERT_(nerror == 0);
    if(nerror!=0){
        return false;
    }
    return true;
}

bool AmplInterface::eval_hes_lag(double *const_x, int nx, double *const_lam, int nc, double *hes_lag, int nnz_hes_lag) {
    ASL_pfgh *asl = asl_;
    _ASSERT_(asl_);
    _ASSERT_(nx == n_var);
    _ASSERT_(nc == n_con);
    _ASSERT_(n_obj == 1);
    _ASSERT_(nnz_hes_lag_ == nnz_hes_lag);

    // NOTE: This must be called AFTER a call to objval and conval
    // (i.e. You must call eval_f and eval_c with the same x before calling this)
    double OW = obj_sense_;
    sphes(hes_lag, -1, &OW, (double *) const_lam);
    return true;
}

void AmplInterface::finalize_solution(int status, double *const_x, int nx, double *const_lam, int nc) {

    ASL_pfgh *asl = asl_;
    _ASSERT_(asl);
    _ASSERT_(const_x && const_lam);
    std::string message;

    if (status == 0) { //SUCCESS
        message = "Optimal Solution Found";
        solve_result_num = 0;
    } else if (status == 2) { //MAXITER_EXCEEDED
        message = "Maximum Number of Iterations Exceeded.";
        solve_result_num = 400;
    } else if (status == 1) { //FAILED
        message = "Iterates diverging; problem might be unbounded.";
        solve_result_num = 300;
    } else {
        message = "Unknown Error";
        solve_result_num = 502;
    }
    char *cmessage = new char[message.length() + 1];
    strcpy(cmessage, message.c_str());

    write_sol(cmessage, const_cast<double *>(const_x), const_cast<double *>(const_lam), 0);
    delete[] cmessage;
}

AmplInterface_file::AmplInterface_file()
   : AmplInterface()
{}

FILE* AmplInterface_file::open_nl(ASL_pfgh *asl, char* stub)
{
    assert_exit(stub, "No .nl file was specified.");
    return jac0dim(stub, (int) strlen(stub));
}

AmplInterface_str::AmplInterface_str(char* nl, size_t size)
   : AmplInterface(),
     nl_content(nl),
     nl_size(size)
{}

FILE* AmplInterface_str::open_nl(ASL_pfgh *asl, char* stub)
{
   // Ignore the stub and use the cached NL file content
   FILE* nl = fmemopen(this->nl_content, this->nl_size, "rb");
   return jac0dim_FILE(nl);
}


extern "C"
{
AmplInterface *EXTERNAL_AmplInterface_new_file(char *nlfilename) {
   AmplInterface* ans = new AmplInterface_file();
   ans->initialize(nlfilename);
   return ans;
}

AmplInterface *EXTERNAL_AmplInterface_new_str(char *nl, size_t size) {
   AmplInterface* ans = new AmplInterface_str(nl, size);
   ans->initialize("membuf.nl");
   return ans;
}

AmplInterface *EXTERNAL_AmplInterface_new(char *nlfilename) {
   return EXTERNAL_AmplInterface_new_file(nlfilename);
}

int EXTERNAL_AmplInterface_n_vars(AmplInterface *p_ai) { return p_ai->get_n_vars(); }

int EXTERNAL_AmplInterface_n_constraints(AmplInterface *p_ai) { return p_ai->get_n_constraints(); }

int EXTERNAL_AmplInterface_nnz_jac_g(AmplInterface *p_ai) { return p_ai->get_nnz_jac_g(); }

int EXTERNAL_AmplInterface_nnz_hessian_lag(AmplInterface *p_ai) { return p_ai->get_nnz_hessian_lag(); }

void EXTERNAL_AmplInterface_get_bounds_info(AmplInterface *p_ai,
                                            double *xl,
                                            double *xu,
                                            int n,
                                            double *gl,
                                            double *gu,
                                            int m) { p_ai->get_bounds_info(xl, xu, n, gl, gu, m); }

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

void EXTERNAL_AmplInterface_x_init(AmplInterface *p_ai, double *invec, int n) { p_ai->get_init_x(invec, n); }

void EXTERNAL_AmplInterface_lam_init(AmplInterface *p_ai, double *invec, int n) {
    p_ai->get_init_multipliers(invec, n);
}

void EXTERNAL_AmplInterface_starting_point(AmplInterface *p_ai, double *x, int nx, double *lam,
                                           int ng) { p_ai->get_starting_point(x, nx, lam, ng); }

bool EXTERNAL_AmplInterface_eval_f(AmplInterface *p_ai, double *invec, int n, double& f)
{ return p_ai->eval_f(invec, n, f); }

bool EXTERNAL_AmplInterface_eval_deriv_f(AmplInterface *p_ai, double *const_x, double *deriv_f, int nx) {
    return p_ai->eval_deriv_f(const_x, deriv_f, nx);
}

void EXTERNAL_AmplInterface_struct_jac_g(AmplInterface *p_ai, int *irow, int *jcol, int nnz_jac_g) {
    p_ai->struct_jac_g(irow, jcol, nnz_jac_g);
}

void EXTERNAL_AmplInterface_struct_hes_lag(AmplInterface *p_ai, int *irow, int *jcol,
                                           int nnz_hes_lag) { p_ai->struct_hes_lag(irow, jcol, nnz_hes_lag); }

bool EXTERNAL_AmplInterface_eval_g(AmplInterface *p_ai, double *const_x, int nx, double *g, int ng) {
    return p_ai->eval_g(const_x, nx, g, ng);
}

bool EXTERNAL_AmplInterface_eval_jac_g(AmplInterface *p_ai, double *const_x, int nx, double *jac_g_values,
                                       int nnz_jac_g) { return p_ai->eval_jac_g(const_x, nx, jac_g_values, nnz_jac_g); }

bool EXTERNAL_AmplInterface_eval_hes_lag(AmplInterface *p_ai, double *const_x, int nx,
                                         double *const_lam, int nc, double *hes_lag, int nnz_hes_lag) {
    return p_ai->eval_hes_lag(const_x, nx, const_lam, nc, hes_lag, nnz_hes_lag);
}

void EXTERNAL_AmplInterface_finalize_solution(AmplInterface *p_ai, int status, double *const_x,
                                              int nx, double *const_lam, int nc) {
    p_ai->finalize_solution(status, const_x, nx, const_lam, nc);
}

void EXTERNAL_AmplInterface_free_memory(AmplInterface *p_ai) { p_ai->~AmplInterface(); }

void EXTERNAL_AmplInterface_map_g_indices(int* invec, int size, int* map, int size_map)
{
    std::map<int, int> g_to_c;

    // populate map
    for(int i=0; i<size_map;++i) {
        g_to_c.insert(std::pair<int, int>(map[i], i));
    }

    for(int i=0; i<size;++i) {
        invec[i] = g_to_c[invec[i]];
    }
}

}

