/**___________________________________________________________________________
 *
 * Pyomo: Python Optimization Modeling Objects
 * Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
 * Under the terms of Contract DE-NA0003525 with National Technology and
 * Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
 * rights in this software.
 * This software is distributed under the 3-clause BSD License.
 * ___________________________________________________________________________
**/
#include "interval.hpp"
#include "mccormick.hpp"
#include <sstream>
#include <string>
typedef mc::Interval I;
typedef mc::McCormick<I> MC;

// Module-level variables as utilities to pass information back to Python
std::string lastException;
std::string lastDisplay;

extern "C"
{
    const char* version_string = "19.11.12";

    // Version number
    const char* get_version() {
        return version_string;
    }

    // Functions to build up an MC object
    void* newVar(double lb, double pt, double ub, int count, int index)
    {
        MC var( I( lb, ub ), pt );
        var.sub(count, index);
        return (void *) new MC(var);
    }

    void* newConstant(double cons)
    {
        return (void *) new MC(cons);
    }

    MC* multiply(MC *var1, MC *var2)
    {
        return (MC*) new MC( (*var1) * (*var2) );
    }

    MC* divide(MC *var1, MC *var2)
    {
        return (MC*) new MC( (*var1) / (*var2) );
    }

    MC* add(MC *var1, MC *var2)
    {
        return (MC*) new MC( (*var1) + (*var2) );
    }

    MC* power(MC *arg1, MC *arg2)
    {
        int exponent = (int)((*arg2).l());
        return (MC*) new MC( pow(*arg1, exponent) );
    }
    MC* powerf(MC *arg1, MC *arg2)
    {
        double exponent = (double)((*arg2).l());
        return (MC*) new MC( pow(*arg1, exponent) );
    }
    MC* powerx(MC *arg1, MC *arg2)
    {
        // exponential is potentially a variable. Using reformulation
        // x^n = exp(n log(x))
        return (MC*) new MC( exp(*arg2 * log(*arg1)) );
    }

    // Other Unary functions
    MC* mc_sqrt(MC *arg1) {return new MC( sqrt(*arg1) );}
    MC* reciprocal(MC *arg1) {return new MC( inv(*arg1) );}
    MC* negation(MC *arg1) {return new MC(0 - *arg1);}
    MC* mc_abs(MC *arg1) {return new MC( fabs(*arg1) );}
    MC* trigSin(MC *arg1) {return new MC( sin(*arg1) );}
    MC* trigCos(MC *arg1) {return new MC( cos(*arg1) );}
    MC* trigTan(MC *arg1) {return new MC( tan(*arg1) );}
    MC* atrigSin(MC *arg1) {return new MC( asin(*arg1) );}
    MC* atrigCos(MC *arg1) {return new MC( acos(*arg1) );}
    MC* atrigTan(MC *arg1) {return new MC( atan(*arg1) );}
    MC* exponential(MC *arg1) {return new MC( exp(*arg1) );}
    MC* logarithm(MC *arg1) {return new MC( log(*arg1) );}

    // Get the MC++ string representation of the MC object
    const char* toString(MC *arg)
    {
        std::ostringstream Fstrm;
        Fstrm << *arg << std::flush;
        lastDisplay = Fstrm.str();
        return lastDisplay.c_str();
    }

    // Lower and upper interval bounds on expression
    double lower(MC *expr) { return (double) (*expr).l(); }
    double upper(MC *expr) { return (double) (*expr).u(); }
    // Concave and convex envelope values for expr at current variable values
    double concave(MC *expr) { return (double) (*expr).cc(); }
    double convex(MC *expr) { return (double) (*expr).cv(); }
    // Subgradients to expr with respect to variable index
    double subcc(MC *expr, int index) { return (double) (*expr).ccsub(index); }
    double subcv(MC *expr, int index) { return (double) (*expr).cvsub(index); }

    // Release pointers when done to avoid memory leaks
    void release(MC *expr)
    {
        delete expr;
    }

    // Catch MC++ exceptions so that we don't core dump,
    // saving the exception message so that Python can retrieve it later.
    MC* try_unary_fcn(MC*(*fcn)(MC*), MC *arg)
    {
        try {
            return fcn(arg);
        } catch (MC::Exceptions &e) {
            lastException = e.what();
            return NULL;
        }
    }
    MC* try_binary_fcn(MC*(*fcn)(MC*, MC*), MC *arg1, MC *arg2)
    {
        try {
            return fcn(arg1, arg2);
        } catch (MC::Exceptions &e) {
            lastException = e.what();
            return NULL;
        }
    }
    const char* get_last_exception_message()
    {
        return lastException.c_str();
    }
}

// Manual compilation commands:
// g++ -I ~/.solvers/MC++/mcpp/src/3rdparty/fadbad++ -I ~/.solvers/MC++/mcpp/src/mc -I /usr/include/python3.6/ -fPIC -O2 -c mcppInterface.cpp
// g++ -shared mcppInterface.o -o mcppInterface.so
