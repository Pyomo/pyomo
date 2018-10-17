#include "interval.hpp"
#include "mccormick.hpp"
typedef mc::Interval I;
typedef mc::McCormick<I> MC;


//Build pyomo expression in MC++
void *createVar(double lb, double pt, double ub, int count, int index)
{
    MC var1( I( lb, ub ), pt );
    var1.sub(count, index);

    void *ans = new MC(var1);

    return ans;
}

void *createConstant(double cons)
{
    void *ans = new MC(cons);

    return ans;
}

MC *mult(MC *var1, MC *var2)
{
    MC F = *var1 * *var2;

    MC *ans = new MC(F);
    return ans;
}

MC *add(MC *var1, MC *var2)
{
    MC F = *var1 + *var2;

    MC *ans = new MC(F);
    return ans;
}

MC *power(MC *var1, MC *var2)
{
    //Extract int value from var2
    //Pow() will only take an integer value
    MC var = *var2;
    double doub = var.l();
    int exponent = (int)doub;

    MC F = pow(*var1, exponent);

    MC *ans = new MC(F);
    return ans;
}


MC *monomial(MC *var1, MC *var2)
{
    MC F = *var1 * *var2;

    MC *ans = new MC(F);
    return ans;
}

MC *reciprocal(MC *var1, MC *var2)
{
    MC F = inv(*var2);

    MC *ans = new MC(F);
    return ans;
}

MC *negation(MC *var1)
{
    MC F = 0 - *var1;

    MC *ans = new MC(F);
    return ans;
}

MC *abs(MC *var1)
{
    MC F = fabs(*var1);

    MC *ans = new MC(F);
    return ans;
}

MC *trigSin(MC *var1)
{
    MC F = sin(*var1);

    MC *ans = new MC(F);
    return ans;
}

MC *trigCos(MC *var1)
{
    MC F = cos(*var1);

    MC *ans = new MC(F);
    return ans;
}

MC *trigTan(MC *var1)
{
    MC F = tan(*var1);

    MC *ans = new MC(F);
    return ans;
}

MC *atrigSin(MC *var1)
{
    MC F = asin(*var1);

    MC *ans = new MC(F);
    return ans;
}

MC *atrigCos(MC *var1)
{
    MC F = acos(*var1);

    MC *ans = new MC(F);
    return ans;
}

MC *atrigTan(MC *var1)
{
    MC F = atan(*var1);

    MC *ans = new MC(F);
    return ans;
}

MC *NPV(MC *var1)
{
    MC F = *var1;

    MC *ans = new MC(F);
    return ans;
}

void *displayOutput(void *ptr)
{
    MC *var  = (MC*) ptr;
    MC F = *var;
    std::cout << "F: " << F << std::endl;
}

MC *exponential(MC *var1)
{
    MC F = exp(*var1);

    MC *ans = new MC(F);
    return ans;
}

//Get usable information from MC++
double lower(MC *expr)
{
    MC F = *expr;
    double Flb = F.l();
    return Flb;
}

double upper(MC *expr)
{
    MC F = *expr;
    double Fub = F.u();
    return Fub;
}

double concave(MC *expr)
{
    MC F = *expr;
    double Fcc = F.cc();
    return Fcc;
}

double convex(MC *expr)
{
    MC F = *expr;
    double Fcv = F.cv();
    return Fcv;
}


double subcc(MC *expr, int index)
{
    MC F = *expr;
    double Fccsub = F.ccsub(index);
    return Fccsub;
}

double subcv(MC *expr, int index)
{
    MC F = *expr;
    double Fcvsub = F.cvsub(index);
    return Fcvsub;
}

extern "C"
{
    void *new_createVar(double lb, double pt, double ub, int count, int index) 
    {
        void *ans = createVar(lb, pt, ub, count, index);
        return ans;
    }


    void *new_createConstant(double cons)
    {
        void *ans = createConstant(cons);
        return ans;
    }

    MC *new_mult(MC *var1, MC *var2)
    {
        MC *ans = mult(var1, var2);
        return ans;
    }

    MC *new_add(MC *var1, MC *var2)
    {
        MC *ans = add(var1, var2);
        return ans;
    }

    MC *new_power(MC *var1, MC *var2)
    {
        MC *ans = power(var1, var2);
        return ans;
    }

    MC *new_monomial(MC *var1, MC *var2)
    {
        MC *ans = monomial(var1, var2);
        return ans;
    }

    MC *new_reciprocal(MC *var1, MC *var2)
    {
        MC *ans = reciprocal(var1, var2);
        return ans;
    }

    MC *new_negation(MC *var1)
    {
        MC *ans = negation(var1);
        return ans;
    }

    MC *new_abs(MC *var1)
    {
        MC *ans = abs(var1);
        return ans;
    }

    MC *new_trigSin(MC *var1)
    {
        MC *ans = trigSin(var1);
        return ans;
    }

    MC *new_trigCos(MC *var1)
    {
        MC *ans = trigCos(var1);
        return ans;
    }

    MC *new_trigTan(MC *var1)
    {
        MC *ans = trigTan(var1);
        return ans;
    }

    MC *new_atrigSin(MC *var1)
    {
        MC *ans = atrigSin(var1);
        return ans;
    }

    MC *new_atrigCos(MC *var1)
    {
        MC *ans = atrigCos(var1);
        return ans;
    }

    MC *new_atrigTan(MC *var1)
    {
        MC *ans = atrigTan(var1);
        return ans;
    }

    MC *new_NPV(MC *var1)
    {
        MC *ans = NPV(var1);
        return ans;
    }

    void *new_displayOutput(void *ptr)
    {
        displayOutput(ptr);
    }

    MC *new_exponential(MC *ptr1)
    {
        MC *ans = exponential(ptr1);
        return ans;
    }

    double new_lower(MC *expr)
    {
        double ans = lower(expr);
        return ans;
    }

    double new_upper(MC *expr)
    {
        double ans = upper(expr);
        return ans;
    }

    double new_concave(MC *expr)
    {
        double ans = concave(expr);
        return ans;
    }

    double new_convex(MC *expr)
    {
        double ans = convex(expr);
        return ans;
    }

    double new_subcc(MC *expr, int index)
    {
        double ans = subcc(expr, index);
        return ans;
    }

        double new_subcv(MC *expr, int index)
    {
        double ans = subcv(expr, index);
        return ans;
    }
}

//g++ -I ~/MC++/mcpp/src/mc -I /usr/include/python2.7/ -fPIC -O2 -c mcppInterface.cpp

//g++ -shared mcppInterface.o -o mcppInterface.so