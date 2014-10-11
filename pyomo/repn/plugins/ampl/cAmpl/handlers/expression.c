#include "expression.h"

int _handle_expression(PyObject * context, PyObject * exp, PyObject ** ampl_repn) {
    // exp_type = type(exp)
    PyObject * exp_type = PyObject_Type(exp);

    // Get various expression types
    PyObject * pyomo_MOD = PyImport_ImportModule("pyomo.core");
    PyObject * base_MOD = PyObject_GetAttrString(pyomo_MOD, "base");
    PyObject * expr_MOD = PyObject_GetAttrString(base_MOD, "expr");
    Py_XDECREF(pyomo_MOD);
    Py_XDECREF(base_MOD);

    // if expr_type is expr._SumExpression:
    PyObject * _SumExpression_CLS = PyObject_GetAttrString(expr_MOD, "_SumExpression");
    if(exp_type == _SumExpression_CLS) {
        Py_XDECREF(_SumExpression_CLS);
        Py_XDECREF(expr_MOD);
        return _handle_sumexp(context, exp, ampl_repn);
    }
    Py_XDECREF(_SumExpression_CLS);
    
    // elif exp_type is expr._ProductExpression:
    PyObject * _ProductExpression_CLS = PyObject_GetAttrString(expr_MOD, "_ProductExpression");
    if(exp_type == _ProductExpression_CLS) {
        Py_XDECREF(_ProductExpression_CLS);
        Py_XDECREF(expr_MOD);
        return _handle_prodexp(context, exp, ampl_repn);
    } 
    Py_XDECREF(_ProductExpression_CLS);
    
    // elif exp_type is expr._PowExpression:
    PyObject * _PowExpression_CLS = PyObject_GetAttrString(expr_MOD, "_PowExpression");
    if(exp_type == _PowExpression_CLS) {
        Py_XDECREF(_PowExpression_CLS);
        Py_XDECREF(expr_MOD);
        return _handle_powexp(context, exp, ampl_repn);
    }
    Py_XDECREF(_PowExpression_CLS);
    
    // elif exp_type is expr._IntrinsicFunctionExpression:
    PyObject * _IntrinsicFunctionExpression_CLS = PyObject_GetAttrString(expr_MOD, "_IntrinsicFunctionExpression");
    if(exp_type == _IntrinsicFunctionExpression_CLS) {
        Py_XDECREF(_IntrinsicFunctionExpression_CLS);
        Py_XDECREF(expr_MOD);
        return _handle_ifexp(context, exp, ampl_repn);
    }
    Py_XDECREF(_IntrinsicFunctionExpression_CLS);
    
    // else:
    // raise ValueError, "Unsupported expression type: "+str(exp)
    PyObject * error_msg = PyString_FromString("Unsupported expression type: ");
    PyObject * exp_str = PyObject_Str(exp);
    if(exp_str == NULL) {
        exp_str = PyString_FromString("<nil>");
    }
    PyString_ConcatAndDel(&error_msg, exp_str);
    PyErr_SetObject(PyExc_ValueError, error_msg);

    // Free types
    Py_XDECREF(expr_MOD);

    // Raise the error
    return ERROR;
}
