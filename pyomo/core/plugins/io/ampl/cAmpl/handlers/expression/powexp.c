#include "powexp.h"

int _handle_powexp(PyObject * context, PyObject * exp, PyObject ** ampl_repn) {
    // result object
    PyObject * _result = NULL;

    // assert(len(exp._args) == 2)
    PyObject * exp__args = PyObject_GetAttrString(exp, "_args");
    Py_ssize_t len_exp__args = PySequence_Length(exp__args);
    if(len_exp__args != 2) {
        PyErr_SetString(PyExc_AssertionError, "");
        return ERROR;
    }

    // base_repn = generate_ampl_repn(exp._args[0])
    PyObject * exp__args_0 = PySequence_GetItem(exp__args, 0);
    PyObject * base_repn = recursive_generate_ampl_repn(context, exp__args_0);
    if(base_repn == NULL) return ERROR;
    Py_DECREF(exp__args_0);

    // exponent_repn = generate_ampl_repn(exp._args[1])
    PyObject * exp__args_1 = PySequence_GetItem(exp__args, 1);
    PyObject * exponent_repn = recursive_generate_ampl_repn(context, exp__args_1);
    if(exponent_repn == NULL) return ERROR;
    Py_DECREF(exp__args_1);

    Py_DECREF(exp__args);

    // if base_repn.is_constant() and exponent_repn.is_constant():

    // opt
    PyObject * br_ic = PyObject_CallMethod(base_repn, "is_constant", "()");
    PyObject * er_ic = PyObject_CallMethod(exponent_repn, "is_constant", "()");
    PyObject * br__constant = PyObject_GetAttrString(base_repn, "_constant");
    PyObject * er__constant = PyObject_GetAttrString(exponent_repn, "_constant");

    if(PyObject_IsTrue(br_ic) && PyObject_IsTrue(er_ic)) {
        // ampl_repn._constant = base_repn._constant**exponent_repn._constant
        _result = PyNumber_Power(br__constant, er__constant, Py_None);
        PyObject_SetAttrString(*ampl_repn, "_constant", _result);
        Py_DECREF(_result); _result = NULL;

    // elif exponent_repn.is_constant() and exponent_repn._constant == 1.0:
    } else if(PyObject_IsTrue(er_ic) && PyFloat_AS_DOUBLE(er__constant) == 1.0) {
        // ampl_repn = base_repn
        Py_DECREF(*ampl_repn);
        *ampl_repn = base_repn;
        Py_INCREF(*ampl_repn);

    // elif exponent_repn.is_constant() and exponent_repn._constant == 0.0:
    } else if(PyObject_IsTrue(er_ic) && PyFloat_AS_DOUBLE(er__constant) == 0.0) {
        // ampl_repn._constant = 1.0
        PyObject * _one = Py_BuildValue("f", 1.0);
        PyObject_SetAttrString(*ampl_repn, "_constant", _one);
        Py_DECREF(_one);

    // else:
    } else {
        // instead, let's just return the expression we are given and only
        // use the ampl_repn for the vars
        // ampl_repn._nonlinear_expr = exp
        PyObject_SetAttrString(*ampl_repn, "_nonlinear_expr", exp);

        // ampl_repn._nonlinear_vars = base_repn._nonlinear_vars
        PyObject * base_repn__nlv = PyObject_GetAttrString(base_repn, "_nonlinear_vars");
        PyObject_SetAttrString(*ampl_repn, "_nonlinear_vars", base_repn__nlv);
        Py_DECREF(base_repn__nlv);

        // opt
        PyObject * ampl_repn__nlv = PyObject_GetAttrString(*ampl_repn, "_nonlinear_vars");
        PyObject * _update = PyString_FromString("update");

        // ampl_repn._nonlinear_vars.update(exponent_repn._nonlinear_vars)
        PyObject * exponent_repn__nlv = PyObject_GetAttrString(exponent_repn, "_nonlinear_vars");
        PyObject_CallMethodObjArgs(ampl_repn__nlv, _update, exponent_repn__nlv, NULL);
        Py_DECREF(exponent_repn__nlv);

        // ampl_repn._nonlinear_vars.update(base_repn._linear_terms_var)
        PyObject * base_repn__ltv = PyObject_GetAttrString(base_repn, "_linear_terms_var");
        PyObject_CallMethodObjArgs(ampl_repn__nlv, _update, base_repn__ltv, NULL);
        Py_DECREF(base_repn__ltv);

        // ampl_repn._nonlinear_vars.update(exponent_repn._linear_terms_var)
        PyObject * exponent_repn__ltv = PyObject_GetAttrString(exponent_repn, "_linear_terms_var");
        PyObject_CallMethodObjArgs(ampl_repn__nlv, _update, exponent_repn__ltv, NULL);
        Py_DECREF(exponent_repn__ltv);

        // cleanup
        Py_DECREF(_update);
        Py_DECREF(ampl_repn__nlv);
    }

    // cleanup
    Py_DECREF(br_ic);
    Py_DECREF(er_ic);
    Py_DECREF(br__constant);
    Py_DECREF(er__constant);

    // cleanup
    Py_DECREF(base_repn);
    Py_DECREF(exponent_repn);

    assert((*ampl_repn)->ob_refcnt == 1);
    return TRUE;
}
