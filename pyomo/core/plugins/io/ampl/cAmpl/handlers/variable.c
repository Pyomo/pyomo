#include "variable.h"

int _handle_variable(PyObject * context, PyObject * exp, PyObject ** ampl_repn) {
    // if exp.fixed:
    PyObject * exp_fixed = PyObject_GetAttrString(exp, "fixed");
    if(exp_fixed != NULL && PyObject_IsTrue(exp_fixed)) {
        // ampl_repn._constant = exp.value
        PyObject * exp_value = PyObject_GetAttrString(exp, "value");
        PyObject_SetAttrString(*ampl_repn, "_constant", exp_value);

        // return ampl_repn
        Py_DECREF(exp_value);
        Py_DECREF(exp_fixed);
        return TRUE;
    }

    PyObject * exp_name = PyObject_GetAttrString(exp, "name");

    // ampl_repn._linear_terms_coef[exp.name] = 1.0
    PyObject * _linear_terms_coef = PyObject_GetAttrString(*ampl_repn, "_linear_terms_coef");
    PyObject * one = Py_BuildValue("f", 1.0);
    PyDict_SetItem(_linear_terms_coef, exp_name, one);
    Py_DECREF(_linear_terms_coef);

    // ampl_repn._linear_terms_var[exp.name] = exp
    PyObject * _linear_terms_var = PyObject_GetAttrString(*ampl_repn, "_linear_terms_var");
    PyDict_SetItem(_linear_terms_var, exp_name, exp);
    Py_DECREF(_linear_terms_var);

    // Should return generated ampl_repn
    Py_DECREF(exp_name);
    return TRUE;
}
