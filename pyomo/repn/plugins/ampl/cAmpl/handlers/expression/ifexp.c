#include "ifexp.h"
#include "../../cAmpl.h"

int _handle_ifexp(PyObject * context, PyObject * exp, PyObject ** ampl_repn) {
    // assert(len(exp._args) == 1)
    PyObject * exp__args = PyObject_GetAttrString(exp, "_args");
    Py_ssize_t len_exp__args = PySequence_Length(exp__args);
    if(len_exp__args != 1) {
        PyErr_SetString(PyExc_AssertionError, "");
        return ERROR;
    }

    // child_repn = generate_ampl_repn(exp._args[0])
    PyObject * exp__args_0 = PySequence_GetItem(exp__args, 0);
    PyObject * child_repn = recursive_generate_ampl_repn(context, exp__args_0);
    if(child_repn == NULL) return ERROR;
    Py_DECREF(exp__args_0);
    Py_DECREF(exp__args);

    // ampl_repn._nonlinear_expr = exp
    PyObject_SetAttrString(*ampl_repn, "_nonlinear_expr", exp);

    // ampl_repn._nonlinear_vars = child_repn._nonlinear_vars
    PyObject * child_repn__nlv = PyObject_GetAttrString(child_repn, "_nonlinear_vars");
    PyObject_SetAttrString(*ampl_repn, "_nonlinear_vars", child_repn__nlv);
    Py_DECREF(child_repn__nlv);

    // ampl_repn._nonlinear_vars.update(child_repn._linear_terms_var)
    PyObject * ampl_repn__nlv = PyObject_GetAttrString(*ampl_repn, "_nonlinear_vars");
    PyObject * child_repn__ltv = PyObject_GetAttrString(child_repn, "_linear_terms_var");
    PyObject_CallMethod(ampl_repn__nlv, "update", "(O)", child_repn__ltv);
    Py_DECREF(child_repn__ltv);
    Py_DECREF(ampl_repn__nlv);

    Py_DECREF(child_repn);

    assert((*ampl_repn)->ob_refcnt == 1);
    return TRUE;
}
