#include "fixedval.h"

int _handle_fixedval(PyObject * context, PyObject * exp, PyObject ** ampl_repn) {
    // ampl_repn._constant = exp.value
    PyObject * exp_value = PyObject_GetAttrString(exp, "value");
    PyObject_SetAttrString(*ampl_repn, "_constant", exp_value);
    Py_DECREF(exp_value);

    return TRUE;
}
