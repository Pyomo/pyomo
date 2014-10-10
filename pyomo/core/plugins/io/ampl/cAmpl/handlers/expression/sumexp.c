#include "sumexp.h"
#include "../../cAmpl.h"

int _handle_sumexp(PyObject * context, PyObject * exp, PyObject ** ampl_repn) {
    // Objects usable for temp calculations
    PyObject * _result, * _result2;

    // ampl_repn._constant = exp._const
    PyObject * exp__const = PyObject_GetAttrString(exp, "_const");
    PyObject_SetAttrString(*ampl_repn, "_constant", exp__const);
    Py_DECREF(exp__const);

    // ampl_repn._nonlinear_expr = None
    PyObject_SetAttrString(*ampl_repn, "_nonlinear_expr", Py_None);

    // for i in xrange(len(exp._args)):
    PyObject * exp__args = PyObject_GetAttrString(exp, "_args");
    Py_ssize_t len_exp__args = PySequence_Length(exp__args);
    Py_ssize_t i;
    PyObject * exp__coef = PyObject_GetAttrString(exp, "_coef"); // opt
    for(i = 0; i < len_exp__args; i++) {
        // exp_coef = exp._coef[i]
        PyObject * exp_coef = PySequence_GetItem(exp__coef, i);

        // child_repn = generate_ampl_repn(exp._args[i])
        PyObject * exp__args_i = PySequence_GetItem(exp__args, i);
        PyObject * child_repn = recursive_generate_ampl_repn(context, exp__args_i);
        if(child_repn == NULL) {
            Py_DECREF(exp__args_i);
            Py_DECREF(child_repn);
            Py_DECREF(exp_coef);
            Py_DECREF(exp__coef);
            Py_DECREF(exp__args);
            return ERROR;
        }
        Py_DECREF(exp__args_i);

        // adjust the constant
        // ampl_repn._constant += exp_coef * child_repn._constant
        PyObject * child_repn__constant = PyObject_GetAttrString(child_repn, "_constant");
        _result = PyNumber_Multiply(exp_coef, child_repn__constant);
        PyObject * ampl_repn__constant = PyObject_GetAttrString(*ampl_repn, "_constant");
        _result2 = PyNumber_Add(ampl_repn__constant, _result);
        PyObject_SetAttrString(*ampl_repn, "_constant", _result2);
        Py_DECREF(_result2); _result2 = NULL;
        Py_DECREF(ampl_repn__constant);
        Py_DECREF(_result); _result = NULL;
        Py_DECREF(child_repn__constant);

        // adjust the linear terms
        // for (var_name,var) in child_repn._linear_terms_var.iteritems():
        PyObject * child_repn__ltv = PyObject_GetAttrString(child_repn, "_linear_terms_var");
        PyObject * child_repn__ltc = PyObject_GetAttrString(child_repn, "_linear_terms_coef");
        PyObject * var_name, * var;
        Py_ssize_t pos = 0;
        while(PyDict_Next(child_repn__ltv, &pos, &var_name, &var)) {
            // if var_name in ampl_repn._linear_terms_var:
            PyObject * ampl_repn__ltv = PyObject_GetAttrString(*ampl_repn, "_linear_terms_var");
            PyObject * ampl_repn__ltc = PyObject_GetAttrString(*ampl_repn, "_linear_terms_coef");

            // opt
            PyObject * child_repn__ltc_vn = PyDict_GetItem(child_repn__ltc, var_name);
            _result = PyNumber_Multiply(exp_coef, child_repn__ltc_vn);

            if(PyMapping_HasKey(ampl_repn__ltv, var_name)) {
                // ampl_repn._linear_terms_coef[var_name] += exp_coef * child_repn._linear_terms_coef[var_name]
                PyObject * ampl_repn__ltc_vn = PyDict_GetItem(ampl_repn__ltc, var_name);
                _result2 = PyNumber_Add(ampl_repn__ltc_vn, _result);

                PyDict_SetItem(ampl_repn__ltc, var_name, _result2);

                Py_DECREF(ampl_repn__ltc_vn);
                Py_DECREF(_result2); _result2 = NULL;
            // else:
            } else {
                // ampl_repn._linear_terms_var[var_name] = var
                PyDict_SetItem(ampl_repn__ltv, var_name, var);

                // ampl_repn._linear_terms_coef[var_name] = exp_coef*child_repn._linear_terms_coef[var_name]
                PyDict_SetItem(ampl_repn__ltc, var_name, _result);
            }

            Py_DECREF(child_repn__ltc_vn);
            Py_DECREF(_result); _result = NULL;

            Py_DECREF(ampl_repn__ltv);
            Py_DECREF(ampl_repn__ltc);
        }

        Py_DECREF(child_repn__ltv);
        Py_DECREF(child_repn__ltc);

        // adjust the nonlinear terms
        // if not child_repn._nonlinear_expr is None:
        PyObject * child_repn__nle = PyObject_GetAttrString(child_repn, "_nonlinear_expr");
        if(child_repn__nle != Py_None) {
            // if ampl_repn._nonlinear_expr is None:
            PyObject * ampl_repn__nle = PyObject_GetAttrString(*ampl_repn, "_nonlinear_expr");

            // opt
            _result = Py_BuildValue("(OO)", exp_coef, child_repn__nle);

            if(ampl_repn__nle == Py_None) {
                // ampl_repn._nonlinear_expr = [(exp_coef, child_repn._nonlinear_expr)]
                _result2 = Py_BuildValue("[O]", _result);
                PyObject_SetAttrString(*ampl_repn, "_nonlinear_expr", _result2);
                Py_DECREF(_result2); _result2 = NULL;
            // else:
            } else {
                // ampl_repn._nonlinear_expr.append((exp_coef, child_repn._nonlinear_expr))
                PyList_Append(ampl_repn__nle, _result);
            }
            Py_DECREF(_result); _result = NULL;
            Py_DECREF(ampl_repn__nle);
        }
        Py_DECREF(child_repn__nle);
        
        // adjust the nonlinear vars
        // ampl_repn._nonlinear_vars.update(child_repn._nonlinear_vars)
        PyObject * ampl_repn__nlv = PyObject_GetAttrString(*ampl_repn, "_nonlinear_vars");
        PyObject * child_repn__nlv = PyObject_GetAttrString(child_repn, "_nonlinear_vars");
        PyObject_CallMethod(ampl_repn__nlv, "update", "(O)", child_repn__nlv);
        Py_DECREF(child_repn__nlv);
        Py_DECREF(ampl_repn__nlv);

        // cleanup
        Py_DECREF(exp_coef);
        Py_DECREF(child_repn);
    }

    Py_DECREF(exp__coef);
    Py_DECREF(exp__args);

    assert((*ampl_repn)->ob_refcnt == 1);
    return TRUE;
}
