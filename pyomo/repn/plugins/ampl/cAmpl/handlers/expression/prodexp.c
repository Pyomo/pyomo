#include "prodexp.h"
#include "../../cAmpl.h"

// C NOTE: this function is potentially dangerous, since in some cases it
//         uses the PyFloat_AS_DOUBLE function instead of a safer downcast
//         or a PyObject_RichCompareBool call. In testing, this saved up
//         to 90% of the function call time, but it has implicit limitations
//         when the represented PyFloat object is outside the range of a C
//         double. Users experiencing crashes should use the pure Python
//         implementation instead.

int _handle_prodexp(PyObject * context, PyObject * exp, PyObject ** ampl_repn) {
    // tmp
    PyObject * _result = NULL;
    PyObject * _result2 = NULL;
    Py_ssize_t i;

    // denom=1.0
    PyObject * denom = Py_BuildValue("f", 1.0);
    
    // for e in exp._denominator:
    PyObject * exp__denominator = PyObject_GetAttrString(exp, "_denominator");
    Py_ssize_t len_exp__denominator = PySequence_Length(exp__denominator);
    for(i = 0; i < len_exp__denominator; i++) {
        PyObject * e = PySequence_GetItem(exp__denominator, i);

        // if e.fixed_value():
        PyObject * e_fixed_value = PyObject_CallMethod(e, "fixed_value", NULL);
        if(PyObject_IsTrue(e_fixed_value) == TRUE) {
            // denom *= e.value
            PyObject * e_value = PyObject_GetAttrString(e, "value");
            _result = PyNumber_Multiply(denom, e_value);
            Py_DECREF(denom);
            denom = _result;
            _result = NULL;
            Py_DECREF(e_value);

        // elif e.is_constant():
        } else {
            PyObject * e_is_constant = PyObject_CallMethod(e, "is_constant", "()");
            if(PyObject_IsTrue(e_is_constant) == TRUE) {
                // denom *= e()
                PyObject * e_ = PyObject_CallObject(e, NULL);
                _result = PyNumber_Multiply(denom, e_);
                Py_DECREF(denom);
                denom = _result;
                _result = NULL;
                Py_DECREF(e_);

            // else:
            } else {
                // ampl_repn._nonlinear_expr = exp
                PyObject_SetAttrString(*ampl_repn, "_nonlinear_expr", exp);

                // break
                Py_DECREF(e_is_constant);
                Py_DECREF(e_fixed_value);
                Py_DECREF(e);
                break;
            }
            Py_DECREF(e_is_constant);
        }
        Py_DECREF(e_fixed_value);

        // if denom == 0.0:
        if(PyFloat_AS_DOUBLE(denom) == 0.0) {
            // print "Divide-by-zero error - offending sub-expression:"
            PySys_WriteStdout("Divide-by-zero error - offending sub-expression:\n");

            // e.pprint()
            PyObject_CallMethod(e, "pprint", NULL);

            // raise ZeroDivisionError
            PyErr_SetString(PyExc_ZeroDivisionError, "");
            Py_DECREF(e);
            return ERROR;
        }

        Py_DECREF(e);
    }
    Py_DECREF(exp__denominator);

    // if not ampl_repn._nonlinear_expr is None:
    PyObject * ampl_repn__nle = PyObject_GetAttrString(*ampl_repn, "_nonlinear_expr");
    if(ampl_repn__nle != Py_None) {
        // opt
        PyObject * _update = PyString_FromString("update");

        // we have a nonlinear expression ... build up all the vars
        // for e in exp._denominator:
        PyObject * exp__denominator = PyObject_GetAttrString(exp, "_denominator");
        Py_ssize_t len_exp__denominator = PySequence_Length(exp__denominator);
        for(i = 0; i < len_exp__denominator; i++) {
            PyObject * e = PySequence_GetItem(exp__denominator, i);

            // arg_repn = generate_ampl_repn(e)
            PyObject * arg_repn = recursive_generate_ampl_repn(context, e);
            Py_DECREF(e);
            if(arg_repn == NULL) return ERROR;

            // opt
            PyObject * ampl_repn__nlv = PyObject_GetAttrString(*ampl_repn, "_nonlinear_vars");

            // ampl_repn._nonlinear_vars.update(arg_repn._linear_terms_var)
            PyObject * arg_repn__ltv = PyObject_GetAttrString(arg_repn, "_linear_terms_var");
            PyObject_CallMethodObjArgs(ampl_repn__nlv, _update, arg_repn__ltv, NULL);
            Py_DECREF(arg_repn__ltv);

            // ampl_repn._nonlinear_vars.update(arg_repn._nonlinear_vars)
            PyObject * arg_repn__nlv = PyObject_GetAttrString(arg_repn, "_nonlinear_vars");
            PyObject_CallMethodObjArgs(ampl_repn__nlv, _update, arg_repn__nlv, NULL);
            Py_DECREF(arg_repn__nlv);

            Py_DECREF(ampl_repn__nlv);
            Py_DECREF(arg_repn);
        }
        Py_DECREF(exp__denominator);

        // for e in exp._numerator;
        PyObject * exp__numerator = PyObject_GetAttrString(exp, "_numerator");
        Py_ssize_t len_exp__numerator = PySequence_Length(exp__numerator);
        for(i = 0; i < len_exp__numerator; i++) {
            PyObject * e = PySequence_GetItem(exp__numerator, i);

            // arg_repn = generate_ampl_repn(e)
            PyObject * arg_repn = recursive_generate_ampl_repn(context, e);
            if(arg_repn == NULL) return ERROR;
            Py_DECREF(e);

            // opt
            PyObject * ampl_repn__nlv = PyObject_GetAttrString(*ampl_repn, "_nonlinear_vars");

            // ampl_repn._nonlinear_vars.update(arg_repn._linear_terms_var)
            PyObject * arg_repn__ltv = PyObject_GetAttrString(arg_repn, "_linear_terms_var");
            PyObject_CallMethodObjArgs(ampl_repn__nlv, _update, arg_repn__ltv, NULL);
            Py_DECREF(arg_repn__ltv);

            // ampl_repn._nonlinear_vars.update(arg_repn._nonlinear_vars)
            PyObject * arg_repn__nlv = PyObject_GetAttrString(arg_repn, "_nonlinear_vars");
            PyObject_CallMethodObjArgs(ampl_repn__nlv, _update, arg_repn__nlv, NULL);
            Py_DECREF(arg_repn__nlv);

            Py_DECREF(ampl_repn__nlv);
            Py_DECREF(arg_repn);
        }
        Py_DECREF(exp__numerator);

        Py_DECREF(_update);
        return TRUE;
    }
    Py_DECREF(ampl_repn__nle);

    // OK, the denominator is a constant
    // build up the ampl_repns for the numerator
    // C NOTE: we break from Python objects a bit here for the counters
    // n_linear_args = 0
    int n_linear_args = 0;

    // n_nonlinear_args = 0
    int n_nonlinear_args = 0;

    // arg_repns = list();
    PyObject * arg_repns = Py_BuildValue("[]");

    // for i in xrange(len(exp._numerator)):
    // C NOTE: don't actually care about the xrange
    PyObject * exp__numerator = PyObject_GetAttrString(exp, "_numerator");
    Py_ssize_t len_exp__numerator = PySequence_Length(exp__numerator);
    for(i = 0; i < len_exp__numerator; i++) {
        // e = exp._numerator[i]
        PyObject * e = PySequence_GetItem(exp__numerator, i);

        // e_repn = generate_ampl_repn(e)
        PyObject * e_repn = recursive_generate_ampl_repn(context, e);
        Py_DECREF(e);
        if(e_repn == NULL) return ERROR;

        // arg_repns.append(e_repn)
        PyList_Append(arg_repns, e_repn);

        // check if the expression is not nonlinear else it is nonlinear
        // if not e_repn._nonlinear_expr is None:
        PyObject * e_repn__nle = PyObject_GetAttrString(e_repn, "_nonlinear_expr");
        if(e_repn__nle != Py_None) {
            // n_nonlinear_args += 1
            n_nonlinear_args += 1;

        // check whether the expression is constant or else it is linear
        // elif not ((len(e_repn._linear_terms_var) == 0) and (e_repn._nonlinear_expr is None)):
        } else {
            PyObject * e_repn__ltv = PyObject_GetAttrString(e_repn, "_linear_terms_var");
            Py_ssize_t len_e_repn__ltv = PyDict_Size(e_repn__ltv);
            if(!(len_e_repn__ltv == 0 && e_repn__nle == Py_None)) {
                // n_linear_args += 1
                n_linear_args += 1;
            }
            Py_DECREF(e_repn__ltv);
        }
        Py_DECREF(e_repn__nle);
        Py_DECREF(e_repn);
    }
    Py_DECREF(exp__numerator);

    // is_nonlinear = False;
    // C NOTE: more trickery to avoid extraneous PyObjects
    int is_nonlinear = FALSE;

    // if n_linear_args > 1 or n_nonlinear_args > 0:
    if(n_linear_args > 1 || n_nonlinear_args > 0) {
        is_nonlinear = TRUE;
    }

    // if is_nonlinear is True:
    if(is_nonlinear == TRUE) {
        // do like AMPL and simply return the expression
        // without extracting the potentially linear part
        // ampl_repn = ampl_representation()
        *ampl_repn = new_ampl_representation();

        // ampl_repn._nonlinear_expr = exp
        PyObject_SetAttrString(*ampl_repn, "_nonlinear_expr", exp);

        // for repn in arg_repns:
        Py_ssize_t len_arg_repns = PySequence_Length(arg_repns);
        for(i = 0; i < len_arg_repns; i++) {
            PyObject * repn = PySequence_GetItem(arg_repns, i);

            // opt
            PyObject * ampl_repn__nlv = PyObject_GetAttrString(*ampl_repn, "_nonlinear_vars");

            // ampl_repn._nonlinear_vars.update(repn._linear_terms_var)
            PyObject * repn__ltv = PyObject_GetAttrString(repn, "_linear_terms_var");
            PyObject_CallMethod(ampl_repn__nlv, "update", "(O)", repn__ltv);
            Py_DECREF(repn__ltv);

            // ampl_repn._nonlinear_vars.update(repn._nonlinear_vars)
            PyObject * repn__nlv = PyObject_GetAttrString(repn, "_nonlinear_vars");
            PyObject_CallMethod(ampl_repn__nlv, "update", "(O)", repn__nlv);
            Py_DECREF(repn__nlv);

            // cleanup
            Py_DECREF(ampl_repn__nlv);
            Py_DECREF(repn);
        }

        // return ampl_repn
        Py_DECREF(arg_repns);
        return TRUE;

    // else:
    } else {
        // is linear or constant
        // ampl_repn = current_repn = arg_repns[0]
        PyObject * current_repn = PySequence_GetItem(arg_repns, 0);
        Py_DECREF(*ampl_repn); // C NOTE: decref current repn so we don't leak on next line
        *ampl_repn = current_repn;

        // for i in xrange(1, len(arg_repns)):
        // C NOTE: don't care about the xrange
        Py_ssize_t len_arg_repns = PySequence_Length(arg_repns);
        for(i = 1; i < len_arg_repns; i++) {
            // e_repn = arg_repns[i]
            PyObject * e_repn = PySequence_GetItem(arg_repns, i);

            // ampl_repn = ampl_representation()
            Py_DECREF(*ampl_repn);
            *ampl_repn = new_ampl_representation();

            // const_c * const_e
            // ampl_repn._constant = current_repn._constant * e_repn._constant
            PyObject * current_repn__constant = PyObject_GetAttrString(current_repn, "_constant");
            PyObject * e_repn__constant = PyObject_GetAttrString(e_repn, "_constant");
            _result = PyNumber_Multiply(current_repn__constant, e_repn__constant);
            PyObject_SetAttrString(*ampl_repn, "_constant", _result);
            Py_DECREF(_result); _result = NULL;

            // const_e * L_c
            // if e_repn._constant != 0.0:
            if(PyFloat_AS_DOUBLE(e_repn__constant) != 0.0) {
                // opt
                PyObject * current_repn__ltv = PyObject_GetAttrString(current_repn, "_linear_terms_var");
                PyObject * current_repn__ltc = PyObject_GetAttrString(current_repn, "_linear_terms_coef");
                PyObject * ampl_repn__ltv = PyObject_GetAttrString(*ampl_repn, "_linear_terms_var");
                PyObject * ampl_repn__ltc = PyObject_GetAttrString(*ampl_repn, "_linear_terms_coef");

                // for(var_name, var) in current_repn._linear_terms_var.iteritems():
                PyObject * var_name, * var;
                i = 0;
                while(PyDict_Next(current_repn__ltv, &i, &var_name, &var)) {
                    // ampl_repn._linear_terms_coef[var_name] = current_repn._linear_terms_coef[var_name] * e_repn._constant
                    PyObject * cr__ltc_vn = PyDict_GetItem(current_repn__ltc, var_name);
                    _result = PyNumber_Multiply(cr__ltc_vn, e_repn__constant);
                    Py_DECREF(cr__ltc_vn);
                    PyDict_SetItem(ampl_repn__ltc, var_name, _result);
                    Py_DECREF(_result); _result = NULL;

                    // ampl_repn._linear_terms_var[var_name] = var
                    PyDict_SetItem(ampl_repn__ltv, var_name, var);
                }

                Py_DECREF(current_repn__ltv);
                Py_DECREF(current_repn__ltc);
                Py_DECREF(ampl_repn__ltv);
                Py_DECREF(ampl_repn__ltc);
            }

            // const_c * L_e
            // if current_repn._constant != 0.0:
            if(PyFloat_AS_DOUBLE(current_repn__constant) != 0) {
                // opt
                PyObject * e_repn__ltv = PyObject_GetAttrString(e_repn, "_linear_terms_var");
                PyObject * e_repn__ltc = PyObject_GetAttrString(e_repn, "_linear_terms_coef");
                PyObject * ampl_repn__ltv = PyObject_GetAttrString(*ampl_repn, "_linear_terms_var");
                PyObject * ampl_repn__ltc = PyObject_GetAttrString(*ampl_repn, "_linear_terms_coef");

                // for (e_var_name,e_var) in e_repn._linear_terms_var.iteritems():
                PyObject * e_var_name, * e_var;
                i = 0;
                while(PyDict_Next(e_repn__ltv, &i, &e_var_name, &e_var)) {
                    // opt
                    PyObject * er__ltc_evn = PyDict_GetItem(e_repn__ltc, e_var_name);
                    _result = PyNumber_Multiply(current_repn__constant, er__ltc_evn);
                    Py_DECREF(er__ltc_evn);

                    // if e_var_name in ampl_repn._linear_terms_var:
                    if(PyMapping_HasKey(ampl_repn__ltv, e_var_name) == TRUE) {
                        // ampl_repn._linear_terms_coef[e_var_name] += current_repn._constant * e_repn._linear_terms_coef[e_var_name]
                        PyObject * _tmp = PyDict_GetItem(ampl_repn__ltc, e_var_name);
                        _result2 = PyNumber_Add(_tmp, _result);
                        PyDict_SetItem(ampl_repn__ltc, e_var_name, _result2);
                        Py_DECREF(_tmp);
                        Py_DECREF(_result2); _result2 = NULL;

                    // else:
                    } else {
                        // ampl_repn._linear_terms_coef[e_var_name] = current_repn._constant * e_repn._linear_terms_coef[e_var_name]
                        PyDict_SetItem(ampl_repn__ltc, e_var_name, _result);
                    }
                    Py_DECREF(_result); _result = NULL;
                }
                
                // cleanup
                Py_DECREF(e_repn__ltv);
                Py_DECREF(e_repn__ltc);
                Py_DECREF(ampl_repn__ltv);
                Py_DECREF(ampl_repn__ltc);
            }

            Py_DECREF(e_repn__constant);
            Py_DECREF(current_repn__constant);
            Py_DECREF(e_repn);

            // current_repn = ampl_repn
            Py_DECREF(current_repn);
            current_repn = *ampl_repn;
        }
    }

    // now deal with the product expression's coefficient that needs
    // to be applied to all parts of the ampl_repn
    // opt
    PyObject * exp_coef = PyObject_GetAttrString(exp, "coef");
    PyObject * quotient = PyNumber_Divide(exp_coef, denom);
    Py_DECREF(exp_coef);

    // ampl_repn._constant *= exp._coef/denom
    PyObject * ampl_repn__constant = PyObject_GetAttrString(*ampl_repn, "_constant");
    _result = PyNumber_Multiply(ampl_repn__constant, quotient);
    Py_DECREF(ampl_repn__constant);
    PyObject_SetAttrString(*ampl_repn, "_constant", _result);
    Py_DECREF(_result); _result = NULL;

    // for var_name in ampl_repn._linear_terms_coef:
    PyObject * ampl_repn__ltc = PyObject_GetAttrString(*ampl_repn, "_linear_terms_coef");
    PyObject * var_name, * _var;
    i = 0;
    // C NOTE: TODO this is scary. is this mutation allowed?
    while(PyDict_Next(ampl_repn__ltc, &i, &var_name, &_var)) {
        _result = PyNumber_Multiply(_var, quotient);
        PyDict_SetItem(ampl_repn__ltc, var_name, _result);
        Py_DECREF(_result); _result = NULL;
    }
    Py_DECREF(ampl_repn__ltc);
    Py_DECREF(quotient);

    // return ampl_repn
    assert((*ampl_repn)->ob_refcnt == 1);
    return TRUE;
}
