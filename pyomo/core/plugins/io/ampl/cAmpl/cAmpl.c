#include <Python.h>
#include "cAmpl.h"
#include "util.h"
#include "handlers.h"

static PyMethodDef cAmplMethods[] = {
    {"generate_ampl_repn", cAmpl_generate_ampl_repn, METH_VARARGS, "Generate an AMPL representation."},
    {NULL, NULL, 0, NULL}
};

static PyObject * pyomo_MOD = NULL;

PyMODINIT_FUNC initcAmpl(void) {
    (void) Py_InitModule("cAmpl", cAmplMethods);
}

/**
 * Python interface function for getting an AMPL representation.
 * Roughly equivalent to calling <tt>cAmpl.generate_ampl_repn(exp)</tt>.
 * Internally, extracts arguments and passes control to
 * internal_generate_ampl_repn().
 *
 * @see internal_generate_ampl_repn()
 * @return A new reference to the generated instance of
 *         <tt>ampl_representation</tt>.
 */
PyObject * cAmpl_generate_ampl_repn(PyObject * self, PyObject * args) {
    PyObject * exp;

    if(!PyArg_ParseTuple(args, "O", &exp)) {
        return NULL;
    }

    return internal_generate_ampl_repn(self, exp);
}

/**
 * Generate an AMPL representation from the given Python expression object.
 *
 * This function is responsible for establishing the framework within which
 * the rest of the cAmpl implementation operates. It covers the highest level
 * of <tt>if-then-else</tt> control flow from the Python implementation and
 * deals with the basic type checking that goes on at that level.
 *
 * This function does not actually manipulate the returned AMPL representation
 * much at all; instead, it performs type checks on the <tt>exp</tt> argument,
 * then passes control to one of several "handlers" which can further modify
 * the representation to return. In this case, the function checks:
 *
 * - If the exp is a variable value (instance of <tt>_VarValue</tt>)
 * - If the exp is a complex expression (instance of <tt>_Expression</tt>)
 * - If the exp is a fixed value (has a <tt>fixed_value()</tt> method)
 *
 * If none of these are true and the function cannot pass control to the
 * associated handler, it raises a Python <tt>ValueError</tt> to alert the
 * user. If the call succeeds, however, the function returns a new AMPL
 * representation (an instance of <tt>ampl_representation</tt>) as a Python
 * object.
 *
 * @param context The Python context within which to operate
 * @param exp The expression to parse. Should be one of <tt>_VarValue</tt>,
 *            <tt>_Expression</tt>, or an object with the <tt>fixed_value</tt>
 *            function defined.
 * @return A new reference to an instance of <tt>ampl_representation</tt> for
 *         the given expression.
 * @see handlers.h
 */
PyObject * internal_generate_ampl_repn(PyObject * context, PyObject * exp) {
    PyObject * ampl_repn = new_ampl_representation();
    PyObject * pyomo_MOD = get_pyomo_module();

    // Find pyomo.core.base for later use
    // Don't decref until end of function!
    PyObject * pyomo_base_MOD = PyObject_GetAttrString(pyomo_MOD, "base");
    Py_DECREF(pyomo_MOD);

    // Flag for when to return parsed expression
    int should_return = FALSE;

    // Variable
    PyObject * pyomo_var_MOD = PyObject_GetAttrString(pyomo_base_MOD, "var");
    PyObject * _VarValue_CLS = PyObject_GetAttrString(pyomo_var_MOD, "_VarValue");

    if(PyObject_IsInstance(exp, _VarValue_CLS)) {
        should_return = _handle_variable(context, exp, &ampl_repn);
    }

    Py_DECREF(pyomo_var_MOD);
    Py_DECREF(_VarValue_CLS);

    if(should_return) {
        Py_DECREF(pyomo_base_MOD);
        return ampl_repn;
    }

    // Expression
    PyObject * pyomo_expr_MOD = PyObject_GetAttrString(pyomo_base_MOD, "expr");
    PyObject * Expression_CLS = PyObject_GetAttrString(pyomo_expr_MOD, "Expression");

    if(PyObject_IsInstance(exp, Expression_CLS)) {
        should_return = _handle_expression(context, exp, &ampl_repn);
    }

    Py_DECREF(pyomo_expr_MOD);
    Py_DECREF(Expression_CLS);

    if(should_return == TRUE) {
        Py_DECREF(pyomo_base_MOD);
        return ampl_repn;
    } else if(should_return == ERROR) {
        Py_DECREF(pyomo_base_MOD);
        return NULL;
    }

    // Constant
    // Function call may return null - be careful with decref, etc.
    PyObject * fixed_value = PyObject_CallMethod(exp, "fixed_value", NULL);

    if(fixed_value && PyObject_IsTrue(fixed_value)) {
        should_return = _handle_fixedval(context, exp, &ampl_repn);
    }

    Py_XDECREF(fixed_value);

    if(should_return == TRUE) {
        Py_DECREF(pyomo_base_MOD);
        return ampl_repn;
    } else if(should_return == ERROR) {
        Py_DECREF(pyomo_base_MOD);
        return NULL;
    }

    // Unrecognized type; raise ValueError
    Py_DECREF(pyomo_base_MOD);
    PyErr_SetString(PyExc_ValueError, "Unexpected expression type");

    return NULL;
}

/**
 * Convenience function to recursively call the primary
 * <tt>generate_ampl_repn</tt> function with a single argument object.
 * Handles the generation of the argument tuple and various
 * error checks. Analogous to having Python recursively call
 * <tt>cAmpl.generate_ampl_repn(exp)</tt> from a running execution
 * of the same function.
 *
 * @param context The Python context within which to execute
 * @param exp The expression for which to generate a representation
 * @return An instance of ampl_representation for the given expression
 */
PyObject * recursive_generate_ampl_repn(PyObject * context, PyObject * exp) {
    if(Py_EnterRecursiveCall(" in AMPL expression generation")) {
        // C NOTE: Will set its own exception
        return NULL;
    }

    PyObject * result_repn = internal_generate_ampl_repn(context, exp);

    Py_LeaveRecursiveCall();
    
    assert(result_repn == NULL || result_repn->ob_refcnt == 1);
    return result_repn; // C NOTE: NULL on error; exception will be set
}

/**
 * Create and return a new instance of the class
 * <tt>pyomo.core.io.ampl.ampl_representation</tt>. 
 * Imports the <tt>pyomo.core</tt> module as necessary.
 *
 * @see get_pyomo_module()
 * @return A new reference to an ampl_representation instance.
 */
PyObject * new_ampl_representation() {
    PyObject * pyomo_MOD = get_pyomo_module();

    // Find the 'ampl' module
    PyObject * io_MOD = PyObject_GetAttrString(pyomo_MOD, "io");
    PyObject * ampl_MOD = PyObject_GetAttrString(io_MOD, "ampl");

    // Get the ampl_representation class and instantiate it
    PyObject * ampl_representation_CLS = PyObject_GetAttrString(ampl_MOD, "ampl_representation");
    PyObject * tuple = Py_BuildValue("()");
    PyObject * ampl_repn = PyInstance_New(ampl_representation_CLS, tuple, NULL);
    Py_DECREF(tuple);

    // Free modules on the import chain
    Py_DECREF(io_MOD);
    Py_DECREF(ampl_MOD);
    Py_DECREF(ampl_representation_CLS);

    // Check reference count
    assert(ampl_repn->ob_refcnt == 1);

    // cleanup
    Py_DECREF(pyomo_MOD);

    // Return the new ampl_representation()
    return ampl_repn;
}

/**
 * Get the <tt>pyomo.core</tt> module. Stores the result in a static 
 * variable and only performs an actual Python import if necessary; either way,
 * increments the reference count of the module before returning.
 *
 * @return a new reference to <tt>pyomo.core</tt>.
 */
PyObject * get_pyomo_module() {
    // Import the 'pyomo.core' module
    if(pyomo_MOD == NULL) {
        if(!(pyomo_MOD = PyImport_ImportModule("pyomo.core"))) {
            printf("import pyomo.core failed!\n");

            PyObject * exc_type, * exc_value, * exc_traceback;
            PyErr_Fetch(&exc_type, &exc_value, &exc_traceback);

            dump_env();

            PyErr_Restore(exc_type, exc_value, exc_traceback);

            return NULL;
        }
    }

    Py_INCREF(pyomo_MOD);
    return pyomo_MOD;
}
