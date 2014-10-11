#include <Python.h>
#include "util.h"

/**
 * Print the current Python environment within which the calling C code
 * is executing, including:
 *
 * - The list of builtins available in the current scope
 * - The list of global variables available
 * - The list of local variables defined in the current scope
 *
 * Flushes standard output before returning.
 */
void dump_env() {
    printf("Builtins: ");
    PyObject_Print(PyEval_GetBuiltins(), stdout, Py_PRINT_RAW);
    printf("\n\n");

    printf("Globals: ");
    PyObject_Print(PyEval_GetGlobals(), stdout, Py_PRINT_RAW);
    printf("\n\n");

    printf("Locals: ");
    PyObject_Print(PyEval_GetLocals(), stdout, Py_PRINT_RAW);
    printf("\n\n");
    fflush(stdout);
}

/**
 * Print the type of the Python object <tt>arg</tt>. Properly handles
 * <tt>NULL</tt> arguments (so any <tt>PyObject *</tt> pointer may be
 * passed), and if the argument is an instance of a class, will also
 * print the class itself.
 *
 * Flushes standard output before returning.
 *
 * @param arg The Python object whose type to print.
 */
void dump_type(PyObject * arg) {
    if(arg == NULL) {
        printf("null\n");
        return;
    }

    PyObject * arg_type = PyObject_Type(arg);
    PyObject_Print(arg_type, stdout, 0);
    if(PyInstance_Check(arg)) {
        printf(": ");
        PyObject * arg_cls = PyObject_GetAttrString(arg, "__class__");
        PyObject_Print(arg_cls, stdout, 0);
        Py_DECREF(arg_cls);
    }
    printf("\n");
    Py_DECREF(arg_type);
    fflush(stdout);
}

/**
 * Print the list of attributes currently defined on the Python object
 * <tt>arg</tt>. Equivalent to calling <tt>print(dir(arg))</tt> in pure
 * Python.
 *
 * Flushes standard output before returning.
 *
 * @param arg The Python object whose attributes to print.
 */
void dump_dir(PyObject * arg) {
    if(arg == NULL) {
        printf("null\n");
        return;
    }
    PyObject * dir_arg = PyObject_Dir(arg);
    PyObject_Print(dir_arg, stdout, 0); printf("\n\n");
    Py_XDECREF(dir_arg);
    fflush(stdout);
}

/**
 * Print the current reference count of the Python object <tt>arg</tt>.
 * Properly handles <tt>NULL</tt> arguments (so any <tt>PyObject *</tt>
 * pointer may be passed).
 *
 * Flushes standard output before returning.
 *
 * @param arg The Python object whose reference count to print.
 * @see dump_refcnt_l()
 * @see dump_refcnt_lw()
 */
void dump_refcnt(PyObject * arg) {
    if(arg == NULL) {
        printf("Null object has no refcount\n");
    } else {
        printf("Object has refcount %ld\n", Py_SAFE_DOWNCAST(arg->ob_refcnt, Py_ssize_t, long));
    }
    fflush(stdout);
}

/**
 * Prints the reference count of the Python object <tt>arg</tt> as named
 * by the string <tt>label</tt>. Useful for debugging when distinguishing
 * between reference counts on multiple objects.
 *
 * Flushes standard output before returning.
 *
 * @param arg The Python object whose reference count to print.
 * @param label The name of the Python object <tt>arg</tt>.
 * @see dump_refcnt()
 * @see dump_refcnt_lw()
 */
void dump_refcnt_l(PyObject * arg, const char * label) {
    if(arg == NULL) {
        printf("Null %s object has no refcount\n", label);
    } else {
        printf("%s object has refcount %ld\n", label, Py_SAFE_DOWNCAST(arg->ob_refcnt, Py_ssize_t, long));
    }
    fflush(stdout);
}

/**
 * Prints the reference count of the Python object <tt>arg</tt> as named
 * by the string <tt>label</tt>. Takes an additional "temporal" label
 * <tt>when</tt>, which specifies (as a human-readable string) the point in
 * program execution when the reference count is printed. Useful for debugging
 * when distinguishing reference counts on the same object across function
 * calls or other code blocks.
 *
 * Flushes standard output before returning.
 *
 * @param arg The Python object whose reference count to print.
 * @param label The name of the Python object <tt>arg</tt>.
 * @param when The label of the time within program execution when the
 *             reference count is printed.
 * @see dump_refcnt()
 * @see dump_refcnt_l()
 */
void dump_refcnt_lw(PyObject * arg, const char * label, const char * when) {
    if(arg == NULL) {
        printf("%s, null %s object has no refcount\n", when, label);
    } else {
        printf("%s, %s object has refcount %ld\n", when, label, Py_SAFE_DOWNCAST(arg->ob_refcnt, Py_ssize_t, long));
    }
    fflush(stdout);
}

/**
 * Print the given message. Flushes standard output before returning.
 * Useful as a faster-to-code alternative to <tt>printf()</tt> and
 * <tt>fflush()</tt>.
 *
 * @param msg The message to print
 * @see dump_msg_va()
 */
void dump_msg(const char * msg) {
    printf(msg);
    fflush(stdout);
}

/**
 * Print the given message, formatted with the given arguments in standard
 * <tt>printf()</tt> style. Flushes standard output before returning. Useful
 * as a faster-to-code alternative to <tt>printf()</tt> and <tt>fflush()</tt>.
 *
 * @param fmt The format string to print
 * @param ... Additional arguments to apply to the format string
 * @see dump_msg()
 */
void dump_msg_va(const char * fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
    fflush(stdout);
}
