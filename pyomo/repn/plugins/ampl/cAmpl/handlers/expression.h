#include <Python.h>
#include "../util.h"
#include "../cAmpl.h"

#include "expression/sumexp.h"
#include "expression/prodexp.h"
#include "expression/powexp.h"
#include "expression/ifexp.h"

int _handle_expression(PyObject * context, PyObject * exp, PyObject ** ampl_repn);
