//#undef NDEBUG
#include <assert.h>

#ifndef _CAMPL_CAMPL_H
#define _CAMPL_CAMPL_H

PyObject * cAmpl_generate_ampl_repn(PyObject * self, PyObject * args);
PyObject * internal_generate_ampl_repn(PyObject * context, PyObject * exp);

PyObject * recursive_generate_ampl_repn(PyObject * context, PyObject * exp);
PyObject * new_ampl_representation();
PyObject * get_pyomo_module();

#endif
