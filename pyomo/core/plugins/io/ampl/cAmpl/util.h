#ifndef _CAMPL_UTIL_H
#define _CAMPL_UTIL_H

void dump_env();
void dump_type(PyObject * arg);
void dump_dir(PyObject * arg);

void dump_refcnt(PyObject * arg);
void dump_refcnt_l(PyObject * arg, const char * label);
void dump_refcnt_lw(PyObject * arg, const char * label, const char * when);

void dump_msg(const char * msg);
void dump_msg_va(const char * fmt, ...);

#define ERROR (-1)
#define FALSE (0)
#define TRUE (1)

#endif
