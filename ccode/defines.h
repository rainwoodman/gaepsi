
#include <Python.h>

#define KLINE_BINS (1024 * 16)
#define KOVERLAP_BINS  64
#ifndef GAEPSI_MODULE_MAIN
  #define NO_IMPORT_ARRAY
  #define NO_IMPORT_UFUNC
#endif
#define PY_ARRAY_UNIQUE_SYMBOL GAEPSI_ARRAY_API
#define PY_UFUNC_UNIQUE_SYMBOL GAEPSI_UFUNC_API
#include <numpy/arrayobject.h>
#include <numpy/ufuncobject.h>
#include <structmember.h>
#define intp npy_intp

#define HIDDEN __attribute__ ((visibility ("hidden")))  
