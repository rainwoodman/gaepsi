#define GAEPSI_MODULE_MAIN

#include "defines.h"
//extern HIDDEN void gadget_initOctTree(PyObject * m);
extern HIDDEN void gadget_initimage(PyObject * m);
//extern HIDDEN void gadget_initscanline(PyObject * m);
extern HIDDEN void gadget_initwarp(PyObject * m);
extern HIDDEN void gadget_initkernel(PyObject * m);
extern HIDDEN void gadget_initrender(PyObject * m);
extern HIDDEN void gadget_initpmin(PyObject * m);
extern HIDDEN void gadget_initcamera(PyObject * m);
extern HIDDEN void gadget_initpeano(PyObject * m);

static PyMethodDef module_methods[] = {
	{NULL}
};

static PyObject * m = NULL;
void init_gaepsiccode (void) {
	import_array();
	import_ufunc();
	m = Py_InitModule3("_gaepsiccode", module_methods, "gaepsi internal ccode module");
//	gadget_initOctTree(m);
	gadget_initimage(m);
	gadget_initcamera(m);
//	gadget_initscanline(m);
	gadget_initwarp(m);
	gadget_initkernel(m);
	gadget_initrender(m);
	gadget_initpmin(m);
	gadget_initcamera(m);
	gadget_initpeano(m);
}
