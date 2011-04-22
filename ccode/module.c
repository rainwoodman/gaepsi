#include <Python.h>
#include <numpy/arrayobject.h>

#define HIDDEN __attribute__ ((visibility ("hidden")))  
extern HIDDEN void gadget_initNDTree(PyObject * m);
extern HIDDEN void gadget_initimage(PyObject * m);
extern HIDDEN void gadget_initremap(PyObject * m);
extern HIDDEN void gadget_initkernel(PyObject * m);
extern HIDDEN void gadget_initrender(PyObject * m);
extern HIDDEN void gadget_initpmin(PyObject * m);

static PyMethodDef module_methods[] = {
	{NULL}
};

static PyObject * m = NULL;
void init_gadgetccode (void) {
	import_array();
	printf("sizeof npyintp %d\n", sizeof(npy_intp));
	m = Py_InitModule3("_gadgetccode", module_methods, "gadget internal ccode module");
	gadget_initNDTree(m);
	gadget_initimage(m);
	gadget_initremap(m);
	gadget_initkernel(m);
	gadget_initrender(m);
	gadget_initpmin(m);
}
