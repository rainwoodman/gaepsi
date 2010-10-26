#include <Python.h>
#include <numpy/arrayobject.h>

#define HIDDEN __attribute__ ((visibility ("hidden")))  
extern HIDDEN void initNDTree(PyObject * m);
extern HIDDEN void initimage(PyObject * m);

static PyMethodDef module_methods[] = {
	{NULL}
};

static PyObject * m = NULL;
void initccode (void) {
	import_array();
	m = Py_InitModule3("ccode", module_methods, "ccode module");
	initNDTree(m);
	initimage(m);
}
