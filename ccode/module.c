#include <Python.h>
#include <numpy/arrayobject.h>

#define HIDDEN __attribute__ ((visibility ("hidden")))  
extern HIDDEN void initNDTree(PyObject * m);

#include "image.h"

static PyMethodDef module_methods[] = {
	{"image", image, METH_KEYWORDS, image_doc_string},
	{NULL}
};

static PyObject * m = NULL;
void initccode (void) {
	import_array();
	m = Py_InitModule3("ccode", module_methods, "ccode module");
	initNDTree(m);
}
