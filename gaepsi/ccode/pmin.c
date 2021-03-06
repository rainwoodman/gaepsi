#include "defines.h"

#define pmin_doc_string "returns the minimum positive value in an array. NaN if all negative"

static double pmind(double v1, double v2) {
	if(v1 > 0.0) {
		if(v2 <= 0.0) return v1;
		if(v1 < v2 || isnan(v2)) return v1;
		else return v2;
	} else {
		if(v2 <= 0.0) return NAN;
		return v2;
	}
}
static PyUFuncGenericFunction generic_functions[] = {NULL, NULL};
static char generic_signatures[] = {NPY_FLOAT, NPY_FLOAT, NPY_FLOAT, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE};
static void * pmin_data[] = {(void*) pmind, (void*)pmind };

void HIDDEN gadget_initpmin(PyObject * m) {
	generic_functions[0] = PyUFunc_ff_f_As_dd_d;
	generic_functions[1] = PyUFunc_dd_d;

	PyObject * pmin_u = PyUFunc_FromFuncAndData(generic_functions,
						pmin_data, generic_signatures,
						2, 2, 1, 
						PyUFunc_None,
						"pmin", pmin_doc_string, 0);
	PyModule_AddObject(m, "pmin", pmin_u);
}
