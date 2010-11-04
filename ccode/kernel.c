#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/ufuncobject.h>
#include "defines.h"

#define HIDDEN __attribute__ ((visibility ("hidden")))  
#define k0_doc_string \
""
#define kline_doc_string \
""
#define koverlap_doc_string \
""
/* these arrays are calculated in python code, the c modules only use them */
float HIDDEN kline[KLINE_BINS];
float HIDDEN koverlap[KOVERLAP_BINS][KOVERLAP_BINS][KOVERLAP_BINS][KOVERLAP_BINS];
float HIDDEN k0f(const float eta) {
	if(eta < 0.5) {
		return 8.0 * M_1_PI * (1.0 - 6 * (1.0 - eta) * eta * eta);
	}
	if(eta < 1.0) {
		float value = 1.0 - eta;
		return 8.0 * M_1_PI * 2.0 * value * value * value;
	}
	return 0.0;
}
double HIDDEN k0d(const double eta) {
	if(eta < 0.5) {
		return 8.0 * M_1_PI * (1.0 - 6 * (1.0 - eta) * eta * eta);
	}
	if(eta < 1.0) {
		double value = 1.0 - eta;
		return 8.0 * M_1_PI * 2.0 * value * value * value;
	}
	return 0.0;
}


float HIDDEN klinef(float d) {
	int ind = d * KLINE_BINS;
	if(ind < 0) ind = -ind;
	if(ind >= KLINE_BINS) ind = KLINE_BINS - 1;
	return kline[ind];
}
double HIDDEN klined(double d) {
	int ind = d * KLINE_BINS;
	if(ind < 0) ind = -ind;
	if(ind >= KLINE_BINS) ind = KLINE_BINS - 1;
	return kline[ind];
}

float HIDDEN koverlapf(float x0, float y0, float x1, float y1) {
	int indx0 = 0.5 * (x0 + 1.0) * KOVERLAP_BINS;
	int indy0 = 0.5 * (y0 + 1.0) * KOVERLAP_BINS;
	int indx1 = 0.5 * (x1 + 1.0) * KOVERLAP_BINS;
	int indy1 = 0.5 * (y1 + 1.0) * KOVERLAP_BINS;
	if(indx1 < 0) return 0;
	if(indy1 < 0) return 0;
	if(indx0 >= KOVERLAP_BINS) return 0;
	if(indy0 >= KOVERLAP_BINS) return 0;
	if(indx0 < 0) indx0 = 0;
	if(indy0 < 0) indy0 = 0;
	if(indx1 >= KOVERLAP_BINS) indx1 = KOVERLAP_BINS - 1;
	if(indy1 >= KOVERLAP_BINS) indy1 = KOVERLAP_BINS - 1;
	return koverlap[indx0][indy0][indx1][indy1];
}
double HIDDEN koverlapd(double x0, double y0, double x1, double y1) {
	int indx0 = 0.5 * (x0 + 1.0) * KOVERLAP_BINS;
	int indy0 = 0.5 * (y0 + 1.0) * KOVERLAP_BINS;
	int indx1 = 0.5 * (x1 + 1.0) * KOVERLAP_BINS;
	int indy1 = 0.5 * (y1 + 1.0) * KOVERLAP_BINS;
	if(indx1 < 0) return 0;
	if(indy1 < 0) return 0;
	if(indx0 >= KOVERLAP_BINS) return 0;
	if(indy0 >= KOVERLAP_BINS) return 0;
	if(indx0 < 0) indx0 = 0;
	if(indy0 < 0) indy0 = 0;
	if(indx1 >= KOVERLAP_BINS) indx1 = KOVERLAP_BINS - 1;
	if(indy1 >= KOVERLAP_BINS) indy1 = KOVERLAP_BINS - 1;
	return koverlap[indx0][indy0][indx1][indy1];
}
static void * k0_data[] = {(void*) k0f, (void*) k0d};
static void * kline_data[] = {(void*) klinef, (void*) klined};
static void * koverlap_data[] = {(void*) koverlapf, (void *) koverlapd};

static PyUFuncGenericFunction generic_functions[] = {NULL, NULL};
static PyUFuncGenericFunction koverlap_functions[] = {NULL, NULL};
static char generic_signatures[] = {PyArray_FLOAT, PyArray_FLOAT, PyArray_DOUBLE, PyArray_DOUBLE};
static char koverlap_signatures[] = {
	PyArray_FLOAT, PyArray_FLOAT, PyArray_FLOAT, PyArray_FLOAT, PyArray_FLOAT,
	PyArray_DOUBLE, PyArray_DOUBLE, PyArray_DOUBLE, PyArray_DOUBLE, PyArray_DOUBLE, };

static PyMethodDef module_methods[] = {
//	{"k0", k0, METH_O, k0_doc_string},
	{NULL}
};

static void PyUFunc_ffff_f(char **args, npy_intp *dimensions, npy_intp *steps, void *func) {
	typedef float (ftype )(float , float, float, float);
	ftype * f = (ftype*) f;
	char * ip1 = args[0], *ip2 = args[1], *ip3 = args[2], *ip4 = args[3],
		*op1 = args[4];
	npy_intp is1 = steps[0], is2 = steps[1], is3 = steps[2], is4 = steps[3], 
		os1 = steps[4];
	npy_intp n = dimensions[0];
	npy_intp i;
	for(i = 0; i < n; i++, ip1+=is1, ip2+=is2, ip3+=is3, ip4+=is4, op1+=os1){
        float in1 = *(float *)ip1;
        float in2 = *(float *)ip2;
        float in3 = *(float *)ip3;
        float in4 = *(float *)ip4;
        *(float *)op1 = f(in1, in2, in3, in4);
	}
}
static void PyUFunc_dddd_d(char **args, npy_intp *dimensions, npy_intp *steps, void *func) {
	typedef double (ftype )(double , double, double, double);
	ftype * f = (ftype*) f;
	char * ip1 = args[0], *ip2 = args[1], *ip3 = args[2], *ip4 = args[3],
		*op1 = args[4];
	npy_intp is1 = steps[0], is2 = steps[1], is3 = steps[2], is4 = steps[3], 
		os1 = steps[4];
	npy_intp n = dimensions[0];
	npy_intp i;
	for(i = 0; i < n; i++, ip1+=is1, ip2+=is2, ip3+=is3, ip4+=is4, op1+=os1){
        double in1 = *(double *)ip1;
        double in2 = *(double *)ip2;
        double in3 = *(double *)ip3;
        double in4 = *(double *)ip4;
        *(double *)op1 = f(in1, in2, in3, in4);
	}
}
void HIDDEN initkernel(PyObject * m) {
	import_array();
	import_ufunc();
	PyObject * thism = Py_InitModule3("kernel", module_methods, "kernel module");
	Py_INCREF(thism);
	PyModule_AddObject(m, "kernel", thism);
	npy_intp kline_dims[] = {KLINE_BINS};
	PyArrayObject * kline_a = (PyArrayObject *)PyArray_SimpleNewFromData(1, kline_dims, NPY_FLOAT, kline);
	npy_intp koverlap_dims[] = {KOVERLAP_BINS, KOVERLAP_BINS, KOVERLAP_BINS, KOVERLAP_BINS};
	PyArrayObject * koverlap_a = (PyArrayObject *)PyArray_SimpleNewFromData(4, koverlap_dims, NPY_FLOAT, koverlap);

	generic_functions[0] =  PyUFunc_f_f;
	generic_functions[1] =  PyUFunc_d_d;
	koverlap_functions[0] =  PyUFunc_ffff_f;
	koverlap_functions[1] =  PyUFunc_dddd_d;
	PyObject * k0_u = PyUFunc_FromFuncAndData(generic_functions, 
					k0_data, generic_signatures, 
					2, 1, 1, 
					PyUFunc_None, 
					"k0", k0_doc_string, 0);
	PyObject * kline_u = PyUFunc_FromFuncAndData(generic_functions, 
					kline_data, generic_signatures, 
					2, 1, 1, 
					PyUFunc_None, 
					"kline", kline_doc_string, 0);
	PyObject * koverlap_u = PyUFunc_FromFuncAndData(koverlap_functions, 
					koverlap_data, koverlap_signatures, 
					2, 4, 1, PyUFunc_None, 
					"koverlap", koverlap_doc_string, 0);
	
	PyModule_AddObject(thism, "k0", k0_u);
	PyModule_AddObject(thism, "kline", kline_u);
	PyModule_AddObject(thism, "koverlap", koverlap_u);
	PyModule_AddObject(thism, "akline", kline_a);
	PyModule_AddObject(thism, "akoverlap", koverlap_a);
}
