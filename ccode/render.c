#include <Python.h>
#include <numpy/arrayobject.h>
#define HIDDEN __attribute__ ((visibility ("hidden")))  

#define color_doc_string \
"color(target, raster, min, max, logscale, cmapr,cmapg, cmapb)\n" \
"cmapr, cmapg, cmapb are floating numbers (0-1.0), target is rgb unsigned int, raster is float or double"

static PyObject * color(PyObject * self, 
	PyObject * args, PyObject * kwds) {
	PyArrayObject *target, *raster;
	PyArrayObject *cmapr, *cmapg, *cmapb, *cmapa;
	float min, max;
	int logscale;
	static char * kwlist[] = {
	"target", "raster", "min", "max","logscale", "cmapr", "cmapg", "cmapb", "cmapa", NULL
	};
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "O!O!ffiO!O!O!O!", kwlist,
		&PyArray_Type, &target, 
		&PyArray_Type, &raster, 
		&min, &max, &logscale,
		&PyArray_Type, &cmapr,
		&PyArray_Type, &cmapg,
		&PyArray_Type, &cmapb,
		&PyArray_Type, &cmapa
		)) return NULL;
	int single_precision = (PyArray_ITEMSIZE(raster) == 4);
	cmapr = (PyArrayObject*) PyArray_Cast(cmapr, NPY_FLOAT);
	cmapg = (PyArrayObject*) PyArray_Cast(cmapg, NPY_FLOAT);
	cmapb = (PyArrayObject*) PyArray_Cast(cmapb, NPY_FLOAT);
	cmapa = (PyArrayObject*) PyArray_Cast(cmapa, NPY_FLOAT);
	int Ncmapbins = PyArray_Size((PyObject*)cmapr);
	float * cmapr_data = PyArray_GETPTR1(cmapr, 0);
	float * cmapg_data = PyArray_GETPTR1(cmapg, 0);
	float * cmapb_data = PyArray_GETPTR1(cmapb, 0);
	float * cmapa_data = PyArray_GETPTR1(cmapa, 0);
	unsigned char * target_data;
	int Nx = PyArray_DIM((PyObject*)raster, 0);
	int Ny = PyArray_DIM((PyObject*)raster, 1);
	int i, j;
	float index_factor = Ncmapbins / (max - min);
	int nans = 0;
	int badvalues = 0;
	for(i = 0; i < Nx; i++) {
	for(j = 0; j < Ny; j++) {
		float raster_value;
		if(single_precision)
			raster_value = *(float*) PyArray_GETPTR2(raster, i, j);
		else
			raster_value = *(double*) PyArray_GETPTR2(raster, i, j);
		if(isnan(raster_value)) {
			nans++;
			continue;
		}
		if(logscale) {
			if(raster_value <= 0.0) {
				badvalues++;
				continue;
			}
			raster_value = log10(raster_value);
		}
		int index = (raster_value - min) * index_factor;
		if(index < 0) index = 0;
		if(index >= Ncmapbins) index = Ncmapbins - 1;
		float a = cmapa_data[index];
		float at = 1.0 - a;
		target_data = PyArray_GETPTR3(target, i, j, 0);
		target_data[0] = (float) target_data[0] * at + 255.0 * cmapr_data[index] * a;
		target_data[1] = (float) target_data[1] * at + 255.0 * cmapg_data[index] * a;
		target_data[2] = (float) target_data[2] * at + 255.0 * cmapb_data[index] * a;
	}
	}
	Py_DECREF(cmapr);
	Py_DECREF(cmapg);
	Py_DECREF(cmapb);
	Py_DECREF(cmapa);
	Py_RETURN_NONE;
}
#define circle_doc_string \
"circle(target, X, Y, V, min, max, cmapr, cmapg, cmapb, cmapa, cmapv)\n" \
"target is an RGB unsigned char array, X,Y,V are float arrays, \n" \
"cmapr,cmapg,cmapb,cmapa are from a Valuemap.table for the rgba colormap\n " \
"cmapv is from a Valuemap.table for the radius map\n"

static PyObject * circle(PyObject * self, 
	PyObject * args, PyObject * kwds) {
	PyArrayObject * target, *X, *Y, *V, *cmapr, *cmapg, *cmapb, *cmapa, * cmapv;
	float min, max, scale;
	int logscale;
	static char * kwlist[] = {
	"target", "X", "Y", "V", "scale", "min", "max", "logscale", "cmapr", "cmapg", "cmapb", "cmapa", "cmapv", NULL
	};
	if(! PyArg_ParseTupleAndKeywords(args, kwds, "O!O!O!O!fffiO!O!O!O!O!", kwlist,
		&PyArray_Type, &target, 
		&PyArray_Type, &X, 
		&PyArray_Type, &Y, 
		&PyArray_Type, &V, 
		&scale,
		&min, &max, &logscale,
		&PyArray_Type, &cmapr,
		&PyArray_Type, &cmapg,
		&PyArray_Type, &cmapb,
		&PyArray_Type, &cmapa,
		&PyArray_Type, &cmapv
		)) return NULL;
	X = (PyArrayObject*) PyArray_Cast(X, NPY_FLOAT);
	Y = (PyArrayObject*) PyArray_Cast(Y, NPY_FLOAT);
	V = (PyArrayObject*) PyArray_Cast(V, NPY_FLOAT);
	cmapr = (PyArrayObject*) PyArray_Cast(cmapr, NPY_FLOAT);
	cmapg = (PyArrayObject*) PyArray_Cast(cmapg, NPY_FLOAT);
	cmapb = (PyArrayObject*) PyArray_Cast(cmapb, NPY_FLOAT);
	cmapa = (PyArrayObject*) PyArray_Cast(cmapa, NPY_FLOAT);
	cmapv = (PyArrayObject*) PyArray_Cast(cmapv, NPY_FLOAT);
	int Ncmapbins = PyArray_Size((PyObject*)cmapr);
	float * cmapr_data = PyArray_GETPTR1(cmapr, 0);
	float * cmapg_data = PyArray_GETPTR1(cmapg, 0);
	float * cmapb_data = PyArray_GETPTR1(cmapb, 0);
	float * cmapa_data = PyArray_GETPTR1(cmapa, 0);
	float * cmapv_data = PyArray_GETPTR1(cmapv, 0);
	int i;
	int N = PyArray_Size((PyObject*)V);
	int DX = PyArray_DIM(target, 0);
	int DY = PyArray_DIM(target, 1);
#define SET(x, y) { \
	if(x>=0 && x<DX && y>=0 && y<DY) {\
		unsigned char * base = (unsigned char*)PyArray_GETPTR3(target, x,y,0); \
		base[0] = at * base[0] + r * a * 255; \
		base[1] = at * base[1] + g * a * 255; \
		base[2] = at * base[2] + b * a * 255; \
	} \
	}
	float index_factor = Ncmapbins / (max - min);
	for(i = 0; i < N; i++) {
		int cx = *(float*)PyArray_GETPTR1(X,i);
		int cy = *(float*)PyArray_GETPTR1(Y,i);
		float value = *(float*)PyArray_GETPTR1(V,i);
		if(isnan(value)) continue;
		
		if(logscale) {
			if(value <= 0.0) continue;
			value = log10(value);
		}
		int index = (value - min) * index_factor;
		if(index < 0) index = 0;
		if(index >= Ncmapbins) index = Ncmapbins - 1;
		int radius = cmapv_data[index] * scale;
		if (radius < 0) radius = 0;
		int error = - radius;
		int x = radius;
		int y = 0;
		float a = cmapa_data[index];
		float at = 1 - a;
		float r = cmapr_data[index];
		float g = cmapg_data[index];
		float b = cmapb_data[index];
		if(radius == 0) {
			SET(cx, cy);
			continue;
		}
		while(x >= y) {
			SET(cx + x, cy + y);
			if(x) SET(cx - x, cy + y);
			if(y) SET(cx + x, cy - y);
			if(x && y) SET(cx - x, cy - y);
			if(x != y) {
				SET(cx + y, cy + x);
				if(y) SET(cx - y, cy + x);
				if(x) SET(cx + y, cy - x);
				if(x && y) SET(cx - y, cy - x);
			}
			error += y;
			++y;
			error += y;
			if(error >= 0) {
				--x;
				error -= x;
				error -= x;
			}
		}
	}

	Py_DECREF(X);
	Py_DECREF(Y);
	Py_DECREF(V);
	Py_DECREF(cmapr);
	Py_DECREF(cmapg);
	Py_DECREF(cmapb);
	Py_DECREF(cmapa);
	Py_DECREF(cmapv);
	Py_RETURN_NONE;
}
static PyMethodDef circle_method = 
	{"circle", circle, METH_KEYWORDS, circle_doc_string };
static PyMethodDef color_method = 
	{"color", color, METH_KEYWORDS, color_doc_string };
void HIDDEN gadget_initrender(PyObject * m) {
	import_array();
	PyObject * circle_f = PyCFunction_New(&circle_method, NULL);
	PyObject * color_f = PyCFunction_New(&color_method, NULL);
	
	PyModule_AddObject(m, "circle", circle_f);
	PyModule_AddObject(m, "color", color_f);
}
