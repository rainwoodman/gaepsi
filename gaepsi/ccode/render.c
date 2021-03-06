#include "defines.h"

#define color_doc_string \
"color(target, raster, min, max, logscale, cmapr,cmapg, cmapb)\n" \
"cmapr, cmapg, cmapb are floating numbers (0-1.0), target is rgb unsigned int, raster is float or double, when the target is RGB, composite the new rasteron top of the original content. when the target is RGBA, overwrites it."

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
	int rgba = (PyArray_DIM(target, 2) == 4);
	cmapr = (PyArrayObject*) PyArray_Cast(cmapr, NPY_FLOAT);
	cmapg = (PyArrayObject*) PyArray_Cast(cmapg, NPY_FLOAT);
	cmapb = (PyArrayObject*) PyArray_Cast(cmapb, NPY_FLOAT);
	cmapa = (PyArrayObject*) PyArray_Cast(cmapa, NPY_FLOAT);
	npy_intp Ncmapbins = PyArray_Size((PyObject*)cmapr);
	float * cmapr_data = PyArray_GETPTR1(cmapr, 0);
	float * cmapg_data = PyArray_GETPTR1(cmapg, 0);
	float * cmapb_data = PyArray_GETPTR1(cmapb, 0);
	float * cmapa_data = PyArray_GETPTR1(cmapa, 0);
	unsigned char * target_data;
	npy_intp Nx = PyArray_DIM((PyObject*)raster, 0);
	npy_intp Ny = PyArray_DIM((PyObject*)raster, 1);
	npy_intp i, j;
	float index_factor = Ncmapbins / (max - min);
	npy_intp nans = 0;
	npy_intp badvalues = 0;
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
		npy_intp index = (raster_value - min) * index_factor;
		if(index < 0) index = 0;
		if(index >= Ncmapbins) index = Ncmapbins - 1;
		if(!rgba) {
			float a = cmapa_data[index];
			float at = 1.0 - a;
			target_data = PyArray_GETPTR3(target, i, j, 0);
			target_data[0] = (float) target_data[0] * at + 255.0 * cmapr_data[index] * a;
			target_data[1] = (float) target_data[1] * at + 255.0 * cmapg_data[index] * a;
			target_data[2] = (float) target_data[2] * at + 255.0 * cmapb_data[index] * a;
		} else {
			target_data = PyArray_GETPTR3(target, i, j, 0);
			target_data[0] = 255.0 * cmapr_data[index];
			target_data[1] = 255.0 * cmapg_data[index];
			target_data[2] = 255.0 * cmapb_data[index];
			target_data[3] = 255.0 * cmapa_data[index];
		}
	}
	}
	Py_DECREF(cmapr);
	Py_DECREF(cmapg);
	Py_DECREF(cmapb);
	Py_DECREF(cmapa);
	Py_RETURN_NONE;
}
#define line_doc_string \
"line(target, X, Y, VX,VY, min, max, cmapr, cmapg, cmapb, cmapa, cmapv)\n" \
"target is an RGB unsigned char array, X,Y,VX,VY are float arrays, \n" \
"cmapr,cmapg,cmapb,cmapa are from a Valuemap.table for the rgba colormap\n " \
"cmapv is from a Valuemap.table for the lenght map\n"
static PyObject * line(PyObject * self, PyObject * args, PyObject * kwds) {
	PyArrayObject * target, *X, *Y, *VX, *VY, * cmapr, * cmapg, * cmapb, * cmapa, * cmapv;
	float min, max, scale;
	int logscale;
	static char * kwlist[] = {
	"target", "X", "Y", "VX", "VY", "scale", "min", "max", "logscale", "cmapr", "cmapg", "cmapb", "cmapa", "cmapv", NULL
	};
	if(! PyArg_ParseTupleAndKeywords(args, kwds, "O!O!O!O!O!fffiO!O!O!O!O!", kwlist,
		&PyArray_Type, &target, 
		&PyArray_Type, &X, 
		&PyArray_Type, &Y, 
		&PyArray_Type, &VX, 
		&PyArray_Type, &VY, 
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
	VX = (PyArrayObject*) PyArray_Cast(VX, NPY_FLOAT);
	VY = (PyArrayObject*) PyArray_Cast(VY, NPY_FLOAT);
	cmapr = (PyArrayObject*) PyArray_Cast(cmapr, NPY_FLOAT);
	cmapg = (PyArrayObject*) PyArray_Cast(cmapg, NPY_FLOAT);
	cmapb = (PyArrayObject*) PyArray_Cast(cmapb, NPY_FLOAT);
	cmapa = (PyArrayObject*) PyArray_Cast(cmapa, NPY_FLOAT);
	cmapv = (PyArrayObject*) PyArray_Cast(cmapv, NPY_FLOAT);
	npy_intp Ncmapbins = PyArray_Size((PyObject*)cmapr);
	float * cmapr_data = PyArray_GETPTR1(cmapr, 0);
	float * cmapg_data = PyArray_GETPTR1(cmapg, 0);
	float * cmapb_data = PyArray_GETPTR1(cmapb, 0);
	float * cmapa_data = PyArray_GETPTR1(cmapa, 0);
	float * cmapv_data = PyArray_GETPTR1(cmapv, 0);
	npy_intp i;
	npy_intp N = PyArray_Size((PyObject*)X);
	npy_intp DX = PyArray_DIM(target, 0);
	npy_intp DY = PyArray_DIM(target, 1);
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
		npy_intp x0 = *(float*)PyArray_GETPTR1(X,i);
		npy_intp y0 = *(float*)PyArray_GETPTR1(Y,i);
		float vx = *(float*)PyArray_GETPTR1(VX,i);
		float vy = *(float*)PyArray_GETPTR1(VY,i);
		float mag = sqrt(vx * vx + vy * vy);
		if(isnan(mag) || mag == 0.0) continue;
		
		float value;
		if(logscale) {
			if(mag <= 0.0) continue;
			value = log10(mag);
		} else {
			value = mag;
		}

		npy_intp index = (value - min) * index_factor;
		if(index < 0) index = 0;
		if(index >= Ncmapbins) index = Ncmapbins - 1;

		npy_intp length = cmapv_data[index] * scale;
		float a = cmapa_data[index];
		float at = 1 - a;
		float r = cmapr_data[index];
		float g = cmapg_data[index];
		float b = cmapb_data[index];
		if (length < 0) length = 0;
		if(length == 0) {
			SET(x0, y0);
			continue;
		}
		/*Bresenham's Line Algorithm */
		npy_intp x1 = x0 + length * vx / mag;
		npy_intp y1 = y0 + length * vy / mag;
		npy_intp dx = abs(x1 - x0);
		npy_intp dy = abs(y1 - y0);
		npy_intp sx = (x0 < x1)?1:-1;
		npy_intp sy = (y0 < y1)?1:-1;
		npy_intp error = dx - dy;
		while(1) {
			SET(x0, y0);
			if(x0 == x1 && y0 == y1) break;
			npy_intp e2 = 2 * error;
			if(e2 > - dy) {
				error -= dy;
				x0 += sx;
			}
			if(e2 < dx) {
				error += dx;
				y0 += sy;
			}
		}
	}

	Py_DECREF(X);
	Py_DECREF(Y);
	Py_DECREF(VX);
	Py_DECREF(VY);
	Py_DECREF(cmapr);
	Py_DECREF(cmapg);
	Py_DECREF(cmapb);
	Py_DECREF(cmapa);
	Py_DECREF(cmapv);
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
	npy_intp Ncmapbins = PyArray_Size((PyObject*)cmapr);
	float * cmapr_data = PyArray_GETPTR1(cmapr, 0);
	float * cmapg_data = PyArray_GETPTR1(cmapg, 0);
	float * cmapb_data = PyArray_GETPTR1(cmapb, 0);
	float * cmapa_data = PyArray_GETPTR1(cmapa, 0);
	float * cmapv_data = PyArray_GETPTR1(cmapv, 0);
	npy_intp i;
	npy_intp N = PyArray_Size((PyObject*)V);
	npy_intp DX = PyArray_DIM(target, 0);
	npy_intp DY = PyArray_DIM(target, 1);
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
		npy_intp cx = *(float*)PyArray_GETPTR1(X,i);
		npy_intp cy = *(float*)PyArray_GETPTR1(Y,i);
		float value = *(float*)PyArray_GETPTR1(V,i);
		if(isnan(value)) continue;
		
		if(logscale) {
			if(value <= 0.0) continue;
			value = log10(value);
		}
		npy_intp index = (value - min) * index_factor;
		if(index < 0) index = 0;
		if(index >= Ncmapbins) index = Ncmapbins - 1;
		npy_intp radius = cmapv_data[index] * scale;
		if (radius < 0) radius = 0;
		npy_intp error = - radius;
		npy_intp x = radius;
		npy_intp y = 0;
		float a = cmapa_data[index];
		float at = 1 - a;
		float r = cmapr_data[index];
		float g = cmapg_data[index];
		float b = cmapb_data[index];
		if(radius == 0) {
			SET(cx, cy);
			continue;
		}
		/*Bresenham's Circle Algorithm */
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
static PyMethodDef line_method = 
	{"line", (PyCFunction) line, METH_KEYWORDS, line_doc_string };
static PyMethodDef circle_method = 
	{"circle", (PyCFunction) circle, METH_KEYWORDS, circle_doc_string };
static PyMethodDef color_method = 
	{"color", (PyCFunction) color, METH_KEYWORDS, color_doc_string };
void HIDDEN gadget_initrender(PyObject * m) {
	PyObject * circle_f = PyCFunction_New(&circle_method, NULL);
	PyObject * color_f = PyCFunction_New(&color_method, NULL);
	PyObject * line_f = PyCFunction_New(&line_method, NULL);
	
	PyModule_AddObject(m, "circle", circle_f);
	PyModule_AddObject(m, "line", line_f);
	PyModule_AddObject(m, "color", color_f);
}
