#include "defines.h"

#define scanline_doc_string \
"keywords: targets, locations, sml, values, src, dir, L"\
" rasterizes into an raster image the sums are calculated but no averaging is done." \
" returns the number of particles actually contributed to the image"
extern HIDDEN float k0f(const float);

static PyObject * scanline(PyObject * self, 
	PyObject * args, PyObject * kwds) {
	static char * kwlist[] = {
		"targets", 
		"locations", "sml", "values", 
		"src", "dir", "L",
		NULL
	};
	PyObject * l_targets, * l_values;
	PyArrayObject * a_src, * a_dir;
	PyArrayObject * locations, * smls;
	float L;
	npy_intp n; /* target counter */
	if(! PyArg_ParseTupleAndKeywords(args, kwds, "O!O!O!O!O!O!f", kwlist,
		&PyList_Type, &l_targets, 
		&PyArray_Type, &locations, 
		&PyArray_Type, &smls, 
		&PyList_Type, &l_values, 
		&PyArray_Type, &a_src, 
		&PyArray_Type, &a_dir, 
		&L)) return NULL;
	
	npy_intp Ntargets = PyList_GET_SIZE(l_targets);
	PyArrayObject *targets[Ntargets];
	PyArrayObject *values[Ntargets];
	for(n = 0; n < Ntargets; n++) {
		targets[n] = (PyArrayObject*)PyList_GET_ITEM(l_targets, n);
		values[n] = (PyArrayObject*)PyArray_Cast((PyArrayObject*)PyList_GET_ITEM(l_values, n), NPY_FLOAT);
	}
	locations = (PyArrayObject*) PyArray_Cast(locations, NPY_FLOAT);
	smls = (PyArrayObject*) PyArray_Cast(smls, NPY_FLOAT);
	a_src = (PyArrayObject*) PyArray_Cast(a_src, NPY_FLOAT);
	a_dir = (PyArrayObject*) PyArray_Cast(a_dir, NPY_FLOAT);

	float src[3];
	float dir[3];
	int d;
	for(d = 0; d < 3; d++) {
		src[d] = *(float*) PyArray_GETPTR1(a_src, d);
		dir[d] = *(float*) PyArray_GETPTR1(a_dir, d);
	}
	
	const npy_intp npar = PyArray_DIM(locations, 0);
	const npy_intp npix = PyArray_DIM(targets[0], 0);
	const int double_precision = (PyArray_ITEMSIZE(targets[0]) != 4);

	float pixc[npix][3];
	npy_intp ipix;
	for(ipix = 0; ipix < npix; ipix++) {
		for(d = 0; d < 3; d++) {
			pixc[ipix][d] = src[d] + dir[d] * ((float)ipix + 0.5) * L / npix;
		}
	}
	npy_intp ipar;
	#pragma omp parallel for private(ipar)
	for(ipar = 0; ipar < npar; ipar++) {
		int d;
		float pos[3];
		for(d = 0; d < 3; d++) {
			pos[d] = *(float*) PyArray_GETPTR2(locations, ipar, d);
		}
		const float sml = *(float*) PyArray_GETPTR1(smls, ipar);
		double dist = 0.0;
		double proj = 0.0;
		for(d = 0; d < 3; d++) {
			const float dd = pos[d] - src[d];
			dist += dd * dd;
			proj += dd * dir[d];
		}
		dist = sqrt(dist);
		const float r0 = sqrt(fabs((sml - dist) * (sml + dist) + proj * proj));
		const float sml3_inv = 1.0 / (sml * sml * sml);
		npy_intp ip0 = ceil((proj - r0) / L * npix);
		npy_intp ip1 = floor((proj + r0) / L * npix);
		if(ip0 < 0) ip0 = 0;
		if(ip0 >= npix ) ip0 = npix - 1;
		if(ip1 < 0) ip1 = 0;
		if(ip1 >= npix ) ip1 = npix - 1;
		npy_intp ip;
		for(ip = ip0; ip <=ip1; ip++) {
			double dist2 = 0;
			for(d = 0; d < 3; d++) {
				const float dd = pos[d] - pixc[ip][d];
				dist2 += dd * dd;
//				printf("pos[%d] = %g pixc[%ld][%d] = %g\n", 
//					d, pos[d], ip, d, pixc[ip][d]);
			}
			const float k = k0f(sqrt(dist2) / sml) * sml3_inv;
//			printf("dist = %g sml = %g k = %g\n", sqrt(dist2), sml, k);
			if(k > 0.0) {
				int n;
				if(double_precision) {
					for(n = 0; n < Ntargets; n++) {
						const float val = *(float*)PyArray_GETPTR1(values[n], ipar);
						double * dst = (double*)PyArray_GETPTR1(targets[n], ip);
						#pragma omp atomic
						*dst += k * val;
					}
				} else {
					for(n = 0; n < Ntargets; n++) {
						const float val = *(float*)PyArray_GETPTR1(values[n], ipar);
						float * dst = (float*)PyArray_GETPTR1(targets[n], ip);
						#pragma omp atomic
						*dst += k * val;
					}
				}
			}
		}
	}
	Py_DECREF(a_src);
	Py_DECREF(a_dir);
	Py_DECREF(locations);
	Py_DECREF(smls);
	for(n = 0; n < Ntargets; n++) {
		Py_DECREF(values[n]);
	}
	return Py_None;
}

static PyMethodDef module_methods[] = {
	{"scanline", (PyCFunction) scanline, METH_KEYWORDS, scanline_doc_string },
	{NULL}
};
void HIDDEN gadget_initscanline(PyObject * m) {
//	PyObject * thism = Py_InitModule3("image", module_methods, "image module");
//	Py_INCREF(thism);
	PyObject * scanline_f = PyCFunction_New(module_methods, NULL);
	PyModule_AddObject(m, "scanline", scanline_f);
}
