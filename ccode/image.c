#include <Python.h>
#include <numpy/arrayobject.h>
#include <time.h>
#include "defines.h"

#define HIDDEN __attribute__ ((visibility ("hidden")))

#define image_doc_string \
"keywords: locations, sml, value,"\
" xmin, ymin, xmax, ymax, zmin, zmax, mask."\
" rasterizes into an raster image the sums are calculated but no averaging is done." \
" returns the number of particles actually contributed to the image"

extern HIDDEN float kline[];
extern HIDDEN float klinesq[];
extern HIDDEN float koverlap[][KOVERLAP_BINS][KOVERLAP_BINS][KOVERLAP_BINS];
static inline npy_intp overlap_index(float f) {
	return f * (KOVERLAP_BINS >> 1) + (KOVERLAP_BINS >> 1);
}
static inline float interp_koverlap(int x0, int y0, int x1, int y1,
	float pxmin, float pymin, float pxmax, float pymax) {

	const float dpx = (pxmax - pxmin) * (KOVERLAP_BINS >> 1);
	const float dpy = (pymax - pymin) * (KOVERLAP_BINS >> 1);

	float addbit = 0;

#define RW(a) ((a) > KOVERLAP_BINS -1)?(KOVERLAP_BINS - 1):(a)
#define LW(a) ((a) < 0)?0:(a)
	float x0y0 = koverlap[x0][y0][x0][y0];
	if(x1 == x0 && y0 == y1) {
		addbit = x0y0 * dpy * dpx ;
		goto exit;
	}
	float x0y1 = koverlap[x0][y1][x0][y1];
	float ldy = (y0 + 1) - pymin * (KOVERLAP_BINS >> 1) - (KOVERLAP_BINS >> 1);
	float rdy = pymax * (KOVERLAP_BINS >> 1) + (KOVERLAP_BINS >> 1) - y1;
	if(x1 == x0 && y1 == y0 + 1) {
		addbit += x0y0 * ldy * dpx;
		addbit += x0y1 * rdy * dpx;
		goto exit;
	}
	float x1y0 = koverlap[x1][y0][x1][y0];
	float ldx = (x0 + 1) - pxmin * (KOVERLAP_BINS >> 1) - (KOVERLAP_BINS >> 1);
	float rdx = pxmax * (KOVERLAP_BINS >> 1) + (KOVERLAP_BINS >> 1) - x1;
	if(x1 == x0 + 1 && y1 == y0) {
		addbit += x0y0 * ldx * (pymax - pymin);
		addbit += x1y0 * rdx * (pymax - pymin);
		goto exit;
	}
	float x1y1 = koverlap[x1][y1][x1][y1];
	if(x1 == x0 + 1 && y1 == y0 + 1) {
		addbit += x0y0 * ldx * ldy;
		addbit += x1y0 * rdx * ldy;
		addbit += x0y1 * ldx * rdy;
		addbit += x1y1 * rdx * rdy;
		goto exit;
	}
	float left = koverlap[x0][RW(y0 + 1)][x0][LW(y1 - 1)];
	if(x1 == x0 && y1 > y0 + 1) {
		addbit += x0y0 * ldy * dpx;
		addbit += x0y1 * rdy * dpx;
		addbit += left * dpx;
		goto exit;
	}
	float top = koverlap[RW(x0 + 1)][y0][LW(x1 - 1)][y0];
	if(x1 > x0 + 1 && y1 == y0) {
		addbit += x0y0 * ldx * dpy;
		addbit += x1y0 * rdx * dpy;
		addbit += top * dpy;
		goto exit;
	}
	float right = koverlap[x1][RW(y0 + 1)][x1][LW(y1 - 1)];
	if(x1 == x0 + 1 && y1 > y0 + 1) {
		addbit += x0y0 * ldx * ldy;
		addbit += x1y0 * rdx * ldy;
		addbit += x0y1 * ldx * rdy;
		addbit += x1y1 * rdx * rdy;
		addbit += left * ldx;
		addbit += right * rdx;
		goto exit;
	}
	float bottom = koverlap[RW(x0 + 1)][y1][LW(x1 - 1)][y1];
	if(x1 > x0 + 1 && y1 == y0 + 1) {
		addbit += x0y0 * ldx * ldy;
		addbit += x1y0 * rdx * ldy;
		addbit += x0y1 * ldx * rdy;
		addbit += x1y1 * rdx * rdy;
		addbit += top * ldy;
		addbit += bottom * rdy;
		goto exit;
	}
	float center = koverlap[RW(x0 + 1)][RW(y0 + 1)][LW(x1 - 1)][RW(y1 - 1)];
	if(x1 > x0 + 1 && y1 > y0 + 1) {
		addbit += x0y0 * ldx * ldy;
		addbit += x1y0 * rdx * ldy;
		addbit += x0y1 * ldx * rdy;
		addbit += x1y1 * rdx * rdy;
		addbit += left * ldx;
		addbit += right * rdx;
		addbit += top * ldy;
		addbit += bottom * rdy;
		addbit += center;
		goto exit;
	}
	printf("unhandled x0=%d, x1=%d, y0=%d, y1=%d", x0, x1, y0, y1);
	exit:
	return addbit;
}
static time_t _ptime_t = 0;
static void ptime(char * str) {
	printf("time = %lf: %s\n", (double) difftime(clock(), _ptime_t)/CLOCKS_PER_SEC, str);
	_ptime_t = clock();
}
static PyObject * image(PyObject * self, 
	PyObject * args, PyObject * kwds) {
	static char * kwlist[] = {
		"targets", 
		"locations", "sml", "values", 
		"xmin", "ymin", "xmax", "ymax",
		"zmin", "zmax",
		"quick", "mask", "boxsize",
		NULL
	};
	PyObject * targets, *Vs;
	PyArrayObject * locations, * S;
	PyArrayObject * mask;
	PyArrayObject * box;
	float xmin, ymin, xmax, ymax, zmin, zmax;
	float boxsizex,boxsizey,boxsizez;
	npy_intp npixelx, npixely;
	npy_intp length;
	npy_intp p; /*particle counter*/
	npy_intp quick;
	npy_intp n; /* target counter */
	if(! PyArg_ParseTupleAndKeywords(args, kwds, "O!O!O!O!ffffffiOO", kwlist,
		&PyList_Type, &targets, 
		&PyArray_Type, &locations, 
		&PyArray_Type, &S, 
		&PyList_Type, &Vs, 
		&xmin, &ymin, &xmax, &ymax,
		&zmin, &zmax,
		&quick, &mask, &box)) return NULL;
	
	npy_intp Ntargets = PyList_GET_SIZE(targets);
	PyArrayObject ** target_arrays = malloc(sizeof(void*) * Ntargets);
	PyArrayObject ** V_arrays = malloc(sizeof(void*) * Ntargets);
	for(n = 0; n < Ntargets; n++) {
		target_arrays[n] = PyList_GET_ITEM(targets, n);
		V_arrays[n] = PyArray_Cast(PyList_GET_ITEM(Vs, n), NPY_FLOAT);
	}
	locations = (PyArrayObject*) PyArray_Cast(locations, NPY_FLOAT);
	if((PyObject*)mask != Py_None)
		mask = (PyArrayObject*) PyArray_Cast(mask, NPY_BOOL);
	if((PyObject*)box != Py_None) {
		box = (PyArrayObject*) PyArray_Cast(box, NPY_FLOAT);
		boxsizex = *((float*)PyArray_GETPTR1(box, 0));
		boxsizey = *((float*)PyArray_GETPTR1(box, 1));
		boxsizez = *((float*)PyArray_GETPTR1(box, 2));
		Py_DECREF(box);
	}
	S = (PyArrayObject*) PyArray_Cast(S, NPY_FLOAT);
	length = PyArray_Size((PyObject*)S);
	npixelx = PyArray_DIM((PyObject*)target_arrays[0], 0);
	npixely = PyArray_DIM((PyObject*)target_arrays[0], 1);

	npy_intp im_dims[] = {npixelx, npixely};

	float psizeX = (xmax - xmin) / npixelx;
	float psizeY = (ymax - ymin) / npixely;

	npy_intp ic = 0;
	npy_intp pc = 0;
	float sml_sum = 0.0;

    #pragma omp parallel private(p)
	{
	npy_intp cache_size = 1024;
	float * cache = malloc(sizeof(float) * cache_size);
    #pragma omp for reduction(+: ic, pc, sml_sum) schedule(dynamic, 1000)
	for(p = 0; p < length; p++) {
		char m = ((PyObject*) mask != Py_None)?(*((char*)PyArray_GETPTR1(mask, p))):1;
		if(!m) continue;
		float x = *((float*)PyArray_GETPTR2(locations, p, 0));
		float y = *((float*)PyArray_GETPTR2(locations, p, 1));
		float z = *((float*)PyArray_GETPTR2(locations, p, 2));
		float sml = *((float*)PyArray_GETPTR1(S, p));
		sml_sum += sml;
		float sml_2 = sml / 2;
		if(x > xmax+sml || x < xmin-sml) {
			if(box == (void*)Py_None) continue;
			x += boxsizex;
			if(x > xmax+sml || x < xmin-sml) {
				x -= boxsizex; x -= boxsizex;
				if(x > xmax+sml || x < xmin-sml) {
				continue;
				}
			}
		}
		if(y > ymax+sml || y < ymin-sml) {
			if(box == (void*)Py_None) continue;
			y += boxsizey;
			if(y > ymax+sml || y < ymin-sml) {
				y -= boxsizey; y -= boxsizey;
				if(y > ymax+sml || y < ymin-sml) {
				continue;
				}
			}
		}
		if(z > zmax+sml_2 || z < zmin-sml_2) {
			if(box == (void*)Py_None) continue;
			z += boxsizez;
			if(z > zmax+sml_2 || z < zmin-sml_2) {
				z -= boxsizez; z -= boxsizez;
				if(z > zmax+sml_2 || z < zmin-sml_2) {
				continue;
				}
			}
		}
		x -= xmin;
		y -= ymin;
		npy_intp i, j;
		float psizeXsml = psizeX / (sml);
		float psizeYsml = psizeY / (sml);
		float psizeXYsml = psizeXsml * psizeYsml;
		float pxmin0 =  - x / sml;
		float pymin0 =  - y / sml;
		float imxmin = - x / sml;
		float imymin = - y / sml;
		float imxmax = imxmin + npixelx * psizeXsml;
		float imymax = imymin + npixely * psizeYsml;
		/* imxmin0, imxmax0, imymin0, imymax0 crops the particle within the image*/	
		if(imxmax < -1) continue;
		if(imymax < -1) continue;
		if(imxmin > 1) continue;
		if(imymin > 1) continue;
		npy_intp imx0 = overlap_index(imxmin);
		npy_intp imx1 = overlap_index(imxmax);
		npy_intp imy0 = overlap_index(imymin);
		npy_intp imy1 = overlap_index(imymax);
		if(imx1 < 0) continue;
		if(imy1 < 0) continue;
		if(imx0 >= KOVERLAP_BINS) continue;
		if(imy0 >= KOVERLAP_BINS) continue;

		if(imx0 < 0) imx0 = 0;
		if(imy0 < 0) imy0 = 0;
		if(imx1 >= KOVERLAP_BINS) imx1 = KOVERLAP_BINS - 1;
		if(imy1 >= KOVERLAP_BINS) imy1 = KOVERLAP_BINS - 1;
		if(imxmin < -1) imxmin = -1;
		if(imymin < -1) imymin = -1;
		if(imxmax > 1) imxmax = 1;
		if(imymax > 1) imymax = 1;
		float norm = interp_koverlap(imx0, imy0, imx1, imy1,
							imxmin, imymin, imxmax, imymax);
		
		if(norm == 0.0) continue;
	//	printf("%d: %d %d %d %d %f\n", p, imx0, imy0, imx1, imy1, koverlap[imx0][imy0][imx1][imy1]);
	//	printf("%d: %f %f %f %f %f\n", p, imxmin, imymin, imxmax, imymax, norm);
		npy_intp ipixelmin = floor((x - sml) / psizeX);
		npy_intp ipixelmax = ceil((x + sml) / psizeX);
		npy_intp jpixelmin = floor((y - sml) / psizeY);
		npy_intp jpixelmax = ceil((y + sml) / psizeY);

		ic++;
#define PIXEL_IN_IMAGE (i >=0 && i < npixelx && j >=0 && j < npixely)
		
		if(ipixelmin < 0) ipixelmin = 0;
		if(ipixelmax >= npixelx) ipixelmax = npixelx - 1;
		if(jpixelmin < 0) jpixelmin = 0;
		if(jpixelmax >= npixely) jpixelmax = npixely - 1;

		npy_intp k;
		npy_intp desired_cache_size = (ipixelmax - ipixelmin + 1) * (jpixelmax - jpixelmin + 1);
		if(desired_cache_size > cache_size) {
			while(desired_cache_size > cache_size) {
					cache_size *= 2;
			}
			free(cache);
			cache = malloc(sizeof(float) * cache_size);
		}

		k = 0;
		float sum = 0;
		for(i = ipixelmin; i <= ipixelmax; i++)  {
			float pxmin = pxmin0 + i * psizeXsml;
			float pxmax = pxmin + psizeXsml;
			npy_intp x0 = pxmin * (KOVERLAP_BINS >> 1) + (KOVERLAP_BINS >> 1);
			npy_intp x1 = pxmax * (KOVERLAP_BINS >> 1) + (KOVERLAP_BINS >> 1);
			if(x1 < 0 || x0 >= KOVERLAP_BINS) {
				for(j = jpixelmin; j <= jpixelmax; j++) {
					cache[k++] = 0.0;
				}
				continue;
			}
			if(x0 < 0) x0 = 0;
			if(x1 >= KOVERLAP_BINS) x1 = KOVERLAP_BINS - 1;
			for(j = jpixelmin; j <= jpixelmax; j++) {

				float pymin = pymin0 + psizeYsml * j;
				float pymax = pymin + psizeYsml;
				npy_intp y0 = pymin * (KOVERLAP_BINS >> 1) + (KOVERLAP_BINS >> 1);
				npy_intp y1 = pymax * (KOVERLAP_BINS >> 1) + (KOVERLAP_BINS >> 1);
				if(y0 >= KOVERLAP_BINS || y1 < 0 ) {
					cache[k++] = 0.0;
					continue;
				}
				pc++;
		
				/* possible if pxmax == 2.0 or pymax == 2.0*/
				if(y0 < 0) y0 = 0;
				if(y1 >= KOVERLAP_BINS) y1 = KOVERLAP_BINS - 1;

				float addbit;
				if((x1 - x0 < 2 && y1 - y0 < 2)) {
				/*this branch deal with high resolution (pixelsize <= KOVERLAP size)*/
					float centerx = (pxmax + pxmin);
					float centery = (pymax + pymin);
				/*hpotf is slower than sqrt*/
					float d = 0.5 * sqrt(centerx*centerx+centery*centery) * KLINE_BINS;
					npy_intp dfloor = d;
					npy_intp dceil = d + 1;
					if(dceil < KLINE_BINS) {
						addbit = (kline[dceil] * (d - dfloor) + kline[dfloor] * (dceil - d)) * psizeXYsml;
					} else {
						addbit = 0.0;
					}
				} else {
				/*this branch deal with low resolution ( pixelsize >> KOVERLAP size)*/
					if(quick) {
						addbit = koverlap[x0][y0][x1][y1];
					} else {
						addbit = interp_koverlap(x0, y0, x1, y1, pxmin, pymin, pxmax, pymax);
					}
				}
				cache[k++] = addbit;
				sum += addbit;
			}
		}
		if(sum == 0) continue;
		float fac = norm / sum;
		npy_intp n;
		for(n = 0; n < Ntargets; n++) {
			int single_precision = (PyArray_ITEMSIZE(target_arrays[n]) == 4);
			k = 0;
			float value = *((float*)PyArray_GETPTR1(V_arrays[n], p));
			value *= fac;
			if(single_precision) {
				for(i = ipixelmin; i <= ipixelmax; i++) {
				for(j = jpixelmin; j <= jpixelmax; j++) {
					#pragma omp atomic
					*((float*)PyArray_GETPTR2(target_arrays[n],i,j)) += value * cache[k];
					k++;
				}
				}
			} else {
				for(i = ipixelmin; i <= ipixelmax; i++) {
				for(j = jpixelmin; j <= jpixelmax; j++) {
					#pragma omp atomic
					*((double*)PyArray_GETPTR2(target_arrays[n],i,j)) += (double)( value * cache[k]);
					k++;
				}
				}
			}
		}
	}
	free(cache);
	}
#if 0
	ptime("render");
	printf("ic = %d pc = %d \n", ic, pc);
   printf("sml_sum = %f\n", sml_sum);
#endif
	for(n = 0; n < Ntargets; n++) {
		Py_DECREF(V_arrays[n]);
	}
	free(target_arrays);
	free(V_arrays);
 	Py_DECREF(S);
 	Py_DECREF(locations);
	if(mask != Py_None)
		Py_DECREF(mask);
	return PyInt_FromLong(ic);
}
static PyMethodDef module_methods[] = {
	{"image", image, METH_KEYWORDS, image_doc_string },
	{NULL}
};
void HIDDEN gadget_initimage(PyObject * m) {
	import_array();
//	PyObject * thism = Py_InitModule3("image", module_methods, "image module");
//	Py_INCREF(thism);
	PyObject * image_f = PyCFunction_New(module_methods, NULL);
//	PyModule_AddObject(m, "image", thism);
	PyModule_AddObject(m, "image", image_f);
}
