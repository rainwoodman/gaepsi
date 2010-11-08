#include <Python.h>
#include <numpy/arrayobject.h>
#include <time.h>
#include "defines.h"

#define HIDDEN __attribute__ ((visibility ("hidden")))  

#define image_doc_string \
"keywords: locations, sml, value,"\
" xmin, ymin, xmax, ymax, npixelx, npixely, zmin, zmax."\
" returns an image of given size."

extern HIDDEN float kline[];
extern HIDDEN float koverlap[][KOVERLAP_BINS][KOVERLAP_BINS][KOVERLAP_BINS];
static inline int overlap_index(float f) {
	return f * (KOVERLAP_BINS / 2) + KOVERLAP_BINS / 2;
}
static inline float interp_koverlap(int x0, int y0, int x1, int y1,
	float pxmin, float pymin, float pxmax, float pymax) {

	const float dpx = (pxmax - pxmin) * (KOVERLAP_BINS / 2);
	const float dpy = (pymax - pymin) * (KOVERLAP_BINS / 2);

	float addbit = 0;

#define RW(a) ((a) > KOVERLAP_BINS -1)?(KOVERLAP_BINS - 1):(a)
#define LW(a) ((a) < 0)?0:(a)
	float x0y0 = koverlap[x0][y0][x0][y0];
	if(x1 == x0 && y0 == y1) {
		addbit = x0y0 * dpy * dpx ;
		goto exit;
	}
	float x0y1 = koverlap[x0][y1][x0][y1];
	float ldy = (y0 + 1) - pymin * (KOVERLAP_BINS / 2) - (KOVERLAP_BINS / 2);
	float rdy = pymax * (KOVERLAP_BINS / 2) + (KOVERLAP_BINS / 2) - y1;
	if(x1 == x0 && y1 == y0 + 1) {
		addbit += x0y0 * ldy * dpx;
		addbit += x0y1 * rdy * dpx;
		goto exit;
	}
	float x1y0 = koverlap[x1][y0][x1][y0];
	float ldx = (x0 + 1) - pxmin * (KOVERLAP_BINS / 2) - (KOVERLAP_BINS / 2);
	float rdx = pxmax * (KOVERLAP_BINS / 2) + (KOVERLAP_BINS / 2) - x1;
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
		"locations", "sml", "value", 
		"xmin", "ymin", "xmax", "ymax",
		"npixelx", "npixely", "zmin", "zmax",
		"quick", NULL
	};
	PyArrayObject * locations, * S, *V;
	float xmin, ymin, xmax, ymax, zmin, zmax;
	int npixelx, npixely;
	int length;
	int p;
	int quick;
	if(! PyArg_ParseTupleAndKeywords(args, kwds, "O!O!O!ffffiiffi", kwlist,
		&PyArray_Type, &locations, 
		&PyArray_Type, &S, 
		&PyArray_Type, &V, 
		&xmin, &ymin, &xmax, &ymax,
		&npixelx, &npixely, &zmin, &zmax,
		&quick)) return NULL;

	locations = (PyArrayObject*) PyArray_Cast(locations, NPY_FLOAT);
	S = (PyArrayObject*) PyArray_Cast(S, NPY_FLOAT);
	V = (PyArrayObject*) PyArray_Cast(V, NPY_FLOAT);
	length = PyArray_Size((PyObject*)S);
	npy_intp im_dims[] = {npixelx, npixely};
	PyArrayObject * result = (PyArrayObject*)PyArray_ZEROS(2, im_dims, NPY_FLOAT, 0);
	float psizeX = (xmax - xmin) / npixelx;
	float psizeY = (ymax - ymin) / npixely;

	float smlmax = 0.0;
	for(p = 0; p < length; p++) {
		float sml = *((float*)PyArray_GETPTR1(S, p));
		if(sml > smlmax) smlmax = sml;
	}
	int ic = 0;
	int pc = 0;

	int cache_size = 1024;
	float * cache = malloc(sizeof(float) * cache_size);

	for(p = 0; p < length; p++) {
		float x = *((float*)PyArray_GETPTR2(locations, p, 0));
		float y = *((float*)PyArray_GETPTR2(locations, p, 1));
		float z = *((float*)PyArray_GETPTR2(locations, p, 2));
		if(x > xmax+smlmax || x < xmin-smlmax) continue;
		if(y > ymax+smlmax || y < ymin-smlmax) continue;
		if(z > zmax || z < zmin) continue;
		float sml = *((float*)PyArray_GETPTR1(S, p));
		x -= xmin;
		y -= ymin;
		int i, j;
		float psizeXsml = psizeX / (sml);
		float psizeYsml = psizeY / (sml);
		float pxmin0 =  - x / sml;
		float pymin0 =  - y / sml;
		float imxmin = - x / sml;
		float imymin = - x / sml;
		float imxmax = imxmin + npixelx * psizeXsml;
		float imymax = imymin + npixely * psizeYsml;
		/* imxmin0, imxmax0, imymin0, imymax0 crops the particle within the image*/	
		if(imxmax < -1) continue;
		if(imymax < -1) continue;
		if(imxmin > 1) continue;
		if(imymin > 1) continue;
		int imx0 = overlap_index(imxmin);
		int imx1 = overlap_index(imxmax);
		int imy0 = overlap_index(imymin);
		int imy1 = overlap_index(imymax);
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
		printf("%d: %d %d %d %d %f\n", p, imx0, imy0, imx1, imy1, koverlap[imx0][imy0][imx1][imy1]);
		printf("%d: %f %f %f %f %f\n", p, imxmin, imymin, imxmax, imymax, norm);
		int ipixelmin = floor((x - sml) / psizeX);
		int ipixelmax = ceil((x + sml) / psizeX);
		int jpixelmin = floor((y - sml) / psizeY);
		int jpixelmax = ceil((y + sml) / psizeY);
		float value = *((float*)PyArray_GETPTR1(V, p));

		ic++;
#define PIXEL_IN_IMAGE (i >=0 && i < npixelx && j >=0 && j < npixely)
		
		if(ipixelmin < 0) ipixelmin = 0;
		if(ipixelmax >= npixelx) ipixelmax = npixelx - 1;
		if(jpixelmin < 0) jpixelmin = 0;
		if(jpixelmax >= npixely) jpixelmax = npixely - 1;

		int k;
		int desired_cache_size = (ipixelmax - ipixelmin + 1) * (jpixelmax - jpixelmin + 1);
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
			int x0 = pxmin * KOVERLAP_BINS / 2 + KOVERLAP_BINS / 2;
			int x1 = pxmax * KOVERLAP_BINS / 2 + KOVERLAP_BINS / 2;
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
				int y0 = pymin * KOVERLAP_BINS / 2 + KOVERLAP_BINS / 2;
				int y1 = pymax * KOVERLAP_BINS / 2 + KOVERLAP_BINS / 2;
				if(y0 >= KOVERLAP_BINS || y1 < 0 ) {
					cache[k++] = 0.0;
					continue;
				}
				pc++;

				/* possible if pxmax == 2.0 or pymax == 2.0*/
				if(y0 < 0) y0 = 0;
				if(y1 >= KOVERLAP_BINS) y1 = KOVERLAP_BINS - 1;

				float addbit = 0.0;
				if((x1 - x0 < 2 && y1 - y0 < 2)) {
					float centerx = (pxmax + pxmin);
					float centery = (pymax + pymin);
					float d = 0.5 * sqrt(centerx * centerx + centery * centery) * KLINE_BINS;
					
					int dfloor = floor(d);
					int dceil = ceil(d);
					if(dfloor >= KLINE_BINS) dfloor = KLINE_BINS - 1;
					if(dceil >= KLINE_BINS) dceil = KLINE_BINS - 1;
					float value = (kline[dceil] - kline[dfloor]) * (d - dfloor) + kline[dfloor];
					addbit = value * (pxmax - pxmin) * (pymax - pymin);
				} else {
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
		printf("k = %d, desired = %d\n",k, desired_cache_size);
		float fac = norm / sum;
		k = 0;
		for(i = ipixelmin; i <= ipixelmax; i++)  {
			for(j = jpixelmin; j <= jpixelmax; j++) {
				*((float*)PyArray_GETPTR2(result,i,j)) += value * cache[k] * fac;
					
				k++;
			}
		}
	}
#if 0
	ptime("render");
	printf("ic = %d pc = %d \n", ic, pc);
#endif
	free(cache);
 	Py_DECREF(S);
 	Py_DECREF(locations);
 	Py_DECREF(V);
	return (PyObject*)result;
}

static PyMethodDef module_methods[] = {
	{"image", image, METH_KEYWORDS, image_doc_string },
	{NULL}
};
void HIDDEN initimage(PyObject * m) {
	import_array();
	PyObject * thism = Py_InitModule3("image", module_methods, "image module");
	Py_INCREF(thism);
	PyModule_AddObject(m, "image", thism);
}
