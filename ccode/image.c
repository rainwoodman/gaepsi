#include <Python.h>
#include <numpy/arrayobject.h>
#include <time.h>
#define HIDDEN __attribute__ ((visibility ("hidden")))  

#define image_doc_string \
"keywords: locations, sml, value,"\
" xmin, ymin, xmax, ymax, npixelx, npixely, zmin, zmax."\
" kernel_box_values, kernel_box_bins, kernel_box_deta" \
" returns an image of given size."
/*
#define KERNEL_BOX_VALUES4(x0,y0,x1,y1) \
	(*(float*)(PyArray_GETPTR4(kernel_box_values, x0,y0,x1,y1)))*/
#define KERNEL_BOX_VALUES4(x0,y0,x1,y1) \
	(((float*)(kernel_box_values->data))[((((x0) * bins + (y0)) * bins + (x1)) * bins) + (y1)])

static inline float find_kernel_quick(PyArrayObject * kernel_box_values, int bins, int x0,  int y0, int x1, int y1) {
	return KERNEL_BOX_VALUES4(x0,y0,x1,y1);
}

static inline float find_kernel(PyArrayObject * kernel_box_values, int bins, int x0, int y0, int x1, int y1,
	float pxmin, float pymin, float pxmax, float pymax) {

	float addbit = 0;

#define RW(a) ((a) > bins -1)?(bins - 1):(a)
#define LW(a) ((a) < 0)?0:(a)
	float x0y0 = KERNEL_BOX_VALUES4(x0, y0, x0, y0);
	if(x1 == x0 && y0 == y1) {
		addbit = x0y0 * (pymax - pymin) * (pxmax - pxmin) ;
		return addbit;
	}
	float x0y1 = KERNEL_BOX_VALUES4(x0, y1, x0, y1);
	float ldy = (y0 + 1) - pymin;
	float rdy = pymax - y1;
	if(x1 == x0 && y1 == y0 + 1) {
		addbit += x0y0 * ldy * (pxmax - pxmin) ;
		addbit += x0y1 * rdy * (pxmax - pxmin) ;
		return addbit;
	}
	float x1y0 = KERNEL_BOX_VALUES4(x1, y0, x1, y0);
	float ldx = (x0 + 1) - pxmin;
	float rdx = pxmax - x1;
	if(x1 == x0 + 1 && y1 == y0) {
		addbit += x0y0 * ldx * (pymax - pymin);
		addbit += x1y0 * rdx * (pymax - pymin);
		return addbit;
	}
	float x1y1 = KERNEL_BOX_VALUES4(x1, y1, x1, y1);
	if(x1 == x0 + 1 && y1 == y0 + 1) {
		addbit += x0y0 * ldx * ldy;
		addbit += x1y0 * rdx * ldy;
		addbit += x0y1 * ldx * rdy;
		addbit += x1y1 * rdx * rdy;
		return addbit;
	}
	float left = KERNEL_BOX_VALUES4(x0, RW(y0 + 1), x0, LW(y1 - 1));
	if(x1 == x0 && y1 > y0 + 1) {
		addbit += x0y0 * ldy * (pxmax - pxmin);
		addbit += x0y1 * rdy * (pxmax - pxmin);
		addbit += left * (pxmax - pxmin);
		return addbit;
	}
	float top = KERNEL_BOX_VALUES4(RW(x0 + 1), y0, LW(x1 - 1), y0);
	if(x1 > x0 + 1 && y1 == y0) {
		addbit += x0y0 * ldx * (pymax - pymin);
		addbit += x1y0 * rdx * (pymax - pymin);
		addbit += top * (pymax - pymin);
		return addbit;
	}
	float right = KERNEL_BOX_VALUES4(x1, RW(y0 + 1), x1, LW(y1 - 1));
	if(x1 == x0 + 1 && y1 > y0 + 1) {
		addbit += x0y0 * ldx * ldy;
		addbit += x1y0 * rdx * ldy;
		addbit += x0y1 * ldx * rdy;
		addbit += x1y1 * rdx * rdy;
		addbit += left * ldx;
		addbit += right * rdx;
		return addbit;
	}
	float bottom = KERNEL_BOX_VALUES4(RW(x0 + 1), y1, LW(x1 - 1), y1);
	if(x1 > x0 + 1 && y1 == y0 + 1) {
		addbit += x0y0 * ldx * ldy;
		addbit += x1y0 * rdx * ldy;
		addbit += x0y1 * ldx * rdy;
		addbit += x1y1 * rdx * rdy;
		addbit += top * ldy;
		addbit += bottom * rdy;
		return addbit;
	}
	float center = KERNEL_BOX_VALUES4(RW(x0 + 1), RW(y0 + 1), LW(x1 - 1), RW(y1 - 1));
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
		return addbit;
	}
	printf("unhandled x0=%d, x1=%d, y0=%d, y1=%d", x0, x1, y0, y1);
	return 0.0;
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
		"kernel_box_values", "kernel_box_bins", "kernel_box_deta",
		"quick"
	};
	PyArrayObject * locations, * S, *V;
	PyArrayObject * kernel_box_values;
	float xmin, ymin, xmax, ymax, zmin, zmax;
	int npixelx, npixely;
	int bins;
	int length;
	float deta;
	int p;
	int quick;
	if(! PyArg_ParseTupleAndKeywords(args, kwds, "O!O!O!ffffiiffO!ifi", kwlist,
		&PyArray_Type, &locations, 
		&PyArray_Type, &S, 
		&PyArray_Type, &V, 
		&xmin, &ymin, &xmax, &ymax,
		&npixelx, &npixely, &zmin, &zmax,
		&PyArray_Type, 
		&kernel_box_values, &bins, &deta,
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

	int cache_size = 1048576;
	float * cache = malloc(cache_size * sizeof(float));
	
	for(p = 0; p < length; p++) {
		float x = *((float*)PyArray_GETPTR2(locations, p, 0));
		float y = *((float*)PyArray_GETPTR2(locations, p, 1));
		float z = *((float*)PyArray_GETPTR2(locations, p, 2));
		if(x > xmax+smlmax || x < xmin-smlmax) continue;
		if(y > ymax+smlmax || y < ymin-smlmax) continue;
		if(z > zmax || z < zmin) continue;
		float sml = *((float*)PyArray_GETPTR1(S, p));
		float value = *((float*)PyArray_GETPTR1(V, p));
		x -= xmin;
		y -= ymin;
		int ipixelmin = floor((x - sml) / psizeX);
		int ipixelmax = ceil((x + sml) / psizeX);
		int jpixelmin = floor((y - sml) / psizeY);
		int jpixelmax = ceil((y + sml) / psizeY);
		int i, j;
		ic++;
		float psizeXsml = psizeX / (sml * deta);
		float psizeYsml = psizeY / (sml * deta);
		float pxmin0 =  (- x / sml + 1.0) / deta;
		float pymin0 =  (- y / sml + 1.0) / deta;
		float sum = 0.0;
		int k = 0;
		int desired_cache_size = (ipixelmax - ipixelmin + 1) * (jpixelmax - jpixelmin + 1);
		if(desired_cache_size > cache_size) {
			while(desired_cache_size > cache_size) {
				cache_size *= 2;
			}
			free(cache);
			cache = malloc(sizeof(float) * cache_size);
			printf("growing cache to %d\n", cache_size);
		}
		for(i = ipixelmin; i <= ipixelmax; i++)  {
			float pxmin = pxmin0 + i * psizeXsml;
			float pxmax = pxmin + psizeXsml;
			int x0 = pxmin;
			int x1 = pxmax;
			if(x1 < 0 || x0 >=bins) {
				for(j = jpixelmin; j <= jpixelmax; j++) {
					cache[k++] = 0.0;
				}
				continue;
			}
			if(x0 < 0) x0 = 0;
			if(x1 < 0) x1 = 0;
			if(x0 >= bins) x0 = bins - 1;
			if(x1 >= bins) x1 = bins - 1;
			for(j = jpixelmin; j <= jpixelmax; j++) {

				float pymin = pymin0 + psizeYsml * j;
				float pymax = pymin + psizeYsml;
				int y0 = pymin;
				int y1 = pymax;
				if(y0 > bins || y1 < 0 ) {
					cache[k++] = 0.0;
					continue;
				}
				pc++;

				/* possible if pxmax == 2.0 or pymax == 2.0*/
				if(y0 < 0) y0 = 0;
				if(y1 < 0) y1 = 0;
				if(y0 >= bins) y0 = bins - 1;
				if(y1 >= bins) y1 = bins - 1;

				float addbit = quick
					?
					KERNEL_BOX_VALUES4(x0, y0, x1, y1)
					:
					find_kernel(kernel_box_values, bins, x0, y0, x1, y1, pxmin, pymin, pxmax, pymax);
				sum += addbit;
				cache[k++] = addbit;
			}
		}
		k = 0;
		float suminv = 1.0 / sum;
		for(i = ipixelmin; i <= ipixelmax; i++)  {
			if(i < 0 || i >= npixelx) {
				k += (jpixelmax - jpixelmin + 1);
				continue;
			} 
			for(j = jpixelmin; j <= jpixelmax; j++) {
				if(j < 0 || j >= npixely) {
					k++; continue;
				}
				*((float*)PyArray_GETPTR2(result,i,j)) += value * cache[k++] * suminv;
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
