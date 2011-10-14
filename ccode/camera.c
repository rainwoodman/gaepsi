#include <Python.h>
#include <numpy/arrayobject.h>
#include <time.h>
#include "defines.h"

#define HIDDEN __attribute__ ((visibility ("hidden")))

typedef struct _Camera {
	npy_intp displaydim[2];  /* pixels*/
	float pos[3];
	float boxsize[3];
	float far;  /* far cut of distant objects */
	float near; /* near cut of closeby objects */
	float Fov; /* FOV angle */
	float ctnFov; /* ctn of fov used to return effective sml*/
	float dir[3];
	float up[3];
	float matrix[4][4];
	float pix_area;
	PyObject ** sph;
	PyObject ** raster;
	int sph_length;
} Camera;

static float interp(const Camera * cam, const npy_intp ipix, const npy_intp jpix, const float pixpos[2], const float pixsml[2]);
static float camera_transform(const Camera * cam, const float pos[4], const float sml, float NDCpos[2]);
static float splat(const Camera * cam, const float NDCpos[2], 
	const float NDCsml, float ** cache, size_t * cache_size, 
	npy_intp ipixlim[2], npy_intp jpixlim[2]);
static const float quad_linear(const float x, const float y, const float z, const float w);
static float interp_overlap( const float x0f, const float y0f, const float x1f, const float y1f);

static const inline float fclipf(const float v, const float min, const float max) {
	return fmin(fmax(v, min), max);
}

static const inline npy_intp idim(const npy_intp i1, const npy_intp i2) {
	return ((i1 - i2 < 0) - 1) & (i1 - i2);
}
static const inline npy_intp imin(const npy_intp i1, const npy_intp i2) {
	return (i1 <i2)?i1:i2;
}
static const inline npy_intp imax(const npy_intp i1, const npy_intp i2) {
	return (i1 >i2)?i1:i2;
}
static void inline matrixmul(float m1[4][4], float m2[4][4], float out[4][4]) {
	int i, j, k;
	for(i = 0; i < 4; i ++) {
		for(j = 0; j < 4; j++) {
			out[i][j] = 0;
		}
	}
	for(i = 0; i < 4; i ++) {
		for(j = 0; j < 4; j++) {
			for(k = 0; k < 4; k++) {
			out[i][k] += m1[i][j] * m2[j][k];
			}
		}
	}
}
static inline void crossproduct(const float v1[3], const float v2[3], float out[3]) {
	out[0] = v1[1] * v2[2] - v2[1] * v1[2];
	out[1] = v1[2] * v2[0] - v2[2] * v1[0];
	out[2] = v1[0] * v2[1] - v2[0] * v1[1];
}
static void normalize(float v[3]) {
	double s = 0.0;
	s = 1./ sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
	v[0] *= s;
	v[1] *= s;
	v[2] *= s;
}

static PyObject * camera(PyObject * self, PyObject * args, PyObject * kwds) {
	static char * kwlist[] = {
		"raster", 
		"sph", 
		"locations", "sml",
		"dir", "up",
		"near", "far", "Fov", "dim", 
		"pos", "mask",
		NULL
	};
	PyObject * Lraster, * Lsph, * Alocations, * Asmls, * Adim, 
		* Apos, * Amask, * Adirection, *Aup;
	float near, far, Fov;
	if(! PyArg_ParseTupleAndKeywords(args, kwds, "O!O!O!O!O!O!fffO!O!O", kwlist,
		&PyList_Type, &Lraster, 
		&PyList_Type, &Lsph, 
		&PyArray_Type, &Alocations, 
		&PyArray_Type, &Asmls, 
		&PyArray_Type, &Adirection,
		&PyArray_Type, &Aup,
		&near, &far, &Fov,
		&PyArray_Type, &Adim, 
		&PyArray_Type, &Apos, 
		&Amask)) return NULL;
	Camera c = {0};

	if(Amask != Py_None)
		Amask = PyArray_Cast((PyArrayObject*)Amask, NPY_BOOL);

	Alocations = PyArray_Cast((PyArrayObject*)Alocations, NPY_FLOAT);
	Adirection = PyArray_Cast((PyArrayObject*)Adirection, NPY_FLOAT);
	Aup = PyArray_Cast((PyArrayObject*)Aup, NPY_FLOAT);
	Asmls = PyArray_Cast((PyArrayObject*)Asmls, NPY_FLOAT);
	Apos = PyArray_Cast((PyArrayObject*)Apos, NPY_FLOAT);
	Adim = PyArray_Cast((PyArrayObject*)Adim, NPY_INTP);

	c.sph_length = PyList_GET_SIZE(Lsph);
	c.sph = PyMem_New(PyObject*, PyList_GET_SIZE(Lsph));
	c.raster = PyMem_New(PyObject*, PyList_GET_SIZE(Lraster));
	

	c.displaydim[0] = *(npy_intp*)PyArray_GETPTR1(Adim, 0);
	c.displaydim[1] = *(npy_intp*)PyArray_GETPTR1(Adim, 1);
	c.pix_area = 1.0 / (c.displaydim[0] * c.displaydim[1]);
	c.far = far;
	c.near = near;
	c.Fov = Fov;
	c.ctnFov = 1.0 / tan(Fov);
	c.dir[0] = *(float*)PyArray_GETPTR1(Adirection, 0);
	c.dir[1] = *(float*)PyArray_GETPTR1(Adirection, 1);
	c.dir[2] = *(float*)PyArray_GETPTR1(Adirection, 2);

	c.up[0] = *(float*)PyArray_GETPTR1(Aup, 0);
	c.up[1] = *(float*)PyArray_GETPTR1(Aup, 1);
	c.up[2] = *(float*)PyArray_GETPTR1(Aup, 2);
	c.pos[0] = *(float*)PyArray_GETPTR1(Apos, 0);
	c.pos[1] = *(float*)PyArray_GETPTR1(Apos, 1);
	c.pos[2] = *(float*)PyArray_GETPTR1(Apos, 2);

	float side[3];
	crossproduct(c.dir, c.up, side);
	normalize(side);
	crossproduct(side, c.dir, c.up);
	normalize(c.up);

	float matrix2[4][4] = {{0}};
	matrix2[0][0] = side[0];
	matrix2[0][1] = side[1];
	matrix2[0][2] = side[2];
	matrix2[0][3] = 0;
	matrix2[1][0] = c.up[0];
	matrix2[1][1] = c.up[1];
	matrix2[1][2] = c.up[2];
	matrix2[1][3] = 0;
	matrix2[2][0] = -c.dir[0];
	matrix2[2][1] = -c.dir[1];
	matrix2[2][2] = -c.dir[2];
	matrix2[2][3] = 0;
	matrix2[3][0] = 0;
	matrix2[3][1] = 0;
	matrix2[3][2] = 0;
	matrix2[3][3] = 1;

	float translate[4][4] = {{0}};
	translate[0][0] = 1.0;
	translate[1][1] = 1.0;
	translate[2][2] = 1.0;
	translate[3][3] = 1.0;
	translate[0][3] = -c.pos[0];
	translate[1][3] = -c.pos[1];
	translate[2][3] = -c.pos[2];
	translate[3][3] = 1;

	float matrix3[4][4];
	matrixmul(matrix2, translate, matrix3);

	float persp[4][4] = {{0}};
	persp[0][0] = 1.0 / tan(c.Fov);
	persp[1][1] = 1.0 / tan(c.Fov);
	persp[2][2] = - (c.far + c.near) / (c.far - c.near);
	persp[2][3] = - 2.0 * c.far * c.near / (c.far - c.near);
	persp[3][2] = -1;
	persp[3][3] = 0;

	matrixmul(persp, matrix3, c.matrix);

	int i;
	for(i = 0; i < c.sph_length; i++) {
		c.sph[i] = PyList_GET_ITEM(Lsph, i);
		c.raster[i] = PyList_GET_ITEM(Lraster, i);
	}
	for(i = 0; i < c.sph_length; i++) {
		c.sph[i] = PyArray_Cast((PyArrayObject*)c.sph[i], NPY_FLOAT);
		c.raster[i] = PyArray_Cast((PyArrayObject*)c.raster[i], NPY_FLOAT);
	}
	for(i = 0; i < c.sph_length; i++) {
		Py_DECREF(c.sph[i]);
		Py_DECREF(c.raster[i]);
	}
	
	npy_intp ipar;
	#pragma omp parallel private(ipar)
	{
		float * cache = PyMem_New(float, 1000);
		size_t cache_size = 1000;
		
		#pragma omp for schedule(dynamic, 10)
		for(ipar = 0; ipar < PyArray_Size((PyObject*) Asmls); ipar++) {
			float pos[4];
			int d ;
			for(d = 0; d < 3; d++) {
				pos[d] = *(float*)PyArray_GETPTR2(Alocations, ipar, d);
			}
			pos[3] = 1.0;
			float sml = *(float*)PyArray_GETPTR1(Asmls, ipar);
			float NDCpos[2] = {0};
			float NDCsml = camera_transform(&c, pos, sml, NDCpos);
		//	printf("pos=(%g %g %g) pixpos = (%g %g ) sml = %g \n", pos[0], pos[1], pos[2], NDCpos[0], NDCpos[1], sml);
			if(NDCsml > 0.0) {
				npy_intp ipixlim[2];
				npy_intp jpixlim[2];
				float sum = splat(&c, NDCpos, NDCsml, &cache, &cache_size, ipixlim, jpixlim);
			//	printf("sml = %g sum = %g pix_area = %g x -> %ld %ld y -> %ld %ld\n", sml, sum, c.pix_area, ipixlim[0], ipixlim[1], jpixlim[0], jpixlim[1]);
				int i;
				if(sum > 0.0)
				for(i = 0; i < c.sph_length; i++) {
					npy_intp k = 0;
					float value = *((float*)PyArray_GETPTR1(c.sph[i], ipar));
					int single_precision = (PyArray_ITEMSIZE(c.sph[i]) == 4);
					npy_intp ipix, jpix;
					if(single_precision) {
						for(ipix = ipixlim[0]; ipix <= ipixlim[1]; ipix++) {
						for(jpix = jpixlim[0]; jpix <= jpixlim[1]; jpix++) {
							#pragma omp atomic
							*(float*)PyArray_GETPTR2(c.raster[i], ipix, jpix) += value * cache[k];
							k++;
						}
						}
					} else {
						for(ipix = ipixlim[0]; ipix <= ipixlim[1]; ipix++) {
						for(jpix = jpixlim[0]; jpix <= jpixlim[1]; jpix++) {
							#pragma omp atomic
							*(double*)PyArray_GETPTR2(c.raster[i], ipix, jpix) += value * cache[k];
							k++;
						}
						}
					}
					if(k > cache_size) {
						printf("k > cache_size\n");
						abort();
					}
				}
			}
		}
		PyMem_Del(cache);
	}
	PyMem_Del(c.sph);
	PyMem_Del(c.raster);
	Py_XDECREF(Adirection);
	Py_XDECREF(Aup);
	Py_XDECREF(Amask);
	Py_XDECREF(Apos);
	Py_XDECREF(Alocations);
	Py_XDECREF(Asmls);
	Py_XDECREF(Adim);
	return Py_None;
}

static float camera_transform(const Camera * cam, 
	const float pos[4], const float sml,
	float NDCpos[2]) {
	/* returns 0.0 if the pos is not on the cam, 
     * otherwise fill NDCpos, return the sml in NDC units */
	float tpos[4] = {0.0, 0.0, 0.0, 0.0};
	int i;
	for(i = 0; i < 4; i++) {
		tpos[0] += pos[i] * cam->matrix[0][i];
		tpos[1] += pos[i] * cam->matrix[1][i];
		tpos[2] += pos[i] * cam->matrix[2][i];
		tpos[3] += pos[i] * cam->matrix[3][i];
	}
	for(i = 0; i < 3; i++) {
		tpos[i] /= tpos[3];
		if(tpos[i] < -1.0 || tpos[i] > 1.0) return 0.0;
	}
	NDCpos[0] = tpos[0];
	NDCpos[1] = tpos[1];

	return sml * cam->ctnFov / tpos[3];
}

static float splat(const Camera * cam, const float NDCpos[2], const float NDCsml, float ** cache, size_t * cache_size, npy_intp ipixlim[2], npy_intp jpixlim[2]) {
	/* splats a sph particle to pixels. return 0 if the particle covers too many pixels,
     * assuming it will be too diluted to be shown. 
     * 
     * */
	float pixpos[2];
	pixpos[0] = 0.5 * (NDCpos[0] + 1.0) * cam->displaydim[0];
	pixpos[1] = 0.5 * (NDCpos[1] + 1.0) * cam->displaydim[1];

	float pixsml[2];
	pixsml[0] = 0.5 * NDCsml * cam->displaydim[0];
	pixsml[1] = 0.5 * NDCsml * cam->displaydim[1];

	npy_intp ipix, jpix;
	ipixlim[0] = idim(pixpos[0], pixsml[0]);
	ipixlim[1] = imin(pixpos[0] + pixsml[0], cam->displaydim[0] - 1);
	jpixlim[0] = idim(pixpos[1], pixsml[1]);
	jpixlim[1] = imin(pixpos[1] + pixsml[1], cam->displaydim[1] - 1);

	/* npy_intp overflows with huge smls !*/
	double pixsml_area =  4 * (pixsml[0] + 1)* (pixsml[1] + 1);
	if(pixsml_area > 1024 * 1024) {
		return 0.0;
	}
	if(pixsml_area > *cache_size) {
		while(pixsml_area > *cache_size) {
			(*cache_size) *= 2;
		}
		PyMem_Del(*cache);
		*cache = PyMem_New(float, *cache_size);
	}

	npy_intp k = 0;
	double sum_cache = 0.0;
	for(ipix = ipixlim[0]; ipix <= ipixlim[1]; ipix++) {
	for(jpix = jpixlim[0]; jpix <= jpixlim[1]; jpix++) {
		sum_cache += ((*cache)[k] = interp(cam, ipix, jpix, pixpos, pixsml));
		k++;
	}
	}
	return sum_cache;
}

extern HIDDEN float kline[];
static float interp(const Camera * cam, const npy_intp ipix, const npy_intp jpix, const float pixpos[2], const float pixsml[2]){
/* pos is in the image coordinate. */

	float relx[2];
	float rely[2];

	relx[0] = ((ipix - pixpos[0]) / pixsml[0] + 1.) * 0.5 * KOVERLAP_BINS;
	relx[1] = ((ipix + 1 - pixpos[0]) / pixsml[0] + 1.) * 0.5 * KOVERLAP_BINS;
	rely[0] = ((jpix - pixpos[1]) / pixsml[1] + 1.) * 0.5 * KOVERLAP_BINS;
	rely[1] = ((jpix + 1 - pixpos[1]) / pixsml[1] + 1.) * 0.5 * KOVERLAP_BINS;

	if(relx[1] - relx[0] < 2 && rely[1] - rely[0] < 2) {
		float dx = (ipix + 0.5 - pixpos[0]) / pixsml[0];
		float dy = (jpix + 0.5 - pixpos[1]) / pixsml[1];
		float dist = sqrt(dx * dx + dy * dy);
		dist = dist * KLINE_BINS;
		npy_intp dfloor = dist;
		if(dfloor + 1 < KLINE_BINS) {
			return (kline[dfloor + 1] * (dist - dfloor) + kline[dfloor] * (dfloor + 1 - dist)) / (pixsml[0] * pixsml[1]);
		} else {
			return 0.0;
		}
	} else {
		return interp_overlap(relx[0], rely[0], relx[1], rely[1]);
	}
}
static PyMethodDef module_methods[] = {
	{"camera", (PyCFunction) camera, METH_KEYWORDS, 
	"raster, sph, locations, sml, near, far, F, dim, mask, boxsize",
	},
	{NULL}
};
void HIDDEN gadget_initcamera(PyObject * m) {
	import_array();
	PyObject * camera_func= PyCFunction_New(module_methods, NULL);
	PyModule_AddObject(m, "camera", camera_func);
}

extern HIDDEN float koverlap[][KOVERLAP_BINS][KOVERLAP_BINS][KOVERLAP_BINS];

const float quad_linear(const float x, const float y, const float z, const float w) {
	double sum = 0.0;
	int x0 = x;
	int y0 = y;
	int z0 = ceil(z);
	int w0 = ceil(w);
	if(x0 < 0) x0 = 0;
	if(y0 < 0) y0 = 0;
	if(z0 >= KOVERLAP_BINS - 1) z0 = KOVERLAP_BINS - 1;
	if(w0 >= KOVERLAP_BINS - 1) w0 = KOVERLAP_BINS - 1;

	int x1 = x0 + 1;
	int y1 = y0 + 1;
	int z1 = z0 - 1;
	int w1 = w0 - 1;
	
	if(z0 <= 0) return 0;
	if(w0 <= 0) return 0;
	if(x0 >= KOVERLAP_BINS - 1) return 0;
	if(y0 >= KOVERLAP_BINS - 1) return 0;

	sum += koverlap[x0][y0][z0][w0] * (x1 - x) * (y1 - y) * (z1 - z) * (w1 - w);
	sum += koverlap[x0][y0][z0][w1] * (x1 - x) * (y1 - y) * (z1 - z) * (w - w0);
	sum += koverlap[x0][y0][z1][w0] * (x1 - x) * (y1 - y) * (z - z0) * (w1 - w);
	sum += koverlap[x0][y0][z1][w1] * (x1 - x) * (y1 - y) * (z - z0) * (w - w0);
	sum += koverlap[x0][y1][z0][w0] * (x1 - x) * (y - y0) * (z1 - z) * (w1 - w);
	sum += koverlap[x0][y1][z0][w1] * (x1 - x) * (y - y0) * (z1 - z) * (w - w0);
	sum += koverlap[x0][y1][z1][w0] * (x1 - x) * (y - y0) * (z - z0) * (w1 - w);
	sum += koverlap[x0][y1][z1][w1] * (x1 - x) * (y - y0) * (z - z0) * (w - w0);

	sum += koverlap[x1][y0][z0][w0] * (x - x0) * (y1 - y) * (z1 - z) * (w1 - w);
	sum += koverlap[x1][y0][z0][w1] * (x - x0) * (y1 - y) * (z1 - z) * (w - w0);
	sum += koverlap[x1][y0][z1][w0] * (x - x0) * (y1 - y) * (z - z0) * (w1 - w);
	sum += koverlap[x1][y0][z1][w1] * (x - x0) * (y1 - y) * (z - z0) * (w - w0);
	sum += koverlap[x1][y1][z0][w0] * (x - x0) * (y - y0) * (z1 - z) * (w1 - w);
	sum += koverlap[x1][y1][z0][w1] * (x - x0) * (y - y1) * (z1 - z) * (w - w0);
	sum += koverlap[x1][y1][z1][w0] * (x - x0) * (y - y0) * (z - z0) * (w1 - w);
	sum += koverlap[x1][y1][z1][w1] * (x - x0) * (y - y0) * (z - z0) * (w - w0);
	return sum;
}
static float interp_overlap(const float x0f, const float y0f, const float x1f, const float y1f) {

	int x0 = x0f, y0 = y0f, x1 = x1f, y1 = y1f;

	if(x0 < 0) x0 = 0;
	if(y0 < 0) y0 = 0;
	if(x1 >= KOVERLAP_BINS - 1) x1 = KOVERLAP_BINS - 1;
	if(y1 >= KOVERLAP_BINS - 1) y1 = KOVERLAP_BINS - 1;

	if(x1 <= 0) return 0;
	if(y1 <= 0) return 0;
	if(x0 >= KOVERLAP_BINS - 1) return 0;
	if(y0 >= KOVERLAP_BINS - 1) return 0;

	const float dpx = x1f - x0f;
	const float dpy = y1f - y0f;

	float addbit = 0;

#define RW(a) ((a) > KOVERLAP_BINS -1)?(KOVERLAP_BINS - 1):(a)
#define LW(a) ((a) < 0)?0:(a)
	const float x0y0 = koverlap[x0][y0][x0][y0];
	if(x1 == x0 && y0 == y1) {
		addbit = x0y0 * dpy * dpx ;
		goto exit;
	}
	const float x0y1 = koverlap[x0][y1][x0][y1];
	const float ldy = (y0 + 1) - y0f;
	const float rdy = y1f - y1;
	if(x1 == x0 && y1 == y0 + 1) {
		addbit += x0y0 * ldy * dpx;
		addbit += x0y1 * rdy * dpx;
		goto exit;
	}
	const float x1y0 = koverlap[x1][y0][x1][y0];
	const float ldx = (x0 + 1) - x0f;
	const float rdx = x1f - x1;
	if(x1 == x0 + 1 && y1 == y0) {
		addbit += x0y0 * ldx * dpy;
		addbit += x1y0 * rdx * dpy;
		goto exit;
	}
	const float x1y1 = koverlap[x1][y1][x1][y1];
	if(x1 == x0 + 1 && y1 == y0 + 1) {
		addbit += x0y0 * ldx * ldy;
		addbit += x1y0 * rdx * ldy;
		addbit += x0y1 * ldx * rdy;
		addbit += x1y1 * rdx * rdy;
		goto exit;
	}
	const float left = koverlap[x0][RW(y0 + 1)][x0][LW(y1 - 1)];
	if(x1 == x0 && y1 > y0 + 1) {
		addbit += x0y0 * ldy * dpx;
		addbit += x0y1 * rdy * dpx;
		addbit += left * dpx;
		goto exit;
	}
	const float top = koverlap[RW(x0 + 1)][y0][LW(x1 - 1)][y0];
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
	const float bottom = koverlap[RW(x0 + 1)][y1][LW(x1 - 1)][y1];
	if(x1 > x0 + 1 && y1 == y0 + 1) {
		addbit += x0y0 * ldx * ldy;
		addbit += x1y0 * rdx * ldy;
		addbit += x0y1 * ldx * rdy;
		addbit += x1y1 * rdx * rdy;
		addbit += top * ldy;
		addbit += bottom * rdy;
		goto exit;
	}
	const float center = koverlap[RW(x0 + 1)][RW(y0 + 1)][LW(x1 - 1)][RW(y1 - 1)];
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
