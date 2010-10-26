#include <Python.h>
#include <numpy/arrayobject.h>

#define HIDDEN __attribute__ ((visibility ("hidden")))  

#define image_doc_string \
"keywords: locations, sml, value,"\
" xmin, ymin, xmax, ymax, npixelx, npixely, zmin, zmax."\
" kernel_box_values, kernel_box_bins, kernel_box_deta" \
" returns an image of given size."


static float find_kernel(PyArrayObject * kernel_box_values, int bins, float deta,
	float pxmin, float pymin, float pxmax, float pymax) {
	float deta2 = deta * deta;
#define KERNEL_BOX_VALUES4(x0,y0,x1,y1) \
	(*(float*)(PyArray_GETPTR4(kernel_box_values, x0,y0,x1,y1)))
	float addbit = 0;
	/* particle is within the pixel, 
	   add the entire particle in */
	if(pxmin == 0.0 && pymin == 0.0 && pxmax == 2.0 && pymax == 2.0) {
		addbit = 1.0;
		return addbit;
	}

	/* find which bin the pixel edges sit in */
	int x0 = pxmin / deta;
	int y0 = pymin / deta;
	int x1 = pxmax / deta;
	int y1 = pymax / deta;

	/* possible if pxmax == 2.0 or pymax == 2.0*/
	if(x1 == bins) x1 = bins - 1;
	if(y1 == bins) y1 = bins - 1;

#define RW(a) ((a) > bins -1)?(bins - 1):(a)
#define LW(a) ((a) < 0)?0:(a)
	double x0y0 = KERNEL_BOX_VALUES4(x0, y0, x0, y0);
	if(x1 == x0 && y0 == y1) {
		addbit = x0y0 * (pymax - pymin) * (pxmax - pxmin) / deta2;
		return addbit;
	}
	double x0y1 = KERNEL_BOX_VALUES4(x0, y1, x0, y1);
	double ldy = (y0 + 1) * deta - pymin;
	double rdy = pymax - y1 * deta;
	if(x1 == x0 && y1 == y0 + 1) {
		addbit += x0y0 * ldy * (pxmax - pxmin) / deta2;
		addbit += x0y1 * rdy * (pxmax - pxmin) / deta2;
		return addbit;
	}
	double x1y0 = KERNEL_BOX_VALUES4(x1, y0, x1, y0);
	double ldx = (x0 + 1) * deta - pxmin;
	double rdx = pxmax - x1 * deta;
	if(x1 == x0 + 1 && y1 == y0) {
		addbit += x0y0 * ldx * (pymax - pymin) / deta2;
		addbit += x1y0 * rdx * (pymax - pymin) / deta2;
		return addbit;
	}
	double x1y1 = KERNEL_BOX_VALUES4(x1, y1, x1, y1);
	if(x1 == x0 + 1 && y1 == y0 + 1) {
		addbit += x0y0 * ldx * ldy / deta2;
		addbit += x1y0 * rdx * ldy / deta2;
		addbit += x0y1 * ldx * rdy / deta2;
		addbit += x1y1 * rdx * rdy / deta2;
		return addbit;
	}
	double left = KERNEL_BOX_VALUES4(x0, RW(y0 + 1), x0, LW(y1 - 1));
	if(x1 == x0 && y1 > y0 + 1) {
		addbit += x0y0 * ldy * (pxmax - pxmin) / deta2;
		addbit += x0y1 * rdy * (pxmax - pxmin) / deta2;
		addbit += left * (pxmax - pxmin) / deta;
		return addbit;
	}
	double top = KERNEL_BOX_VALUES4(RW(x0 + 1), y0, LW(x1 - 1), y0);
	if(x1 > x0 + 1 && y1 == y0) {
		addbit += x0y0 * ldx * (pymax - pymin) / deta2;
		addbit += x1y0 * rdx * (pymax - pymin) / deta2;
		addbit += top * (pymax - pymin) / deta;
		return addbit;
	}
	double right = KERNEL_BOX_VALUES4(x1, RW(y0 + 1), x1, LW(y1 - 1));
	if(x1 == x0 + 1 && y1 > y0 + 1) {
		addbit += x0y0 * ldx * ldy / deta2;
		addbit += x1y0 * rdx * ldy / deta2;
		addbit += x0y1 * ldx * rdy / deta2;
		addbit += x1y1 * rdx * rdy / deta2;
		addbit += left * ldx / deta;
		addbit += right * rdx / deta;
		return addbit;
	}
	double bottom = KERNEL_BOX_VALUES4(RW(x0 + 1), y1, LW(x1 - 1), y1);
	if(x1 > x0 + 1 && y1 == y0 + 1) {
		addbit += x0y0 * ldx * ldy / deta2;
		addbit += x1y0 * rdx * ldy / deta2;
		addbit += x0y1 * ldx * rdy / deta2;
		addbit += x1y1 * rdx * rdy / deta2;
		addbit += top * ldy / deta;
		addbit += bottom * rdy / deta;
		return addbit;
	}
	double center = KERNEL_BOX_VALUES4(RW(x0 + 1), RW(y0 + 1), LW(x1 - 1), RW(y1 - 1));
	if(x1 > x0 + 1 && y1 > y0 + 1) {
		addbit += x0y0 * ldx * ldy / deta2;
		addbit += x1y0 * rdx * ldy / deta2;
		addbit += x0y1 * ldx * rdy / deta2;
		addbit += x1y1 * rdx * rdy / deta2;
		addbit += left * ldx / deta;
		addbit += right * rdx / deta;
		addbit += top * ldy / deta;
		addbit += bottom * rdy / deta;
		addbit += center;
		return addbit;
	}
	printf("unhandled x0=%d, x1=%d, y0=%d, y1=%d", x0, x1, y0, y1);
	return 0.0;
}
static PyObject * image(PyObject * self, 
	PyObject * args, PyObject * kwds) {
	static char * kwlist[] = {
		"locations", "sml", "value", 
		"xmin", "ymin", "xmax", "ymax",
		"npixelx", "npixely", "zmin", "zmax",
		"kernel_box_values", "kernel_box_bins", "kernel_box_deta"
	};
	PyArrayObject * locations, * S, *V;
	PyArrayObject * kernel_box_values;
	float xmin, ymin, xmax, ymax, zmin, zmax;
	int npixelx, npixely;
	int kernel_box_bins;
	int length;
	float kernel_box_deta;
	int p;
	if(! PyArg_ParseTupleAndKeywords(args, kwds, "O!O!O!ffffiiffO!if", kwlist,
		&PyArray_Type, &locations, 
		&PyArray_Type, &S, 
		&PyArray_Type, &V, 
		&xmin, &ymin, &xmax, &ymax,
		&npixelx, &npixely, &zmin, &zmax,
		&PyArray_Type, &kernel_box_values, &kernel_box_bins, &kernel_box_deta)) return NULL;

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
	for(p = 0; p < length; p++) {
		float x = *((float*)PyArray_GETPTR2(locations, p, 0));
		float y = *((float*)PyArray_GETPTR2(locations, p, 1));
		float z = *((float*)PyArray_GETPTR2(locations, p, 2));
		if(z > zmax || z < zmin) continue;
		if(x > xmax+smlmax || x < xmin-smlmax) continue;
		if(y > ymax+smlmax || y < ymin-smlmax) continue;
		float sml = *((float*)PyArray_GETPTR1(S, p));
		float value = *((float*)PyArray_GETPTR1(V, p));
		x -= xmin;
		y -= ymin;
		int ipixelmin = floor(x - sml) / psizeX;
		int ipixelmax = ceil(x + sml) / psizeX;
		int jpixelmin = floor(y - sml) / psizeY;
		int jpixelmax = ceil(y + sml) / psizeY;
		int i, j;
		ic++;
		if(ipixelmin < 0 ) ipixelmin = 0;
		if(jpixelmin < 0 ) jpixelmin = 0;
		if(ipixelmax >= npixelx) ipixelmax = npixelx - 1;
		if(jpixelmax >= npixely) jpixelmax = npixely - 1;
		for(i = ipixelmin; i < ipixelmax; i++)
		for(j = jpixelmin; j < jpixelmax; j++) {
			
			float * pixel = (float*)PyArray_GETPTR2(result, i, j);
			float pxmin = i * psizeX;
			float pxmax = pxmin + psizeX;
			float pymin = j * psizeY;
			float pymax = pymin + psizeY;
			pxmin = (pxmin - x) / sml + 1.0;
			pxmax = (pxmax - x) / sml + 1.0;
			pymin = (pymin - y) / sml + 1.0;
			pymax = (pymax - y) / sml + 1.0;
			pc++;
			if(pymax < 0.0) continue;
			if(pxmax < 0.0) continue;
			if(pymin > 2.0) continue;
			if(pxmin > 2.0) continue;

			if(pxmin < 0.0) pxmin = 0.0;
			if(pymin < 0.0) pymin = 0.0;
			if(pxmax > 2.0) pxmax = 2.0;
			if(pymax > 2.0) pymax = 2.0;

			float addbit = find_kernel(kernel_box_values, kernel_box_bins, kernel_box_deta, pxmin, pymin, pxmax, pymax);
			*pixel += value * addbit;
		}
	}
	printf("ic = %d pc = %d \n", ic, pc);
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
