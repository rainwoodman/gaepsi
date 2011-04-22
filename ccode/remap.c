#include <Python.h>
#include <numpy/arrayobject.h>
#define HIDDEN __attribute__ ((visibility ("hidden")))  

#define doc_string \
"keywords: POS, ROWVECTORS, BOX, MIN, MAX"\
"modifies the POS in place such that each point is shifted into the BOX by integer times of ROWVECTORS."

static inline int inbox(float pos[], float box[], int D) {
	int i;
#define ERR 1e-6
	for(i = 0; i < D; i++) {
		if(pos[i] < 0.0 - ERR) return 0;
		if(pos[i] > box[i] + ERR) return 0;
	}
	return 1;
}
static inline void tryshift(float original[], float shifted[], 
	float rowvectors[3][3], int I[], int D) {
	int i, j;	
	for(i = 0; i < D; i++) {
		shifted[i] = original[i];
		for(j = 0; j < D; j++) {
	 		shifted[i] += rowvectors[j][i] * I[j];
		}
	}
}
static PyObject * shift(PyObject * self, 
	PyObject * args, PyObject * kwds) {
	PyArrayObject * pos, * arowvectors, *abox, *amin, *amax;
	static char * kwlist[] = {"POS", "ROWVECTORS", "BOX", "MIN", "MAX", NULL};
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "O!O!O!O!O!", kwlist,
		&PyArray_Type, &pos,
		&PyArray_Type, &arowvectors,
		&PyArray_Type, &abox,
		&PyArray_Type, &amin,
		&PyArray_Type, &amax))
		return NULL;
	
	int D = PyArray_DIMS(pos)[1];
	npy_intp length = PyArray_DIMS(pos)[0];
	int i,j;
	npy_intp p;
	amin = (PyArrayObject *) PyArray_Cast(amin, NPY_INT);
	amax = (PyArrayObject *) PyArray_Cast(amax, NPY_INT);
	abox = (PyArrayObject *) PyArray_Cast(abox, NPY_FLOAT);
	arowvectors = (PyArrayObject *) PyArray_Cast(arowvectors, NPY_FLOAT);
	int min[3] = {0}, max[3] = {0};
	float box[3];
	float rowvectors[3][3];
	for(i = 0; i < D; i++) {
		min[i] = *((int*)PyArray_GETPTR1(amin, i));
		max[i] = *((int*)PyArray_GETPTR1(amax, i));
		box[i] = *((float*)PyArray_GETPTR1(abox, i));
		for(j = 0; j < D; j++) {
			rowvectors[i][j] = *((float*)PyArray_GETPTR2(arowvectors, i, j));
		}
	}
#if 0
	printf("box = %f %f %f\n", box[0], box[1], box[2]);
	printf("min = %d %d %d\n", min[0], min[1], min[2]);
	printf("max = %d %d %d\n", max[0], max[1], max[2]);
	printf("rowvectors = %f %f %f\n", rowvectors[0][0], rowvectors[0][1], rowvectors[0][2]);
	printf("rowvectors = %f %f %f\n", rowvectors[1][0], rowvectors[1][1], rowvectors[1][2]);
	printf("rowvectors = %f %f %f\n", rowvectors[2][0], rowvectors[2][1], rowvectors[2][2]);
	printf("D = %d\n", D);
#endif
	int I[3] = {0,0,0};
	int failed_count = 0;
	for(p = 0; p < length; p++) {
		float * ppos[3];
		float shifted[3];
		float original[3];
		for(i = 0; i < D; i++) {
			ppos[i] = (float *) PyArray_GETPTR2(pos, p, i);
			original[i] = *(ppos[i]);
		}
		
		tryshift(original, shifted, rowvectors, I, D);
		if(!inbox(shifted, box, D)) {
			int newI[3];
			for(newI[0] = min[0]; newI[0] <= max[0]; newI[0]++)
			for(newI[1] = min[1]; newI[1] <= max[1]; newI[1]++)
			for(newI[2] = min[2]; newI[2] <= max[2]; newI[2]++) {
				tryshift(original, shifted, rowvectors, newI, D);
				if(inbox(shifted, box, D)) {
					for(i = 0; i < D; i++) {
						I[i] = newI[i];
					}
					goto out_of_here;
				}
			}
			out_of_here:;
		}
		if(!inbox(shifted, box, D)) {
			printf("%f %f %f to ", original[0], original[1], original[2]);
			printf("%f %f %f faild\n", shifted[0], shifted[1], shifted[2]);
			for(i = 0; i < D; i++) {
				shifted[i] = original[i];
			}
			failed_count++;
		}
		for(i = 0; i < D; i++) {
			*(ppos[i]) = shifted[i];
		}
	}
#if 0
	printf("failed = %d\n", failed_count);
#endif 
	Py_DECREF(abox);
	Py_DECREF(amax);
	Py_DECREF(amin);
	Py_DECREF(arowvectors);
	Py_RETURN_NONE;
}

static PyMethodDef module_methods[] = {
	{"remap_shift", shift, METH_KEYWORDS, doc_string },
	{NULL}
};
void HIDDEN gadget_initremap(PyObject * m) {
	import_array();
//	PyObject * thism = Py_InitModule3("remap", module_methods, "remap module");
//	Py_INCREF(thism);
//	PyModule_AddObject(m, "remap", thism);
	PyObject * remap_f = PyCFunction_New(module_methods, NULL);
	PyModule_AddObject(m, "remap_shift", remap_f);
}
