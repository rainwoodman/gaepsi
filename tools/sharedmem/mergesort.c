#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL MERGESORT_ARRAY_API_20120101
#define PY_UFUNC_UNIQUE_SYMBOL MERGESORT_UFUNC_API_20120101
#include <numpy/arrayobject.h>
#include <numpy/ufuncobject.h>
#include <structmember.h>
#define intp npy_intp
/*************
 *
 * mergesort(data, A, B, out)
 * All needs to be u8
 *
 ****************/
static int cmp_i4(int32_t * p1, int32_t * p2) {
    return (*p1 > *p2) -(*p1 < *p2);
}
static int cmp_i8(int64_t * p1, int64_t * p2) {
    return (*p1 > *p2) -(*p1 < *p2);
}
static int cmp_u8(uint64_t * p1, uint64_t * p2) {
    return (*p1 > *p2) -(*p1 < *p2);
}
static int cmp_u4(uint32_t * p1, uint32_t * p2) {
    return (*p1 > *p2) -(*p1 < *p2);
}
static int cmp_f4(float * p1, float * p2) {
    return (*p1 > *p2) -(*p1 < *p2);
}
static int cmp_f8(double * p1, double * p2) {
    return (*p1 > *p2) -(*p1 < *p2);
}

static PyObject * permute(PyObject * self,
    PyObject * args, PyObject * kwds) {
    static char * kwlist[] = {
        "array", "index", NULL
    };
    PyArrayObject * array, * index, *out;
    if(!PyArg_ParseTupleAndKeywords(args, kwds, 
        "O!O!", kwlist,
        &PyArray_Type, &array, 
        &PyArray_Type, &index)) return NULL;

  size_t n = PyArray_SIZE(index);
  size_t * p = PyArray_DATA(index);
  char * data = PyArray_DATA(array);
  int itemsize = PyArray_ITEMSIZE(array);

  size_t i, k, pk;

  for (i = 0; i < n; i++)
    {
      k = p[i];
      
      while (k > i) 
        k = p[k];
      
      if (k < i)
        continue ;
      
      /* Now have k == i, i.e the least in its cycle */
      
      pk = p[k];
      
      if (pk == i)
        continue ;
      
      /* shuffle the elements of the cycle */
      
      {
        unsigned int a;

        char t[itemsize];
        
        memmove(t, & data[i * itemsize], itemsize);
      
        while (pk != i)
          {
            memmove(&data[k * itemsize], & data[pk * itemsize], itemsize);
            k = pk;
            pk = p[k];
          };
        
        memmove(&data[k * itemsize], t, itemsize);
      }
    }
    Py_INCREF(array);
    return (PyObject*) array;
}

static PyObject * merge(PyObject * self,
     PyObject * args, PyObject * kwds) {

    static char * kwlist[] = {
        "data", "A", "B", "out", NULL
    };

    static struct {
        int type;
        int (*func)(void *,  void*);
    } cmp_table[] = {
        {NPY_FLOAT32, cmp_f4},
        {NPY_FLOAT64, cmp_f8},
        {NPY_INT32, cmp_i4},
        {NPY_UINT32, cmp_u4},
        {NPY_INT64, cmp_i8},
        {NPY_UINT64, cmp_u8},
        {0, NULL},
    };
    PyArrayObject * data, * A, * B, * out;
    if(!PyArg_ParseTupleAndKeywords(args, kwds, 
        "O!O!O!O!", kwlist,
        &PyArray_Type, &data, 
        &PyArray_Type, &A, 
        &PyArray_Type, &B, 
        &PyArray_Type, &out)) return NULL;

    intptr_t i = 0;
    int (*func)(void *, void*) = NULL;
    for(i = 0; cmp_table[i].func != NULL; i++) {
        if(cmp_table[i].type == PyArray_TYPE(data)) {
            func = cmp_table[i].func;
        }
    }
    if(func == NULL) {
        PyErr_SetString(PyExc_TypeError, "data format is unknown, taking u4 u8 i4 i8 f4 f8 only");
        return NULL;
    }
    size_t sizeA = PyArray_SIZE(A);
    size_t sizeB = PyArray_SIZE(B);
    void * dataptr = PyArray_DATA(data);
    int64_t * Aptr = PyArray_DATA(A);
    int64_t * Aend = Aptr + sizeA;

    int64_t * Bptr = PyArray_DATA(B);
    int64_t * Bend = Bptr + sizeB;
    int64_t * Optr = PyArray_DATA(out);

    #define VA PyArray_GETPTR1(data, *Aptr)
    #define VB PyArray_GETPTR1(data, (*Bptr) + sizeA)

    Py_BEGIN_ALLOW_THREADS
    while(Aptr < Aend|| Bptr < Bend) {
        while(Aptr < Aend && (Bptr == Bend || func(VA, VB) <= 0)) {
            *Optr = *Aptr;
            Aptr++;
            Optr++;
            //printf("adding from A, i = %ld, k = %ld v = %ld\n", i, k, v);
        }
        while(Bptr < Bend && (Aptr == Aend || func(VA, VB) >= 0)) {
            *Optr = *Bptr + sizeA;
            Bptr++;
            Optr++;
            //printf("adding from B, j = %ld, k = %ld v = %ld\n", j, k, v);
        }
    }
    Py_END_ALLOW_THREADS
    Py_INCREF(Py_None);
    return Py_None;
}

static PyMethodDef module_methods[] = {
	{"merge", (PyCFunction) merge, METH_KEYWORDS, 
    "merge(data, A, B, out) A, B, out are sorted arg indices, data is the unsorted data.\n"
    "len(data) = len(A) + len(B) = len(out)\n"
    "data[:len(A)][A] is sorted\n"
    "data[len(A):][B] is sorted\n"
    "in other words both A and B are 0 started indices\n"
    "everything must be continueous in memory!",},
	{"permute", (PyCFunction) permute, METH_KEYWORDS, 
    "permute(array, index) array = array[index] with O(1) storage. index has to be u8. array has to be 1d.\n"},
	{NULL}
};
static PyObject * m = NULL;
void init_mergesort(void) {
    import_array();
    import_ufunc();
	m = Py_InitModule3("_mergesort", module_methods, "mergesort module for parallel mergesort");
	PyObject * mergefunc = PyCFunction_New(module_methods, NULL);
	PyObject * permutefunc = PyCFunction_New(module_methods + 1, NULL);
	PyModule_AddObject(m, "merge", mergefunc);
	PyModule_AddObject(m, "permute", permutefunc);
}
