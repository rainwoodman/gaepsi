#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/ufuncobject.h>
#include <time.h>
#include "defines.h"

#define HIDDEN __attribute__ ((visibility ("hidden")))

static npy_int64 peano_hilbert_key(npy_int x, npy_int y, npy_int z);

static void PyUFunc_ppp_l(char **args, npy_intp *dimensions, npy_intp *steps, void *func) {
	typedef npy_intp (ftype )(npy_int , npy_int, npy_int);
	ftype * f = (ftype*) func;
	char * ip1 = args[0], *ip2 = args[1], *ip3 = args[2], *op1 = args[3];
	npy_intp is1 = steps[0], is2 = steps[1], is3 = steps[2], os1 = steps[3];
	npy_intp n = dimensions[0];
	npy_intp i;
	for(i = 0; i < n; i++, ip1+=is1, ip2+=is2, ip3+=is3, op1+=os1){
        npy_intp in1 = *(npy_intp *)ip1;
        npy_intp in2 = *(npy_intp *)ip2;
        npy_intp in3 = *(npy_intp *)ip3;
        *(npy_int64 *)op1 = f(in1, in2, in3);
	}
}
static void PyUFunc_ccc_l(char **args, npy_intp *dimensions, npy_intp *steps, void *func) {
	typedef npy_intp (ftype )(npy_int, npy_int, npy_int);
	ftype * f = (ftype*) func;
	char * ip1 = args[0], *ip2 = args[1], *ip3 = args[2], *op1 = args[3];
	npy_intp is1 = steps[0], is2 = steps[1], is3 = steps[2], os1 = steps[3];
	npy_intp n = dimensions[0];
	npy_intp i;
	for(i = 0; i < n; i++, ip1+=is1, ip2+=is2, ip3+=is3, op1+=os1){
        char in1 = *(char *)ip1;
        char in2 = *(char *)ip2;
        char in3 = *(char *)ip3;
        *(npy_int64 *)op1 = f(in1, in2, in3);
	}
}

static void PyUFunc_sss_l(char **args, npy_intp *dimensions, npy_intp *steps, void *func) {
	typedef npy_intp (ftype )(npy_int, npy_int, npy_int);
	ftype * f = (ftype*) func;
	char * ip1 = args[0], *ip2 = args[1], *ip3 = args[2], *op1 = args[3];
	npy_intp is1 = steps[0], is2 = steps[1], is3 = steps[2], os1 = steps[3];
	npy_intp n = dimensions[0];
	npy_intp i;
	for(i = 0; i < n; i++, ip1+=is1, ip2+=is2, ip3+=is3, op1+=os1){
        npy_int16 in1 = *(npy_int16 *)ip1;
        npy_int16 in2 = *(npy_int16 *)ip2;
        npy_int16 in3 = *(npy_int16 *)ip3;
        *(npy_int64 *)op1 = f(in1, in2, in3);
	}
}

static void PyUFunc_iii_l(char **args, npy_intp *dimensions, npy_intp *steps, void *func) {
	typedef npy_intp (ftype )(npy_int, npy_int, npy_int);
	ftype * f = (ftype*) func;
	char * ip1 = args[0], *ip2 = args[1], *ip3 = args[2], *op1 = args[3];
	npy_intp is1 = steps[0], is2 = steps[1], is3 = steps[2], os1 = steps[3];
	npy_intp n = dimensions[0];
	npy_intp i;
	for(i = 0; i < n; i++, ip1+=is1, ip2+=is2, ip3+=is3, op1+=os1){
        npy_int32 in1 = *(npy_int32 *)ip1;
        npy_int32 in2 = *(npy_int32 *)ip2;
        npy_int32 in3 = *(npy_int32 *)ip3;
        *(npy_int64 *)op1 = f(in1, in2, in3);
	}
}

static void PyUFunc_lll_l(char **args, npy_intp *dimensions, npy_intp *steps, void *func) {
	typedef npy_intp (ftype )(npy_int, npy_int, npy_int);
	ftype * f = (ftype*) func;
	char * ip1 = args[0], *ip2 = args[1], *ip3 = args[2], *op1 = args[3];
	npy_intp is1 = steps[0], is2 = steps[1], is3 = steps[2], os1 = steps[3];
	npy_intp n = dimensions[0];
	npy_intp i;
	for(i = 0; i < n; i++, ip1+=is1, ip2+=is2, ip3+=is3, op1+=os1){
        npy_int64 in1 = *(npy_int64 *)ip1;
        npy_int64 in2 = *(npy_int64 *)ip2;
        npy_int64 in3 = *(npy_int64 *)ip3;
        *(npy_int64 *)op1 = f(in1, in2, in3);
	}
}

static PyUFuncGenericFunction generic_functions[] = {
	PyUFunc_ccc_l,
	PyUFunc_sss_l,
	PyUFunc_iii_l,
	PyUFunc_lll_l};

static char generic_signatures[] = {
	NPY_INT8, NPY_INT8, NPY_INT8, NPY_INTP,
	NPY_INT16, NPY_INT16, NPY_INT16, NPY_INTP,
	NPY_INT32, NPY_INT32, NPY_INT32, NPY_INTP,
	NPY_INT64, NPY_INT64, NPY_INT64, NPY_INTP,
};
static void* generic_data[] = {
	(void*) peano_hilbert_key,
	(void*) peano_hilbert_key,
	(void*) peano_hilbert_key,
	(void*) peano_hilbert_key,
};

void HIDDEN gadget_initpeano(PyObject * m) {
	import_array();
	import_ufunc();

	PyObject * peanokey_ufunc = PyUFunc_FromFuncAndData(generic_functions, 
					generic_data, generic_signatures, 
					4, 3, 1, 
					PyUFunc_None, 
					"peanohilbert", "peano hilbert key, 32 bit precision", 0);
	
	PyModule_AddObject(m, "peanohilbert", peanokey_ufunc);
}

/*  The following rewrite of the original function
 *  peano_hilbert_key_old() has been written by MARTIN REINECKE. 
 *  It is about a factor 2.3 - 2.5 faster than Volker's old routine!
 */
const unsigned char rottable3[48][8] = {
  {36, 28, 25, 27, 10, 10, 25, 27},
  {29, 11, 24, 24, 37, 11, 26, 26},
  {8, 8, 25, 27, 30, 38, 25, 27},
  {9, 39, 24, 24, 9, 31, 26, 26},
  {40, 24, 44, 32, 40, 6, 44, 6},
  {25, 7, 33, 7, 41, 41, 45, 45},
  {4, 42, 4, 46, 26, 42, 34, 46},
  {43, 43, 47, 47, 5, 27, 5, 35},
  {33, 35, 36, 28, 33, 35, 2, 2},
  {32, 32, 29, 3, 34, 34, 37, 3},
  {33, 35, 0, 0, 33, 35, 30, 38},
  {32, 32, 1, 39, 34, 34, 1, 31},
  {24, 42, 32, 46, 14, 42, 14, 46},
  {43, 43, 47, 47, 25, 15, 33, 15},
  {40, 12, 44, 12, 40, 26, 44, 34},
  {13, 27, 13, 35, 41, 41, 45, 45},
  {28, 41, 28, 22, 38, 43, 38, 22},
  {42, 40, 23, 23, 29, 39, 29, 39},
  {41, 36, 20, 36, 43, 30, 20, 30},
  {37, 31, 37, 31, 42, 40, 21, 21},
  {28, 18, 28, 45, 38, 18, 38, 47},
  {19, 19, 46, 44, 29, 39, 29, 39},
  {16, 36, 45, 36, 16, 30, 47, 30},
  {37, 31, 37, 31, 17, 17, 46, 44},
  {12, 4, 1, 3, 34, 34, 1, 3},
  {5, 35, 0, 0, 13, 35, 2, 2},
  {32, 32, 1, 3, 6, 14, 1, 3},
  {33, 15, 0, 0, 33, 7, 2, 2},
  {16, 0, 20, 8, 16, 30, 20, 30},
  {1, 31, 9, 31, 17, 17, 21, 21},
  {28, 18, 28, 22, 2, 18, 10, 22},
  {19, 19, 23, 23, 29, 3, 29, 11},
  {9, 11, 12, 4, 9, 11, 26, 26},
  {8, 8, 5, 27, 10, 10, 13, 27},
  {9, 11, 24, 24, 9, 11, 6, 14},
  {8, 8, 25, 15, 10, 10, 25, 7},
  {0, 18, 8, 22, 38, 18, 38, 22},
  {19, 19, 23, 23, 1, 39, 9, 39},
  {16, 36, 20, 36, 16, 2, 20, 10},
  {37, 3, 37, 11, 17, 17, 21, 21},
  {4, 17, 4, 46, 14, 19, 14, 46},
  {18, 16, 47, 47, 5, 15, 5, 15},
  {17, 12, 44, 12, 19, 6, 44, 6},
  {13, 7, 13, 7, 18, 16, 45, 45},
  {4, 42, 4, 21, 14, 42, 14, 23},
  {43, 43, 22, 20, 5, 15, 5, 15},
  {40, 12, 21, 12, 40, 6, 23, 6},
  {13, 7, 13, 7, 41, 41, 22, 20}
};

const unsigned char subpix3[48][8] = {
  {0, 7, 1, 6, 3, 4, 2, 5},
  {7, 4, 6, 5, 0, 3, 1, 2},
  {4, 3, 5, 2, 7, 0, 6, 1},
  {3, 0, 2, 1, 4, 7, 5, 6},
  {1, 0, 6, 7, 2, 3, 5, 4},
  {0, 3, 7, 4, 1, 2, 6, 5},
  {3, 2, 4, 5, 0, 1, 7, 6},
  {2, 1, 5, 6, 3, 0, 4, 7},
  {6, 1, 7, 0, 5, 2, 4, 3},
  {1, 2, 0, 3, 6, 5, 7, 4},
  {2, 5, 3, 4, 1, 6, 0, 7},
  {5, 6, 4, 7, 2, 1, 3, 0},
  {7, 6, 0, 1, 4, 5, 3, 2},
  {6, 5, 1, 2, 7, 4, 0, 3},
  {5, 4, 2, 3, 6, 7, 1, 0},
  {4, 7, 3, 0, 5, 6, 2, 1},
  {6, 7, 5, 4, 1, 0, 2, 3},
  {7, 0, 4, 3, 6, 1, 5, 2},
  {0, 1, 3, 2, 7, 6, 4, 5},
  {1, 6, 2, 5, 0, 7, 3, 4},
  {2, 3, 1, 0, 5, 4, 6, 7},
  {3, 4, 0, 7, 2, 5, 1, 6},
  {4, 5, 7, 6, 3, 2, 0, 1},
  {5, 2, 6, 1, 4, 3, 7, 0},
  {7, 0, 6, 1, 4, 3, 5, 2},
  {0, 3, 1, 2, 7, 4, 6, 5},
  {3, 4, 2, 5, 0, 7, 1, 6},
  {4, 7, 5, 6, 3, 0, 2, 1},
  {6, 7, 1, 0, 5, 4, 2, 3},
  {7, 4, 0, 3, 6, 5, 1, 2},
  {4, 5, 3, 2, 7, 6, 0, 1},
  {5, 6, 2, 1, 4, 7, 3, 0},
  {1, 6, 0, 7, 2, 5, 3, 4},
  {6, 5, 7, 4, 1, 2, 0, 3},
  {5, 2, 4, 3, 6, 1, 7, 0},
  {2, 1, 3, 0, 5, 6, 4, 7},
  {0, 1, 7, 6, 3, 2, 4, 5},
  {1, 2, 6, 5, 0, 3, 7, 4},
  {2, 3, 5, 4, 1, 0, 6, 7},
  {3, 0, 4, 7, 2, 1, 5, 6},
  {1, 0, 2, 3, 6, 7, 5, 4},
  {0, 7, 3, 4, 1, 6, 2, 5},
  {7, 6, 4, 5, 0, 1, 3, 2},
  {6, 1, 5, 2, 7, 0, 4, 3},
  {5, 4, 6, 7, 2, 3, 1, 0},
  {4, 3, 7, 0, 5, 2, 6, 1},
  {3, 2, 0, 1, 4, 5, 7, 6},
  {2, 5, 1, 6, 3, 4, 0, 7}
};

/*! This function computes a Peano-Hilbert key for an integer triplet (x,y,z),
  *  with x,y,z in the range between 0 and 2^bits-1.
  */
npy_int64 peano_hilbert_key(int x, int y, int z)
{
  const int bits = 20;
  int mask;
  unsigned char rotation = 0;
  npy_int64 key = 0;

  for(mask = 1 << (bits - 1); mask > 0; mask >>= 1)
    {
      unsigned char pix = ((x & mask) ? 4 : 0) | ((y & mask) ? 2 : 0) | ((z & mask) ? 1 : 0);

      key <<= 3;
      key |= subpix3[rotation][pix];
      rotation = rottable3[rotation][pix];
    }

  return key;
}
