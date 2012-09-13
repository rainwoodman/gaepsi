# cython:cdivision=True
import numpy
cimport numpy
from libc.stdint cimport *
from libc.stdlib cimport qsort, const_void, calloc
from libc.string cimport memcpy
from cpython.ref cimport Py_XDECREF, Py_XINCREF
from cpython.object cimport PyObject, PyTypeObject
from npydtype cimport *
from numpy cimport PyUFunc_FromFuncAndData, PyUFunc_None
cimport npyiter

cdef class fckeyobject:
  cdef fckey_t value
  def __str__(self):
    return hex(<ufckey_t>self.value)
  def __repr__(self):
    return hex(<ufckey_t>self.value)

DEF BITS = 40

cdef extern from 'fillingcurve_internal.c':
  cdef fckey_t _xyz2ind (ipos_t x, ipos_t y, ipos_t z) nogil 
  cdef void _ind2xyz (fckey_t ind, ipos_t* x, ipos_t* y, ipos_t* z) nogil

numpy.import_array()
numpy.import_ufunc()

fckeytype = _register_dtype(fckeyobject)

cdef int f2i(double scale[4], double pos[3], ipos_t point[3]) nogil:
    cdef int d
    cdef double f
    for d in range(3):
      f = (pos[d] - scale[d]) * scale[3]
      if f < 0 or f >= (<ipos_t>1 << BITS) : return 0
      else: point[d] = <ipos_t> f
    return 1

cdef void i2f(double scale[4], ipos_t point[3], double pos[3]) nogil:
    cdef int d
    for d in range(3):
      pos[d] = point[d] / scale[3] + scale[d]

cdef void i2fc(ipos_t ipos[3], fckey_t * key) nogil:
    key[0] = _xyz2ind(ipos[0], ipos[1], ipos[2])

cdef void fc2i(fckey_t * key, ipos_t ipos[3]) nogil:
    _ind2xyz(key[0], &ipos[0], &ipos[1], &ipos[2])

cdef void i2f0(double scale[4], ipos_t point[3], double pos[3]) nogil:
    cdef int d
    for d in range(3):
      pos[d] = point[d] / scale[3]

def decode(numpy.ndarray fckey, out1=None, out2=None, out3=None, scale=None):
  cdef double min[4]
  cdef int out_float = 0
  if scale is None:
    iter = numpy.nditer([fckey, out1, out2, out3],
      op_flags=[['readonly']] + [['writeonly', 'allocate']] * 3,
      op_dtypes =[fckeytype, 'i8', 'i8', 'i8'],
      flags = ['external_loop', 'buffered', 'zerosize_ok'])
  else:
    iter = numpy.nditer([fckey, out1, out2, out3],
      op_flags=[['readonly']] + [['writeonly', 'allocate']] * 3,
      op_dtypes =[fckeytype, 'f8', 'f8', 'f8'],
      flags = ['external_loop', 'buffered', 'zerosize_ok'])
    min[0] = scale[0]
    min[1] = scale[1]
    min[2] = scale[2]
    min[3] = scale[3]
    out_float = 1

  cdef npyiter.CIter citer
  cdef size_t size = npyiter.init(&citer, iter)
  cdef ipos_t ipos[3]
  cdef double fpos[3]
  with nogil:
    while size > 0:
      while size > 0:
        fc2i((<fckey_t*>(citer.data[0])), ipos)
        if not out_float:
          (<ipos_t*>(citer.data[1]))[0] = ipos[0]
          (<ipos_t*>(citer.data[2]))[0] = ipos[1]
          (<ipos_t*>(citer.data[3]))[0] = ipos[2]
        else:
          i2f(min, ipos, fpos)
          (<double*>(citer.data[1]))[0] = fpos[0]
          (<double*>(citer.data[2]))[0] = fpos[1]
          (<double*>(citer.data[3]))[0] = fpos[2]
           
        npyiter.advance(&citer)
        size = size - 1
      size = npyiter.next(&citer)
  return iter.operands[1], iter.operands[2], iter.operands[3]

def encode(numpy.ndarray X, numpy.ndarray Y, numpy.ndarray Z, out=None, scale=None):
  cdef ipos_t MAX = ((<ipos_t>1) << BITS) - 1
  cdef double min[4]
  cdef int in_float = 0
  if scale is None:
    iter = numpy.nditer([X, Y, Z, out],
      op_flags=[['readonly']] * 3 +  [['writeonly', 'allocate']],
      op_dtypes =['i8', 'i8', 'i8', fckeytype],
      flags = ['external_loop', 'buffered', 'zerosize_ok'])
  else:
    iter = numpy.nditer([X, Y, Z, out],
      op_flags=[['readonly']] * 3 +  [['writeonly', 'allocate']],
      op_dtypes =['f8', 'f8', 'f8', fckeytype],
      flags = ['external_loop', 'buffered', 'zerosize_ok'])
    min[0] = scale[0]
    min[1] = scale[1]
    min[2] = scale[2]
    min[3] = scale[3]
    in_float = 1

  cdef npyiter.CIter citer
  cdef size_t size = npyiter.init(&citer, iter)
  cdef ipos_t ipos[3]
  cdef double fpos[3]
  with nogil:
    while size > 0:
      while size > 0:
        if not in_float:
          ipos[0] = (<ipos_t*>(citer.data[0]))[0]
          ipos[1] = (<ipos_t*>(citer.data[1]))[0]
          ipos[2] = (<ipos_t*>(citer.data[2]))[0]
          i2fc(ipos, (<fckey_t*>(citer.data[3])))
        else:
          fpos[0] = (<double*>(citer.data[0]))[0]
          fpos[1] = (<double*>(citer.data[1]))[0]
          fpos[2] = (<double*>(citer.data[2]))[0]
          if 0 != f2i(min, fpos, ipos):
            (<fckey_t*>(citer.data[3]))[0] = <fckey_t> -1
          else:
            i2fc(ipos, <fckey_t*>(citer.data[3]))

        npyiter.advance(&citer)
        size = size - 1
      size = npyiter.next(&citer)
  return iter.operands[3]

cdef numpy.dtype _register_dtype(typeobj):
  cdef PyArray_Descr * descr = <PyArray_Descr*>calloc(1, sizeof(PyArray_Descr))
  cdef PyArray_ArrFuncs * f = <PyArray_ArrFuncs *>calloc(1, sizeof(PyArray_ArrFuncs))

  f.getitem = <void*>_getitem
  f.setitem = <void*>_setitem
  f.copyswapn = <void*>_copyswapn
  f.copyswap = <void*>_copyswap
  f.argsort[0] = <void*> _argsort
  f.argsort[1] = <void*> _argsort
  f.argsort[2] = <void*> _argsort
  f.sort[0] = <void*> _sort
  f.sort[1] = <void*> _sort
  f.sort[2] = <void*> _sort
  f.argmax = <void*> _argmax
  f.argmin = <void*> _argmin
  f.compare = <void*> _compare

  cdef int typenum = register_dtype(descr, f,
     dict(elsize=16, kind='i', byteorder='=', type='z', alignment=8, typeobj=typeobj, metadata={})
  )

  register_safe_castfuncs(typenum, {
           'i8': <intptr_t>_upcasti8, 'i4': <intptr_t>_upcasti4,
           'i2': <intptr_t>_upcasti2, 'i1': <intptr_t>_upcasti1,
           'u8': <intptr_t>_upcastu8, 'u4': <intptr_t>_upcastu4,
           'u2': <intptr_t>_upcastu2, 'u1': <intptr_t>_upcastu1,
           })
  register_ufuncs(typenum, <void*>_op_zzz, [typenum, typenum, typenum], {
     numpy.bitwise_and: '&', numpy.bitwise_or: '|', 
     numpy.bitwise_xor: '^', numpy.add: '+',
     numpy.subtract: '-'   , numpy.multiply: '*',
     numpy.divide: '/', numpy.left_shift: 'l',
     numpy.right_shift: 'r',
      })
  register_ufuncs(typenum, <void*>_op_zzb, [typenum, typenum, numpy.NPY_BOOL], {
     numpy.less: '<', numpy.less_equal: ',',
     numpy.greater: '>', numpy.greater_equal: '.',
     numpy.equal: '=',
  })
  register_ufuncs(typenum, <void*>_op_zz, [typenum, typenum], {
     numpy.invert: '~', numpy.negative: '\\',
  })
  register_ufuncs(typenum, <void*>_op_zz, [typenum, typenum], {
     numpy.invert: '~', numpy.negative: '\\',
  })
  return numpy.PyArray_DescrFromType(typenum)

# Functions that implements the dtype
cdef void _copyswapn(char *dest, intptr_t dstride, char *src, intptr_t sstride, intptr_t n, int swap, void *arr) nogil:
  cdef intptr_t i = 0
  cdef fckey_t tmp
  cdef char *a, *b
  if not swap:
    if sizeof(fckey_t) == dstride and sizeof(fckey_t) == sstride:
      memcpy(dest, src, sizeof(fckey_t) * n)
    else:
      while i < n:
        (<fckey_t*>dest)[0] = (<fckey_t*>src)[0]
        dest = dest + dstride
        src = src + sstride
  else:
    while i < n:
      a = dest
      b = src + sizeof(fckey_t) - 1
      while b >= src:
        a[0] = b[0]
        b = b - 1
        a = a + 1
      dest = dest + dstride
      src = src + sstride

cdef void _copyswap(fckey_t *dest, fckey_t *src, int swap, void * NOT_USED) nogil:
  cdef char t, *a, *b
  cdef int i
  if swap:
    i = 0
    a = <char*> dest
    b = <char*> src + sizeof(fckey_t) - 1
    while i < sizeof(fckey_t):
      a[0] = b[0]
      a = a + 1
      b = b - 1
      i = i + 1
  else:
    dest[0] = src[0]
#    (<char *>NULL)[0] = 100
#    with gil: print 'copyswap', hex(dest[0]), hex(src[0])
    

cdef int _argmax(fckey_t * data, intptr_t n, intptr_t * max_ind, void* NOT_USED) nogil:
   max_ind[0] = 0
   cdef fckey_t max = data[0]
   cdef intptr_t i = 1
   while i < n:
     if data[i] > max:
       max = data[i]
       max_ind[0] = i
     i = i + 1
   return 0
cdef int _argmin(fckey_t * data, intptr_t n, intptr_t * min_ind, void* NOT_USED) nogil:
   min_ind[0] = 0
   cdef fckey_t min = data[0]
   cdef intptr_t i = 1
   while i < n:
     if data[i] < min:
       min = data[i]
       min_ind[0] = i
     i = i + 1
   return 0
cdef int _compare(fckey_t * v, fckey_t * u, void * NOT_USED) nogil:
    return (v[0] > u[0]) - (v[0] < u[0])

cdef int _qsortcmp(fckey_t * v, fckey_t * u) nogil:
    return (v[0] > u[0]) - (v[0] < u[0])

cdef int _sort(fckey_t *v, intptr_t num, void *NOT_USED) nogil:
    qsort(v, num, sizeof(fckey_t), <int (*)(const_void *, const_void *) nogil>_qsortcmp)
    return 0

cdef PyObject *_getitem(fckey_t * data, void * NOT_USED):
    raise RuntimeError('getitem shall never be called')
#    cdef KeyScalarObject ret = _FCKey(data[0])
#    Py_XINCREF(<PyObject*>ret)
    return None

cdef int _setitem(object item, fckey_t * data, void * NOT_USED):
    if isinstance(item, fckeyobject):
      data[0] = item.value
    else:
      data[0] = item
    return 0

cdef int _argsort(fckey_t *v, intptr_t* tosort, intptr_t num, void *NOT_USED) nogil:
    cdef fckey_t vp
    cdef intptr_t *pl, *pr
    cdef intptr_t *stack[128], **sptr=stack, *pm, *pi, *pj, *pk, vi
    cdef intptr_t tmp
    pl = tosort;
    pr = tosort + num - 1;

    while True:
        while pr - pl > 17:
            #/* quicksort partition */
            pm = pl + ((pr - pl) >> 1);
            if v[pm[0]]<v[pl[0]]: tmp = pm[0]; pm[0] = pl[0]; pl[0] = tmp
            if v[pr[0]]<v[pm[0]]: tmp = pr[0]; pr[0] = pm[0]; pm[0] = tmp
            if v[pm[0]]<v[pl[0]]: tmp = pm[0]; pm[0] = pl[0]; pl[0] = tmp
            vp = v[pm[0]]
            pi = pl
            pj = pr - 1
            tmp = pm[0]; pm[0] = pj[0]; pj[0] = tmp
            while True:
                pi = pi + 1
                while v[pi[0]] < vp: pi = pi + 1
                pj = pj - 1
                while vp < v[pj[0]]: pj = pj - 1
                if pi >= pj: break
                tmp = pi[0]; pi[0] = pj[0]; pj[0] = tmp
            pk = pr - 1
            tmp = pi[0]; pi[0] = pk[0]; pk[0] = tmp
            # push largest partition on stack
            if pi - pl < pr - pi:
                sptr[0] = pi + 1;
                sptr[1] = pr;
                sptr = sptr + 2
                pr = pi - 1;
            else:
                sptr[0] = pl;
                sptr[1] = pi - 1;
                sptr = sptr + 2
                pl = pi + 1;

        # insertion sort
        pi = pl + 1
        while pi <= pr:
            vi = pi[0]
            vp = v[vi]
            pj = pi
            pk = pi - 1
            while pj > pl and vp < v[pk[0]]:
                pj[0] = pk[0]
                pj = pj - 1
                pk = pk - 1
            pj[0] = vi;
            pi = pi + 1
        if sptr == stack: break

        sptr = sptr - 1
        pr = sptr[0]
        sptr = sptr - 1
        pl = sptr[0]

    return 0


cdef void _op_zzb(char** args, intptr_t* dimensions, intptr_t* steps, int op) nogil:
  cdef fckey_t * src
  cdef fckey_t * op1
  cdef char * dst
  cdef intptr_t i = 0
  while i < dimensions[0]:
    src = <fckey_t *>args[0]
    op1 = <fckey_t *>args[1]
    dst = <char *>args[2]
    if op == '<':
      dst[0] = src[0] < op1[0]
    elif op == ',':
      dst[0] = src[0] <= op1[0]
    elif op == '>':
      dst[0] = src[0] > op1[0]
    elif op == '.':
      dst[0] = src[0] >= op1[0]
    elif op == '=':
      dst[0] = src[0] == op1[0]

    args[0] = args[0] + steps[0]
    args[1] = args[1] + steps[1]
    args[2] = args[2] + steps[2]
    i = i + 1

cdef void _op_zz(char** args, intptr_t* dimensions, intptr_t* steps, int op) nogil:
  cdef fckey_t * src
  cdef fckey_t * dst
  cdef intptr_t i = 0
  while i < dimensions[0]:
    src = <fckey_t *>args[0]
    dst = <fckey_t *>args[1]
    if op == '\\': 
      dst[0] = - src[0]
    if op == '~': 
      dst[0] = ~ src[0]
    args[0] = args[0] + steps[0]
    args[1] = args[1] + steps[1]
    i = i + 1

cdef void _op_zzz(char** args, intptr_t* dimensions, intptr_t* steps, int op) nogil:
  cdef fckey_t * src
  cdef fckey_t * op1
  cdef fckey_t * dst
  cdef intptr_t i = 0
  while i < dimensions[0]:
    src = <fckey_t *>args[0]
    op1 = <fckey_t *>args[1]
    dst = <fckey_t *>args[2]
    if op == '\\': 
      op1[0] = - src[0]
    if op == '~': 
      op1[0] = ~ src[0]
    elif op == '&': 
      dst[0] = src[0] & op1[0]
    elif op == '|':
      dst[0] = src[0] | op1[0]
    elif op == '^':
      dst[0] = src[0] ^ op1[0]
    elif op == '<':
      dst[0] = src[0] < op1[0]
    elif op == ',':
      dst[0] = src[0] <= op1[0]
    elif op == '>':
      dst[0] = src[0] > op1[0]
    elif op == '.':
      dst[0] = src[0] >= op1[0]
    elif op == '=':
      dst[0] = src[0] == op1[0]
    elif op == 'l':
      dst[0] = src[0] << op1[0]
    elif op == 'r':
      dst[0] = src[0] >> op1[0]
    elif op == '+':
      dst[0] = src[0] + op1[0]
    elif op == '-':
      dst[0] = src[0] - op1[0]
    elif op == '*':
      dst[0] = src[0] * op1[0]
    elif op == '/':
      dst[0] = src[0] / op1[0]

    args[0] = args[0] + steps[0]
    args[1] = args[1] + steps[1]
    args[2] = args[2] + steps[2]
    i = i + 1

cdef void _upcasti8(int64_t *src, fckey_t * dst, intptr_t n, void * NOTUSED, void * NOT_USED2) nogil:
   cdef intptr_t i = 0
   while i < n:
     dst[i] = src[i]
     i = i + 1
cdef void _upcasti4(int32_t *src, fckey_t * dst, intptr_t n, void * NOTUSED, void * NOT_USED2) nogil:
   cdef intptr_t i = 0
   while i < n:
     dst[i] = src[i]
     i = i + 1
cdef void _upcasti2(int16_t *src, fckey_t * dst, intptr_t n, void * NOTUSED, void * NOT_USED2) nogil:
   cdef intptr_t i = 0
   while i < n:
     dst[i] = src[i]
     i = i + 1
cdef void _upcasti1(int8_t *src, fckey_t * dst, intptr_t n, void * NOTUSED, void * NOT_USED2) nogil:
   cdef intptr_t i = 0
   while i < n:
     dst[i] = src[i]
     i = i + 1

cdef void _upcastu8(uint64_t *src, fckey_t * dst, intptr_t n, void * NOTUSED, void * NOT_USED2) nogil:
   cdef intptr_t i = 0
   while i < n:
     dst[i] = src[i]
     i = i + 1
cdef void _upcastu4(uint32_t *src, fckey_t * dst, intptr_t n, void * NOTUSED, void * NOT_USED2) nogil:
   cdef intptr_t i = 0
   while i < n:
     dst[i] = src[i]
     i = i + 1
cdef void _upcastu2(uint16_t *src, fckey_t * dst, intptr_t n, void * NOTUSED, void * NOT_USED2) nogil:
   cdef intptr_t i = 0
   while i < n:
     dst[i] = src[i]
     i = i + 1
cdef void _upcastu1(uint8_t *src, fckey_t * dst, intptr_t n, void * NOTUSED, void * NOT_USED2) nogil:
   cdef intptr_t i = 0
   while i < n:
     dst[i] = src[i]
     i = i + 1
