import numpy
def join_recarrays(*args):
  if len(args) == 1 and isinstance(args[0] , list):
    args = args[0]

  descr = []

  for a in args:
    if isinstance(a, tuple):
      descr += [(a[0], a[1].dtype.str)]
    else:
      descr += a.dtype.descr

  newrec = numpy.empty(args[0].shape, dtype=dtype(descr))
  
  for a in args:
    if isinstance(a, tuple):
      newrec[a[0]] = a[1]
    else:
      for field in a.dtype.fields:
        newrec[field] = a[field]

  return newrec

def bincount2d(x, y, weights=None, shape=None):
  if shape is None:
    shape = (numpy.max(x)+1, numpy.max(y)+1)
  ind = numpy.ravel_multi_index((x,y), shape, mode='clip')
  out = numpy.bincount(ind, weights, minlength=shape[0] * shape[1])
  return out.reshape(*shape)
