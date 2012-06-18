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

def array_split(ary,indices_or_sections,axis = 0):
    """
    Split an array into multiple sub-arrays.

    The only difference from numpy.array_split is we do not apply the
    kludge that 'fixes' the 0 length array dimentions. We try to preserve
    the original shape as much as possible, and only slice along axis

    Please refer to the ``split`` documentation.  The only difference
    between these functions is that ``array_split`` allows
    `indices_or_sections` to be an integer that does *not* equally
    divide the axis.

    See Also
    --------
    split : Split array into multiple sub-arrays of equal size.

    Examples
    --------
    >>> x = np.arange(8.0)
    >>> np.array_split(x, 3)
        [array([ 0.,  1.,  2.]), array([ 3.,  4.,  5.]), array([ 6.,  7.])]

    """
    try:
        Ntotal = ary.shape[axis]
    except AttributeError:
        Ntotal = len(ary)
    try: # handle scalar case.
        Nsections = len(indices_or_sections) + 1
        div_points = [0] + list(indices_or_sections) + [Ntotal]
    except TypeError: #indices_or_sections is a scalar, not an array.
        Nsections = int(indices_or_sections)
        if Nsections <= 0:
            raise ValueError('number sections must be larger than 0.')
        Neach_section,extras = divmod(Ntotal,Nsections)
        section_sizes = [0] + \
                        extras * [Neach_section+1] + \
                        (Nsections-extras) * [Neach_section]
        div_points = numpy.array(section_sizes).cumsum()

    sub_arys = []
    sary = numpy.swapaxes(ary,axis,0)
    for i in range(Nsections):
        st = div_points[i]; end = div_points[i+1]
        sub_arys.append(numpy.swapaxes(sary[st:end],axis,0))

    return sub_arys
