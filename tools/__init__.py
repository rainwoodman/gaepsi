
def bindmethods(obj, locals, before=None, after=None):
  """ bind methods of obj to locals namespace 
      methods starting with _ are ignored
      # before can update kwargs before the method is called
      def before(args, kwargs): pass
      def after(args, kwargs): pass
  """
  from types import MethodType
  for x in dir(obj):
    m = getattr(obj, x)
    if not x.startswith('_') and isinstance(m, MethodType):
      locals[x] = _wrapfunc(m, before=before, after=after)

def _wrapfunc(m, before=None, after=None):
    """ wraps a bound method, creates a unbound method
        that calls m and calls before / after respectively
        'before' and 'after' are called with a dictionary of
        the actually args.
        'before' can update the args
    """
    import inspect
    from types import MethodType
    args, varargs, varkw, defaults = inspect.getargspec(m)

    D = {}
   
    if defaults is not None:
      D = dict(zip(reversed(args),reversed(defaults)))

    P = list(args[1:]) # remove self
    A = [ '%s=%s' % (a, repr(D[a])) if a in D else a for a in P]

    if varargs is not None:
      A += ['*%s' % varargs]
      P += ['*%s' % varargs]
      
    if varkw is not None:
      A += ['**%s' % varkw]
      P += ['**%s' % varkw]

    def callargs(m, *args, **kwargs):
      d = inspect.getcallargs(m, *args, **kwargs)
      del d['self']
      if varargs: 
        args = d[varargs]
        del d[varargs]
      else:
        args = []
      if varkw: 
        kwargs = d[varkw]
        del d[varkw]
      else:
        kwargs = {}
      d.update(kwargs)
      return  args, d
    L = {'m':m, 'after':after, 'before': before, 'callargs': callargs}

    code = """def %(name)s(%(args)s):
         varargs, kwargs = callargs(m, %(params)s)
         if before: before(varargs, kwargs)
         ret = m(*varargs, **kwargs)
         if after: after(varargs, kwargs)
         return ret
    """ % {'name':m.__name__, 
           'args':','.join(A), 
           'params':','.join(P)}
    eval(compile(code, '', 'exec', 0, True), L)
    func = L[m.__name__]
    func.__doc__ = m.__doc__
    func.source = code
    del L[m.__name__]
    return func

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

def savetxt(fname, X, fmt='%.18e', delimiter=' ', newline='\n', preamble=None, comment_character='#', header=None):
    """
    Save an array to a text file.

    Parameters
    ----------
    fname : filename or file handle
        If the filename ends in ``.gz``, the file is automatically saved in
        compressed gzip format.  `loadtxt` understands gzipped files
        transparently.
    X : array_like
        Data to be saved to a text file.
    fmt : str or sequence of strs
        A single format (%10.5f), a sequence of formats, or a
        multi-format string, e.g. 'Iteration %d -- %10.5f', in which
        case `delimiter` is ignored.
    delimiter : str
        Character separating columns.
    newline : str
        .. versionadded:: 1.5.0

        Character separating lines.
    preamble : str or sequence of strs, optional
        If specified, the content of the strings will be added at the top
        of the file. Each line will be preceded by the `comment_character` string
        and terminated by `newline`. In default configuration, numpy.loadtxt
        recognizes the preamble as a comment and, thus, ignores it.
    comment_character : str, optional
        The string which is intended to introduce the lines of the `preamble`
        (default is '#'). Please note that numpy.loadtxt uses the `comments`
        keyword to refer to this string.
    header : bool or sequence of strs, optional
        The column names; will be added after the `preamble` (if specified),
        but before the data. If header is 'True', the names will either
        be inferred from `X`.dtype.names or default names (f1, f2..fn)
        will be used. If a sequence of strs is given, these
        will be used as names.


    See Also
    --------
    save : Save an array to a binary file in NumPy ``.npy`` format
    savez : Save several arrays into a ``.npz`` compressed archive

    Notes
    -----
    Further explanation of the `fmt` parameter
    (``%[flag]width[.precision]specifier``):

    flags:
        ``-`` : left justify

        ``+`` : Forces to preceed result with + or -.

        ``0`` : Left pad the number with zeros instead of space (see width).

    width:
        Minimum number of characters to be printed. The value is not truncated
        if it has more characters.

    precision:
        - For integer specifiers (eg. ``d,i,o,x``), the minimum number of
          digits.
        - For ``e, E`` and ``f`` specifiers, the number of digits to print
          after the decimal point.
        - For ``g`` and ``G``, the maximum number of significant digits.
        - For ``s``, the maximum number of characters.

    specifiers:
        ``c`` : character

        ``d`` or ``i`` : signed decimal integer

        ``e`` or ``E`` : scientific notation with ``e`` or ``E``.

        ``f`` : decimal floating point

        ``g,G`` : use the shorter of ``e,E`` or ``f``

        ``o`` : signed octal

        ``s`` : string of characters

        ``u`` : unsigned decimal integer

        ``x,X`` : unsigned hexadecimal integer

    This explanation of ``fmt`` is not complete, for an exhaustive
    specification see [1]_.
    
    Explanation of `preamble` and `header`:
        Both preamble and header are used to add information on the
        data at the top of the file.
        In default configuration, the
        preamble preserves compatibility of the
        output file with numpy.loadtxt, while the header not necessarily
        does.
        The header is a single line containing, for example, the name of each
        column. While this is very desirable, it may become a problem when
        the data is recovered with numpy.loadtxt, which
        (in default configuration) does not recognize the header and probably
        fails, because it cannot convert the header entries to the target type.
        Additionally, a preamble can be used to put an introductory text
        at the top of the file.

    References
    ----------
    .. [1] `Format Specification Mini-Language
           <http://docs.python.org/library/string.html#
           format-specification-mini-language>`_, Python Documentation.

    Examples
    --------
    >>> x = y = z = np.arange(0.0,5.0,1.0)
    >>> np.savetxt('test.out', x, delimiter=',')   # X is an array
    >>> np.savetxt('test.out', (x,y,z))   # x,y,z equal sized 1D arrays
    >>> np.savetxt('test.out', x, fmt='%1.4e')   # use exponential notation
    >>> recar = np.zeros(3, dtype={'names':['col1', 'col2'], 'formats':['i4','f4']})
    >>> np.savetxt('test.out', recar, header=True)
    >>> np.savetxt('test.out', recar, preamble=['This is the most','important result.'])
    >>> np.savetxt('test.out', recar, header=['Apples', 'Oranges'])
    """
    import sys
    import numpy as np
    from numpy.compat import asbytes, asstr, asbytes_nested, bytes
    def _is_string_like(v):
        try: v + ''
        except: return False
        return True

    # Py3 conversions first
    if isinstance(fmt, bytes):
        fmt = asstr(fmt)
    delimiter = asstr(delimiter)

    own_fh = False
    if _is_string_like(fname):
        own_fh = True
        if fname.endswith('.gz'):
            import gzip
            fh = gzip.open(fname, 'wb')
        else:
            if sys.version_info[0] >= 3:
                fh = open(fname, 'wb')
            else:
                fh = open(fname, 'w')
    elif hasattr(fname, 'seek'):
        fh = fname
    else:
        raise ValueError('fname must be a string or file handle')

    try:
        X = np.asarray(X)

        # Handle 1-dimensional arrays
        if X.ndim == 1:
            # Common case -- 1d array of numbers
            if X.dtype.names is None:
                X = np.atleast_2d(X).T
                ncol = 1

            # Complex dtype -- each field indicates a separate column
            else:
                ncol = 0
                for desc in X.dtype.descr:
                    if len(desc) == 3: ncol = ncol + desc[2]
                    else : ncol = ncol + 1
        else:
            ncol = X.shape[1]

        # `fmt` can be a string with multiple insertion points or a
        # list of formats.  E.g. '%10.5f\t%10d' or ('%10.5f', '$10d')
        if type(fmt) in (list, tuple):
            if len(fmt) != ncol:
                raise AttributeError('fmt has wrong shape.  %s' % str(fmt))
            format = asstr(delimiter).join(map(asstr, fmt))
        elif type(fmt) is str:
            if fmt.count('%') == 1:
                fmt = [fmt, ]*ncol
                format = delimiter.join(fmt)
            elif fmt.count('%') != ncol:
                raise AttributeError('fmt has wrong number of %% formats.  %s'
                                     % fmt)
            else:
                format = fmt

        if preamble is not None:
            # A comment object was specified.
            # Check whether the comment_character is valid.
            if not _is_string_like(comment_character):
                raise ValueError("'comment_character' has to be a string.")
            # If the comment is a plane string, make it an element of a list
            # to be iterated over below.
            if _is_string_like(preamble):
                preamble = [preamble]
            # Iterate over comment argument if possible.
            if hasattr(preamble, '__iter__'):
                for co in preamble:
                    # Check whether individual elements are string-like
                    if _is_string_like(co):
                        # Remove trailing newline character and split string
                        # at occurrence of newline character to avoid new
                        # preamble lines without introducing comment character.
                        co = co.rstrip(newline)
                        cosplit = co.split(newline)
                        for c in cosplit:
                            fh.write(asbytes(''.join((comment_character,c,newline))))
   
        if header is not None:
          # Check whether column names are given as strings, or names
          # must be infered from dtype.
          if isinstance(header, bool):
              if header:
                  # If possible, infer column names from dtype and use default
                  # (f1, f2 .. fn) otherwise.
                  if X.dtype.names is not None:
                    column_names = X.dtype.names
                  else:
                    column_names = []
                    for i in range(ncol):
                      column_names.append(''.join(('f',str(i+1))))
          # Check whether column names are given as a list/tuple and check number.
          elif isinstance(header, list) or isinstance(header, tuple):
            if len(header) != ncol:
              raise ValueError(''.join(("Table has %i column(s) but header has %i entrie(s)" % (ncol,len(header)))))
            column_names = header
          # Write column names (seperated by delimiter).
          headerline = ''
          for colname in column_names:
              headerline = ''.join((headerline, colname, delimiter))
          headerline = headerline.rstrip(delimiter)
          fh.write(asbytes(''.join((headerline,newline))))
 

        for row in X:
            fh.write(asbytes(format % tuple(row) + newline))
    finally:
        if own_fh:
            fh.close()
