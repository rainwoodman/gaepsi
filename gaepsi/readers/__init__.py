import numpy
import warnings
from gaepsi.tools import virtarray
from types import MethodType
from gaepsi.io import CFile, F77File

def is_string_like(v):
  try: v + ''
  except: return False
  return True

def get_reader(reader, *args, **kwargs):
  return Reader(reader, *args, **kwargs)

def Reader(reader, **kwargs):
  """ first assume reader is a module name and import Reader class from it.
      if failed assume reader is a qualified classname and import 
      the module and class.
  """
  if is_string_like(reader) :
    if reader == "hdf5":
      return ReaderObjHDF5()
    try:
      module = __import__('gaepsi.readers.%s' % reader, globals(), {}, [''], 0)
      modulename = reader
      classname = 'Snapshot'
    except Exception as e:
      modulename, classname = reader.rsplit('.', 1)
      module = __import__('gaepsi.readers.%s' % modulename, globals(), {}, [classname], 0)
      
    if not hasattr(module, classname):
      raise ImportError('Reader class %s not found in %s', classname, modulename)
    reader = getattr(module, classname)

  if not isinstance(reader, ReaderObj):
    if hasattr(reader, '__call__'):
      reader = reader(**kwargs)
    reader = ReaderObj(reader)
  # will be picked up when creating the constant
    reader.extra_kwargs = kwargs
  return reader

class ConstBase:
  """
     a constant can be at the following locations:
     the header, (saved in the snapshot file)
     the class,  (may be converted to something in the header)
     or extra.   (lost)
  """
  def __init__(self, header, init=False, extra={}):
    """ if init is True, the values are initialized """
    self._header = header
    # extra is the extra values
    self._extra = extra.copy()
    if init:
      for item in self._header.dtype.names:
        if hasattr(self, item):
          attr = getattr(self, item)
          if isinstance(attr, (tuple, basestring)): continue
          self._header[item] = attr
      
  def __iter__(self):
    def generator():
      for name in sorted(numpy.unique(dir(self) \
         + list(self._header.dtype.names) \
         + list(self._extra.keys()))):
        if name[0] == '_': continue
        yield name
    return generator()
    
  def __contains__(self, item):
    return item in self._header.dtype.names or hasattr(self, item) or item in self._extra

  def __getitem__(self, item):
    if item in self._extra:
      return self._extra[item]
    if item in self._header.dtype.names:
      return self._header[item]
    elif hasattr(self, item):
      attr = getattr(self, item)
      if isinstance(attr, basestring):
        return self._header[attr]
      elif isinstance(attr, tuple) or isinstance(attr, list):
        dtype, get, set = attr
        return virtarray(None, dtype, MethodType(get, self, None), MethodType(set, self, None))
      else:
        return attr

  def __setitem__(self, item, value):
    if item is Ellipsis:
      assert isinstance(value, ConstBase)
      self._header[...] = value._header
      self._extra = value._extra.copy()
    elif item in self._header.dtype.names:
      self._header[item] = value
    elif hasattr(self, item):
      attr = getattr(self, item)
      if isinstance(attr, basestring):
        self._header[attr] = value
      elif isinstance(attr, tuple) or isinstance(attr, list):
        dtype, get, set = attr
        virtarray(None, dtype, MethodType(get, self, None), MethodType(set, self, None))[...] = value
      else:
        raise IndexError("can't set %s" % item)
    else:
      self._extra[item] = value

  def __str__(self):
    s = []
    for name in self:
      if name[0] == '_': continue
      if name in self._header.dtype.names:
        s += ['%s(header) = %s' % (name, str(self[name]))]
      elif name in self._extra:
        s += ['%s(extra) = %s' % (name, str(self[name]))]
      else:
        s += ['%s(class) = %s' % (name, str(self[name]))]
    return '\n'.join(s)

class ReaderObj(object):
  filedict = {'F': F77File, 'C': CFile }
  properties = {
      'format': ('F', lambda x: x),
      'header': (None, numpy.dtype),
      'schema': (None, lambda x: x),
      'constants': ({}, lambda x: type('constants', (x, ConstBase, object), {})),
      'endian': ('<', lambda x: x),
    }

  def __init__(cls, config):
    missing = []

    for p in ReaderObj.properties:
      desc = ReaderObj.properties[p]
      value, constructor = desc
      if hasattr(config, p):
        cls.__dict__[p] = constructor(getattr(config, p))
      else:
        if value is not None:
          cls.__dict__[p] = constructor(value)
        else:
          missing.append(p)

    if missing:
      raise ValueError("missing class properties in %s: %s" %
       (config.__name__, ', '.join(missing)))

    cls.file_class = cls.filedict[cls.format]

  def __str__(cls):
    return str(cls.__dict__)
""" unused

  def write_header(cls, snapshot):
    file = cls.file_class(snapshot.file, endian=cls.endian, mode='r+')
    file.seek(0)
    file.write_record(snapshot.header)
    
  def create_structure(cls, snapshot):
    file = cls.file_class(snapshot.file, endian=cls.endian, mode='r+')
    cls.update_offsets(snapshot)
    for s in cls:
        if not snapshot.sizes[s.name] == None:
          file.seek(snapshot.offsets[s.name])
          file.create_record(s.dtype.base, snapshot.sizes[s.name] // s.dtype.base.itemsize)
    file.seek(0)
    file.write_record(snapshot.header)

   """ 

class ReaderObjHDF5(object):
  def __init__(cls):
    pass
  def __str__(cls):
    return "<HDF5Reader>"
