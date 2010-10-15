from numpy import dtype
from numpy import ndarray
from numpy import NaN
class _BlockSchema:
  def __init__(self, name, dtype, ptypes):
    self.name = name;
    self.dtype = dtype;
    self.ptypes = ptypes;
    self.conditions = [];

class Schema(list):
  def __init__(self, bl, header, padding=256):
    """
     Schema([
      ('pos', ('f4', 3), [0, 1, 4, 5]),
      ('mass', 'f4', [0, 4, 5])],
      [
      ('Nparticle', ('u4', 6)),
      ('mass', ('f8', 6)),
      ('time', 'f8'),
      ('redshift', 'f8')
      ], padding=256);
      padding: padding of the header.
    """
    list.__init__(self)
    self.__dict = {}

    for x in bl :
      sb = _BlockSchema(x[0], dtype(x[1]), x[2])
      for i in range(3, len(x)) :
        sb.conditions.append(x[i])
      self.append(sb)
      self.__dict[x[0]] = sb
      dt = dtype(header)
      if padding != None:
        size = dt.itemsize
        if size > padding: raise Error("header can't exceed 256 bytes")
        if size == padding: self.header = dt
        else :
          header.append(('unused', ('c', padding - size)))
      self.header = dtype(header)

  def get(self, name) :
    return self.__dict[name]

  def has_key(self, name) :
    return self.__dict.has_key(name)

  def reindex(self, snapshot, name):
    """ reindex data block 'name' in P['all']['name'] into P[ptype]['name']"""
    Nstart = 0
    blockschema = self.__dict[name]
    for ptype in blockschema.ptypes:
      N = snapshot.Nparticle[ptype]
      snapshot.P[ptype][name] = ndarray(N, blockschema.dtype, snapshot.P['all'][name].data,
          blockschema.dtype.itemsize * Nstart)
      Nstart += N

  def open(self, snapshot, fname):
    if hasattr(fname, 'read_record') :
      snapshot.file = fname;
    else:
      raise Error("override this method")

  def update_meta(self, snapshot):
    raise Error("override this method")

  def pad(self, original_size):
    raise Error("override this method")

  def post_init(self, snapshot) :
    pass

  def update_offsets(self, snapshot):
    pos = self.pad(self.header.itemsize);
    for blockschema in self:
      cease_existing = False
      for condition in blockschema.conditions:
        if snapshot.header[condition] == 0 : cease_existing = True

      if cease_existing :
        snapshot.sizes[blockschema.name] = None
        snapshot.offsets[blockschema.name] = None
        continue

      N = 0L;
      for ptype in blockschema.ptypes:
        N = N + snapshot.Nparticle[ptype];

      size = N * blockschema.dtype.itemsize;
      snapshot.sizes[blockschema.name] = size;
      snapshot.offsets[blockschema.name] = pos;

      if size != 0 :
        pos += self.pad(size);

