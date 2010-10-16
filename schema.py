from numpy import dtype
from numpy import ndarray
from numpy import NaN

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
      sb = dict(name = x[0], dtype = dtype(x[1]), ptypes = x[2], conditions=[])
      for i in range(3, len(x)) :
        sb['conditions'].append(x[i])
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

  def open(self, snapshot, fname):
    if hasattr(fname, 'read_record') :
      snapshot.file = fname;
    else:
      raise Error("override this method")

  def update_meta(self, snapshot):
    raise Error("override this method")

  def post_init(self, snapshot) :
    pass

  def update_offsets(self, snapshot):
    blockpos = snapshot.file.get_size(snapshot.header.size));
    for bs in self:
      cease_existing = False
      for cond in bs['conditions']:
        if snapshot.header[cond] == 0 : cease_existing = True

      if cease_existing :
        snapshot.sizes[bs['name']] = None
        snapshot.offsets[bs['name']] = None
        continue

      for ptype in bs.ptypes:
        N += snapshot.Nparticle[ptype]
      blocksize = N * bs.dtype.itemsize

      snapshot.sizes[bs.name] = blocksize
      snapshot.offsets[bs.name] = blockpos
      if size != 0 : 
        blockpos += self.get_size(blocksize);

