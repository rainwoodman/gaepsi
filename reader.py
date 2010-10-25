from numpy import dtype
from numpy import array
from numpy import NaN
from snapshot import Snapshot
class Reader:
  def __init__(self, file_class, header, schemas):
    self.header = dtype(header)
    self.schemas = [dict(name = sch[0], 
                    dtype=dtype(sch[1]), 
                    ptypes=sch[2], 
                    conditions=sch[3]) for sch in schemas]
    self.file_class = file_class
    self.hash = {}
    for s in self.schemas:
      self.hash[s['name']] = s

  def open(self, file):
    snapshot = Snapshot()
    self.prepare(snapshot, file)
    return snapshot

  def prepare(self, snapshot, file):
    snapshot.file = self.file_class(file)
    snapshot.reader = self
    snapshot.header = snapshot.file.read_record(self.header, 1)[0]
    self.setup_constants(snapshot)
    self.update_offsets(snapshot)

  def setup_constants(self, snapshot):
    snapshot.C['G'] = 43007.1
    snapshot.C['H'] = 0.1

    for c,v in self.constants(snapshot).items():
      snapshot.C[c] = v
    snapshot.N = snapshot.C['N']

  def constants(self, snapshot):
    return dict(
      N = snapshot.header['N'])

  def update_offsets(self, snapshot):
    blockpos = self.file_class.get_size(self.header.itemsize);
    for s in self.schemas:
      name = s['name']
      cease_existing = False
      for cond in s['conditions']:
        if snapshot.header[cond] == 0 : cease_existing = True

      if cease_existing :
        snapshot.sizes[name] = None
        snapshot.offsets[name] = None
        continue
      N = 0
      for ptype in s['ptypes']:
        N += snapshot.N[ptype]
      blocksize = N * s['dtype'].itemsize

      snapshot.sizes[name] = blocksize
      snapshot.offsets[name] = blockpos
      if blocksize != 0 : 
        blockpos += self.file_class.get_size(blocksize);


  def load(self, snapshot, name, ptype=None):
    if ptype == None: ptype = 'all'

    if snapshot[ptype].has_key(name) : return

    sch = self.hash[name]
    snapshot.file.seek(snapshot.offsets[name])
    length = snapshot.sizes[name] // sch['dtype'].itemsize
    if ptype == 'all':
      if snapshot.sizes[name] != 0 :
        snapshot.P['all'][name] = snapshot.file.read_record(sch['dtype'], length)
      else :
        snapshot.P['all'][name] = None
    else :
      if not ptype in sch['ptypes'] : 
        snapshot.P[ptype][name] = None
        return
      offset = 0
      for i in range(6):
        if i in sch['ptypes'] and i < ptype :
          offset += snapshot.N[i]
      snapshot.P[ptype][name] = snapshot.file.read_record(sch['dtype'], length, offset, snapshot.N[ptype])

