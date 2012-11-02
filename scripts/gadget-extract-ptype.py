#! /usr/bin/env python
import argparse
import numpy
import sharedmem
from gaepsi.snapshot import Snapshot
from gaepsi.meshindex import MeshIndex

#sharedmem.set_debug(True)
def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("filename")
  parser.add_argument("-f", "--format", dest='format', required=True)
  parser.add_argument("-p", "--ptype", dest='ptypes', 
              action='append', required=False, type=int)
  parser.add_argument("-o", "--output", dest='output', 
              required=True)
  parser.add_argument("--maxnpar", dest='maxnpar', default=20*1024*1024,
              required=False, type=int)
  parser.add_argument("-F", "--filter", dest='filter', 
              required=False, type=lambda x: x.split(':'))
  group = parser.add_argument_group()
  group.add_argument("-m", "--meshindex", dest='meshindex', 
              required=False, type=MeshIndex.fromfile)
  x = group.add_mutually_exclusive_group()
  x.add_argument("--origin", nargs='+', dest='origin', 
              type=float, required=False)
  x.add_argument("--center", nargs='+', dest='center', 
              type=float, required=False)
  group.add_argument("--boxsize", nargs='+', dest='boxsize', 
              type=float, required=False)
  args = parser.parse_args()

  origin = None
  boxsize = None

  def open(template, fid, create=False):
    if '%d' in template:
      snap = Snapshot(template % fid, args.format, create=create)
    else:
      snap = Snapshot(template, args.format, create=create)
    return snap

  def filter(snap, ptype, origin, boxsize):
    if snap.C['N'][ptype] == 0: return None, 0
    if origin is None or boxsize is None:
      return None, snap.C['N'][ptype]
    tail = origin + boxsize
    pos = snap[ptype, 'pos']
    iter = numpy.nditer([pos[:, 0], pos[:, 1], pos[:, 2], None],
          op_dtypes=[None, None, None, '?'],
          op_flags=[['readonly']] * 3 + [['writeonly', 'allocate']],
          flags=['external_loop', 'buffered'])
    for x, y, z, m in iter:
      m[...] = \
        (x >= origin[0]) & (y >= origin[1]) & (z >= origin[2]) \
      & (x <= tail  [0]) & (y <= tail  [1]) & (z <= tail  [2])
      
    return iter.operands[3], iter.operands[3].sum()
    
  def select(snap, ptype, block, mask):
    if mask is None:
      result = snap[ptype, block]
    else:
      result = snap[ptype, block][mask]
    del snap[ptype, block]
    return result

  snap0 = open(args.filename, 0)
  nptypes = snap0.reader.schema.nptypes

  if args.ptypes is None:
    args.ptypes = range(nptypes)

  if '%d' in args.filename:
    Nfile = snap0.C['Nfiles']
  else:
    Nfile = 1

  fids = range(Nfile)

  if args.boxsize is not None:
    boxsize = numpy.empty(3, dtype='f8')
    boxsize[:] = args.boxsize
    if args.origin is None and args.center is None:
      parser.print_help()
      parser.exit()
    else:
      origin = numpy.empty(3, dtype='f8')
      if args.origin is not None:
        origin[:] = args.origin
      else:
        origin[:] = args.center
        origin[:] = origin - boxsize * 0.5

      if args.meshindex is not None:
        fids = args.meshindex.cut(origin, boxsize)
      args.meshindex = None

  defaultheader = snap0.header
  Ntot = snap0.C['Ntot']
  Ntot_out = numpy.zeros(nptypes, dtype='i8')

  with sharedmem.Pool(use_threads=True) as pool:
    def work(fid):
      snap = open(args.filename, fid)
      N_out = numpy.zeros(nptypes, dtype='i8')
      for ptype in args.ptypes:
        mask, count = filter(snap, ptype, origin, boxsize)
        N_out[ptype] = count
      return N_out
    def reduce(N_out):
      Ntot_out[...] = N_out + Ntot_out
    pool.map(work, fids, callback=reduce)

  Nfile_out = (Ntot_out.sum() // args.maxnpar + 1)


  if Nfile_out > 1 and '%d' not in args.output:
    args.output += '.%d'

  outputs = [open(args.output, fid, create=True) for fid in range(Nfile_out)]
  written = numpy.zeros((Nfile_out, nptypes), dtype='i8')
  free = numpy.empty_like(written)
  writing = numpy.zeros((Nfile_out, nptypes), dtype='i8')

  cursor = numpy.zeros(nptypes, dtype='i8')

  for i, output in enumerate(outputs):
    output.header[...] = defaultheader
    output.C['Ntot'] = Ntot_out
    output.C['N'] = (Ntot_out * (i + 1) // Nfile_out) \
                 -  (Ntot_out * i // Nfile_out)
    free[i] = output.C['N']
    output.C['Nfiles'] = Nfile_out
    output.create_structure()

  with sharedmem.Pool(use_threads=True) as pool:
    def work(fid):
      snap = open(args.filename, fid)

      for ptype in args.ptypes:
        mask, count = filter(snap, ptype, origin, boxsize)
        if count == 0: continue
        with pool.lock:
          cumfree = free[:, ptype].cumsum()
          last_output = cumfree.searchsorted(count, side='left')
          first_output = cumfree.searchsorted(0, side='right')
          table = numpy.zeros(last_output - first_output + 1, ('i8', 4))
          outputid, istart, len, ostart = table.T
          outputid[...] = range(first_output, last_output+1)
          ostart[...] = written[first_output:last_output+1, ptype]
          len[:-1] = free[first_output:last_output, ptype]
          len[-1] = count - len[:-1].sum()
          istart[1:] = len.cumsum()[:-1]

          writing[first_output:last_output+1, ptype] += len
          written[first_output:last_output+1, ptype] += len
          free[first_output:last_output+1, ptype] -= len

          for i in range(first_output, last_output + 1):
            for block in snap.reader.schema:
              outputs[i].alloc(block, ptype)

        for block in snap.reader.schema:
          if (ptype, block) not in snap: continue
          towrite = select(snap, ptype, block, mask)
          for id, i, l, o in table:
            outputs[id][ptype, block][o:o+l] = towrite[i:i+l]

      with pool.lock:
        writing[first_output:last_output+1, ptype] -= len

        for i, output in enumerate(outputs):
          if (free[i] == 0).all() and (writing[i] == 0).all() and outputs[i] is not None:
            print output[0, 'mass']
            output.save_all()
            outputs[i] = None
            output = None
            print 'saving ', i
      return
    pool.map(work, fids)

if __name__ == '__main__':
  main()
