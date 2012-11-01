#! /usr/bin/env python
import argparse
import numpy
import sharedmem
from gaepsi.snapshot import Snapshot
from gaepsi.meshindex import MeshIndex
def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("filename")
  parser.add_argument("-f", "--format", dest='format', required=True)
  parser.add_argument("-p", "--ptype", dest='ptypes', 
              action='append', required=True, type=int)
  parser.add_argument("-o", "--output", dest='output', 
              required=True)
  parser.add_argument("--maxnpar", dest='maxnpar', default=20*1024*1024,
              required=False, type=int)
  parser.add_argument("-F", "--filter", dest='filter', 
              required=False, type=lambda x: x.split(':'))
  group = parser.add_argument_group()
  group.add_argument("-m", "--meshindex", dest='meshindex', 
              required=False, type=MeshIndex.fromfile)
  group.add_argument("--origin", nargs='+', dest='origin', 
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
    return result

  snap0 = open(args.filename, 0)

  if '%d' in args.filename:
    Nfile = snap0.C['Nfiles']
  else:
    Nfile = 1

  fids = range(Nfile)

  if args.origin is not None:
    if args.boxsize is None:
      parser.print_help()
      parser.exit()
    else:
      origin = numpy.empty(3, dtype='f8')
      boxsize = numpy.empty(3, dtype='f8')
      origin[:] = args.origin
      boxsize[:] = args.boxsize
      if args.meshindex is not None:
        fids = args.meshindex.cut(origin, boxsize)
  
  defaultheader = snap0.header
  Ntot = snap0.C['Ntot']
  Ntot_out = numpy.zeros(snap0.reader.schema.nptypes, dtype='i8')

  with sharedmem.Pool(use_threads=True) as pool:
    def work(fid):
      snap = open(args.filename, fid)
      N_out = numpy.zeros(snap0.reader.schema.nptypes, dtype='i8')
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

  print Nfile_out, args.output

  outputs = [open(args.output, fid, create=True) for fid in range(Nfile_out)]
  written = [numpy.zeros_like(output.C['N']) for output in outputs]
  cursor = numpy.zeros(snap0.reader.schema.nptypes, dtype='i8')

  for i, output in enumerate(outputs):
    output.header[...] = defaultheader
    output.C['Ntot'] = Ntot_out
    output.C['N'] = (Ntot_out * (i + 1) // Nfile_out) \
                 -  (Ntot_out * i // Nfile_out)
    output.C['Nfile'] = Nfile_out
    output.create_structure()

  with sharedmem.Pool(use_threads=True) as pool:
    def work(fid):
      snap = open(args.filename, fid)
      finished_outputs = []

      for ptype in args.ptypes:
        mask, count = filter(snap, ptype, origin, boxsize)
        for block in snap.reader.schema:
          # prefetch in parallel
          if (ptype, block) in snap:
            snap[ptype, block]
   
        with pool.lock:
          istart = 0
          oldcursor = cursor[ptype]
          while True:
            if count == 0: break
            i = cursor[ptype]
            free = (outputs[i].C['N'] - written[i])[ptype]
            ostart = written[i][ptype]
            if free > count:
              for block in snap.reader.schema:
                if (ptype, block) in snap:
                  towrite = select(snap, ptype, block, mask)
                  outputs[i][ptype, block][ostart:ostart+count] = towrite
              written[i][ptype] += count
              count = 0
            else:
              for block in snap.reader.schema:
                if (ptype, block) in snap:
                  towrite = select(snap, ptype, block, mask)
                  outputs[i][ptype, block][ostart:ostart+free] = towrite[istart:free]
              written[i][ptype] += free
              istart += free
              count -= free
              cursor[ptype] = cursor[ptype] + 1
          for i in range(oldcursor, cursor[ptype]):
            print i, written[i], outputs[i].C['N']
            if (written[i] == outputs[i].C['N']).all():
              finished_outputs.append(i)
        for i in finished_outputs:
          outputs[i].save_all()
          print outputs[i][ptype, 'pos']
      return

    pool.map(work, fids)

  print Nfile_out, Ntot, Ntot_out

if __name__ == '__main__':
  main()
