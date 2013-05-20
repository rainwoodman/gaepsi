#! python
import sharedmem
from sys import argv
import argparse
import numpy
from gaepsi.snapshot import Snapshot
from gaepsi.meshindex import MeshIndex
#sharedmem.set_debug(True)
def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("-f", dest='format', required=True)
  parser.add_argument("filename")
  parser.add_argument("-N", dest='N', default=64, required=False, type=int)
  parser.add_argument("-o", dest='output', default=None, required=True)
  args = parser.parse_args()
  
  if '%d' in args.filename:
    snap0 = Snapshot(args.filename % 0, args.format)
    Nfile = snap0.C['Nfiles']
  else:
    snap0 = Snapshot(args.filename, args.format)
    Nfile = 1

  boxsize = snap0.C['boxsize']
  m = MeshIndex(N=args.N, Nd=3, boxsize=boxsize)

  with sharedmem.Pool(use_threads=False) as pool:
    def work(fid):
      if '%d' in args.filename:
        snap = Snapshot(args.filename % fid, args.format)
      else:
        snap = Snapshot(args.filename, args.format)
      print 'start', snap.file
      pos = snap['pos']
      print 'read', snap.file
      return m.set(fid, pos) 
    l = pool.map(work, range(Nfile))

  m.compile(dict(l))
  m.tofile(args.output)

if __name__ == '__main__':
  main()
