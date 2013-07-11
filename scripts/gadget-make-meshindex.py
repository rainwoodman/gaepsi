# deprecated
#! /usr/bin/env python
raise Depcerated
import sharedmem
from sys import argv
import argparse
import numpy
from gaepsi.compiledbase import fillingcurve
from gaepsi.snapshot import Snapshot
from gaepsi.meshindex import MeshIndex
from gaepsi.tools import packarray

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("-f", dest='format', default='genic', required=False)
  parser.add_argument("filename")
  # n is bits per chanel
  parser.add_argument("-n", dest='bitsperaxis', default=6, required=False, type=int)
  parser.add_argument("-o", dest='output', default=None, required=True)
  args = parser.parse_args()
  
  bits = args.bitsperaxis * 3

  if '%d' in args.filename:
    snap0 = Snapshot(args.filename % 0, args.format)
    Nfile = snap0.C['Nfiles']
  else:
    snap0 = Snapshot(args.filename, args.format)
    Nfile = 1

  boxsize = snap0.C['boxsize']
  C = []
  F = []
  with sharedmem.Pool(use_threads=True) as pool:
    def work(fid):
      if '%d' in args.filename:
        snap = Snapshot(args.filename % fid, args.format)
      else:
        snap = Snapshot(args.filename, args.format)
      N = snap.C['N'].sum()
      x,y,z = snap[None, 'pos'].T
      del snap
      scale = fillingcurve.scale(0, boxsize)
      zkey = fillingcurve.encode(x, y, z, scale=scale)
      del x, y, z
      dig = numpy.uint32(zkey >> (fillingcurve.bits * 3 - bits))
  
      bincount = numpy.bincount(dig)
      
      return (fid, bincount.nonzero()[0])

    def reduce(res):
      fid, active = res
      F.append(fid)
      C.append(list(active))
      print fid, len(active)
    pool.map(work, range(Nfile), callback=reduce)

  F = numpy.array(F, dtype='u4')
  l = [len(a) for a in C]
  C = numpy.concatenate(C)
  fid = numpy.repeat(F, l)
  arg = C.argsort()
  fid = fid[arg]
  count = numpy.bincount(C, minlength=1<<bits)
  
  index = packarray(fid, count)
  map = MeshIndex(0, boxsize, args.bitsperaxis, index)
  map.tofile(args.output)

if __name__ == '__main__':
  main()
