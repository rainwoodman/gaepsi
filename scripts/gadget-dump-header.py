#! python

from sys import argv
import argparse
from gaepsi.snapshot import Snapshot

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("-f", dest='format', default='genic', required=False)
  parser.add_argument("filename")
  parser.add_argument("-p", nargs=2, default=[], dest='put', action='append', 
             required=False)
  parser.add_argument("-g", dest='get', default=[], action='append', required=False)
  args = parser.parse_args()
  snap = Snapshot(args.filename, args.format)
  if len(args.put) == 0 and len(args.get) == 0:
    for name in snap.C:
      print name, '=', snap.C[name]
    return

  for field, value in args.put:
      snap.C[field] = value
      print field, 'set to', snap.C[field]
      snap.save_header()
  for field in args.get:
      print snap.C[field]

main()
