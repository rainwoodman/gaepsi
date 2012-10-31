#! python

from sys import argv
import argparse
from gaepsi.snapshot import Snapshot

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("-f", dest='format', default='genic', required=False)
  parser.add_argument("filename")
  parser.add_argument("field", default=None, nargs='?')
  parser.add_argument("-s", dest='value', default=None, required=False)
  args = parser.parse_args()
  snap = Snapshot(args.filename, args.format)
  if args.field is not None:
    if args.value is not None:
      snap.C[args.field] = args.value
      print args.field, 'set to', snap.C[args.field]
      snap.save_header()
    else:
      print snap.C[args.field]
  else:
    for name in snap.C:
      print name, '=', snap.C[name]

main()
