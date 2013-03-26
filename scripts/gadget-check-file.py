#! python
from gaepsi.snapshot import Snapshot
from sys import argv

for filename in argv[2:]:
  try:
    snap = Snapshot(filename, argv[1])
    snap.check()
    print snap.file, 'ok'
  except IOError as e:
    print filename, e

