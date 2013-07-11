#! python
from gaepsi.snapshot import Snapshot
from sys import argv

template = Snapshot(argv[2], argv[1])
for filename in argv[2:]:
  try:
    snap = Snapshot(filename, argv[1], template=template)
    snap.check()
    print snap.file, 'ok'
  except IOError as e:
    print filename, e

