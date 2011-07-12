#! python
from gaepsi.snapshot import Snapshot
from sys import argv

try:
  snap = Snapshot(argv[2], argv[1])
  snap.check()
  print snap.file.name, 'ok'
except IOError as e:
  print argv[2], e

