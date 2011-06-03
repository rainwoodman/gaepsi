#! python

from sys import argv

from gaepsi.snapshot import Snapshot

snap = Snapshot(argv[1], argv[2])
for name in snap.header.dtype.names:
  print name, snap.header[name]
