#! python

from sys import argv

from gaepsi.snapshot import Snapshot

snap = Snapshot(argv[2], argv[1])
for name in snap.header.dtype.names:
  print name, snap.header[name]
