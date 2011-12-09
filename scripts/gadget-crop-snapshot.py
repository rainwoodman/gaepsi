from sys import argv
from numpy import fromstring, newaxis
from gaepsi.tools.crop import crop_snapshot

if __name__ == "__main__":
  snapname = argv[2]
  format = argv[1]
  map = argv[3]

  center = fromstring(argv[4], sep=',')
  size = fromstring(argv[5], sep=',')
  output = 'snapshot.out'
  crop_snapshot(center, size, map, format, snapname, output)

