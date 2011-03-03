import gzip
from numpy import fromstring
from numpy import arange

def tozipfile(filename, array, patch_size=1048576):
  """writes an array to a zip file"""
  file = gzip.open(filename, 'w')
  for i in range(0, len(array), patch_size):
    file.write(array[i:i+patch_size].tostring())
  file.close()

def fromzipfile(filename, array, patch_size=1048576):
  """fills an array with data from a zip file. The array must be already allocated"""
  file = gzip.open(filename, 'r')
  for i in range(0, len(array), patch_size):
    str = file.read(size = patch_size * array.itemsize)
    data = fromstring(str, dtype=array.dtype)
    array[i:i+patch_size] = data

