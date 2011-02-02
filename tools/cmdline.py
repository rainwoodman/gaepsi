from optparse import OptionParser, OptionValueError
from numpy import fromstring, matrix
def parsearray(option, opt_str, value, parser, *args, **kwargs):
  try:
    arr = fromstring(value, dtype=kwargs['dtype'], sep=kwargs['sep'])
    if kwargs['len'] and (kwargs['len'] != len(arr)):
      raise OptionValueError("length of argument doesn't match with %d" % kwargs['len'])
    setattr(parser.values, option.dest, arr)
  except:
    raise OptionValueError("failed parsing array: %s %s" % ( opt_str, value))
def parsematrix(option, opt_str, value, parser, *args, **kwargs):
  try:
    mat = matrix(value)
  except:
    raise OptionValueError("failed parsing matrix %s %s" % (opt_str, value))
  if kwargs['shape'] and (kwargs['shape'] != mat.shape):
    raise OptionValueError("shape %s doesn't match with %s" % (mat.shape, kwargs['shape']))
  setattr(parser.values, option.dest, mat)
 
