from numpy import fromfile
from numpy import zeros, dtype
from numpy import uint8, array, linspace
from scipy.weave import inline

def icmap(levels, cmap, bins):
  cmap = array(cmap)
  detail = zeros(bins)
  for i in range(len(cmap) -1):
    start = int(levels[i] * (bins - 1))
    end = int(levels[i + 1] * (bins -1))
    detail[start:end] = linspace(cmap[i], cmap[i+1], end - start, endpoint=False)
  detail[-1] = cmap[-1]
  return detail

def render(array, min, max, logscale=True, levels = [0, 0.2, 0.4, 0.6, 1.0], rmap =[0, 0.5, 1.0, 1.0, 0.2], gmap=[0, 0.0, 0.5, 1.0, 0.2], bmap=[0.0, 0.0, 0.0, 0.3, 1.0]):

  image = zeros(array.shape, dtype=dtype(('u1', 3)))
  image_flat = image.reshape((-1, 3))
  array_flat = array.ravel()
  bins = 1024 * 16
  ri = uint8(icmap(levels, rmap, bins) * 255)
  gi = uint8(icmap(levels, gmap, bins) * 255)
  bi = uint8(icmap(levels, bmap, bins) * 255)
  print 'start rendering'
  ccode = r"""
int i, j, _bins = bins; 
int log = logscale;
float _min = min;
float _max = max;
for(i = 0; i < Nimage_flat[0]; i++) {
    float v = ARRAY_FLAT1(i);
	if(log) {
		if ( v <= 0.0 ) 
			v = -30;
		else 
			v = log10f(v);
	}
	float value = (v - _min) / (_max - _min);
	int index = value * _bins;
	if(index < 0) index = 0;
	if(index >= _bins) index = _bins - 1;
	IMAGE_FLAT2(i,0) = RI1(index);
	IMAGE_FLAT2(i,1) = GI1(index);
	IMAGE_FLAT2(i,2) = BI1(index);
}
  """
  inline(ccode, ["image_flat", "array_flat", "min", "max", 'rmap', 'gmap', 'bmap', 'ri', 'gi', 'bi', 'bins', 'logscale'])
  return image

"""
itmp = '%d.im'
otmp = '%d.raw'
for i in range(16):
  arr = fromfile(file(itmp % i, 'r'), dtype='f4')
  image = render(arr, min = 0.0, max = 1.0, levels=[0.0, 0.02, 0.2, 0.4, 1.0])
  image.tofile(file(otmp %i, 'w+'))
"""
