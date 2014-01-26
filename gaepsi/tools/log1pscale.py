import numpy as np
from numpy import ma

from matplotlib.cbook import dedent
from matplotlib.ticker import (NullFormatter, ScalarFormatter,
                               LogFormatterMathtext)
from matplotlib.ticker import (NullLocator, LogLocator, AutoLocator,
                               SymmetricalLogLocator)
from matplotlib.transforms import Transform, IdentityTransform
from matplotlib import docstring
from matplotlib.scale import ScaleBase, register_scale
from matplotlib.scale import _mask_non_positives, _clip_non_positives

class Log1pScale(ScaleBase):
    """
    A log(1 + x) scale.  Care is taken so bad values are not plotted.

    For computational efficiency (to push as much as possible to Numpy
    C code in the common cases), this scale provides different
    transforms depending on the base of the logarithm:

       - base 10 (:class:`Base10Transform`)
       - arbitrary base (:class:`BasexTransform`)
    """

    name = 'log1p'

    class Log1pTransform(Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True
        has_inverse = True

        def __init__(self, base, nonpos):
            Transform.__init__(self)
            self.base = base
            if nonpos == 'mask':
                self._handle_nonpos = _mask_non_positives
            else:
                self._handle_nonpos = _clip_non_positives

        def transform_non_affine(self, a):
            a = a + 1
            a = self._handle_nonpos(a * self.base)
            if isinstance(a, ma.MaskedArray):
                return ma.log(a) / np.log(self.base)
            return np.log(a) / np.log(self.base)

        def inverted(self):
            return Log1pScale.InvertedLog1pTransform(self.base)

    class InvertedLog1pTransform(Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True
        has_inverse = True

        def __init__(self, base):
            Transform.__init__(self)
            self.base = base

        def transform_non_affine(self, a):
            return ma.power(self.base, a) / self.base - 1

        def inverted(self):
            return Log1pScale.Log1pTransform(self.base)

    def __init__(self, axis, **kwargs):
        """
        *basex*/*basey*:
           The base of the logarithm

        *nonposx*/*nonposy*: ['mask' | 'clip' ]
          non-positive values in *x* or *y* can be masked as
          invalid, or clipped to a very small positive number

        *subsx*/*subsy*:
           Where to place the subticks between each major tick.
           Should be a sequence of integers.  For example, in a log10
           scale: ``[2, 3, 4, 5, 6, 7, 8, 9]``

           will place 8 logarithmically spaced minor ticks between
           each major tick.
        """
        if axis.axis_name == 'x':
            base = kwargs.pop('basex', 10.0)
            subs = kwargs.pop('subsx', None)
            nonpos = kwargs.pop('nonposx', 'mask')
        else:
            base = kwargs.pop('basey', 10.0)
            subs = kwargs.pop('subsy', None)
            nonpos = kwargs.pop('nonposy', 'mask')

        if nonpos not in ['mask', 'clip']:
            raise ValueError("nonposx, nonposy kwarg must be 'mask' or 'clip'")

        self._transform = self.Log1pTransform(base, nonpos)

        self.base = base
        self.subs = subs

    def set_default_locators_and_formatters(self, axis):
        """
        Set the locators and formatters to specialized versions for
        log scaling.
        """
        axis.set_major_locator(AutoLocator())
        axis.set_major_formatter(ScalarFormatter())
        axis.set_minor_locator(NullLocator())
        axis.set_minor_formatter(NullFormatter())

    def get_transform(self):
        """
        Return a :class:`~matplotlib.transforms.Transform` instance
        appropriate for the given logarithm base.
        """
        return self._transform

    def limit_range_for_scale(self, vmin, vmax, minpos):
        """
        Limit the domain to positive values.
        """
        return (vmin <= -1.0 and minpos or vmin,
                vmax <= -1.0 and minpos or vmax)

register_scale(Log1pScale)
