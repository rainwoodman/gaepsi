from numpy.distutils.core import setup, Extension
from numpy import get_include
import monkey

setup(name="gaepsi", version="0.2",
      author="Yu Feng",
      author_email="yfeng1@andrew.cmu.edu",
      description="Visualization and Analysis toolkit for SPH Cosmology code GADGET",
      url="http://github.com/rainwoodman/gaepsi",
      download_url="http://web.phys.cmu.edu/~yfeng1/gaepsi/gaepsi-0.2.tar.gz",
      zip_safe=False,
      install_requires=['cython', 'numpy', 'sharedmem', 'chealpy'],
      requires=['numpy', 'sharedmem', 'chealpy'],
      package_dir = {'gaepsi': '.'},
      packages = [
        'gaepsi', 'gaepsi.cosmology', 'gaepsi.readers', 'gaepsi.tools', 'gaepsi.compiledbase'
      ],
      scripts = [ 'scripts/gadget-render.py', 
                  'scripts/gadget-mklayers.py', 
                  'scripts/gadget-hist.py',
                  'scripts/gadget-dump-header.py',
                  'scripts/gadget-check-file.py',
                  'scripts/gadget-gen-snapshot.py',
                  'scripts/gadget-crop-snapshot.py',
                 ],
      ext_modules = [
        Extension("gaepsi.%s" % name, 
             [ name.replace('.', '/') + '.pyx',],
             extra_compile_args=['-O0', '-g', '-Dintp=npy_intp'],
             libraries=[],
             include_dirs=[get_include(), 'cosmology', 'compiledbase'],
             depends = extra
        ) for name, extra in [
         ('compiledbase._fast', []),
         ('compiledbase._field', []),
         ('compiledbase.camera', []),
         ('compiledbase.ztree', ['compiledbase/npyiter.pxd',]),
#         ('compiledbase.zfof', []),
         ('compiledbase.query', []),
         ('compiledbase.ngbquery', []),
         ('compiledbase.rayquery', []),
         ('compiledbase.fillingcurve', []),
         ('cosmology._cosmology', []),
         ('cosmology._qlf', ['cosmology/qlf_calculator.c']),
        ]
      ])

