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
      package_dir = {'gaepsi': 'gaepsi'},
      packages = [
        'gaepsi', 'gaepsi.cosmology', 'gaepsi.readers', 'gaepsi.tools', 'gaepsi.compiledbase'
      ],
      scripts = [  
                  'scripts/gadget-dump-header.py',
                  'scripts/gadget-make-meshindex.py',
                  'scripts/gadget-extract-ptype.py',
                  'scripts/gadget-check-file.py',
                 ],
      ext_modules = [
        Extension("gaepsi.%s" % name, 
             [ 'gaepsi/' + name.replace('.', '/') + '.pyx',],
             extra_compile_args=['-O0', '-g', '-Dintp=npy_intp'],
             libraries=[],
             include_dirs=[get_include(), 'gaepsi/cosmology', 'gaepsi/compiledbase'],
             depends = extra
        ) for name, extra in [
         ('compiledbase._fast', []),
         ('compiledbase.geometry', []),
#         ('compiledbase._field', []), orphan, pending removal?
         ('compiledbase.camera', []),
         ('compiledbase.ztree', ['gaepsi/compiledbase/npyiter.pxd',]),
#         ('compiledbase.zfof', []),
         ('compiledbase.query', []),
         ('compiledbase.ngbquery', []),
         ('compiledbase.rayquery', []),
         ('compiledbase.fillingcurve', []),
         ('cosmology._cosmology', []),
         ('cosmology._qlf', ['gaepsi/cosmology/qlf_calculator.c']),
        ]
      ])

