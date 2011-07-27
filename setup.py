from numpy.distutils.core import setup, Extension
from numpy import get_include
setup(name="gaepsi", version="1.0",
      package_dir = {'gaepsi': '.'},
      packages = [
        'gaepsi', 'gaepsi.constant', 'gaepsi.plot', 'gaepsi.readers', 'gaepsi.tools'
      ],
      scripts = [ 'scripts/gadget-render.py', 
                  'scripts/gadget-mklayers.py', 
                  'scripts/gadget-hist.py',
                  'scripts/gadget-dump-header.py',
                  'scripts/gadget-check-file.py',
                  'scripts/gadget-gen-snapshot.py',
                 ],
      libraries=[('pluecker', {'sources':['ccode/pluecker.f90']}),
                 ],
      ext_modules = [
        Extension("gaepsi._gaepsiccode", 
             ["ccode/module.c", 
              "ccode/image.c", 
              "ccode/octtree.c",
              "ccode/remap.c",
              "ccode/kernel.c",
              "ccode/render.c",
              "ccode/pmin.c",
              "ccode/sml.c",
             ], 
             extra_compile_args=['-O3', '-fopenmp'],
             libraries=['gomp', 'pluecker'],
             include_dirs=[get_include()],
             depends = ["ccode/defines.h"]
        ),
      ])

