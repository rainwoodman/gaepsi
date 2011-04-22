from distutils.core import setup, Extension
from numpy import get_include
setup(name="gadget", version="1.0",
      package_dir = {'gadget': '.'},
      packages = [
        'gadget', 'gadget.constant', 'gadget.plot', 'gadget.readers', 'gadget.tools'
      ],
      scripts = [ 'scripts/gadget-render.py', 
                  'scripts/gadget-mklayers.py', 
                  'scripts/gadget-hist.py',
                  'scripts/gadget-dump-header.py'],
      ext_modules = [
        Extension("gadget._gadgetccode", 
             ["ccode/module.c", 
              "ccode/image.c", 
              "ccode/ndtree.c",
              "ccode/remap.c",
              "ccode/kernel.c",
              "ccode/render.c",
              "ccode/pmin.c",
             ], 
             extra_compile_args=['-O3', '-fopenmp'],
             libraries=['gomp'],
             include_dirs=[get_include()],
             depends = ["ccode/defines.h"]
        ),
      ])

