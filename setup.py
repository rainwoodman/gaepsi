from distutils.core import setup, Extension
from numpy import get_include
setup(name="gadget", version="1.0",
      packages = [
        '', 'constant', 'plot', 'readers'
      ],
      ext_modules = [
        Extension("ccode", 
             ["ccode/module.c", 
              "ccode/image.c", 
              "ccode/ndtree.c"], 
             include_dirs=[get_include()],
        ),
      ])

