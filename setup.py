from distutils.core import setup, Extension
from numpy import get_include
setup(name="gadget", version="1.0",
      packages = [
        '', 'constant', 'plot', 'schemadefs'
      ],
      ext_modules = [
        Extension("quadtree", ["ccode/quadtree.c"], 
             include_dirs=[get_include()]
        ),
        Extension("octtree", ["ccode/octtree.c"], 
             include_dirs=[get_include()]
        )
      ])

