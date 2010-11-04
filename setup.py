from distutils.core import setup, Extension
from numpy import get_include
setup(name="gadget", version="1.0",
      package_dir = {'gadget': '.'},
      packages = [
        'gadget', 'gadget.constant', 'gadget.plot', 'gadget.readers', 'gadget.tools'
      ],
      ext_modules = [
        Extension("gadget.ccode", 
             ["ccode/module.c", 
              "ccode/image.c", 
              "ccode/ndtree.c",
              "ccode/remap.c",
              "ccode/kernel.c",
             ], 
             include_dirs=[get_include()],
        ),
      ])

