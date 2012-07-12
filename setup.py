from numpy.distutils.core import setup, Extension
from numpy import get_include
setup(name="gaepsi", version="1.0",
      package_dir = {'gaepsi': '.'},
      packages = [
        'gaepsi', 'gaepsi.cosmology', 'gaepsi.ccode', 'gaepsi.readers', 'gaepsi.tools', 'gaepsi.tools.sharedmem', 'gaepsi.cython'
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
             [ name.replace('.', '/') + '.c',],
             extra_compile_args=['-O3'],
             libraries=[],
             include_dirs=[get_include()],
             depends = extra
        ) for name, extra in [
         ('cython._fast', []),
         ('cython._field', []),
         ('cython._camera', []),
         ('cython.ztree', []),
         ('cython.zfof', []),
         ('cython.zorder', ['cython/zorder_internal.c']),
         ('cython.zquery', ['cython/zquery_internal.c']),
         ('cosmology._cosmology', []),
         ('tools.sharedmem.listtools', []),
         ('tools.sharedmem._mergesort', []),
        ]
      ])

