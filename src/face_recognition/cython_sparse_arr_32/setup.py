from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(
  name = "facedist32",
  cmdclass = {"build_ext": build_ext},
  ext_modules =
  [
    Extension("facedist32",
              ["facedist32.pyx"],
              extra_compile_args = ["-O3", "-fopenmp", "-xAVX"],
              extra_link_args=['-qopenmp']
              )
  ]
)

