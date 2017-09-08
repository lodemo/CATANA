from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(
  name = "facedist",
  cmdclass = {"build_ext": build_ext},
  ext_modules =
  [
    Extension("facedist",
              ["facedist.pyx"],
              extra_compile_args = ["-O3", "-fopenmp", "-xAVX"],
              extra_link_args=['-qopenmp']
              )
  ]
)

