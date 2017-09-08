from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(
  name = "facedistsplit32",
  cmdclass = {"build_ext": build_ext},
  ext_modules =
  [
    Extension("facedistsplit32",
              ["facedistsplit32.pyx"],
              extra_compile_args = ["-O3", "-fopenmp", "-xAVX"],
              extra_link_args=['-qopenmp']
              )
  ]
)

