import os
import sys
import subprocess
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        
        # Diwajibkan oleh setuptools agar library diletakkan pada folder yang benar
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DMSHQC_ENABLE_NATIVE_OPT=OFF", # Pastikan hermetic di CI/CD
            "-DMSHQC_ENABLE_IPO=ON"
        ]

        build_args = ['-j', str(os.cpu_count() or 2)]

        # Menerima Injeksi Variabel Lingkungan dari CI/CD
        env_cmake_args = os.environ.get('CMAKE_ARGS')
        if env_cmake_args:
            cmake_args.extend(env_cmake_args.split())

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)

setup(
    name="mshqc",
    version="1.0.0",
    author="Muhamad Syahrul Hidayat",
    description="Modern Quantum Chemistry Library (Pure MLIR JIT Backend)",
    ext_modules=[CMakeExtension('_mshqc')],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
    python_requires=">=3.8",
)
