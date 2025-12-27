"""
Setup script for MSHQC Python bindings
File: /home/syahrul/mshqc/setup.py
"""

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import os
import subprocess

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        try:
            subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build extensions")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        
        # [FIX] Get pybind11 cmake path
        import pybind11
        pybind11_cmake_path = pybind11.get_cmake_dir()

        # CMake config
        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-DPYTHON_EXECUTABLE={sys.executable}',
            f'-Dpybind11_DIR={pybind11_cmake_path}',  # [FIX] Point to pybind11 config
            '-DCMAKE_BUILD_TYPE=Release',
        ]
        
        build_args = ['--config', 'Release']
        
        # Set threads
        if 'CMAKE_BUILD_PARALLEL_LEVEL' not in os.environ:
            build_args += ['-j4']
        
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
            
        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get('CXXFLAGS', ''),
            self.distribution.get_version())

        # Configure
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, 
                            cwd=self.build_temp, env=env)
        # Build
        subprocess.check_call(['cmake', '--build', '.'] + build_args, 
                            cwd=self.build_temp)

# Read long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mshqc",
    version="1.0.0",
    author="MSHQC Team",
    description="Multi-State High-Quality Calculations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=['mshqc'],
    package_dir={'': 'python'},
    ext_modules=[CMakeExtension('mshqc._mshqc')],
    cmdclass=dict(build_ext=CMakeBuild),
    install_requires=[
        'numpy>=1.20.0',
        'scipy>=1.7.0',
        'pybind11>=2.6.0', # Wajib ada di sini
    ],
    python_requires='>=3.7',
)