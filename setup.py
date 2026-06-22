"""
Setup script for MSHQC Python bindings
File: /workspaces/mshqc/setup.py
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
        # extdir adalah folder staging tempat file .whl dirakit
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        
        # Pastikan output dari cmake masuk tepat ke dalam folder 'mshqc'
        if not extdir.endswith("mshqc"):
            extdir = os.path.join(extdir, "mshqc")
        os.makedirs(extdir, exist_ok=True)
        
        import nanobind
        nanobind_cmake_path = nanobind.cmake_dir()

        # CMake config
        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-DCMAKE_RUNTIME_OUTPUT_DIRECTORY={extdir}', # Menangkap libmshqc.so
            f'-DPYTHON_EXECUTABLE={sys.executable}',
            f'-Dnanobind_DIR={nanobind_cmake_path}',  
            '-DCMAKE_BUILD_TYPE=Release',
        ]
        
        build_args = ['--config', 'Release']
        
        # Set eksekusi paralel OpenMP/Kompilasi
        if 'CMAKE_BUILD_PARALLEL_LEVEL' not in os.environ:
            build_args += ['-j4']
        
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
            
        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get('CXXFLAGS', ''),
            self.distribution.get_version())

        # Configure & Build
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)

# Read long description safely
try:
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = "MSHQC Quantum Mechanics Library"

setup(
    name="mshqc",
    version="1.0.0",
    author="Muhamad Syahrul Hidayat",
    description="Multi-State High-Quality Calculations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=['mshqc'],
    package_dir={'': 'python'},
    
    # Memastikan file biner terbungkus dengan aman (zip_safe=False sangat penting untuk C++)
    package_data={
        "mshqc": ["*.so", "*.pyi"],
    },
    include_package_data=True,
    zip_safe=False,
    
    ext_modules=[CMakeExtension('mshqc._mshqc')],
    cmdclass=dict(build_ext=CMakeBuild),
    install_requires=[
        'numpy>=1.20.0',
        'scipy>=1.7.0',
        'nanobind>=1.9.0',
    ],
    python_requires='>=3.8',
)