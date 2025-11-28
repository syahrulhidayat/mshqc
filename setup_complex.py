import os
import sys
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import numpy

try:
    import pybind11
except ImportError:
    # pybind11 will be installed by pip if not available
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pybind11>=2.12"])
    import pybind11

class BuildExt(build_ext):
    def build_extensions(self):
        if sys.platform.startswith('win'):
            for ext in self.extensions:
                ext.extra_compile_args.append('/std:c++17')
                ext.extra_compile_args.append('/O2')
        else:
            for ext in self.extensions:
                ext.extra_compile_args.append('-std=c++17')
                ext.extra_compile_args.append('-O3')
    def run(self):
        for ext in self.extensions:
            self.build_cmake(ext)
        super().run()

    def build_cmake(self, ext):
        build_temp = pathlib.Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)

        cmake_args = [
            f"-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DMSHQC_PYTHON=ON",
            f"-DMSHQC_WITH_LIBINT2={os.environ.get('MSHQC_WITH_LIBINT2','ON')}",
            f"-DMSHQC_WITH_LIBCINT={os.environ.get('MSHQC_WITH_LIBCINT','OFF')}",
        ]

        # Windows toolchain
        if sys.platform.startswith("win"):
            cmake_args += ["-A", "x64"]

        # Prefix include untuk numpy/pybind11
        env = os.environ.copy()
        env["CXXFLAGS"] = env.get("CXXFLAGS", "") + f" -I{numpy.get_include()} -I{pybind11.get_include()}"

# Configure & build static lib mshqc
        subprocess.check_call(["cmake", "."] + cmake_args, cwd=build_temp, env=env)
        subprocess.check_call(["cmake", "--build", ".", "--config", "Release", "-j"], cwd=build_temp)

        # Cari hasil lib mshqc
        # Asumsikan CMake install tidak dipakai; kita link pakai objects saat build ekstensi
        # Alternatif: gunakan file .a/.lib hasil add_library(mshqc STATIC ...)
        # Lokasi umum:
        lib_candidates = []
        for name in ["mshqc", "libmshqc"]:
            for ext in [".a", ".lib", ".so", ".dylib"]:
                p = next(build_temp.rglob(f"*{name}*{ext}"), None)
                if p and p.exists():
                    lib_candidates.append(str(p))
        if not lib_candidates:
            print("Peringatan: tidak menemukan library mshqc terkompilasi, lanjutkan link via objek CMake-built.")

        # Simpan lokasi build untuk dipakai kompilasi ekstensi
        ext._mshqc_build_dir = str(build_temp)
        ext._mshqc_libs = lib_candidates

class PyExt(Extension):
    pass

# Compile all source files
sources = []
for root, _, files in os.walk('src'):
    for file in files:
        if file.endswith('.cc'):
            sources.append(os.path.join(root, file))

ext = PyExt(
    "mshqc._core",
    sources=["python/bindings.cc"] + sources,  # binding pybind11 + all sources
    include_dirs=[
        "include",
        "src",
        numpy.get_include(),
        pybind11.get_include(),
    ],
    language="c++",
)

def setup_kwargs():
    # Link options cross-platform
    extra_compile_args = ["-std=c++17", "-O3"]
    extra_link_args = []

    if sys.platform.startswith("win"):
        extra_compile_args = ["/std:c++17", "/O2"]
    else:
        # OpenMP opsional
        if os.environ.get("MSHQC_WITH_OPENMP", "ON") == "ON":
            extra_compile_args += ["-fopenmp"]
            extra_link_args += ["-fopenmp"]

    return dict(
        name="mshqc",
        version="0.1.0",
        description="Python bindings for MSH-QC quantum mechanics library",
try:
            with open("README.md", "r", encoding="utf-8") as f:
                long_desc = f.read()
        except:
            long_desc = ""
        long_description_content_type="text/markdown",
        author="Muhamad Sahrul Hidayat",
        license="MIT",
        url="https://github.com/syahrulhidayat/mshqc",
        packages=["mshqc"],
        package_dir={"": "python"},
        package_data={"mshqc": ["py.typed"]},
        include_package_data=True,
        ext_modules=[ext],
        cmdclass={"build_ext": CMakeBuild},
        zip_safe=False,
        python_requires=">=3.8",
        install_requires=["numpy>=1.22", "pybind11>=2.12"],
        options={
            "build_ext": {
                "inplace": False
            }
        },
        # Hint untuk compiler/linker args
        extra_compile_args=extra_compile_args,
    )

if __name__ == "__main__":
    setup(**setup_kwargs())