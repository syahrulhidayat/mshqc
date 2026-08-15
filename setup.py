import os
import sys
import glob
import shlex
import shutil
import subprocess
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = ""):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self) -> None:
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError("CMake must be installed and accessible in PATH to build extensions.")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext: CMakeExtension) -> None:
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        
        # Resolusi hierarki direktori modul mshqc
        if not extdir.endswith("mshqc"):
            extdir = os.path.join(extdir, "mshqc")
        os.makedirs(extdir, exist_ok=True)

        import nanobind
        nanobind_cmake_path = nanobind.cmake_dir()

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-Dnanobind_DIR={nanobind_cmake_path}",
            "-DCMAKE_BUILD_TYPE=Release",
        ]

        # Injeksi argumen CMake kustom dari environment variable (umum pada sistem build HPC)
        env_cmake_args = os.environ.get("CMAKE_ARGS")
        if env_cmake_args:
            cmake_args.extend(shlex.split(env_cmake_args))

        # Generator Ninja direkomendasikan untuk menghindari inefisiensi re-evaluasi target pada Make
        if shutil.which("ninja"):
            cmake_args.extend(["-G", "Ninja"])

        parallel_level = os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL", str(os.cpu_count() or 4))
        build_args = ["--config", "Release", "-j", parallel_level]

        os.makedirs(self.build_temp, exist_ok=True)

        env = os.environ.copy()
        # Injeksi metadata versi dengan flag C++ preprocessor
        env["CXXFLAGS"] = f'{env.get("CXXFLAGS", "")} -DVERSION_INFO=\\"{self.distribution.get_version()}\\"'

        # Eksekusi pipeline kompilasi
        subprocess.check_call(["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(["cmake", "--build", "."] + build_args, cwd=self.build_temp)

        # Pemindahan artifak biner menggunakan copy2 untuk mempertahankan file timestamp dan metadata
        # Mencegah invalidasi cache yang prematur saat re-packaging
        for filepath in glob.glob(os.path.join(self.build_temp, "**/*.so"), recursive=True):
            shutil.copy2(filepath, extdir)

        for filepath in glob.glob(os.path.join(ext.sourcedir, "python", "mshqc", "*.so")):
            if "_mshqc" in os.path.basename(filepath):
                shutil.copy2(filepath, extdir)

# Eliminasi definisi redundan: name, version, author, dll. harus berada di pyproject.toml
setup(
    ext_modules=[CMakeExtension("mshqc._mshqc")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
)