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

        if not extdir.endswith("mshqc"):
            extdir = os.path.join(extdir, "mshqc")
        os.makedirs(extdir, exist_ok=True)

        import nanobind
        nanobind_cmake_path = nanobind.cmake_dir()

        # Cari semua kemungkinan prefix tempat pip menginstal mkl-devel & mkl-include
        real_exe_root = os.path.dirname(os.path.dirname(os.path.realpath(sys.executable)))
        exe_root = os.path.dirname(os.path.dirname(os.path.abspath(sys.executable)))
        candidate_roots = [
            sys.prefix,
            sys.base_prefix,
            exe_root,
            real_exe_root,
            os.environ.get("MKLROOT", ""),
            os.environ.get("CONDA_PREFIX", ""),
            "/usr/local",
        ]

        mkl_lib_path = ""
        mkl_inc_path = ""
        for root in candidate_roots:
            if not root:
                continue
            lib_dir = os.path.join(root, "lib")
            inc_dir = os.path.join(root, "include")
            if not mkl_lib_path and os.path.isdir(lib_dir):
                matches = glob.glob(os.path.join(lib_dir, "libmkl_rt.so*"))
                if matches:
                    # Buat symlink libmkl_rt.so jika hanya ada libmkl_rt.so.2
                    so_link = os.path.join(lib_dir, "libmkl_rt.so")
                    if not os.path.exists(so_link):
                        try:
                            os.symlink(os.path.basename(matches[0]), so_link)
                            mkl_lib_path = so_link
                        except OSError:
                            mkl_lib_path = matches[0]
                    else:
                        mkl_lib_path = so_link
            if not mkl_inc_path and os.path.isfile(os.path.join(inc_dir, "mkl.h")):
                mkl_inc_path = inc_dir

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY={extdir}",
            f"-DPython_ROOT_DIR={real_exe_root}",
            f"-DPython3_ROOT_DIR={real_exe_root}",
            f"-DPython_EXECUTABLE={sys.executable}",
            f"-DPython3_EXECUTABLE={sys.executable}",
            f"-Dnanobind_DIR={nanobind_cmake_path}",
            "-DCMAKE_BUILD_TYPE=Release",
        ]

        if mkl_lib_path:
            cmake_args.append(f"-DMKL_LIBRARY={mkl_lib_path}")
        if mkl_inc_path:
            cmake_args.append(f"-DMKL_INCLUDE_DIR={mkl_inc_path}")

        env_cmake_args = os.environ.get("CMAKE_ARGS")
        if env_cmake_args:
            cmake_args.extend(shlex.split(env_cmake_args))

        if shutil.which("ninja"):
            cmake_args.extend(["-G", "Ninja"])

        parallel_level = os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL", str(os.cpu_count() or 4))
        build_args = ["--config", "Release", "-j", parallel_level]

        os.makedirs(self.build_temp, exist_ok=True)

        env = os.environ.copy()
        env["CXXFLAGS"] = f'{env.get("CXXFLAGS", "")} -DVERSION_INFO=\\"{self.distribution.get_version()}\\"'

        subprocess.check_call(["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(["cmake", "--build", "."] + build_args, cwd=self.build_temp)

        for filepath in glob.glob(os.path.join(self.build_temp, "**/*.so"), recursive=True):
            shutil.copy2(filepath, extdir)

        for filepath in glob.glob(os.path.join(ext.sourcedir, "python", "mshqc", "*.so")):
            if "_mshqc" in os.path.basename(filepath):
                shutil.copy2(filepath, extdir)

        src_basis_dir = os.path.join(ext.sourcedir, "data", "basis")
        if os.path.isdir(src_basis_dir):
            pkg_src_basis_dir = os.path.join(ext.sourcedir, "python", "mshqc", "basis")
            dst_wheel_basis_dir = os.path.join(extdir, "basis")

            os.makedirs(pkg_src_basis_dir, exist_ok=True)
            os.makedirs(dst_wheel_basis_dir, exist_ok=True)

            for gbs_file in glob.glob(os.path.join(src_basis_dir, "*.gbs")):
                shutil.copy2(gbs_file, pkg_src_basis_dir)
                if os.path.abspath(pkg_src_basis_dir) != os.path.abspath(dst_wheel_basis_dir):
                    shutil.copy2(gbs_file, dst_wheel_basis_dir)


setup(
    ext_modules=[CMakeExtension("mshqc._mshqc")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
)