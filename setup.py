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
        real_exe_root = os.path.dirname(os.path.dirname(os.path.realpath(sys.executable)))
        exe_root = os.path.dirname(os.path.dirname(os.path.abspath(sys.executable)))

        candidate_roots = [
            sys.prefix,
            sys.base_prefix,
            exe_root,
            real_exe_root,
            os.environ.get("MKLROOT", ""),
            os.environ.get("CONDA_PREFIX", ""),
            "C:/deps/Library" if sys.platform == "win32" else "/usr/local",
            "C:/deps" if sys.platform == "win32" else "",
        ]
        mkl_lib_path = ""
        mkl_inc_path = ""
        for root in candidate_roots:
            if not root:
                continue
            lib_dirs = [os.path.join(root, "lib"), os.path.join(root, "Library", "lib")]
            inc_dirs = [os.path.join(root, "include"), os.path.join(root, "Library", "include")]
            if not mkl_lib_path:
                for lib_dir in lib_dirs:
                    if not os.path.isdir(lib_dir):
                        continue
                    if sys.platform == "win32":
                        matches = glob.glob(os.path.join(lib_dir, "mkl_rt*.lib"))
                        if matches:
                            mkl_lib_path = matches[0]
                            break
                    elif sys.platform == "darwin":
                        matches = glob.glob(os.path.join(lib_dir, "libmkl_rt*.dylib"))
                        if matches:
                            mkl_lib_path = matches[0]
                            break
                    else:
                        matches = glob.glob(os.path.join(lib_dir, "libmkl_rt.so*"))
                        if matches:
                            so_link = os.path.join(lib_dir, "libmkl_rt.so")
                            if not os.path.exists(so_link):
                                try:
                                    os.symlink(os.path.basename(matches[0]), so_link)
                                    mkl_lib_path = so_link
                                except OSError:
                                    mkl_lib_path = matches[0]
                            else:
                                mkl_lib_path = so_link
                            break
            if not mkl_inc_path:
                for inc_dir in inc_dirs:
                    if os.path.isfile(os.path.join(inc_dir, "mkl.h")):
                        mkl_inc_path = inc_dir
                        break

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY={extdir}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE={extdir}",
            f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE={extdir}",
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
        if sys.platform != "win32":
            env["CXXFLAGS"] = f'{env.get("CXXFLAGS", "")} -DVERSION_INFO=\\"{self.distribution.get_version()}\\"'.strip()

        subprocess.check_call(["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(["cmake", "--build", "."] + build_args, cwd=self.build_temp)

        for ext_pat in ("**/*.so", "**/*.pyd", "**/*.dylib"):
            for filepath in glob.glob(os.path.join(self.build_temp, ext_pat), recursive=True):
                if "_mshqc" in os.path.basename(filepath):
                    shutil.copy2(filepath, extdir)
        for ext_pat in ("*.so", "*.pyd", "*.dylib"):
            for filepath in glob.glob(os.path.join(ext.sourcedir, "python", "mshqc", ext_pat)):
                if "_mshqc" in os.path.basename(filepath):
                    shutil.copy2(filepath, extdir)

        basis_candidates = [
            os.path.join(ext.sourcedir, "data", "basis"),
            os.path.join(os.path.dirname(ext.sourcedir), "mshqc-basis", "data", "basis"),
            os.path.join(os.path.expanduser("~"), "mshqc-basis", "data", "basis"),
        ]
        src_basis_dir = ""
        for cand in basis_candidates:
            if os.path.isdir(cand) and glob.glob(os.path.join(cand, "*.gbs")):
                src_basis_dir = cand
                break
        if src_basis_dir:
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