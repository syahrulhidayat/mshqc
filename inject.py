
from pathlib import Path

p = Path("python/bindings.cc")
text = p.read_text(encoding="utf-8")

old = '.def("build", &PetiteList::build)'
new = (
    '.def("build", nb::overload_cast<>(&PetiteList::build))\n'
    '        .def("build", nb::overload_cast<const Eigen::MatrixXd&, double>(&PetiteList::build),\n'
    '             nb::arg("schwarz"), nb::arg("cutoff") = 1e-12)'
)

if old in text:
    p.write_text(text.replace(old, new, 1), encoding="utf-8")
    print("[OK] Overload PetiteList::build di python/bindings.cc berhasil diperbaiki.")
elif "nb::overload_cast<>(&PetiteList::build)" in text:
    print("[SKIP] python/bindings.cc sudah terpatch.")
else:
    raise SystemExit("[ERROR] Baris .def(\"build\", &PetiteList::build) tidak ditemukan.")
