#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "ndarray.hpp"

namespace py = pybind11;

PYBIND11_MODULE(native_backend, m) {
    m.doc() = "GoTorch native C++ backend";
    m.attr("__backend_type__") = "cpp";

    py::class_<NDArray>(m, "NDArray")
        .def(py::init<>())
        .def(py::init([](const std::vector<float>& data,
                         const std::vector<size_t>& shape,
                         const std::vector<size_t>& strides,
                         size_t offset) {
            return std::make_unique<NDArray>(data, shape, strides, offset);
        }),
        py::arg("data"),
        py::arg("shape"),
        py::arg("strides") = std::vector<size_t>{},
        py::arg("offset") = 0)
        .def_property_readonly("shape", [](const NDArray& self) {
            py::tuple t(self.shape.size());
            for (size_t i = 0; i < self.shape.size(); ++i) {
                t[i] = self.shape[i];
            }
            return t;
        })
        .def_property_readonly("strides", [](const NDArray& self) {
            py::tuple t(self.strides.size());
            for (size_t i = 0; i < self.strides.size(); ++i) {
                t[i] = self.strides[i];
            }
            return t;
        })
        .def_property_readonly("offset", [](const NDArray& self) {
            return self.offset;
        })
        .def_property_readonly("data", [](const NDArray& self) {
            if (!self.storage) {
                return std::vector<float>{};
            }
            return self.storage->data;
        })
        .def("add", &NDArray::add, py::arg("addend"))
        .def("__add__", [](const NDArray& self, py::object other) -> py::object {
            if (py::isinstance<NDArray>(other)) {
                return py::cast(self.add(other.cast<const NDArray&>()));
            }
            return py::reinterpret_borrow<py::object>(Py_NotImplemented);
        })
        .def("__radd__", [](const NDArray& self, py::object other) -> py::object {
            if (py::isinstance<NDArray>(other)) {
                return py::cast(other.cast<const NDArray&>().add(self));
            }
            return py::reinterpret_borrow<py::object>(Py_NotImplemented);
        })
        .def("sub", &NDArray::sub, py::arg("subtrahend"))
        .def("__sub__", [](const NDArray& self, py::object other) -> py::object {
            if (py::isinstance<NDArray>(other)) {
                return py::cast(self.sub(other.cast<const NDArray&>()));
            }
            return py::reinterpret_borrow<py::object>(Py_NotImplemented);
        })
        .def("__rsub__", [](const NDArray& self, py::object other) -> py::object {
            if (py::isinstance<NDArray>(other)) {
                return py::cast(other.cast<const NDArray&>().sub(self));
            }
            return py::reinterpret_borrow<py::object>(Py_NotImplemented);
        })
        .def("transpose", &NDArray::transpose, py::arg("dim0") = 0, py::arg("dim1") = 1)
        .def("matmul", &NDArray::matmul, py::arg("other"))
        .def("is_contiguous", &NDArray::is_contiguous)
        .def("ones_like", &NDArray::ones_like)
        .def("zeros_like", &NDArray::zeros_like)
        .def("size", &NDArray::size)
        .def("__getitem__", [](const NDArray& self, py::object index) -> float {
            if (py::isinstance<py::int_>(index)) {
                return self({index.cast<size_t>()});
            }
            if (py::isinstance<py::tuple>(index)) {
                auto tup = index.cast<py::tuple>();
                std::vector<size_t> idxs;
                idxs.reserve(tup.size());
                for (size_t i = 0; i < tup.size(); ++i) {
                    idxs.push_back(tup[i].cast<size_t>());
                }
                return self(idxs);
            }
            throw py::type_error("Index must be int or tuple of ints");
        })
        .def("__setitem__", [](NDArray& self, py::object index, float val) {
            if (py::isinstance<py::int_>(index)) {
                self({index.cast<size_t>()}) = val;
                return;
            }
            if (py::isinstance<py::tuple>(index)) {
                auto tup = index.cast<py::tuple>();
                std::vector<size_t> idxs;
                idxs.reserve(tup.size());
                for (size_t i = 0; i < tup.size(); ++i) {
                    idxs.push_back(tup[i].cast<size_t>());
                }
                self(idxs) = val;
                return;
            }
            throw py::type_error("Index must be int or tuple of ints");
        })
        .def("__repr__", [](const NDArray& self) {
            py::tuple t(self.shape.size());
            for (size_t i = 0; i < self.shape.size(); ++i) {
                t[i] = self.shape[i];
            }
            return "<NDArray shape=" + py::str(t).cast<std::string>() + ">";
        });
}
