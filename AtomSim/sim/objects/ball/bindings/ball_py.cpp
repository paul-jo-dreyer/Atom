#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include "ball/core/contact.hpp"
#include "ball/core/dynamics.hpp"
#include "ball/core/integrators.hpp"

namespace py = pybind11;

PYBIND11_MODULE(ball_py, m) {
    m.doc() = "ball: passive sphere dynamics with linear damping + impulse contact";

    using Scalar = double;
    using Dyn    = ball::BallDynamics<Scalar>;

    py::class_<ball::Params<Scalar>>(m, "Params")
        .def(py::init<>())
        .def_readwrite("radius",      &ball::Params<Scalar>::radius)
        .def_readwrite("mass",        &ball::Params<Scalar>::mass)
        .def_readwrite("restitution", &ball::Params<Scalar>::restitution)
        .def_readwrite("damping",     &ball::Params<Scalar>::damping);

    py::class_<Dyn>(m, "BallDynamics")
        .def(py::init<>())
        .def_readwrite("params", &Dyn::params)
        .def("continuous", &Dyn::continuous);

    m.def("rk4_step",
          &ball::rk4_step<Dyn, Scalar>,
          py::arg("dynamics"), py::arg("state"), py::arg("dt"));

    m.def("exact_step",
          &ball::exact_step<Scalar>,
          py::arg("params"), py::arg("state"), py::arg("dt"));

    m.def("apply_contact_impulse",
          &ball::apply_contact_impulse<Scalar>,
          py::arg("params"), py::arg("state"), py::arg("normal"));
}
