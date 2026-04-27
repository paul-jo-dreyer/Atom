#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include "diff_drive/core/dynamics.hpp"
#include "diff_drive/core/integrators.hpp"
#include "diff_drive/core/linearize.hpp"

namespace py = pybind11;

PYBIND11_MODULE(diff_drive_py, m) {
    m.doc() = "diff_drive: skid-steer dynamics core (Python bindings)";

    using Scalar = double;
    using Dyn = diff_drive::DiffDriveDynamics<Scalar>;

    py::class_<diff_drive::Params<Scalar>>(m, "Params")
        .def(py::init<>())
        .def_readwrite("track_width", &diff_drive::Params<Scalar>::track_width)
        .def_readwrite("tau_motor",   &diff_drive::Params<Scalar>::tau_motor);

    py::class_<Dyn>(m, "DiffDriveDynamics")
        .def(py::init<>())
        .def_readwrite("params", &Dyn::params)
        .def("continuous", &Dyn::continuous);

    py::class_<diff_drive::Jacobians<Scalar>>(m, "Jacobians")
        .def_readonly("A", &diff_drive::Jacobians<Scalar>::A)
        .def_readonly("B", &diff_drive::Jacobians<Scalar>::B);

    m.def("rk4_step",
          &diff_drive::rk4_step<Dyn, Scalar>,
          py::arg("dynamics"),
          py::arg("state"),
          py::arg("control"),
          py::arg("dt"));

    m.def("linearize", &diff_drive::linearize<Scalar>);
}
