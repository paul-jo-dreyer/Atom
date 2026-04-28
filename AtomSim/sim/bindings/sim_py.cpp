#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "sim/ball.hpp"
#include "sim/robot.hpp"
#include "sim/types.hpp"
#include "sim/world.hpp"

namespace py = pybind11;

PYBIND11_MODULE(sim_py, m) {
    m.doc() = "AtomSim simulation: World, Robot, Ball + their configs";

    // --- nested params (re-bound here so this module is self-contained) ---

    py::class_<diff_drive::Params<float>>(m, "DiffDriveParams")
        .def(py::init<>())
        .def_readwrite("track_width", &diff_drive::Params<float>::track_width)
        .def_readwrite("tau_motor",   &diff_drive::Params<float>::tau_motor);

    py::class_<ball::Params<float>>(m, "BallParams")
        .def(py::init<>())
        .def_readwrite("radius",      &ball::Params<float>::radius)
        .def_readwrite("mass",        &ball::Params<float>::mass)
        .def_readwrite("restitution", &ball::Params<float>::restitution)
        .def_readwrite("damping",     &ball::Params<float>::damping);

    // --- enums ---

    py::enum_<sim::BodyType>(m, "BodyType")
        .value("Kinematic", sim::BodyType::Kinematic)
        .value("Dynamic",   sim::BodyType::Dynamic);

    // --- configs ---

    py::class_<sim::WorldConfig>(m, "WorldConfig")
        .def(py::init<>())
        .def_readwrite("field_x_half",   &sim::WorldConfig::field_x_half)
        .def_readwrite("field_y_half",   &sim::WorldConfig::field_y_half)
        .def_readwrite("goal_y_half",    &sim::WorldConfig::goal_y_half)
        .def_readwrite("goal_extension", &sim::WorldConfig::goal_extension)
        .def_readwrite("gravity_x",      &sim::WorldConfig::gravity_x)
        .def_readwrite("gravity_y",      &sim::WorldConfig::gravity_y)
        .def_readwrite("substeps",       &sim::WorldConfig::substeps);

    py::class_<sim::RobotConfig>(m, "RobotConfig")
        .def(py::init<>())
        .def_readwrite("body_type",         &sim::RobotConfig::body_type)
        .def_readwrite("chassis_side",      &sim::RobotConfig::chassis_side)
        .def_readwrite("mass",              &sim::RobotConfig::mass)
        .def_readwrite("yaw_inertia",       &sim::RobotConfig::yaw_inertia)
        .def_readwrite("x0",                &sim::RobotConfig::x0)
        .def_readwrite("y0",                &sim::RobotConfig::y0)
        .def_readwrite("theta0",            &sim::RobotConfig::theta0)
        .def_readwrite("manipulator_parts", &sim::RobotConfig::manipulator_parts)
        .def_readwrite("dynamics_params",   &sim::RobotConfig::dynamics_params);

    py::class_<sim::BallConfig>(m, "BallConfig")
        .def(py::init<>())
        .def_readwrite("x0",              &sim::BallConfig::x0)
        .def_readwrite("y0",              &sim::BallConfig::y0)
        .def_readwrite("vx0",             &sim::BallConfig::vx0)
        .def_readwrite("vy0",             &sim::BallConfig::vy0)
        .def_readwrite("field_k",         &sim::BallConfig::field_k)
        .def_readwrite("dynamics_params", &sim::BallConfig::dynamics_params);

    // --- main classes ---

    py::class_<sim::World>(m, "World")
        .def(py::init<const sim::WorldConfig&>(), py::arg("config") = sim::WorldConfig{})
        .def("step", &sim::World::step, py::arg("dt"))
        .def_property_readonly("config", &sim::World::config,
                               py::return_value_policy::reference_internal);

    py::class_<sim::Robot>(m, "Robot")
        .def(py::init<sim::World&, const sim::RobotConfig&>(),
             py::arg("world"), py::arg("config"),
             py::keep_alive<1, 2>())   // keep World alive while Robot exists
        .def("pre_step",  &sim::Robot::pre_step,  py::arg("wheel_cmd"), py::arg("dt"))
        .def("post_step", &sim::Robot::post_step)
        .def("set_state", &sim::Robot::set_state, py::arg("state"))
        .def_property_readonly("state", &sim::Robot::state)
        .def_property_readonly("config", &sim::Robot::config,
                               py::return_value_policy::reference_internal)
        .def_property_readonly("body_type", &sim::Robot::body_type);

    py::class_<sim::Ball>(m, "Ball")
        .def(py::init<sim::World&, const sim::BallConfig&>(),
             py::arg("world"), py::arg("config"),
             py::keep_alive<1, 2>())
        .def("pre_step",  &sim::Ball::pre_step,  py::arg("dt"))
        .def("post_step", &sim::Ball::post_step)
        .def("set_state", &sim::Ball::set_state, py::arg("state"))
        .def_property_readonly("state", &sim::Ball::state)
        .def_property_readonly("config", &sim::Ball::config,
                               py::return_value_policy::reference_internal);
}
