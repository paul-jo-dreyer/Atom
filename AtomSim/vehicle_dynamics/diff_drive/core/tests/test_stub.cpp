#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "dynamics.hpp"
#include "integrators.hpp"
#include "linearize.hpp"

#include <cmath>

TEST_CASE("dynamics: zero state with symmetric command accelerates forward") {
    diff_drive::DiffDriveDynamics<double> dyn;
    diff_drive::State<double> x = diff_drive::State<double>::Zero();
    diff_drive::Control<double> u;
    u << 1.0, 1.0;

    const auto dx = dyn.continuous(x, u);
    CHECK(dx[diff_drive::V] > 0.0);
    CHECK(dx[diff_drive::OMEGA] == doctest::Approx(0.0));
}

TEST_CASE("rk4: integrating forward command moves the body") {
    diff_drive::DiffDriveDynamics<double> dyn;
    diff_drive::State<double> x = diff_drive::State<double>::Zero();
    diff_drive::Control<double> u;
    u << 1.0, 1.0;

    auto x1 = diff_drive::rk4_step(dyn, x, u, 0.01);
    CHECK(x1[diff_drive::V] > 0.0);
    CHECK(x1[diff_drive::V] < 1.0);
    CHECK(std::isfinite(x1[diff_drive::PX]));
}

TEST_CASE("linearize: produces finite A, B at origin") {
    diff_drive::DiffDriveDynamics<double> dyn;
    diff_drive::State<double> x = diff_drive::State<double>::Zero();
    diff_drive::Control<double> u = diff_drive::Control<double>::Zero();

    auto J = diff_drive::linearize(dyn, x, u);
    CHECK(J.A.allFinite());
    CHECK(J.B.allFinite());
    CHECK(J.A(diff_drive::V, diff_drive::V) == doctest::Approx(-1.0 / dyn.params.tau_motor));
}
