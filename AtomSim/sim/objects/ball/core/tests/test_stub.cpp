#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "contact.hpp"
#include "dynamics.hpp"
#include "integrators.hpp"

#include <cmath>

TEST_CASE("free flight: linear damping decays velocity exponentially") {
    ball::BallDynamics<double> dyn;
    ball::State<double> x;
    x << 0.0, 0.0, 1.0, 0.0;

    auto x1 = ball::exact_step(dyn.params, x, 1.0);
    // After 1 s with damping = 0.5, vx = exp(-0.5)
    CHECK(x1[ball::VX] == doctest::Approx(std::exp(-0.5)));
    CHECK(x1[ball::VY] == doctest::Approx(0.0));
    CHECK(x1[ball::PX] == doctest::Approx((1.0 - std::exp(-0.5)) / 0.5));
}

TEST_CASE("rk4 and exact_step agree on the linear ODE") {
    ball::BallDynamics<double> dyn;
    ball::State<double> x;
    x << 0.0, 0.0, 1.0, 0.5;

    const double dt = 0.01;
    const int N = 100;
    auto x_rk4 = x, x_exact = x;
    for (int i = 0; i < N; ++i) {
        x_rk4   = ball::rk4_step(dyn, x_rk4, dt);
        x_exact = ball::exact_step(dyn.params, x_exact, dt);
    }
    CHECK((x_rk4 - x_exact).norm() < 1e-7);
}

TEST_CASE("contact: head-on collision reflects with restitution") {
    ball::Params<double> params;
    params.restitution = 0.8;
    ball::State<double> x;
    x << 1.0, 0.0, -1.0, 0.0;
    ball::Vec2<double> n;
    n << 1.0, 0.0;

    auto x_after = ball::apply_contact_impulse(params, x, n);
    // vx_new = vx - (1 + e)(vx . n) n_x = -1 - 1.8 * (-1) * 1 = 0.8
    CHECK(x_after[ball::VX] == doctest::Approx(0.8));
    CHECK(x_after[ball::VY] == doctest::Approx(0.0));
    CHECK(x_after[ball::PX] == doctest::Approx(1.0));
    CHECK(x_after[ball::PY] == doctest::Approx(0.0));
}

TEST_CASE("contact: separating velocity is unchanged") {
    ball::Params<double> params;
    ball::State<double> x;
    x << 0.0, 0.0, 1.0, 0.3;
    ball::Vec2<double> n;
    n << 1.0, 0.0;  // ball moving in +x, normal in +x → already separating

    auto x_after = ball::apply_contact_impulse(params, x, n);
    CHECK((x_after - x).norm() == doctest::Approx(0.0));
}

TEST_CASE("contact: oblique elastic collision preserves tangential velocity") {
    ball::Params<double> params;
    params.restitution = 1.0;  // perfectly elastic
    ball::State<double> x;
    x << 0.0, 0.0, -1.0, 1.0;  // diagonal incoming
    ball::Vec2<double> n;
    n << 1.0, 0.0;

    auto x_after = ball::apply_contact_impulse(params, x, n);
    CHECK(x_after[ball::VX] == doctest::Approx(1.0));   // normal flipped
    CHECK(x_after[ball::VY] == doctest::Approx(1.0));   // tangential preserved
}
