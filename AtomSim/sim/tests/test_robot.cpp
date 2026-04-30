#include <doctest/doctest.h>

#include "robot.hpp"
#include "world.hpp"

#include <cmath>

namespace {

sim::RobotConfig kinematic_default() {
    sim::RobotConfig cfg;
    cfg.body_type    = sim::BodyType::Kinematic;
    cfg.chassis_side = 0.06f;
    cfg.dynamics_params.track_width = 0.10f;
    cfg.dynamics_params.tau_motor   = 0.05f;
    // simple flat paddle, body-local
    cfg.manipulator_parts = {{
        {0.03f, -0.015f},
        {0.04f, -0.015f},
        {0.04f,  0.015f},
        {0.03f,  0.015f},
    }};
    return cfg;
}

sim::RobotConfig dynamic_default() {
    auto cfg = kinematic_default();
    cfg.body_type   = sim::BodyType::Dynamic;
    cfg.mass        = 0.3f;
    cfg.yaw_inertia = 5.0e-4f;
    return cfg;
}

}  // namespace

TEST_CASE("Robot (kinematic) advances forward under symmetric command") {
    sim::World world{};
    sim::Robot robot(world, kinematic_default());

    diff_drive::Control<float> u;
    u << 0.5f, 0.5f;

    const float dt = 0.01f;
    for (int i = 0; i < 50; ++i) {
        robot.pre_step(u, dt);
        world.step(dt);
        robot.post_step();
    }

    const auto& s = robot.state();
    CHECK(s[diff_drive::PX] > 0.05f);                       // moved forward
    CHECK(std::abs(s[diff_drive::PY])    < 1.0e-4f);        // no lateral drift
    CHECK(std::abs(s[diff_drive::THETA]) < 1.0e-4f);        // no yaw
    CHECK(s[diff_drive::V]               > 0.0f);
    CHECK(std::abs(s[diff_drive::OMEGA]) < 1.0e-4f);
}

TEST_CASE("Robot (kinematic) yaws under differential command") {
    sim::World world{};
    sim::Robot robot(world, kinematic_default());

    diff_drive::Control<float> u;
    u << -0.5f, 0.5f;   // pure CCW rotation

    const float dt = 0.01f;
    for (int i = 0; i < 100; ++i) {
        robot.pre_step(u, dt);
        world.step(dt);
        robot.post_step();
    }

    const auto& s = robot.state();
    CHECK(s[diff_drive::THETA] > 0.0f);                     // yawed CCW
    CHECK(std::abs(s[diff_drive::PX]) < 1.0e-4f);           // no translation
    CHECK(std::abs(s[diff_drive::PY]) < 1.0e-4f);
}

TEST_CASE("Robot (dynamic) advances forward and tracks v_cmd asymptotically") {
    // 50 steps = 0.5 s = 10 τ. Plenty for the motor lag to settle, well
    // before the robot reaches the wall at x = +0.375 m (would take ~0.7 s
    // at full v_cmd, plus the chassis half-side and manipulator reach).
    sim::World world{};
    sim::Robot robot(world, dynamic_default());

    diff_drive::Control<float> u;
    u << 0.5f, 0.5f;

    const float dt = 0.01f;
    for (int i = 0; i < 50; ++i) {
        robot.pre_step(u, dt);
        world.step(dt);
        robot.post_step();
    }

    const auto& s = robot.state();
    CHECK(s[diff_drive::PX] > 0.15f);                       // moved well forward
    CHECK(s[diff_drive::V]  > 0.45f);                       // body velocity within ~10% of v_cmd
    CHECK(s[diff_drive::V]  < 0.55f);
    CHECK(std::abs(s[diff_drive::PY])    < 1.0e-3f);        // no lateral drift (clamped)
    CHECK(std::abs(s[diff_drive::OMEGA]) < 1.0e-3f);
}

TEST_CASE("Robot (dynamic) physically stops at a wall") {
    // Drive forward into the right wall and verify the body comes to rest
    // (not bouncing wildly, not phasing through). This exercises Box2D's
    // contact resolution — the whole point of going dynamic.
    //
    // Y-offset by 0.10 m so the robot impacts the upper-right wall segment
    // rather than passing through the goal mouth (default goal_y_half=0.06).
    sim::World world{};
    auto cfg = dynamic_default();
    cfg.y0 = 0.10f;
    sim::Robot robot(world, cfg);

    diff_drive::Control<float> u;
    u << 0.5f, 0.5f;

    const float dt = 0.01f;
    for (int i = 0; i < 200; ++i) {
        robot.pre_step(u, dt);
        world.step(dt);
        robot.post_step();
    }

    const auto& s = robot.state();
    CHECK(s[diff_drive::PX] < 0.375f);                       // didn't tunnel through
    CHECK(s[diff_drive::PX] > 0.30f);                        // got close to the wall
    CHECK(s[diff_drive::PY] == doctest::Approx(0.10f).epsilon(0.05));  // no lateral drift
    CHECK(std::abs(s[diff_drive::V]) < 1.0e-3f);             // came to rest
}

TEST_CASE("Robot (dynamic) reports wall contact via contact_points()") {
    // Same scenario as "physically stops at a wall": drive straight into
    // the upper-right field wall (y0=0.10 keeps us out of the goal mouth).
    // Once the chassis is jammed, we expect at least one contact point
    // tagged with CATEGORY_WALL, with non-zero normal impulse and a
    // surface normal pointing roughly back along -x (force-on-us
    // convention: the wall is shoving the robot in -x).
    sim::World world{};
    auto cfg = dynamic_default();
    cfg.y0 = 0.10f;
    sim::Robot robot(world, cfg);

    diff_drive::Control<float> u;
    u << 0.5f, 0.5f;  // both wheels forward → drive straight at the wall

    const float dt = 0.01f;
    for (int i = 0; i < 200; ++i) {
        robot.pre_step(u, dt);
        world.step(dt);
        robot.post_step();
    }

    const auto contacts = robot.contact_points();
    REQUIRE(!contacts.empty());

    bool saw_wall_contact = false;
    for (const auto& c : contacts) {
        if (!(c.other_category & sim::CATEGORY_WALL)) continue;
        saw_wall_contact = true;

        // Force-on-us convention: normal should point AWAY from the wall
        // (robot is at x ≈ +0.34 m, wall is at x = +0.375 m, so the wall
        // pushes us in the -x direction).
        CHECK(c.normal_x < -0.5f);                         // mostly -x
        CHECK(std::abs(c.normal_y) < 0.5f);                // not y-aligned

        // Active interaction this step → totalNormalImpulse > 0.
        CHECK(c.normal_impulse > 0.0f);

        // Contact point should be at (or just past) the right wall x.
        // Allow a little slack for Box2D's slop tolerance.
        CHECK(c.point_x > 0.35f);
        CHECK(c.point_x < 0.40f);

        // Penetration depth: in steady-state pinning, separation is small
        // negative (a sliver of overlap). Mainly check it's not wildly out
        // of the expected unit range.
        CHECK(c.separation < 0.01f);
        CHECK(c.separation > -0.05f);
    }
    CHECK(saw_wall_contact);
}

TEST_CASE("Robot (dynamic) reports no contacts mid-field") {
    // Sanity check the "empty" case — drive forward briefly, well inside
    // the field. No contacts should be reported.
    sim::World world{};
    sim::Robot robot(world, dynamic_default());

    diff_drive::Control<float> u;
    u << 0.2f, 0.2f;

    const float dt = 0.01f;
    for (int i = 0; i < 20; ++i) {
        robot.pre_step(u, dt);
        world.step(dt);
        robot.post_step();
    }

    CHECK(robot.contact_points().empty());
}

TEST_CASE("Robot (kinematic) honors set_state") {
    sim::World world{};
    sim::Robot robot(world, kinematic_default());

    diff_drive::State<float> s;
    s << 0.1f, -0.05f, 1.0f, 0.0f, 0.0f;
    robot.set_state(s);

    CHECK(robot.state()[diff_drive::PX]    == doctest::Approx(0.1f));
    CHECK(robot.state()[diff_drive::PY]    == doctest::Approx(-0.05f));
    CHECK(robot.state()[diff_drive::THETA] == doctest::Approx(1.0f));

    // Box2D's pose is in scaled units (user metres × kBox2dScale).
    const b2Transform xf = b2Body_GetTransform(robot.body_id());
    CHECK(xf.p.x == doctest::Approx(0.1f * sim::kBox2dScale));
    CHECK(xf.p.y == doctest::Approx(-0.05f * sim::kBox2dScale));
}
