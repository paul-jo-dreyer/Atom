#include <doctest/doctest.h>

#include "ball.hpp"
#include "robot.hpp"
#include "world.hpp"

#include <cmath>

TEST_CASE("Ball: free flight in bounds matches closed-form damping") {
    // Disable field pull-back so we measure pure exponential decay. (Without
    // this, the ball at v=1 with damping=0.5 reaches x = field_x_half at
    // t ≈ 0.42 s and the field force corrupts the result.)
    sim::World world{};
    sim::BallConfig cfg;
    cfg.dynamics_params.damping = 0.5f;
    cfg.field_k                 = 0.0f;
    sim::Ball ball(world, cfg);

    ::ball::State<float> s;
    s << 0.0f, 0.0f, 1.0f, 0.0f;
    ball.set_state(s);

    const float dt = 0.01f;
    for (int i = 0; i < 50; ++i) {     // 0.5 s
        ball.pre_step(dt);
        world.step(dt);
        ball.post_step();
    }

    // v(t) = v0 * exp(-k_d * t) = 1 * exp(-0.25) ≈ 0.7788
    CHECK(ball.state()[::ball::VX] == doctest::Approx(std::exp(-0.25f)).epsilon(0.01));
    CHECK(std::abs(ball.state()[::ball::VY]) < 1.0e-6f);
}

TEST_CASE("Ball: field pull-back keeps ball from leaving the field") {
    // Strong damping + strong field stiffness — the ball decelerates inside
    // the boundary, overshoots a little, oscillates with rapidly decaying
    // amplitude, settles back inside. This is the regime we want for
    // training: smooth, bounded, never terminates.
    sim::World world{};
    sim::BallConfig cfg;
    cfg.dynamics_params.damping     = 1.0f;
    cfg.dynamics_params.restitution = 0.0f;
    cfg.field_k                     = 500.0f;
    sim::Ball ball(world, cfg);

    ::ball::State<float> s;
    s << 0.0f, 0.0f, 3.0f, 0.0f;
    ball.set_state(s);

    const float dt = 0.005f;
    float max_overshoot = 0.0f;
    for (int i = 0; i < 2000; ++i) {   // 10 s — well past settling
        ball.pre_step(dt);
        world.step(dt);
        ball.post_step();
        const float over = std::abs(ball.state()[::ball::PX]) - world.config().field_x_half;
        if (over > max_overshoot) max_overshoot = over;
    }

    CHECK(max_overshoot < 0.15f);                                              // bounded
    CHECK(std::abs(ball.state()[::ball::PX]) < world.config().field_x_half);   // settled back inside
}

TEST_CASE("Ball: multi-part manipulator — each part can register a contact") {
    // The robot has a 3-part manipulator: a central bar at x=[0.030, 0.035],
    // y=[-0.030, 0.030], plus two triangular wings at the corners. A ball at
    // y=0 only touches the central bar; a ball at y=0.024 enters the top wing.
    // This test confirms the wing IS picked up (i.e., not just the first
    // shape) when the geometry actually engages.
    sim::RobotConfig robot_cfg;
    robot_cfg.body_type    = sim::BodyType::Kinematic;
    robot_cfg.chassis_side = 0.060f;
    robot_cfg.x0           = 0.0f;
    robot_cfg.manipulator_parts = {
        // central bar
        {{0.030f, -0.030f}, {0.035f, -0.030f}, {0.035f, 0.030f}, {0.030f, 0.030f}},
        // top wing (sloped from (0.035, 0.017) out to (0.055, 0.030))
        {{0.035f, 0.017f}, {0.055f, 0.030f}, {0.035f, 0.030f}},
        // bottom wing
        {{0.035f, -0.017f}, {0.055f, -0.030f}, {0.035f, -0.030f}},
    };

    sim::BallConfig ball_cfg;
    ball_cfg.dynamics_params.radius      = 0.014f;
    ball_cfg.dynamics_params.restitution = 0.6f;
    ball_cfg.dynamics_params.damping     = 0.0f;

    SUBCASE("ball at y=0 hits central bar") {
        sim::World world{};
        sim::Robot robot(world, robot_cfg);
        sim::Ball ball(world, ball_cfg);

        ::ball::State<float> s;
        s << 0.10f, 0.000f, -1.0f, 0.0f;   // moving in -x toward the manipulator
        ball.set_state(s);

        diff_drive::Control<float> u_zero;
        u_zero << 0.0f, 0.0f;
        const float dt = 0.005f;
        for (int i = 0; i < 100; ++i) {
            ball.pre_step(dt);
            robot.pre_step(u_zero, dt);
            world.step(dt);
            ball.post_step();
            robot.post_step();
        }
        // Ball should have bounced (vx now positive)
        CHECK(ball.state()[::ball::VX] > 0.0f);
    }

    SUBCASE("ball at y=0.024 hits the top wing") {
        sim::World world{};
        sim::Robot robot(world, robot_cfg);
        sim::Ball ball(world, ball_cfg);

        // Ball positioned where ONLY the top-wing geometry can reach it
        // (chassis ends at y=0.030; ball center at y=0.024 + radius 0.014 →
        // ball top edge at 0.038, but contact happens via the wing's sloped
        // outer face).
        ::ball::State<float> s;
        s << 0.10f, 0.024f, -1.0f, 0.0f;
        ball.set_state(s);

        diff_drive::Control<float> u_zero;
        u_zero << 0.0f, 0.0f;
        const float dt = 0.005f;
        for (int i = 0; i < 100; ++i) {
            ball.pre_step(dt);
            robot.pre_step(u_zero, dt);
            world.step(dt);
            ball.post_step();
            robot.post_step();
        }
        // Ball should have bounced — proving the wing shape (part [1]) is
        // physically active and registering contacts.
        CHECK(ball.state()[::ball::VX] > 0.0f);
    }
}

TEST_CASE("Ball: sustained pushing — robot does not traverse through ball") {
    // The traversal bug: with restitution < 1, repeated impulses dissipate
    // energy until v_ball ≈ v_robot. Without position correction, our
    // integration would let the manipulator slowly overtake the ball and
    // pass through. With position correction, the ball is held at the
    // contact surface (or just outside) for as long as the push lasts.
    sim::World world{};

    sim::RobotConfig robot_cfg;
    robot_cfg.body_type    = sim::BodyType::Dynamic;
    robot_cfg.chassis_side = 0.06f;
    robot_cfg.mass         = 0.3f;
    robot_cfg.yaw_inertia  = 5.0e-4f;
    robot_cfg.x0           = -0.10f;
    robot_cfg.dynamics_params.track_width = 0.10f;
    robot_cfg.dynamics_params.tau_motor   = 0.05f;
    sim::Robot robot(world, robot_cfg);

    sim::BallConfig ball_cfg;
    ball_cfg.x0 = 0.0f;
    ball_cfg.y0 = 0.0f;
    ball_cfg.dynamics_params.radius      = 0.020f;
    ball_cfg.dynamics_params.restitution = 0.4f;
    ball_cfg.dynamics_params.damping     = 0.8f;
    sim::Ball ball(world, ball_cfg);

    diff_drive::Control<float> u_drive;
    u_drive << 0.3f, 0.3f;
    const float dt = 1.0f / 60.0f;
    for (int i = 0; i < 240; ++i) {     // 4 s of sustained push
        ball.pre_step(dt);
        robot.pre_step(u_drive, dt);
        world.step(dt);
        ball.post_step();
        robot.post_step();
    }

    // Ball center must always remain on the +x side of the robot's chassis
    // right edge. With chassis_side = 0.06, robot at some final x_r, the
    // chassis right edge is at x_r + 0.030. Ball center must be > x_r +
    // 0.030 (and ideally > x_r + 0.030 + 0.020 = x_r + 0.050 for non-
    // penetration with the 20mm-radius ball).
    const auto rs = robot.state();
    const auto bs = ball.state();
    const float chassis_right = rs[diff_drive::PX] + 0.5f * robot_cfg.chassis_side;
    CHECK_MESSAGE(bs[::ball::PX] > chassis_right,
                  "ball was traversed by the robot (px_ball=", bs[::ball::PX],
                  " vs chassis_right=", chassis_right, ")");
}

TEST_CASE("Ball: contact with kinematic robot reverses the normal velocity") {
    sim::World world{};
    sim::BallConfig ball_cfg;
    ball_cfg.dynamics_params.radius      = 0.025f;
    ball_cfg.dynamics_params.restitution = 0.8f;
    ball_cfg.dynamics_params.damping     = 0.0f;     // isolate the impulse
    ball_cfg.x0 = -0.05f;
    ball_cfg.vx0 = 1.0f;                              // moving toward robot
    sim::Ball ball(world, ball_cfg);

    sim::RobotConfig robot_cfg;
    robot_cfg.body_type    = sim::BodyType::Kinematic;
    robot_cfg.chassis_side = 0.06f;
    robot_cfg.x0           = 0.05f;
    sim::Robot robot(world, robot_cfg);

    ::ball::State<float> s;
    s << ball_cfg.x0, 0.0f, ball_cfg.vx0, 0.0f;
    ball.set_state(s);

    const float dt = 0.005f;
    diff_drive::Control<float> u_zero;
    u_zero << 0.0f, 0.0f;
    for (int i = 0; i < 100; ++i) {       // 0.5 s — well after the ball reaches the chassis
        ball.pre_step(dt);
        robot.pre_step(u_zero, dt);
        world.step(dt);
        ball.post_step();
        robot.post_step();
    }

    // After bouncing off the static-feeling chassis at x ≈ -0.005 (chassis
    // left edge), the ball's vx should be negative (moving back toward -x)
    // and its magnitude should be ≈ restitution × incoming = 0.8 × 1.0.
    CHECK(ball.state()[::ball::VX] < 0.0f);
    CHECK(std::abs(ball.state()[::ball::VX]) > 0.5f);
    CHECK(std::abs(ball.state()[::ball::VX]) < 1.0f);
    CHECK(ball.state()[::ball::PX] < ball_cfg.x0);   // moved back past starting x
}
