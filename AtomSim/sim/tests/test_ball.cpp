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
