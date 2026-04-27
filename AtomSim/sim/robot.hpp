#pragma once

#include "types.hpp"
#include "world.hpp"

#include "diff_drive/core/dynamics.hpp"
#include "diff_drive/core/types.hpp"

#include <box2d/box2d.h>

namespace sim {

// A diff-drive robot in the simulated world. Composes
// `diff_drive::DiffDriveDynamics` + a Box2D body with a square chassis and
// (optionally) one or more manipulator shapes attached.
//
// `body_type` in the RobotConfig selects the Box2D integration mode:
//
//   Kinematic — the robot's `core/` is the source of truth for body twist;
//   Box2D doesn't push the body. Each pre_step() advances the core via RK4
//   and pushes the new pose to Box2D as a kinematic-body transform. Robot-
//   robot contacts are detected (contact events still fire) but not resolved
//   physically — both bodies inter-penetrate and the trainer is expected to
//   penalize/avoid in software.
//
//   Dynamic — Box2D is the source of truth for body twist post-contact. Each
//   pre_step() reads (v, ω) from Box2D, computes commanded body twist from
//   the wheel command via diff_drive's inverse kinematics, applies a force-
//   based motor lag (F = m·(v_cmd − v)/τ, T = I·(ω_cmd − ω)/τ), and clamps
//   the body-frame lateral velocity to zero (no-slip enforcement). Robot-
//   robot collisions are then physically resolved by Box2D's solver during
//   World::step.
//
// Use pattern (one robot, single step):
//   robot.pre_step(wheel_cmd, dt);   // advance core OR apply forces
//   world.step(dt);                  // Box2D step
//   robot.post_step();               // pull state back from Box2D (dynamic)
class Robot {
public:
    Robot(World& world, const RobotConfig& cfg);
    ~Robot();

    Robot(const Robot&)            = delete;
    Robot& operator=(const Robot&) = delete;
    Robot(Robot&&)                 = delete;
    Robot& operator=(Robot&&)      = delete;

    // Stage 1 of a step: act on the wheel command. In kinematic mode this
    // also advances the core's state and pushes the new pose to Box2D. In
    // dynamic mode it applies forces to the body but does NOT integrate —
    // call World::step(dt) next, then post_step().
    void pre_step(const diff_drive::Control<float>& wheel_cmd, float dt);

    // Stage 2 of a step: pull Box2D's body state back into core (dynamic
    // only; no-op for kinematic).
    void post_step();

    // Current 5-D state, reflecting the most recent pre_step / post_step.
    const diff_drive::State<float>& state() const { return state_; }

    // Reset the state and Box2D body pose simultaneously.
    void set_state(const diff_drive::State<float>& s);

    BodyType body_type() const { return cfg_.body_type; }
    b2BodyId body_id()   const { return body_id_; }

    const RobotConfig& config() const { return cfg_; }

private:
    void create_body(World& world);
    void attach_chassis_shape();
    void attach_manipulator_shapes();
    void clamp_lateral_velocity();

    RobotConfig                          cfg_;
    diff_drive::DiffDriveDynamics<float> dyn_;
    diff_drive::State<float>             state_{diff_drive::State<float>::Zero()};
    b2BodyId                             body_id_;
};

}  // namespace sim
