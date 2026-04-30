#pragma once

#include "types.hpp"
#include "world.hpp"

#include "diff_drive/core/dynamics.hpp"
#include "diff_drive/core/types.hpp"

#include <box2d/box2d.h>

#include <cstdint>
#include <vector>

namespace sim {

// Snapshot of a single contact point on the robot's body. Returned by
// `Robot::contact_points()`. Coordinates and impulses are in physical units
// (metres, N·s) — the Box2D-side `kBox2dScale` factor has already been
// reversed. The `normal` vector follows a force-on-us convention: it points
// from the obstacle TOWARD the robot, so `normal * (normal_impulse / dt)`
// is the world-frame force vector applied to the robot.
//
// `normal_impulse` uses Box2D's `totalNormalImpulse`, which accumulates over
// all internal TGS substeps + restitution within one `World::step` call.
// This is the field Box2D's own docs recommend for "did an interaction
// actually happen this step", and is what reward terms should scale on.
//
// `other_category` is the bitmask of CATEGORY_* bits identifying what the
// other shape is (wall, robot, ball, goal-wall). A single contact will have
// exactly one bit set; reward code can mask against any combination.
struct RobotContactPoint {
    uint64_t other_category = 0;     // CATEGORY_* bitmask of the other shape
    float    point_x        = 0.0f;  // world-frame metres
    float    point_y        = 0.0f;
    float    normal_x       = 0.0f;  // unit vector, points obstacle → robot
    float    normal_y       = 0.0f;
    float    normal_impulse = 0.0f;  // N·s, totalNormalImpulse, last World::step
    float    tangent_impulse = 0.0f; // N·s, friction component along surface
    float    separation     = 0.0f;  // metres; negative ⟹ penetrating
};

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

    // Snapshot the contacts present on the robot's Box2D body at the moment
    // of call. Reflects whatever Box2D resolved during the most recent
    // `World::step`, so the canonical call site is AFTER `world.step(dt)`
    // (after `post_step()` is fine too — neither mutates contact state).
    //
    // Returns one entry per ManifoldPoint per active manifold; a chassis
    // wedged in a corner can produce up to ~4 entries. Speculative contact
    // points (`totalNormalImpulse == 0`) are filtered out — the returned
    // list contains only points where Box2D actually applied an impulse.
    //
    // Empty vector ⟹ no contacts this step.
    std::vector<RobotContactPoint> contact_points() const;

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
