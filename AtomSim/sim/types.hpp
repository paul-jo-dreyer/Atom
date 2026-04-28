#pragma once

#include "ball/core/types.hpp"
#include "diff_drive/core/types.hpp"

#include <array>
#include <cstdint>
#include <vector>

namespace sim {

// Top-level world configuration. Field is centred at the origin.
// Defaults match the agreed soccer-field dimensions: 0.75 m goal-to-goal,
// 0.45 m wide.
//
// Goals open through the left and right field walls — a gap of half-height
// `goal_y_half` centred on y=0, with a chamber extending `goal_extension`
// behind the wall. The goal chamber has its own walls (top, bottom, back)
// in CATEGORY_GOAL_WALL so robots and balls collide with them physically.
// Set goal_y_half = 0 OR goal_extension = 0 to disable goals (continuous
// solid left/right walls).
struct WorldConfig {
    float field_x_half   = 0.375f;   // half of 0.75 m, the goal-to-goal length
    float field_y_half   = 0.225f;   // half of 0.45 m, the field width
    float goal_y_half    = 0.06f;    // half-height of goal opening (0 = no goals)
    float goal_extension = 0.06f;    // depth of goal box behind the wall (0 = no goals)
    float gravity_x      = 0.0f;
    float gravity_y      = 0.0f;     // top-down planar simulation, no gravity
    int   substeps       = 4;        // Box2D 3.x TGS substep count per Step()
};

// Length-scaling factor between user-facing units (meters) and Box2D's
// internal units. We multiply by this when crossing INTO Box2D, divide
// when reading OUT.
//
// Why: Box2D 3.x is tuned for objects in the 0.1–10 m range and bakes a
// 5 mm linear slop into hull computation, contact tolerance, and the
// solver. Our manipulator polygons have features at 5–20 mm, so they get
// silently welded into degeneracy and dropped. Scaling by 10× makes the
// 5 mm-min user feature 50 mm in Box2D's view, comfortably above its 20 mm
// vertex-welding threshold. The scale is purely a Box2D-side convention
// — user inputs/outputs and `core/` stay in meters.
constexpr float kBox2dScale = 10.0f;

// Box2D contact-filter category bits. Each kind of body is one bit; mask bits
// say which categories a body collides with.
//
// Field-perimeter walls (CATEGORY_WALL) collide only with robots — the ball
// passes through them (kept in by the soft pull-back force in Ball::pre_step).
// Goal-box walls (CATEGORY_GOAL_WALL) collide with BOTH the ball and robots,
// since they need to physically bound the goal chamber once a ball enters.
constexpr uint64_t CATEGORY_WALL      = 1ull << 0;
constexpr uint64_t CATEGORY_ROBOT     = 1ull << 1;
constexpr uint64_t CATEGORY_BALL      = 1ull << 2;
constexpr uint64_t CATEGORY_GOAL_WALL = 1ull << 3;

constexpr uint64_t MASK_WALL      = CATEGORY_ROBOT;
constexpr uint64_t MASK_ROBOT     = CATEGORY_WALL | CATEGORY_BALL | CATEGORY_ROBOT | CATEGORY_GOAL_WALL;
constexpr uint64_t MASK_BALL      = CATEGORY_ROBOT | CATEGORY_GOAL_WALL;
constexpr uint64_t MASK_GOAL_WALL = CATEGORY_ROBOT | CATEGORY_BALL;

// Whether a robot's Box2D body is integrated by the solver (dynamic) or has
// its pose imposed externally each step (kinematic). See Phase 1/2 design
// notes.
enum class BodyType {
    Kinematic,
    Dynamic,
};

// Per-robot configuration. `manipulator_parts` is the list of convex
// polygons (each polygon a list of (x, y) vertices in body-local frame) that
// the polygon designer notebook produces.
struct RobotConfig {
    BodyType body_type = BodyType::Kinematic;

    float chassis_side = 0.10f;     // m, side length of the square chassis
    float mass         = 0.5f;      // kg  — only used if body_type == Dynamic
    float yaw_inertia  = 1.0e-3f;   // kg·m² — only used if body_type == Dynamic

    // Initial pose in world frame.
    float x0     = 0.0f;
    float y0     = 0.0f;
    float theta0 = 0.0f;

    // Manipulator shape: zero or more convex polygons in body-local coords.
    std::vector<std::vector<std::array<float, 2>>> manipulator_parts;

    // Vehicle dynamics parameters (track width, motor lag).
    diff_drive::Params<float> dynamics_params{};
};

// Per-ball configuration. Ball physics live in ball::core; this struct adds
// the sim-layer-only knobs (initial state, field pull-back stiffness).
struct BallConfig {
    float x0  = 0.0f;
    float y0  = 0.0f;
    float vx0 = 0.0f;
    float vy0 = 0.0f;

    // Soft-wall stiffness for the field-centering pull-back force. Units:
    // 1/s² — acceleration applied per metre of out-of-bounds penetration.
    // Engages only when |p| > field_half on the corresponding axis.
    float field_k = 50.0f;

    // Ball physics parameters (radius, mass, restitution, damping).
    ::ball::Params<float> dynamics_params{};
};

}  // namespace sim
