#include "robot.hpp"

#include "diff_drive/core/integrators.hpp"

#include <cmath>

namespace sim {

Robot::Robot(World& world, const RobotConfig& cfg) : cfg_(cfg) {
    dyn_.params = cfg_.dynamics_params;

    // Initial state from config.
    state_[diff_drive::PX]    = cfg_.x0;
    state_[diff_drive::PY]    = cfg_.y0;
    state_[diff_drive::THETA] = cfg_.theta0;
    state_[diff_drive::V]     = 0.0f;
    state_[diff_drive::OMEGA] = 0.0f;

    create_body(world);
    attach_chassis_shape();
    attach_manipulator_shapes();

    // For dynamic bodies, set mass after shapes are attached so Box2D's
    // automatic mass recomputation can't override our values. See
    // attach_*_shape's `updateBodyMass = false` flag.
    //
    // Inertia scales by kBox2dScale² because I = m·r² and lengths scale by
    // kBox2dScale on the Box2D side. Mass itself is unscaled (it's mass).
    if (cfg_.body_type == BodyType::Dynamic) {
        b2MassData md{};
        md.mass              = cfg_.mass;
        md.center            = {0.0f, 0.0f};
        md.rotationalInertia = cfg_.yaw_inertia * kBox2dScale * kBox2dScale;
        b2Body_SetMassData(body_id_, md);
    }
}

Robot::~Robot() {
    if (b2Body_IsValid(body_id_)) {
        b2DestroyBody(body_id_);
    }
}

void Robot::create_body(World& world) {
    b2BodyDef def = b2DefaultBodyDef();
    def.type = (cfg_.body_type == BodyType::Dynamic) ? b2_dynamicBody : b2_kinematicBody;
    def.position = {cfg_.x0 * kBox2dScale, cfg_.y0 * kBox2dScale};
    def.rotation = b2MakeRot(cfg_.theta0);
    body_id_ = b2CreateBody(world.world_id(), &def);
}

void Robot::attach_chassis_shape() {
    const float half = 0.5f * cfg_.chassis_side * kBox2dScale;
    b2Polygon chassis = b2MakeBox(half, half);

    b2ShapeDef shape_def = b2DefaultShapeDef();
    shape_def.filter.categoryBits = CATEGORY_ROBOT;
    shape_def.filter.maskBits     = MASK_ROBOT;
    shape_def.density             = 0.0f;
    shape_def.updateBodyMass      = false;
    b2CreatePolygonShape(body_id_, &shape_def, &chassis);
}

void Robot::attach_manipulator_shapes() {
    for (const auto& part : cfg_.manipulator_parts) {
        if (part.size() < 3 || part.size() > 8) {
            continue;  // skip invalid; the loader/designer should have caught this
        }
        b2Vec2 verts[8];
        for (size_t i = 0; i < part.size(); ++i) {
            verts[i] = {part[i][0] * kBox2dScale, part[i][1] * kBox2dScale};
        }
        b2Hull hull = b2ComputeHull(verts, static_cast<int32_t>(part.size()));
        if (hull.count < 3) {
            continue;  // degenerate / collinear (unexpected after scaling); skip
        }
        b2Polygon poly = b2MakePolygon(&hull, 0.0f);

        b2ShapeDef shape_def = b2DefaultShapeDef();
        shape_def.filter.categoryBits = CATEGORY_ROBOT;
        shape_def.filter.maskBits     = MASK_ROBOT;
        shape_def.density             = 0.0f;
        shape_def.updateBodyMass      = false;
        b2CreatePolygonShape(body_id_, &shape_def, &poly);
    }
}

void Robot::pre_step(const diff_drive::Control<float>& wheel_cmd, float dt) {
    if (cfg_.body_type == BodyType::Kinematic) {
        // Source of truth: core. Advance state, push pose to Box2D (scaled).
        state_ = diff_drive::rk4_step(dyn_, state_, wheel_cmd, dt);
        b2Body_SetTransform(body_id_,
                            {state_[diff_drive::PX] * kBox2dScale,
                             state_[diff_drive::PY] * kBox2dScale},
                            b2MakeRot(state_[diff_drive::THETA]));
    } else {
        // Source of truth: Box2D. Read current state (scaled), apply force/
        // torque from the motor-lag controller, let Box2D integrate during
        // World::step.
        const b2Vec2 v_world_b   = b2Body_GetLinearVelocity(body_id_);
        const b2Rot  rot         = b2Body_GetRotation(body_id_);
        // Convert to user units (m/s) for the motor-lag math.
        const float v_long_user  = (v_world_b.x * rot.c + v_world_b.y * rot.s) / kBox2dScale;
        const float omega_b      = b2Body_GetAngularVelocity(body_id_);  // 1/s, no scaling

        const float W   = dyn_.params.track_width;
        const float tau = dyn_.params.tau_motor;

        const float v_cmd     = 0.5f * (wheel_cmd[0] + wheel_cmd[1]);
        const float omega_cmd = (wheel_cmd[1] - wheel_cmd[0]) / W;

        const float a_v_user     = (v_cmd     - v_long_user) / tau;  // body-x accel, m/s²
        const float a_omega_user = (omega_cmd - omega_b)     / tau;  // 1/s²

        // Forces and torques scale into Box2D units: F = m·a (scales by kBox2dScale),
        // T = I·α (I already scaled by kBox2dScale² in SetMassData).
        const float force_mag_b = cfg_.mass * a_v_user * kBox2dScale;
        b2Body_ApplyForceToCenter(body_id_,
                                  {force_mag_b * rot.c, force_mag_b * rot.s},
                                  true);
        b2Body_ApplyTorque(body_id_,
                           cfg_.yaw_inertia * a_omega_user * kBox2dScale * kBox2dScale,
                           true);

        // Hard no-slip enforcement: kill the lateral component of velocity
        // before the solver runs. Direction-only operation — no scaling.
        clamp_lateral_velocity();
    }
}

void Robot::post_step() {
    if (cfg_.body_type == BodyType::Kinematic) {
        return;
    }
    // Pull Box2D's post-step pose and twist (scaled) back into core's state
    // (user units).
    const b2Transform xf       = b2Body_GetTransform(body_id_);
    const b2Vec2      v_world_b = b2Body_GetLinearVelocity(body_id_);
    const float       omega_b   = b2Body_GetAngularVelocity(body_id_);

    state_[diff_drive::PX]    = xf.p.x / kBox2dScale;
    state_[diff_drive::PY]    = xf.p.y / kBox2dScale;
    state_[diff_drive::THETA] = b2Rot_GetAngle(xf.q);
    state_[diff_drive::V]     = (v_world_b.x * xf.q.c + v_world_b.y * xf.q.s) / kBox2dScale;
    state_[diff_drive::OMEGA] = omega_b;
}

void Robot::set_state(const diff_drive::State<float>& s) {
    state_ = s;
    b2Body_SetTransform(body_id_,
                        {state_[diff_drive::PX] * kBox2dScale,
                         state_[diff_drive::PY] * kBox2dScale},
                        b2MakeRot(state_[diff_drive::THETA]));
    if (cfg_.body_type == BodyType::Dynamic) {
        const b2Rot rot   = b2Body_GetRotation(body_id_);
        const float v     = state_[diff_drive::V] * kBox2dScale;
        b2Body_SetLinearVelocity(body_id_,  {v * rot.c, v * rot.s});
        b2Body_SetAngularVelocity(body_id_, state_[diff_drive::OMEGA]);
    }
}

void Robot::clamp_lateral_velocity() {
    const b2Vec2 v_world = b2Body_GetLinearVelocity(body_id_);
    const b2Rot  rot     = b2Body_GetRotation(body_id_);
    const float  v_long  = v_world.x * rot.c + v_world.y * rot.s;
    b2Body_SetLinearVelocity(body_id_, {v_long * rot.c, v_long * rot.s});
}

}  // namespace sim
