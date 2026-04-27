#include "ball.hpp"

#include "ball/core/contact.hpp"
#include "ball/core/integrators.hpp"

namespace sim {

namespace {

inline bool body_id_equals(b2BodyId a, b2BodyId b) {
    return a.index1 == b.index1 && a.world0 == b.world0 && a.generation == b.generation;
}

}  // namespace

Ball::Ball(World& world, const BallConfig& cfg) : world_(&world), cfg_(cfg) {
    state_[::ball::PX] = cfg_.x0;
    state_[::ball::PY] = cfg_.y0;
    state_[::ball::VX] = cfg_.vx0;
    state_[::ball::VY] = cfg_.vy0;

    // Box2D 3.x's broad phase only generates contact pairs where at least
    // one body is dynamic ("Only dynamic proxies collide with kinematic and
    // static proxies"). To get contact events for a kinematic-controlled
    // ball, the body itself must be dynamic — but we still override its
    // pose every step from our own `core/` integrator, so Box2D's solver
    // never gets to actually move the ball. Set linearDamping/gravityScale
    // to zero so Box2D's integration agrees with us between overrides.
    b2BodyDef body_def = b2DefaultBodyDef();
    body_def.type           = b2_dynamicBody;
    body_def.position       = {cfg_.x0, cfg_.y0};
    body_def.linearDamping  = 0.0f;
    body_def.gravityScale   = 0.0f;
    body_id_ = b2CreateBody(world.world_id(), &body_def);

    b2Circle circle;
    circle.center = {0.0f, 0.0f};
    circle.radius = cfg_.dynamics_params.radius;

    b2ShapeDef shape_def = b2DefaultShapeDef();
    shape_def.filter.categoryBits = CATEGORY_BALL;
    shape_def.filter.maskBits     = MASK_BALL;
    shape_def.density             = 0.0f;
    shape_def.updateBodyMass      = false;
    shape_def.enableContactEvents = true;   // we listen for these in post_step
    shape_id_ = b2CreateCircleShape(body_id_, &shape_def, &circle);

    // Set mass explicitly. Required because a zero-mass dynamic body is
    // illegal; the value itself doesn't drive our physics (we override
    // pose/velocity each step) but does affect Box2D's contact response —
    // which we ignore anyway since `core/` is the source of truth.
    b2MassData md{};
    md.mass              = cfg_.dynamics_params.mass;
    md.center            = {0.0f, 0.0f};
    md.rotationalInertia = 1.0e-6f;   // small, never used
    b2Body_SetMassData(body_id_, md);
}

Ball::~Ball() {
    if (b2Body_IsValid(body_id_)) {
        b2DestroyBody(body_id_);
    }
}

void Ball::pre_step(float dt) {
    // 1. Apply pending contact impulses from previous frame. For each
    //    contact, Galilean-shift to the other body's frame, apply the
    //    asymmetric impulse (which assumes the other is immovable), then
    //    shift back. This lets us handle ball-into-robot AND robot-into-
    //    ball uniformly.
    for (const auto& pc : pending_contacts_) {
        ::ball::State<float> shifted = state_;
        shifted[::ball::VX] -= pc.other_velocity[0];
        shifted[::ball::VY] -= pc.other_velocity[1];

        ::ball::Vec2<float> normal;
        normal[0] = pc.normal[0];
        normal[1] = pc.normal[1];
        shifted = ::ball::apply_contact_impulse(cfg_.dynamics_params, shifted, normal);

        state_[::ball::VX] = shifted[::ball::VX] + pc.other_velocity[0];
        state_[::ball::VY] = shifted[::ball::VY] + pc.other_velocity[1];
    }
    pending_contacts_.clear();

    // 2. Soft field-centering force outside the bounds. Linear restoring
    //    acceleration: a = -k * penetration along the violated axis. Applied
    //    as a discrete Euler velocity kick, which is fine since the field
    //    force is zero whenever the ball is in bounds (the common case).
    const float xh = world_->config().field_x_half;
    const float yh = world_->config().field_y_half;
    float ax = 0.0f, ay = 0.0f;
    if      (state_[::ball::PX] >  xh) ax = -cfg_.field_k * (state_[::ball::PX] - xh);
    else if (state_[::ball::PX] < -xh) ax = -cfg_.field_k * (state_[::ball::PX] + xh);
    if      (state_[::ball::PY] >  yh) ay = -cfg_.field_k * (state_[::ball::PY] - yh);
    else if (state_[::ball::PY] < -yh) ay = -cfg_.field_k * (state_[::ball::PY] + yh);
    state_[::ball::VX] += ax * dt;
    state_[::ball::VY] += ay * dt;

    // 3. Integrate the linear-damping ODE via the closed-form step.
    state_ = ::ball::exact_step(cfg_.dynamics_params, state_, dt);

    // 4. Push the new pose AND velocity to Box2D so collision detection sees
    //    a self-consistent state. Velocity matters here because Box2D uses
    //    it to expand swept AABBs in broad phase — without it, fast-moving
    //    balls might have contacts missed.
    b2Body_SetTransform(body_id_,
                        {state_[::ball::PX], state_[::ball::PY]},
                        b2MakeRot(0.0f));
    b2Body_SetLinearVelocity(body_id_, {state_[::ball::VX], state_[::ball::VY]});
}

void Ball::post_step() {
    pending_contacts_.clear();

    b2ContactEvents events = b2World_GetContactEvents(world_->world_id());
    for (int i = 0; i < events.beginCount; ++i) {
        const b2ContactBeginTouchEvent& e = events.beginEvents[i];
        const b2BodyId body_a = b2Shape_GetBody(e.shapeIdA);
        const b2BodyId body_b = b2Shape_GetBody(e.shapeIdB);
        const bool ball_is_a = body_id_equals(body_a, body_id_);
        const bool ball_is_b = body_id_equals(body_b, body_id_);
        if (!ball_is_a && !ball_is_b) {
            continue;
        }
        // Manifold normal points from shape A to shape B by Box2D convention.
        // For the impulse formula we need the normal pointing INTO the ball.
        b2Vec2 n = e.manifold.normal;
        if (ball_is_a) {
            n.x = -n.x;
            n.y = -n.y;
        }
        const b2BodyId other_body = ball_is_a ? body_b : body_a;
        const b2Vec2   v_other    = b2Body_GetLinearVelocity(other_body);
        pending_contacts_.push_back({{n.x, n.y}, {v_other.x, v_other.y}});
    }
}

void Ball::set_state(const ::ball::State<float>& s) {
    state_ = s;
    b2Body_SetTransform(body_id_,
                        {state_[::ball::PX], state_[::ball::PY]},
                        b2MakeRot(0.0f));
    b2Body_SetLinearVelocity(body_id_, {state_[::ball::VX], state_[::ball::VY]});
}

}  // namespace sim
