#include "ball.hpp"

#include "ball/core/contact.hpp"
#include "ball/core/integrators.hpp"

#include <limits>

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
    body_def.position       = {cfg_.x0 * kBox2dScale, cfg_.y0 * kBox2dScale};
    body_def.linearDamping  = 0.0f;
    body_def.gravityScale   = 0.0f;
    body_id_ = b2CreateBody(world.world_id(), &body_def);

    b2Circle circle;
    circle.center = {0.0f, 0.0f};
    circle.radius = cfg_.dynamics_params.radius * kBox2dScale;

    b2ShapeDef shape_def = b2DefaultShapeDef();
    shape_def.filter.categoryBits = CATEGORY_BALL;
    shape_def.filter.maskBits     = MASK_BALL;
    shape_def.density             = 0.0f;
    shape_def.updateBodyMass      = false;
    shape_def.enableContactEvents = true;   // (no longer used; we poll instead)
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
    //    shift back. This handles ball-into-robot AND robot-into-ball
    //    uniformly. All velocities here are in user (m/s); the Box2D-side
    //    scaling happens at the SetLinearVelocity call below.
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

    // 1b. Position correction: push the ball out of any penetration with
    //     other bodies, using the contact data Box2D produced last step.
    //     Without this, sustained contact causes traversal — the impulse
    //     formula loses energy (restitution < 1), so ball velocity decays
    //     toward the pusher's velocity, and our integration alone cannot
    //     guarantee non-penetration. Separation comes from Box2D in scaled
    //     units; we divide by kBox2dScale to push state in user metres.
    {
        const int capacity = b2Body_GetContactCapacity(body_id_);
        if (capacity > 0) {
            std::vector<b2ContactData> contacts(static_cast<std::size_t>(capacity));
            const int n = b2Body_GetContactData(body_id_, contacts.data(), capacity);
            for (int i = 0; i < n; ++i) {
                const b2ContactData& cd = contacts[static_cast<std::size_t>(i)];
                float min_sep = std::numeric_limits<float>::max();
                for (int j = 0; j < cd.manifold.pointCount; ++j) {
                    const float s = cd.manifold.points[j].separation;
                    if (s < min_sep) {
                        min_sep = s;
                    }
                }
                if (min_sep >= 0.0f) {
                    continue;  // speculative-only, no actual penetration
                }
                const b2BodyId body_a   = b2Shape_GetBody(cd.shapeIdA);
                const bool     ball_is_a = body_id_equals(body_a, body_id_);
                b2Vec2 n = cd.manifold.normal;
                if (ball_is_a) {
                    n.x = -n.x;
                    n.y = -n.y;
                }
                const float depth_user = (-min_sep) / kBox2dScale;
                state_[::ball::PX] += depth_user * n.x;
                state_[::ball::PY] += depth_user * n.y;
            }
        }
    }

    // 2. Soft field-centering force outside the bounds. Linear restoring
    //    acceleration: a = -k * penetration along the violated axis. Applied
    //    as a discrete Euler velocity kick in user units (m/s, m).
    const float xh = world_->config().field_x_half;
    const float yh = world_->config().field_y_half;
    float ax = 0.0f, ay = 0.0f;
    if      (state_[::ball::PX] >  xh) ax = -cfg_.field_k * (state_[::ball::PX] - xh);
    else if (state_[::ball::PX] < -xh) ax = -cfg_.field_k * (state_[::ball::PX] + xh);
    if      (state_[::ball::PY] >  yh) ay = -cfg_.field_k * (state_[::ball::PY] - yh);
    else if (state_[::ball::PY] < -yh) ay = -cfg_.field_k * (state_[::ball::PY] + yh);
    state_[::ball::VX] += ax * dt;
    state_[::ball::VY] += ay * dt;

    // 3. Integrate the linear-damping ODE via the closed-form step (user units).
    state_ = ::ball::exact_step(cfg_.dynamics_params, state_, dt);

    // 4. Push pose AND velocity to Box2D, both scaled. Velocity matters for
    //    Box2D's broad-phase swept AABB expansion.
    b2Body_SetTransform(body_id_,
                        {state_[::ball::PX] * kBox2dScale,
                         state_[::ball::PY] * kBox2dScale},
                        b2MakeRot(0.0f));
    b2Body_SetLinearVelocity(body_id_,
                             {state_[::ball::VX] * kBox2dScale,
                              state_[::ball::VY] * kBox2dScale});
}

void Ball::post_step() {
    pending_contacts_.clear();

    // Poll all current contacts on the ball body. Filter on manifold's
    // `separation` to ignore Box2D's speculative-collision skin — only
    // react to actual penetration.
    //
    // We queue an impulse on EVERY frame penetration is active. The impulse
    // formula is self-limiting (no-op when v_rel · n ≥ 0), so re-applying
    // each frame doesn't over-energize but DOES handle sustained-push as
    // damping decays the ball's velocity back below the pusher's.
    const int capacity = b2Body_GetContactCapacity(body_id_);
    if (capacity == 0) {
        return;
    }
    std::vector<b2ContactData> contacts(static_cast<std::size_t>(capacity));
    const int n_contacts = b2Body_GetContactData(body_id_, contacts.data(), capacity);

    for (int i = 0; i < n_contacts; ++i) {
        const b2ContactData& cd = contacts[static_cast<std::size_t>(i)];

        float min_sep = std::numeric_limits<float>::max();
        for (int j = 0; j < cd.manifold.pointCount; ++j) {
            const float s = cd.manifold.points[j].separation;
            if (s < min_sep) {
                min_sep = s;
            }
        }
        if (min_sep >= 0.0f) {
            continue;
        }

        const b2BodyId body_a   = b2Shape_GetBody(cd.shapeIdA);
        const b2BodyId body_b   = b2Shape_GetBody(cd.shapeIdB);
        const bool     ball_is_a = body_id_equals(body_a, body_id_);
        const bool     ball_is_b = body_id_equals(body_b, body_id_);
        if (!ball_is_a && !ball_is_b) {
            continue;
        }

        b2Vec2 n = cd.manifold.normal;
        if (ball_is_a) {
            n.x = -n.x;
            n.y = -n.y;
        }
        const b2BodyId other_body = ball_is_a ? body_b : body_a;
        const b2Vec2   v_other_b  = b2Body_GetLinearVelocity(other_body);
        // Convert other body's velocity to user units before storing — the
        // Galilean shift in pre_step subtracts it from `state_` velocities,
        // which are user-units.
        pending_contacts_.push_back({
            {n.x, n.y},
            {v_other_b.x / kBox2dScale, v_other_b.y / kBox2dScale}
        });
    }
}

void Ball::set_state(const ::ball::State<float>& s) {
    state_ = s;
    b2Body_SetTransform(body_id_,
                        {state_[::ball::PX] * kBox2dScale,
                         state_[::ball::PY] * kBox2dScale},
                        b2MakeRot(0.0f));
    b2Body_SetLinearVelocity(body_id_,
                             {state_[::ball::VX] * kBox2dScale,
                              state_[::ball::VY] * kBox2dScale});
}

}  // namespace sim
