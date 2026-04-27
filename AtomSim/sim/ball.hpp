#pragma once

#include "types.hpp"
#include "world.hpp"

#include "ball/core/types.hpp"

#include <box2d/box2d.h>

#include <array>
#include <vector>

namespace sim {

// A passive ball in the simulated world. The ball's `core/` is the source of
// truth for state — Box2D is used only for contact detection. Each step:
//
//   1. pre_step(dt):
//      a. Apply pending contact impulses from the previous step's
//         post_step() (one-frame deferred — invisible at dt = 10 ms).
//      b. Apply the soft field-centering force when out of bounds (no
//         bouncing on walls — the user's design choice for smoother
//         training gradients).
//      c. Integrate via ball::exact_step (closed-form, linear damping).
//      d. Push the new pose to Box2D as a kinematic transform.
//
//   2. world.step(dt) — Box2D detects contacts.
//
//   3. post_step():
//      Read contact-begin events from Box2D, store the relevant unit normals
//      (pointing into the ball) for the next pre_step's impulse application.
//
// The ball's Box2D body has CATEGORY_BALL / MASK_BALL filter, so it never
// sees the field walls — by design. It DOES collide with robots; those are
// the contacts the impulse model handles.
class Ball {
public:
    Ball(World& world, const BallConfig& cfg);
    ~Ball();

    Ball(const Ball&)            = delete;
    Ball& operator=(const Ball&) = delete;
    Ball(Ball&&)                 = delete;
    Ball& operator=(Ball&&)      = delete;

    void pre_step(float dt);
    void post_step();

    const ::ball::State<float>& state() const { return state_; }
    void set_state(const ::ball::State<float>& s);

    const BallConfig& config()  const { return cfg_; }
    b2BodyId          body_id() const { return body_id_; }

private:
    World*                              world_;
    BallConfig                          cfg_;
    ::ball::State<float>                state_{::ball::State<float>::Zero()};
    b2BodyId                            body_id_;
    b2ShapeId                           shape_id_;

    // Pending contacts collected in post_step(), applied at the start of the
    // next pre_step(). Each holds the contact unit normal (pointing INTO the
    // ball) and the world-frame velocity of the OTHER body at contact time.
    // Storing the other's velocity is what lets the impulse formula handle
    // the case where the ball is stationary and the robot is moving — we
    // Galilean-shift to the other's frame, apply the asymmetric impulse,
    // shift back.
    struct PendingContact {
        std::array<float, 2> normal;
        std::array<float, 2> other_velocity;
    };
    std::vector<PendingContact> pending_contacts_;
};

}  // namespace sim
