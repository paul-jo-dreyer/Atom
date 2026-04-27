#pragma once

#include "types.hpp"

namespace ball {

// Apply a contact impulse to the ball's velocity, reflecting the normal
// component with coefficient of restitution `params.restitution`. Tangential
// component passes through unchanged (frictionless contact). No-op if the
// ball is already separating from the surface (v . n >= 0).
//
// `n` must be a unit-length outward normal pointing from the contact surface
// into the ball. The asymmetric formula assumes the contacted body is
// immovable (static wall, kinematic robot). For ball-vs-ball contact a
// symmetric two-body formula is needed and lives elsewhere.
template <typename Scalar>
State<Scalar> apply_contact_impulse(const Params<Scalar>& params,
                                    const State<Scalar>& x,
                                    const Vec2<Scalar>& n) {
    const Scalar vn = x[VX] * n[0] + x[VY] * n[1];
    if (vn >= Scalar(0)) {
        return x;
    }
    const Scalar dv = -(Scalar(1) + params.restitution) * vn;
    State<Scalar> x_after = x;
    x_after[VX] += dv * n[0];
    x_after[VY] += dv * n[1];
    return x_after;
}

}  // namespace ball
