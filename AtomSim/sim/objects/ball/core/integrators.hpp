#pragma once

#include "types.hpp"

#include <cmath>

namespace ball {

template <typename Dynamics, typename Scalar>
State<Scalar> rk4_step(const Dynamics& dyn,
                       const State<Scalar>& x,
                       Scalar dt) {
    const State<Scalar> k1 = dyn.continuous(x);
    const State<Scalar> k2 = dyn.continuous((x + (dt / Scalar(2)) * k1).eval());
    const State<Scalar> k3 = dyn.continuous((x + (dt / Scalar(2)) * k2).eval());
    const State<Scalar> k4 = dyn.continuous((x + dt * k3).eval());
    return x + (dt / Scalar(6)) * (k1 + Scalar(2) * k2 + Scalar(2) * k3 + k4);
}

// Closed-form step for the linear-damping ODE. Equivalent to RK4 in shape but
// exact for any dt — no truncation error. Use this in production; rk4_step is
// kept primarily for cross-checking and shape parity with diff_drive.
template <typename Scalar>
State<Scalar> exact_step(const Params<Scalar>& params,
                         const State<Scalar>& x,
                         Scalar dt) {
    using std::exp;
    const Scalar k     = params.damping;
    const Scalar decay = exp(-k * dt);
    // disp = (1 - exp(-k*dt))/k has a removable singularity at k=0; the
    // limit is dt. Guard against k == 0 (also cheap-correct for k near 0
    // since `decay` is then ≈ 1 and 1 - decay underflows to 0 cleanly).
    const Scalar disp  = (k == Scalar(0)) ? dt : (Scalar(1) - decay) / k;
    State<Scalar> x_next;
    x_next[PX] = x[PX] + x[VX] * disp;
    x_next[PY] = x[PY] + x[VY] * disp;
    x_next[VX] = x[VX] * decay;
    x_next[VY] = x[VY] * decay;
    return x_next;
}

}  // namespace ball
