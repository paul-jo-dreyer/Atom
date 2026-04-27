#pragma once

#include "types.hpp"

namespace diff_drive {

template <typename Dynamics, typename Scalar>
State<Scalar> euler_step(const Dynamics& dyn,
                         const State<Scalar>& x,
                         const Control<Scalar>& u,
                         Scalar dt) {
    return x + dt * dyn.continuous(x, u);
}

template <typename Dynamics, typename Scalar>
State<Scalar> rk4_step(const Dynamics& dyn,
                       const State<Scalar>& x,
                       const Control<Scalar>& u,
                       Scalar dt) {
    const State<Scalar> k1 = dyn.continuous(x, u);
    const State<Scalar> k2 = dyn.continuous((x + (dt / Scalar(2)) * k1).eval(), u);
    const State<Scalar> k3 = dyn.continuous((x + (dt / Scalar(2)) * k2).eval(), u);
    const State<Scalar> k4 = dyn.continuous((x + dt * k3).eval(), u);
    return x + (dt / Scalar(6)) * (k1 + Scalar(2) * k2 + Scalar(2) * k3 + k4);
}

}  // namespace diff_drive
