#pragma once

#include "types.hpp"

#include <cmath>

namespace diff_drive {

template <typename Scalar>
struct DiffDriveDynamics {
    Params<Scalar> params{};

    State<Scalar> continuous(const State<Scalar>& x, const Control<Scalar>& u) const {
        using std::cos;
        using std::sin;

        const Scalar theta = x[THETA];
        const Scalar v     = x[V];
        const Scalar omega = x[OMEGA];

        const Scalar v_left  = u[0];
        const Scalar v_right = u[1];
        const Scalar v_cmd_avg = (v_left + v_right) / Scalar(2);
        const Scalar omega_cmd = (v_right - v_left) / params.track_width;

        State<Scalar> dx;
        dx[PX]    = v * cos(theta);
        dx[PY]    = v * sin(theta);
        dx[THETA] = omega;
        dx[V]     = (v_cmd_avg - v) / params.tau_motor;
        dx[OMEGA] = (omega_cmd - omega) / params.tau_motor;
        return dx;
    }
};

}  // namespace diff_drive
