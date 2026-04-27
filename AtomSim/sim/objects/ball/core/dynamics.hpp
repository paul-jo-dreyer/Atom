#pragma once

#include "types.hpp"

namespace ball {

template <typename Scalar>
struct BallDynamics {
    Params<Scalar> params{};

    State<Scalar> continuous(const State<Scalar>& x) const {
        State<Scalar> dx;
        dx[PX] = x[VX];
        dx[PY] = x[VY];
        dx[VX] = -params.damping * x[VX];
        dx[VY] = -params.damping * x[VY];
        return dx;
    }
};

}  // namespace ball
