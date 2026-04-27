#pragma once

#include "dynamics.hpp"
#include "types.hpp"

#include <cmath>

namespace diff_drive {

template <typename Scalar>
struct Jacobians {
    Eigen::Matrix<Scalar, kStateDim, kStateDim>   A;
    Eigen::Matrix<Scalar, kStateDim, kControlDim> B;
};

template <typename Scalar>
Jacobians<Scalar> linearize(const DiffDriveDynamics<Scalar>& dyn,
                            const State<Scalar>& x,
                            const Control<Scalar>& u) {
    using std::cos;
    using std::sin;
    (void)u;

    const Scalar theta = x[THETA];
    const Scalar v     = x[V];
    const Scalar tau   = dyn.params.tau_motor;
    const Scalar W     = dyn.params.track_width;

    Jacobians<Scalar> J;
    J.A.setZero();
    J.A(PX, THETA) = -v * sin(theta);
    J.A(PX, V)     =  cos(theta);
    J.A(PY, THETA) =  v * cos(theta);
    J.A(PY, V)     =  sin(theta);
    J.A(THETA, OMEGA) = Scalar(1);
    J.A(V, V)         = -Scalar(1) / tau;
    J.A(OMEGA, OMEGA) = -Scalar(1) / tau;

    J.B.setZero();
    J.B(V, 0)     =  Scalar(1) / (Scalar(2) * tau);
    J.B(V, 1)     =  Scalar(1) / (Scalar(2) * tau);
    J.B(OMEGA, 0) = -Scalar(1) / (W * tau);
    J.B(OMEGA, 1) =  Scalar(1) / (W * tau);
    return J;
}

}  // namespace diff_drive
