#pragma once

#include <Eigen/Core>

namespace ball {

constexpr int PX = 0;
constexpr int PY = 1;
constexpr int VX = 2;
constexpr int VY = 3;

constexpr int kStateDim = 4;

template <typename Scalar>
using State = Eigen::Matrix<Scalar, kStateDim, 1>;

template <typename Scalar>
using Vec2 = Eigen::Matrix<Scalar, 2, 1>;

template <typename Scalar>
struct Params {
    Scalar radius      = Scalar(0.05);   // m   (collision geometry)
    Scalar mass        = Scalar(0.05);   // kg  (reserved; not used by current impulse model)
    Scalar restitution = Scalar(0.6);    // [0,1] dimensionless
    Scalar damping     = Scalar(0.5);    // 1/s linear-damping rate constant
};

}  // namespace ball
