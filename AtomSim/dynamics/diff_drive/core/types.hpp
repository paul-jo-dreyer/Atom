#pragma once

#include <Eigen/Core>

namespace diff_drive {

constexpr int PX = 0;
constexpr int PY = 1;
constexpr int THETA = 2;
constexpr int V = 3;
constexpr int OMEGA = 4;

constexpr int kStateDim = 5;
constexpr int kControlDim = 2;

template <typename Scalar>
using State = Eigen::Matrix<Scalar, kStateDim, 1>;

template <typename Scalar>
using Control = Eigen::Matrix<Scalar, kControlDim, 1>;

template <typename Scalar>
struct Params {
    Scalar track_width = Scalar(0.2);
    Scalar tau_motor   = Scalar(0.05);
};

}  // namespace diff_drive
