#pragma once

#include <cmath> 
#include <RcppEigen.h>

namespace glmmr {
namespace maths {

using namespace Eigen;

inline double obj_fun(const MatrixXd &A, 
                      const VectorXd &U2) {
  return U2.transpose() * A * U2;
}

inline double c_obj_fun(const MatrixXd &M, 
                        const VectorXd &C) {
  // this is the objective function c-optimal
  //MatrixXd M_inv = M.inverse();
  return C.transpose() * M * C;
}

inline double c_d_deriv(const MatrixXd &M1, 
                        const MatrixXd &M2, 
                        const VectorXd &C) {
  // this is the directional derivative from M1 to M2 c-optimal
  MatrixXd M_inv = M1.inverse();
  return C.transpose() * M_inv * (M1 - M2) * M_inv * C;
}

inline double c_deriv(const MatrixXd &M, 
                      const VectorXd &C) {
  // this is the directional derivative from M1 to M2 c-optimal
  MatrixXd M_inv = M.inverse();
  return (-1 * M_inv * C * C.transpose() * M_inv).norm();
}

inline MatrixXd gen_m(const MatrixXd &X, 
                             const MatrixXd &A) {
  //generate information matrix
  return X.transpose() * A * X;
}


}

}
