#ifndef OPTIMMATHS_H
#define OPTIMMATHS_H

#include <cmath> 
#include <RcppEigen.h>
#include <glmmr.h>

namespace glmmr {
namespace maths {

inline double obj_fun(const Eigen::MatrixXd &A, 
                      const Eigen::VectorXd &U2) {
  return U2.transpose() * A * U2;
}

inline double c_obj_fun(const Eigen::MatrixXd &M, 
                        const Eigen::VectorXd &C) {
  // this is the objective function c-optimal
  Eigen::MatrixXd M_inv = M.inverse();
  return C.transpose() * M_inv * C;
}

inline double c_d_deriv(const Eigen::MatrixXd &M1, 
                        const Eigen::MatrixXd &M2, 
                        const Eigen::VectorXd &C) {
  // this is the directional derivative from M1 to M2 c-optimal
  Eigen::MatrixXd M_inv = M1.inverse();
  return C.transpose() * M_inv * (M1 - M2) * M_inv * C;
}

inline double c_deriv(const Eigen::MatrixXd &M, 
                      const Eigen::VectorXd &C) {
  // this is the directional derivative from M1 to M2 c-optimal
  Eigen::MatrixXd M_inv = M.inverse();
  return (-1 * M_inv * C * C.transpose() * M_inv).norm();
}

inline Eigen::MatrixXd gen_m(const Eigen::MatrixXd &X, 
                             const Eigen::MatrixXd &A) {
  //generate information matrix
  return X.transpose() * A * X;
}


}

}

#endif