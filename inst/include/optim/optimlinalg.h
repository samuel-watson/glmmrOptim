#pragma once

#include <cmath> 
#include <RcppEigen.h>

namespace glmmr {
namespace algo {

//removes index from square matrix
inline Eigen::MatrixXd remove_one_many_mat(const Eigen::MatrixXd &A, 
                                           const Eigen::ArrayXi &i) {
  
  Eigen::ArrayXi isort = i;
  std::sort(isort.data(),isort.data()+isort.size(), std::greater<int>());
  Eigen::MatrixXd A2 = A;
  
  for(int j=0; j<i.size(); j++){
    Eigen::MatrixXd A3(A2.rows()-1,A2.cols()-1);
    double d = A2(isort(j), isort(j));
    Eigen::VectorXd b(A2.rows()-1);
    if(isort(j) == A2.rows()){
      b = A2.block(0,isort(j),A2.rows()-1,1);
      A3 = A2.block(0,0,A2.rows()-1,A2.rows()-1);
    } else if(isort(j) == 0){
      b = A2.block(1,isort(j),A2.rows()-1,1);
      A3 = A2.block(1,1,A2.rows()-1,A2.rows()-1);
    } else {
      b.segment(0,isort(j)) = A2.block(0,isort(j),isort(j),1);
      b.segment(isort(j),A2.rows()-1-isort(j)) = A2.block(isort(j)+1,isort(j),A2.rows()-1-isort(j),1);
      A3.block(0,0,isort(j),isort(j)) = A2.block(0,0,isort(j),isort(j));
      A3.block(0,isort(j),isort(j),A2.rows()-1-isort(j)) = A2.block(0,isort(j)+1,isort(j),A2.rows()-1-isort(j));
      A3.block(isort(j),0,A2.rows()-1-isort(j),isort(j)) = A2.block(isort(j)+1,0,A2.rows()-1-isort(j),isort(j));
      A3.block(isort(j),isort(j),A2.rows()-1-isort(j),A2.rows()-1-isort(j)) = A2.block(isort(j)+1,isort(j)+1,A2.rows()-1-isort(j),A2.rows()-1-isort(j));
    }
    A3 -= (1.0/d)*b*b.transpose();
    A2 = A3;
  }
  
  return A2;
}

// removes observations and calculates value of objective function
inline double remove_one_many(const Eigen::MatrixXd &A, 
                              const Eigen::ArrayXi &i,
                              const Eigen::VectorXd &u) {
  
  Eigen::MatrixXd A2 = glmmr::algo::remove_one_many_mat(A,i);
  int n = u.size()-i.size();
  Eigen::VectorXd u2(n);
  bool isin;
  int iter = 0;
  
  for(int j=0; j< i.size(); j++){
    isin = (i == j).any();
    if(!isin){
      u2(iter) = u(j);
      iter++;
    }
  }
  return u2.transpose()*A2 * u2;
}


inline Eigen::MatrixXd add_one_mat(const Eigen::MatrixXd &A, 
                                   double sigma_jj, 
                                   const Eigen::VectorXd &f) {
  
  Eigen::MatrixXd A2 = Eigen::MatrixXd::Zero(A.rows() + 1, A.rows() + 1);
  A2.block(0,0,A.rows(),A.cols()) = A;
  A2(A2.rows() - 1, A2.rows() - 1) = 1 / sigma_jj;
  
  Eigen::VectorXd u1 = Eigen::VectorXd::Zero(f.size()+1);
  u1.segment(0,f.size()) = f;
  Eigen::VectorXd v1 = Eigen::VectorXd::Zero(u1.size());
  v1(v1.size()-1) = 1.0;
  
  A2 -= ((A2 * u1) * (v1.transpose() * A2)) * (1.0/(1.0 + (v1.transpose() * A2) * u1));
  A2 -= ((A2 * v1) * (u1.transpose() * A2)) * (1.0/(1.0 + (u1.transpose() * A2) * v1));
  
  return A2;
}


inline double add_one(const Eigen::MatrixXd &A, 
                      double sigma_jj, 
                      const Eigen::VectorXd &f,
                      const Eigen::VectorXd &u) {
  
  Eigen::MatrixXd A2 = glmmr::algo::add_one_mat(A,sigma_jj,f);
  return u.transpose()*A2*u;
}

inline Eigen::ArrayXi uvec_minus(const Eigen::ArrayXi &v, 
                                 int rm_idx) {
  int n = v.size();
  if (rm_idx == 0) return v.tail(n-1);
  if (rm_idx == n-1) return v.head(n-1);
  Eigen::ArrayXi res(v.size()-1);
  res.head(rm_idx) = v.head(rm_idx);
  res.tail(n-1-rm_idx) = v.tail(n-1-rm_idx);
  return res;
}

}

}
