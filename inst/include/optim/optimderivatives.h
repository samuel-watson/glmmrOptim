#pragma once

#include <RcppEigen.h>
#include <glmmr/covariance.hpp>

// [[Rcpp::depends(glmmrBase)]]

namespace glmmr{

using namespace Eigen;

class OptimDerivatives {
public: 
  std::vector<int> gaussian;
  std::vector<glmmr::MatrixField<MatrixXd> > FirstOrderDerivatives;
  std::vector<glmmr::MatrixField<MatrixXd> > SecondOrderDerivatives;
  OptimDerivatives(){};
  
  void addDesign(glmmr::Covariance& cov, bool is_gaussian){
    glmmr::MatrixField<MatrixXd> first;
    glmmr::MatrixField<MatrixXd> second;
    int curr_size = FirstOrderDerivatives.size();
    std::vector<MatrixXd> derivs;
    cov.derivatives(derivs,2);
    int R = cov.npar();
    gaussian.push_back((int)(is_gaussian));
    for(int i = 0; i < R; i++){
      first.add(derivs[1+i]);
    }
    FirstOrderDerivatives.push_back(first);
    for(int i = 0; i < R; i++){
      for(int j = i; j < R; j++){
        int scnd_idx = i + j*(R-1) - j*(j-1)/2;
        second.add(derivs[R+1+scnd_idx]);
      }
    }
    SecondOrderDerivatives.push_back(second);
  };
};

}
