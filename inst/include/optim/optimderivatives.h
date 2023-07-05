#ifndef OPTIMDERIVATIVES_H
#define OPTIMDERIVATIVES_H

#include <RcppEigen.h>
#include <glmmr.h>
#include "matrixfield.h"

namespace glmmr{

using namespace Eigen;

class OptimDerivatives {
public: 
  std::vector<int> gaussian;
  std::vector<glmmr::MatrixField<MatrixXd> > FirstOrderDerivatives;
  std::vector<glmmr::MatrixField<MatrixXd> > SecondOrderDerivatives;
  OptimDerivatives(){};
  
  void addDesign(glmmr::ModelBits& model){
    glmmr::MatrixField<MatrixXd> first;
    glmmr::MatrixField<MatrixXd> second;
    int curr_size = FirstOrderDerivatives.size();
    std::vector<MatrixXd> derivs;
    model.covariance.derivatives(derivs,2);
    int R = model.covariance.npar();
    gaussian.push_back((int)(model.family.family == "gaussian"));
    for(int i = 0; i < R; i++){
      first.add(model.covariance.Z()*derivs[1+i]*model.covariance.Z().transpose());
    }
    if(gaussian[curr_size]){
      first.add(model.data.variance.matrix().asDiagonal()*MatrixXd::Identity(model.n(),model.n()));
    }
    FirstOrderDerivatives.push_back(first);
    
    for(int i = 0; i < R; i++){
      for(int j = i; j < R; j++){
        int scnd_idx = i + j*(R-1) - j*(j-1)/2;
        second.add(model.covariance.Z()*derivs[R+1+scnd_idx]*model.covariance.Z().transpose());
      }
    }
    SecondOrderDerivatives.push_back(second);
    // Rcpp::Rcout << "\nField size: " << FirstOrderDerivatives[0].size();
    // Rcpp::Rcout << "\n2 Field size: " << SecondOrderDerivatives[0].size();
    // Rcpp::Rcout << "\nMat 1 size: " << FirstOrderDerivatives[0].rows(0) << " x " << FirstOrderDerivatives[0].cols(0);
    // Rcpp::Rcout << "\nMat1: " << FirstOrderDerivatives[0](0);
  };
};

}

#endif