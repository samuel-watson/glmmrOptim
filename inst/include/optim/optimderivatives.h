#ifndef OPTIMDERIVATIVES_H
#define OPTIMDERIVATIVES_H

#include <RcppEigen.h>
#include <glmmr.h>

namespace glmmr{

using namespace Eigen;

class OptimDerivatives {
public: 
  std::vector<int> gaussian;
  std::vector<std::vector<MatrixXd> > FirstOrderDerivatives;
  std::vector<std::vector<MatrixXd> > SecondOrderDerivatives;
  OptimDerivatives(){};
  
  void addDesign(const glmmr::ModelBits& model){
    FirstOrderDerivatives.resize(FirstOrderDerivatives.size()+1);
    SecondOrderDerivatives.resize(SecondOrderDerivatives.size()+1);
    int curr_size = FirstOrderDerivatives.size();
    std::vector<MatrixXd> derivs;
    model.covariance.derivatives(derivs,2);
    int R = model.covariance.npar();
    gaussian.push_back((int)model.family.family == "gaussian");
    for(int i = 0; i < R; i++){
      FirstOrderDerivatives[curr_size-1].push_back(model.covariance.Z()*derivs[1+i]*model.covariance.Z().transpose());
    }
    if(gaussian){
      FirstOrderDerivatives[curr_size-1].push_back(model.data.variance.matrix().asDiagonal()*MatrixXd::Identity(model.n(),model.n()));
    }
    
    for(int i = 0; i < R; i++){
      for(int j = i; j < R; j++){
        int scnd_idx = i + j*(R-1) - j*(j-1)/2;
        SecondOrderDerivatives[curr_size-1].push_back(model.covariance.Z()*derivs[R+1+scnd_idx]*model.covariance.Z().transpose());
      }
    }
  };
};

}

#endif