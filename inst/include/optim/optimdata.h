#pragma once

#include <RcppEigen.h>
#include <glmmr/matrixfield.h>

namespace glmmr{

using namespace Eigen;
using namespace Rcpp;

class OptimData {
  public: 
    glmmr::MatrixField<VectorXd> C_list_;
    glmmr::MatrixField<MatrixXd> D_list_;
    glmmr::MatrixField<MatrixXd> X_all_list_;
    glmmr::MatrixField<MatrixXd> Z_all_list_;
    const MatrixXd W_all_diag_;
    glmmr::MatrixField<MatrixXd> V0_list_;
    VectorXd weights_;
    ArrayXi max_obs_;
    const ArrayXi exp_cond_;
    OptimData(const glmmr::MatrixField<VectorXd> &C_list, 
              const glmmr::MatrixField<MatrixXd> &X_list, 
              const glmmr::MatrixField<MatrixXd> &Z_list, 
              const glmmr::MatrixField<MatrixXd> &D_list,
              const MatrixXd& w_diag,
              const glmmr::MatrixField<MatrixXd> &V0_list,
              const ArrayXi& max_obs,
              const VectorXd& weights,
              const ArrayXi& exp_cond) :  
      C_list_(C_list), 
      D_list_(D_list),
      X_all_list_(X_list),
      Z_all_list_(Z_list),
      W_all_diag_(w_diag),
      V0_list_(V0_list),
      weights_(weights), 
      max_obs_(max_obs),
      exp_cond_(exp_cond){};
    
    OptimData(const Rcpp::List &C_list, 
              const Rcpp::List &X_list, 
              const Rcpp::List &Z_list, 
              const Rcpp::List &D_list,
              const MatrixXd& w_diag,
              const Rcpp::List &V0_list,
              const ArrayXi& max_obs,
              const VectorXd& weights,
              const ArrayXi& exp_cond) :  
      W_all_diag_(w_diag),
      weights_(weights), 
      max_obs_(max_obs),
      exp_cond_(exp_cond){
      for(int j=0; j<weights.size(); j++){
        C_list_.add(VectorXd(Rcpp::as<Map<VectorXd> >(C_list[j])));
        X_all_list_.add(MatrixXd(Rcpp::as<Map<MatrixXd> >(X_list[j])));
        Z_all_list_.add(MatrixXd(Rcpp::as<Map<MatrixXd> >(Z_list[j])));
        D_list_.add(MatrixXd(Rcpp::as<Map<MatrixXd> >(D_list[j])));
        V0_list_.add(MatrixXd(Rcpp::as<Map<MatrixXd> >(V0_list[j])));
      }
    };
    
};

}
