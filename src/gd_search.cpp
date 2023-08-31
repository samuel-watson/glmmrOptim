#include <glmmr.h>
#include "glmmrOptim.h"

// [[Rcpp::depends(RcppEigen)]]

using namespace Rcpp;

// // [[Rcpp::export]]
// Rcpp::List GradRobustStep(Rcpp::List C_list, 
//                           Rcpp::List X_list, 
//                           Rcpp::List Z_list, 
//                           Rcpp::List D_list, 
//                           SEXP w_diag,
//                           Rcpp::List V0_list,
//                           SEXP max_obs,
//                           SEXP weights,
//                           SEXP exp_cond,
//                           SEXP idx_in,  
//                           int n,
//                           int nmax,
//                           SEXP type,
//                           bool robust_log = false,
//                           bool trace = true,
//                           bool uncorr = false,
//                           bool bayes = false) {
//   Eigen::MatrixXd w_diag_ = as<Eigen::MatrixXd>(w_diag);
//   Eigen::ArrayXi max_obs_ = as<Eigen::ArrayXi>(max_obs);
//   Eigen::VectorXd weights_ = as<Eigen::VectorXd>(weights);
//   Eigen::ArrayXi exp_cond_ = as<Eigen::ArrayXi>(exp_cond);
//   glmmr::OptimData data(C_list,X_list,Z_list,D_list,w_diag_,V0_list,max_obs_,weights_,exp_cond_);
//   Eigen::ArrayXi idx_in_ = as<Eigen::ArrayXi>(idx_in);
//   Eigen::ArrayXi type_ = as<Eigen::ArrayXi>(type);
//   glmmr::OptimDesign hc(idx_in_, n,data,nmax,robust_log, trace, uncorr, bayes);
//   int k = type_.size();
//   for(int i=0; i<k; i++){
//     if(type_(i)==1){
//       hc.local_search();
//     } else if(type_(i)==2){
//       hc.greedy_search();
//     } else if(type_(i)==3){
//       hc.reverse_greedy_search();
//     } else {
//       Rcpp::stop("Type must be 1,2,3");
//     }
//   }
//   return Rcpp::List::create(Rcpp::Named("idx_in") = hc.idx_in_,
//                             Rcpp::Named("best_val_vec") = hc.val_,
//                             Rcpp::Named("func_calls") = hc.fcalls_,
//                             Rcpp::Named("mat_ops") = hc.matops_,
//                             Rcpp::Named("bayes") = hc.bayes_);
// }

// [[Rcpp::export]]
SEXP CreateOptimData(Rcpp::List C_list, 
                     Rcpp::List X_list, 
                     Rcpp::List Z_list, 
                     Rcpp::List D_list, 
                     SEXP w_diag,
                     Rcpp::List V0_list,
                     SEXP max_obs,
                     SEXP weights,
                     SEXP exp_cond){
  Eigen::MatrixXd w_diag_ = as<Eigen::MatrixXd>(w_diag);
  Eigen::ArrayXi max_obs_ = as<Eigen::ArrayXi>(max_obs);
  Eigen::VectorXd weights_ = as<Eigen::VectorXd>(weights);
  Eigen::ArrayXi exp_cond_ = as<Eigen::ArrayXi>(exp_cond);
  XPtr<glmmr::OptimData> ptr(new glmmr::OptimData(C_list,X_list,Z_list,D_list,w_diag_,V0_list,max_obs_,weights_,exp_cond_),true);
  return ptr;
}

// [[Rcpp::export]]
SEXP CreateOptim(SEXP dataptr,
                 SEXP derivptr,
                          SEXP idx_in,  
                          int n,
                          int nmax,
                          bool robust_log = false,
                          bool trace = true,
                          bool kr = false,
                          bool uncorr = false,
                          bool bayes = false) {
  Eigen::ArrayXi idx_in_ = as<Eigen::ArrayXi>(idx_in);
  XPtr<glmmr::OptimData> dptr(dataptr);
  XPtr<glmmr::OptimDerivatives> pptr(derivptr);
  XPtr<glmmr::OptimDesign> ptr(new glmmr::OptimDesign(idx_in_, n,*dptr,*pptr,nmax,robust_log, trace, uncorr, kr, bayes),true);
  return ptr;
}

// [[Rcpp::export]]
SEXP CreateDerivatives(){
  XPtr<glmmr::OptimDerivatives> ptr(new glmmr::OptimDerivatives(),true);
  return ptr;
}

// [[Rcpp::export]]
Rcpp::List FindOptimumDesign(SEXP dptr_,SEXP type_){
  Eigen::ArrayXi type = as<Eigen::ArrayXi>(type_);
  XPtr<glmmr::OptimDesign> ptr(dptr_);
  int k = type.size();
  for(int i=0; i<k; i++){
    if(type(i)==1){
      ptr->local_search();
    } else if(type(i)==2){
      ptr->greedy_search();
    } else if(type(i)==3){
      ptr->reverse_greedy_search();
    } else {
      Rcpp::stop("Type must be 1,2,3");
    }
  }
  return Rcpp::List::create(Rcpp::Named("idx_in") = ptr->idx_in_,
                            Rcpp::Named("best_val_vec") = ptr->val_,
                            Rcpp::Named("func_calls") = ptr->fcalls_,
                            Rcpp::Named("mat_ops") = ptr->matops_,
                            Rcpp::Named("bayes") = ptr->bayes_,
                            Rcpp::Named("kr") = ptr->kr_);
}

// [[Rcpp::export]]
void AddDesignDerivatives(SEXP dptr_, SEXP mptr_, SEXP is_gaussian_){
  bool is_gaussian = as<bool>(is_gaussian_);
  XPtr<glmmr::OptimDerivatives> dptr(dptr_);
  XPtr<glmmr::Covariance> mptr(mptr_);
  dptr->addDesign(*mptr, is_gaussian);
}
