#include "../inst/include/glmmrOptim.h"

// [[Rcpp::depends(RcppEigen)]]


//' Range of combinatorials algorithms to identify approximate optimal GLMM design
//' 
//' Range of combinatorials algorithms to identify approximate optimal GLMM design
//' @param idx_in Integer vector specifying the indexes of the experimental conditions to start from
//' @param n Integer specifying the size of the design to find. For local search, this should be equal to the size of idx_in
//' @param C_list List of C vectors for the c-optimal function, see \link[glmmrOptim]{DesignSpace}
//' @param X_list List of X matrices, where X is the matrix of covariates in the regression model
//' @param Z_list List of Z matrices where Z is the design matrix of random effects terms
//' @param D_list List of D matrices, where D is the covariance matrix of the random effects
//' @param w_diag Matrix with each column corresponding to the diagonal of the individual level variance matrix, see \link[glmmrBase]{Model} for details
//' @param max_obs Vector of integers specifying the maximum number of copies of each experimental condition
//' @param weights Vector specifying the weights of each design
//' @param nmax Integer specifying the maximum number of rows of matrices X and Z during the algorithm.
//' @param exp_cond Vector specifying the experimental condition index of each observation
//' @param V0_list List of prior covariance matrices for the model parameters
//' @param type Integer. 0 = local search algorithm. 1 = local+greedy+local. 2 = local+greedy. 3 = greedy+local
//' @param robust_log Logical indicating to use a weighted sum of log (TRUE) or absolute (FALSE) values of the c-optimal 
//' function when using a robust criterion.
//' @param trace Logical indicating whether to provide detailed output
//' @param uncorr Logical indicating whether to treat all the experimental conditions as uncorrelated (TRUE) or not (FALSE)
//' @param bayes Logical indicating whether to use a Bayesian model with prior distributions on model parameters (TRUE) or a 
//' likelihood based model (FALSE)
//' @return A list containing: a vector of experimental condition indexes in the optimal design, the variance of the optimal design,
//'  the number of function calls and matrix operations, and an indicator for whether a Bayesian model was used.
// [[Rcpp::export]]
Rcpp::List GradRobustStep(Eigen::ArrayXi idx_in,  
                          int n,
                          Rcpp::List C_list, 
                          Rcpp::List X_list, 
                          Rcpp::List Z_list, 
                          Rcpp::List D_list, 
                          Eigen::MatrixXd w_diag,
                          Eigen::ArrayXi max_obs,
                          Eigen::VectorXd weights,
                          Eigen::ArrayXi exp_cond,
                          int nmax,
                          Rcpp::List V0_list,
                          Eigen::ArrayXi type,
                          bool robust_log = false,
                          bool trace = true,
                          bool uncorr = false,
                          bool bayes = false) {
  int ndesign = weights.size();
  glmmr::MatrixField<Eigen::VectorXd> Cfield;
  glmmr::MatrixField<Eigen::MatrixXd> Xfield;
  glmmr::MatrixField<Eigen::MatrixXd> Zfield;
  glmmr::MatrixField<Eigen::MatrixXd> Dfield;
  glmmr::MatrixField<Eigen::MatrixXd> V0field;
  for(int j=0; j<ndesign; j++){
    Cfield.add(Eigen::VectorXd(Rcpp::as<Eigen::Map<Eigen::VectorXd> >(C_list[j])));
    Xfield.add(Eigen::MatrixXd(Rcpp::as<Eigen::Map<Eigen::MatrixXd> >(X_list[j])));
    Zfield.add(Eigen::MatrixXd(Rcpp::as<Eigen::Map<Eigen::MatrixXd> >(Z_list[j])));
    Dfield.add(Eigen::MatrixXd(Rcpp::as<Eigen::Map<Eigen::MatrixXd> >(D_list[j])));
    V0field.add(Eigen::MatrixXd(Rcpp::as<Eigen::Map<Eigen::MatrixXd> >(V0_list[j])));
  }
  
  glmmr::OptimDesign hc(idx_in, n, Cfield, Xfield, Zfield, Dfield,
                        w_diag,max_obs,weights, exp_cond, nmax,
                        V0field, robust_log, trace, uncorr, bayes);
  
  int k = type.size();
  
  
  for(int i=0; i<k; i++){
    switch(type(i))
    {
      case 1:
        hc.local_search();
        break;
      case 2:
        hc.greedy_search();
        break;
      case 3:
        hc.reverse_greedy_search();
        break;
    }
  }

  // if(type==0)hc.local_search();
  // if(type==1){
  //   hc.local_search();
  //   hc.greedy_search();
  //   hc.local_search();
  // }
  // if(type==2){
  //   hc.local_search();
  //   hc.greedy_search();
  // }
  // if(type==3){
  //   hc.greedy_search();
  //   hc.local_search();
  // }
  return Rcpp::List::create(Rcpp::Named("idx_in") = hc.idx_in_,
                            Rcpp::Named("best_val_vec") = hc.val_,
                            Rcpp::Named("func_calls") = hc.fcalls_,
                            Rcpp::Named("mat_ops") = hc.matops_,
                            Rcpp::Named("bayes") = hc.bayes_);
}
