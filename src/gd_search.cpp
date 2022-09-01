#include "../inst/include/glmmroptim.h"
using namespace Rcpp;
using namespace arma;

// [[Rcpp::depends(RcppArmadillo)]]


//' Hill-Climbing algorithm to identify optimal GLMM design
//' 
//' Hill-Climbing algorithm to identify optimal GLMM design
//' @param N Integer specifying number of experimental conditions in the optimal design
//' @param idx_in Integer vector specifying the indexes of the experimental conditions to start from
//' @param n Integer specifying the size of the design to find. For local search, this should be equal to the size of idx_in
//' @param C_list List of C vectors for the c-optimal function, see \link{glmmr}[DesignSpace]
//' @param X_list List of X matrices
//' @param sig_list List of inverse covariance matrices
//' @param weights Vector specifying the weights of each design
//' @param exp_cond Vector specifying the experimental condition index of each observation
//' @param nfix Vector listing the experimental condition indexes that are fixed in the design
//' @param any_fix Integer. 0 = no experimental conditions are fixed, 1 = some experimental conditions are fixed
//' @param type Integer. 0 = local search algorith. 1 = greedy search algorithm.
//' @param rd_mode Integer. Robust objective function, 1=weighted average, 2=minimax
//' @param trace Logical indicating whether to provide detailed output
//' @return A vector of experimental condition indexes in the optimal design
// [[Rcpp::export]]
Rcpp::List GradRobustStep(arma::uvec idx_in, 
                          arma::uword n,
                          Rcpp::List C_list, 
                          Rcpp::List X_list, 
                          Rcpp::List Z_list, 
                          Rcpp::List D_list, 
                          arma::mat w_diag,
                          arma::uvec max_obs,
                          arma::vec weights,
                          arma::uvec exp_cond,
                          arma::uvec nfix, 
                          Rcpp::List V0_list,
                          arma::uword any_fix = 0,
                          arma::uword type = 0,
                          arma::uword rd_mode = 1,
                          bool trace = true,
                          bool uncorr = false,
                          bool bayes = false) {
  arma::uword ndesign = weights.n_elem;
  arma::field<arma::vec> Cfield(ndesign,1);
  arma::field<arma::mat> Xfield(ndesign,1);
  arma::field<arma::mat> Zfield(ndesign,1);
  arma::field<arma::mat> Dfield(ndesign,1);
  arma::field<arma::mat> V0field(ndesign,1);
  for(arma::uword j=0; j<ndesign; j++){
    Cfield(j,0) = as<arma::vec>(C_list[j]);
    Xfield(j,0) = as<arma::mat>(X_list[j]);
    Zfield(j,0) = as<arma::mat>(Z_list[j]);
    Dfield(j,0) = as<arma::mat>(D_list[j]);
    V0field(j,0) = as<arma::mat>(V0_list[j]);
  }
  HillClimbing hc(idx_in, n, Cfield, Xfield, Zfield, Dfield,
                  w_diag,max_obs,
                  weights, exp_cond, any_fix, nfix, V0field, rd_mode, trace, uncorr, bayes);
  if(type==0)hc.local_search();
  if(type==1){
    hc.local_search();
    hc.greedy_search();
    hc.local_search();
  }
  if(type==2){
    hc.local_search();
    hc.greedy_search();
  }
  if(type==3){
    hc.greedy_search();
    hc.local_search();
  }
  return Rcpp::List::create(Named("idx_in") = hc.idx_in_,
                            Named("best_val_vec") = hc.val_,
                            Named("func_calls") = hc.fcalls_,
                            Named("mat_ops") = hc.matops_,
                            Named("bayes") = hc.bayes_);
}
