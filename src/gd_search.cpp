#include <glmmr/covariance.hpp>
#include <glmmr/openmpheader.h>
#include "glmmrOptim.h"

// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::depends(glmmrBase)]]

using namespace Rcpp;


//' Disable or enable parallelised computing
 //' 
 //' By default, the package will use multithreading for many calculations if OpenMP is 
 //' available on the system. For multi-user systems this may not be desired, so parallel
 //' execution can be disabled with this function.
 //' 
 //' @param parallel_ Logical indicating whether to use parallel computation (TRUE) or disable it (FALSE)
 //' @param cores_ Number of cores for parallel execution
 //' @return None, called for effects
 // [[Rcpp::export]]
 void setParallelOptim(SEXP parallel_, int cores_ = 2){
   bool parallel = as<bool>(parallel_);
   if(OMP_IS_USED){
     int a, b; // needed for defines on machines without openmp
     if(!parallel){
       a = 0;
       b = 1;
       omp_set_dynamic(a); 
       omp_set_num_threads(b);
       Eigen::setNbThreads(b);
     } else {
       a = 1;
       b = cores_;
       omp_set_dynamic(a); 
       omp_set_num_threads(b);
       Eigen::setNbThreads(b);
     }
   } 
 }

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

 
 // Inputs: 
 //   A_list: list of p_r x n matrices (A_r = X_r^T L_r^T, precomputed in R)
 //   c_list: list of length-p_r vectors
 //   exp_cond: integer vector length n, 1..K
 //   w: length-R vector of response weights
 //   rho, tol_abs, tol_rel, max_iter
 // Returns: normalised mu (length K)
 
 // [[Rcpp::export]]
 Eigen::VectorXd socp_admm(Rcpp::List A_list, Rcpp::List c_list,
                           Eigen::VectorXi exp_cond, Eigen::VectorXd w,
                           double rho = 1.0, double tol = 1e-6, int max_iter = 2000) {
   const int R = A_list.size();
   const int n = exp_cond.size();
   
   // Precompute per-response: Cholesky of A A^T, particular solution z0, 
   // and a function to apply P = I - A^T (AA^T)^{-1} A
   std::vector<MatrixXd> A(R);
   std::vector<VectorXd> z0(R), z(R), v(R), u(R);
   std::vector<LLT<MatrixXd>> lltAAt(R);
   
   for (int r = 0; r < R; ++r) {
     A[r] = Rcpp::as<MatrixXd>(A_list[r]);
     VectorXd c = Rcpp::as<VectorXd>(c_list[r]);
     lltAAt[r].compute(A[r] * A[r].transpose());
     z0[r] = A[r].transpose() * lltAAt[r].solve(c);
     z[r] = z0[r]; 
     v[r] = z0[r]; 
     u[r] = VectorXd::Zero(n);
   }
   
   // Group indices
   const int K = exp_cond.maxCoeff();
   std::vector<std::vector<int>> groups(K);
   for (int k = 0; k < n; ++k) groups[exp_cond[k]-1].push_back(k);
   
   for (int it = 0; it < max_iter; ++it) {
     // z-update: z_r = z0_r + P_r (v_r - u_r)
     for (int r = 0; r < R; ++r) {
       VectorXd t = v[r] - u[r];
       VectorXd At_t = A[r] * t;
       z[r] = t - A[r].transpose() * lltAAt[r].solve(At_t) + z0[r] 
       - (z0[r] - A[r].transpose() * lltAAt[r].solve(A[r] * z0[r]));
       // simplified: z[r] = z0[r] + (t - A^T (AA^T)^{-1} A t)
     }
     
     // v-update: block soft-threshold per (r, group)
     double primal_res = 0, dual_res = 0;
     for (int r = 0; r < R; ++r) {
       VectorXd v_old = v[r];
       VectorXd q = z[r] + u[r];
       v[r].setZero();
       for (const auto& idx : groups) {
         double nrm = 0;
         for (int k : idx) nrm += q[k]*q[k];
         nrm = std::sqrt(nrm);
         double thr = w[r] / rho;
         if (nrm > thr) {
           double s = 1.0 - thr/nrm;
           for (int k : idx) v[r][k] = s * q[k];
         }
       }
       dual_res += (rho * (v[r] - v_old)).squaredNorm();
     }
     
     // u-update and residuals
     for (int r = 0; r < R; ++r) {
       VectorXd pr = z[r] - v[r];
       primal_res += pr.squaredNorm();
       u[r] += pr;
     }
     
     if (std::sqrt(primal_res) < tol && std::sqrt(dual_res) < tol) break;
   }
   
   // Recover mu
   VectorXd mu = VectorXd::Zero(K);
   for (int k = 0; k < K; ++k) {
     for (const auto& idx : groups[k]) { /* gather */ }
     for (int r = 0; r < R; ++r) {
       double nrm = 0;
       for (int k_idx : groups[k]) nrm += v[r][k_idx]*v[r][k_idx];
       mu[k] += w[r] * std::sqrt(nrm);
     }
   }
   return mu / mu.sum();
 }