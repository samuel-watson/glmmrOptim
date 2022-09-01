#ifndef GLMMROPTIM_H
#define GLMMROPTIM_H

#include <cmath> 
#include <RcppArmadillo.h>
#include <glmmr.h>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace Rcpp;
using namespace arma;

inline double obj_fun(const arma::mat &A, const arma::vec &U2) {
  // this is the directional derivative
  return arma::as_scalar(U2.t() * A * U2);
}

inline double c_obj_fun(arma::mat M, arma::vec C) {
  // this is the objective function c-optimal
  arma::mat M_inv = arma::inv_sympd(M);
  return arma::as_scalar(C.t() * M_inv * C);
}

inline double c_d_deriv(arma::mat M1, arma::mat M2, arma::vec C) {
  // this is the directional derivative from M1 to M2 c-optimal
  arma::mat M_inv = arma::inv_sympd(M1);
  return arma::as_scalar(C.t() * M_inv * (M1 - M2) * M_inv * C);
}

inline double c_deriv(arma::mat M, arma::vec C) {
  // this is the directional derivative from M1 to M2 c-optimal
  arma::mat M_inv = arma::inv_sympd(M);
  return norm(-1 * M_inv * C * C.t() * M_inv,"fro");
}

inline arma::mat gen_m(const arma::mat &X, const arma::mat &A) {
  //generate information matrix
  return X.t() * A * X;
}

inline arma::mat remove_one_many_mat(const arma::mat &A, 
                              const arma::uvec &i) {
  arma::vec idx = arma::linspace(0, A.n_rows - 1, A.n_rows);
  arma::uvec uidx = arma::conv_to<arma::uvec>::from(idx);
  arma::uvec isort = arma::sort(i,"descend"); // sort descending to avoid row conflicts
  arma::mat A2 = A;
  
  for(arma::uword j=0; j<i.n_elem; j++){
    arma::vec idx_new = arma::linspace(0, uidx.n_elem - 1, uidx.n_elem);
    arma::uvec uidx_new = arma::conv_to<arma::uvec>::from(idx_new);
    uidx.shed_row(isort(j));
    uidx_new.shed_row(isort(j));
    double d = A2(isort(j), isort(j));
    arma::vec b = A2.submat(uidx_new, arma::uvec({isort(j)}));
    A2 = A2.submat(uidx_new, uidx_new) - b * b.t() / d;
  }
  
  return A2;
}

inline double remove_one_many(const arma::mat &A, 
                       const arma::uvec &i,
                       const arma::vec &u) {
  arma::vec idx = arma::linspace(0, A.n_rows - 1, A.n_rows);
  arma::uvec uidx = arma::conv_to<arma::uvec>::from(idx);
  arma::uvec isort = arma::sort(i,"descend"); // sort descending to avoid row conflicts
  arma::mat A2 = A;
  for(arma::uword j=0; j<i.n_elem; j++){
    arma::vec idx_new = arma::linspace(0, uidx.n_elem - 1, uidx.n_elem);
    arma::uvec uidx_new = arma::conv_to<arma::uvec>::from(idx_new);
    uidx.shed_row(isort(j));
    uidx_new.shed_row(isort(j));
    double d = A2(isort(j), isort(j));
    arma::vec b = A2.submat(uidx_new, arma::uvec({isort(j)}));
    A2 = A2.submat(uidx_new, uidx_new) - b * b.t() / d;
  }
  return obj_fun(A2, u(uidx));
}

inline double add_one(const arma::mat &A, 
               double sigma_jj, 
               const arma::vec &f,
               const arma::vec &u) {
  arma::mat A2(A.n_rows + 1, A.n_cols + 1, arma::fill::zeros);
  A2.submat(0, 0, A2.n_rows - 2, A2.n_cols - 2) = A;
  A2(A2.n_rows - 1, A2.n_cols - 1) = 1 / sigma_jj;
  
  // step 3: compute K2_inv
  arma::vec u1 = arma::join_cols(f, arma::vec({0}));
  arma::vec v1(u1.n_elem, arma::fill::zeros);
  v1(v1.n_elem - 1) = 1;
  A2 = A2 -
    ((A2 * u1) * (v1.t() * A2)) / (1 + arma::as_scalar((v1.t() * A2) * u1));
  
  // step 4: compute K3_inv
  A2 = A2 -
    ((A2 * v1) * (u1.t() * A2)) / (1 + arma::as_scalar((u1.t() * A2) * v1));
  
  return obj_fun(A2, u);
}


inline arma::mat add_one_mat(const arma::mat &A, 
                             double sigma_jj, 
                             const arma::vec &f) {
  arma::mat A2(A.n_rows + 1, A.n_cols + 1, arma::fill::zeros);
  // step 1: compute A*
  A2.submat(0, 0, A2.n_rows - 2, A2.n_cols - 2) = A;
  for (arma::uword j = 0; j < A2.n_rows - 1; j++) {
    A2(j, A2.n_cols - 1) = 0;
    A2(A2.n_rows - 1, j) = 0;
  }
  A2(A2.n_rows - 1, A2.n_cols - 1) = 1 / sigma_jj;
  
  // step 3: compute K2_inv
  arma::vec u1 = arma::join_cols(f, arma::vec({0}));
  arma::vec v1(u1.n_elem, arma::fill::zeros);
  v1(v1.n_elem - 1) = 1;
  A2 = A2 -
    ((A2 * u1) * (v1.t() * A2)) / (1 + arma::as_scalar((v1.t() * A2) * u1));
  
  // step 4: compute K3_inv
  A2 = A2 -
    ((A2 * v1) * (u1.t() * A2)) / (1 + arma::as_scalar((u1.t() * A2) * v1));
  
  return A2;
}

inline arma::uvec match_uvec(arma::uvec x, arma::uword val){
  arma::uvec match_vec(x.n_elem, fill::value(-1));
  arma::uword count = 0;
  for(arma::uword j=0; j<x.n_elem; j++){
    if(x(j)==val){
      match_vec(count) = j;
      count += 1;
    }
  }
  return match_vec.rows(0,count-1);
}


inline arma::uvec std_setdiff(arma::uvec &x, arma::uvec &y) {
  std::vector<int> a = arma::conv_to<std::vector<int> >::from(arma::sort(x));
  std::vector<int> b = arma::conv_to<std::vector<int> >::from(arma::sort(y));
  std::vector<int> out;
  
  std::set_difference(a.begin(), a.end(), b.begin(), b.end(),
                      std::inserter(out, out.end()));
  
  return arma::conv_to<arma::uvec>::from(out);
}

inline arma::uvec uvec_minus(const arma::uvec &v, arma::uword rm_idx) {
  arma::uword n = v.size();
  if (rm_idx == 0) return v.tail(n-1);
  if (rm_idx == n-1) return v.head(n-1);
  arma::uvec res(v.size()-1);
  res.head(rm_idx) = v.head(rm_idx);
  res.tail(n-1-rm_idx) = v.tail(n-1-rm_idx);
  return res;
}

class HillClimbing {
private:
  const arma::field<arma::vec> C_list_; 
  const arma::field<arma::mat> D_list_;
  const arma::field<arma::mat> X_all_list_; 
  const arma::field<arma::mat> Z_all_list_; 
  const arma::mat W_all_diag_;
  const arma::vec weights_; // weights for each design
  arma::uvec max_obs_;
  arma::uvec curr_obs_;
  const arma::uword nlist_; // number of designs
  arma::uword any_fix_;
  arma::uword n_; //size of the design to find
  arma::uword k_; //unique number of experimental conditions
  arma::uword nmax_;
  arma::uvec p_;
  arma::uvec q_;
  
public:
  arma::uvec idx_in_; 
  arma::uvec idx_in_sub_;
  arma::uvec idx_in_rm_;
  const arma::uvec exp_cond_; //vector linking row to experimental condition number
  arma::uword r_in_design_;
  arma::uword r_in_rm_;
  arma::uvec rows_in_design_;
  arma::uvec count_exp_cond_;
  arma::uvec count_exp_cond_rm_;
  double val_; // overall best value
  double new_val_; // new value
  double rm_val_;
  arma::uword fcalls_;
  arma::uword matops_;
  
  arma::cube A_list_; // inverse sigma matrices
  arma::cube rm1A_list_; // inverse sigma matrices with one removed - initialised to minus one but now needs to resize
  arma::field<arma::mat> M_list_;
  arma::field<arma::mat> M_list_sub_;
  arma::field<arma::mat> V0_list_;
  const arma::uvec nfix_; //the indexes of the experimental conditions to keep
  const arma::uword rd_mode_; // robust designs mode: 1 == weighted, 2 == minimax.
  
  bool trace_;
  bool uncorr_;
  bool bayes_;
  
public:
  HillClimbing(arma::uvec idx_in, 
               arma::uword n,
               arma::field<arma::vec> C_list, 
               arma::field<arma::mat> X_list, 
               arma::field<arma::mat> Z_list, 
               arma::field<arma::mat> D_list,
               arma::mat w_diag,
               arma::uvec max_obs,
               arma::vec weights,
               arma::uvec exp_cond,
               arma::uword any_fix,
               arma::uvec nfix,
               arma::field<arma::mat> V0_list,
               arma::uword rd_mode = 0, 
               bool trace=false,
               bool uncorr=false,
               bool bayes = false) :
  C_list_(C_list), 
  D_list_(D_list),
  X_all_list_(X_list),
  Z_all_list_(Z_list),
  W_all_diag_(w_diag),
  weights_(weights), 
  max_obs_(max_obs),
  curr_obs_(max_obs.n_elem,fill::zeros),
  nlist_(weights_.n_elem),
  any_fix_(any_fix),
  n_(n), 
  k_(max_obs.n_elem),
  nmax_(2*ceil(X_all_list_(0,0).n_rows/k_)*n_),
  p_(nlist_,fill::zeros),
  q_(nlist_,fill::zeros),
  idx_in_(idx_in),
  idx_in_sub_(idx_in),
  idx_in_rm_(idx_in),
  exp_cond_(exp_cond),
  r_in_design_(0),
  r_in_rm_(0),
  rows_in_design_(nmax_,arma::fill::zeros),
  count_exp_cond_(nmax_,arma::fill::zeros),
  count_exp_cond_rm_(nmax_,arma::fill::zeros),
  val_(0.0), 
  new_val_(0.0),
  rm_val_(0.0),
  fcalls_(0),
  matops_(0),
  A_list_(nmax_,nmax_,nlist_,fill::zeros), 
  rm1A_list_(nmax_,nmax_,nlist_,fill::zeros),
  M_list_(nlist_),
  M_list_sub_(nlist_),
  V0_list_(V0_list),
  nfix_(nfix), 
  rd_mode_(rd_mode), 
  trace_(trace),
  uncorr_(uncorr),
  bayes_(bayes){
    build_XZ();
  }
  
  arma::uvec join_idx(const arma::uvec &idx, arma::uword elem) {
    return arma::join_cols(idx, arma::uvec({elem}));
  }
  
  void build_XZ(){
    //Rcpp::Rcout << "\nStart XZ";
    idx_in_ = sort(idx_in_);
    for(arma::uword i=0; i<idx_in_.n_elem; i++){
      curr_obs_(idx_in_(i)-1)++;
    }
    idx_in_sub_ = idx_in_;
    arma::vec idx = arma::linspace(0, nmax_-1, nmax_);
    rows_in_design_ = arma::conv_to<arma::uvec>::from(idx);
    arma::vec vals(nlist_);
    arma::uword rowcount;
    for(arma::uword j=0; j<nlist_;j++){
      p_(j) = X_all_list_(j,0).n_cols;
      q_(j) = Z_all_list_(j,0).n_cols;
      arma::mat X(nmax_,p_(j),fill::zeros);
      arma::mat Z(nmax_,q_(j),fill::zeros);
      arma::vec w_diag(nmax_);
      rowcount = 0;
      for(arma::uword k=0; k<idx_in_.n_elem;k++){
        arma::uvec rowstoincl = find(exp_cond_ == idx_in_(k));
        for(arma::uword l=0;l<rowstoincl.n_elem;l++){
          X.row(rowcount) = X_all_list_(j,0).row(rowstoincl(l));
          Z.row(rowcount) = Z_all_list_(j,0).row(rowstoincl(l));
          w_diag(rowcount) = W_all_diag_(rowstoincl(l),j);
          if(j==0)count_exp_cond_(k)++;
          rowcount++;
        }
      }
      if(j==0)r_in_design_ = rowcount;
      arma::mat tmp = Z.head_rows(rowcount)*D_list_(j,0)*Z.head_rows(rowcount).t();
      tmp.diag() += w_diag.head(rowcount);
      M_list_(j,0) = X.head_rows(rowcount).t() * tmp.i() * X.head_rows(rowcount);
      M_list_sub_(j,0) = M_list_(j,0);
      if(uncorr_){
        vals(j) = bayes_ ? c_obj_fun( M_list_(j,0) + V0_list_(j,0), C_list_(j,0)) : c_obj_fun( M_list_(j,0), C_list_(j,0));
      } else {
        A_list_.slice(j).submat(0,0,r_in_design_-1,r_in_design_-1) = tmp.i();
        vals(j) = bayes_ ? c_obj_fun( X.head_rows(rowcount).t() * A_list_.slice(j).submat(0,0,r_in_design_-1,r_in_design_-1) * X.head_rows(rowcount) + V0_list_(j,0), C_list_(j,0)) : c_obj_fun( X.head_rows(rowcount).t() * A_list_.slice(j).submat(0,0,r_in_design_-1,r_in_design_-1) * X.head_rows(rowcount), C_list_(j,0));
      }
      
    }
    new_val_ = rd_mode_ == 1 ? arma::dot(vals, weights_) : vals.max();
    if(trace_)Rcpp::Rcout << "\nval: " << new_val_;
  }
  
  void local_search(){
    if (trace_) Rcpp::Rcout << "\nLocal search";
    int i = 0;
    double diff = -1.0;
    while(diff < 0){
      i++;
      val_ = new_val_;
      if (trace_) Rcpp::Rcout << "\nIter " << i << ": Var: " << val_;
      // evaluate the swaps
      arma::mat val_swap(k_,k_,fill::value(10000));
      for(arma::uword j=1; j < k_+1; j++){
        if(any(idx_in_ == j)){
          arma::vec val_in_vec = eval(true,j);
          val_swap.row(j-1) = val_in_vec.t();
        }
      }
      
      double newval = val_swap.min();
      diff = newval - val_;
      if (trace_) Rcpp::Rcout << " diff: " << diff;
      if(diff < 0){
        arma::uword swap_sort = val_swap.index_min();
        arma::uword target = floor(swap_sort/k_); 
        arma::uword rm_target = (swap_sort) - target*k_;
        if(uncorr_){
          rm_obs_uncor(rm_target+1);
          new_val_ = add_obs_uncor(target+1,true,true);
        } else {
          rm_obs(rm_target+1);
          new_val_ = add_obs(target+1,true,true);
        }
      }
    }
  }
  
  void greedy_search(){
    // step 1: find optimal smallest design
    int i = 0;
    if (trace_) Rcpp::Rcout << "\nidx: " << idx_in_.t();
    if (trace_) Rcpp::Rcout << "\nGreedy search: " << n_;
    arma::uword idxcount = idx_in_.n_elem;
    while(idxcount < n_){
      i++;
      idxcount++;
      val_ = new_val_;
      if (trace_) Rcpp::Rcout << "\nIter " << i << " size: " << idxcount << " Var: " << val_ ;
      arma::vec val_swap = eval(false);
      arma::uword swap_sort = val_swap.index_min();
      if (trace_) Rcpp::Rcout << " adding " << swap_sort+1;
      if(uncorr_){
        new_val_ = add_obs_uncor(swap_sort+1,false,true);
      } else {
        new_val_ = add_obs(swap_sort+1,false,true); 
      }
    }
  }
  
private:
  arma::uvec get_rows(arma::uword idx){
    arma::uword start = (idx == 0) ? 0 : sum(count_exp_cond_.subvec(0,idx-1));
    arma::uword end = start + count_exp_cond_(idx) - 1;
    return rows_in_design_.subvec(start,end);
  }
  
  arma::uvec get_all_rows(arma::uvec idx){
    // get the rows corresponding to the given vector of experimental conditions
    arma::uvec rowidx(nmax_);
    arma::uword count = 0;
    for(arma::uword i = 0; i < idx.n_elem; i++){
      arma::uvec addidx = find(exp_cond_ == idx(i));
      for(arma::uword j = 0; j < addidx.n_elem; j++){
        rowidx(count) = addidx(j);
        count++;
      }
    }
    return rowidx.head(count);
  }
  
  arma::uvec idx_minus(arma::uword idx){
    arma::uword start = (idx == 0) ? 0 : sum(count_exp_cond_.subvec(0,idx-1));
    arma::uword end = start + count_exp_cond_(idx) - 1;
    arma::uvec idxrtn(r_in_design_-(end-start+1));
    idxrtn.head(start) = rows_in_design_.head(start);
    if(end < r_in_design_-1)idxrtn.tail(r_in_design_-end) = rows_in_design_.subvec(end+1,r_in_design_-1);
    return rows_in_design_.subvec(start,end);
  }
  
  void rm_obs(arma::uword outobs){
    arma::uvec rm_cond = find(idx_in_== outobs);
    arma::uvec rowstorm = get_rows(rm_cond(0));
    idx_in_rm_ = uvec_minus(idx_in_,rm_cond(0));
    arma::uvec idxexist = get_all_rows(idx_in_rm_);
    for (arma::uword idx = 0; idx < nlist_; ++idx) {
      matops_++;
      arma::mat A1 = A_list_.slice(idx).submat(0,0,r_in_design_-1,r_in_design_-1);
      const arma::mat rm1A = remove_one_many_mat(A1, rowstorm);
      if(idx==0)r_in_rm_ = rm1A.n_rows;
      rm1A_list_.slice(idx).submat(0,0,r_in_rm_-1,r_in_rm_-1) = rm1A;
      M_list_sub_(idx,0) = X_all_list_(idx,0).rows(idxexist).t()*rm1A*X_all_list_(idx,0).rows(idxexist);
    }
    
    count_exp_cond_rm_.head(rm_cond(0)) = count_exp_cond_.head(rm_cond(0));
    if(rm_cond(0)>=idx_in_.n_elem - 1){
      count_exp_cond_rm_(rm_cond(0)) = count_exp_cond_(rm_cond(0)+1);
    } else {
      count_exp_cond_rm_.subvec(rm_cond(0),idx_in_.n_elem-2) = count_exp_cond_.subvec(rm_cond(0)+1,idx_in_.n_elem-1);
    }
  }
  
  void rm_obs_uncor(arma::uword outobs){
    arma::uvec rm_cond = find(idx_in_ == outobs);
    arma::uvec rowstorm = find(exp_cond_ == outobs);
    for(arma::uword j=0; j<nlist_;j++){
      arma::mat X(rowstorm.n_elem,p_(j),fill::zeros);
      arma::mat Z(rowstorm.n_elem,q_(j),fill::zeros);
      arma::vec w_diag(rowstorm.n_elem);
      
      for(arma::uword l=0;l<rowstorm.n_elem;l++){
        X.row(l) = X_all_list_(j,0).row(rowstorm(l));
        Z.row(l) = Z_all_list_(j,0).row(rowstorm(l));
        w_diag(l) = W_all_diag_(rowstorm(l),j);
      }
      
      arma::mat tmp = Z*D_list_(j,0)*Z.t();
      tmp.diag() += w_diag;
      M_list_sub_(j,0) = M_list_(j,0) - X.t() * tmp.i() * X;
    }
    idx_in_rm_ = uvec_minus(idx_in_,rm_cond(0));
    count_exp_cond_rm_.head(rm_cond(0)) = count_exp_cond_.head(rm_cond(0));
    if(rm_cond(0)>=idx_in_.n_elem - 1){
      count_exp_cond_rm_(rm_cond(0)) = count_exp_cond_(rm_cond(0)+1);
    } else {
      count_exp_cond_rm_.subvec(rm_cond(0),idx_in_.n_elem-2) = count_exp_cond_.subvec(rm_cond(0)+1,idx_in_.n_elem-1);
    }
  }
  
  double add_obs(arma::uword inobs,
                 bool userm = true,
                 bool keep = false){
    arma::uvec rowstoadd = find(exp_cond_ == inobs);
    arma::uvec idxvec = userm ? idx_in_rm_ : idx_in_;
    arma::uvec idxexist = get_all_rows(idxvec);
    arma::uword n_to_add = rowstoadd.n_elem;
    arma::uword n_already_in = idxexist.n_elem;
    arma::uvec idx_in_vec(n_already_in + n_to_add, fill::zeros);
    arma::vec vals(nlist_);
    bool issympd = true;
    arma::uword r_in_design_tmp_ = r_in_design_;
    for (arma::uword idx = 0; idx < nlist_; ++idx) {
      arma::mat M;
      arma::mat A = userm ? rm1A_list_.slice(idx).submat(0,0,r_in_rm_-1,r_in_rm_-1) : 
        A_list_.slice(idx).submat(0,0,r_in_design_tmp_-1,r_in_design_tmp_-1);
      if(!keep){
        arma::mat Z1 = Z_all_list_(idx,0).rows(idxexist);
        arma::mat Z2 = Z_all_list_(idx,0).rows(rowstoadd);
        arma::mat sig112 = (Z2*D_list_(idx,0)*Z1.t());
        arma::mat sig112A = sig112*A;
        arma::mat sig2 = Z2*D_list_(idx,0)*Z2.t();
        sig2.diag() += W_all_diag_(rowstoadd);
        arma::mat S = sig2 - sig112A*sig112.t();
        arma::mat X12 = X_all_list_(idx,0).rows(rowstoadd) - sig112A*X_all_list_(idx,0).rows(idxexist);
        M = userm ? M_list_sub_(idx,0) + X12.t() * S.i() * X12 : M_list_(idx,0) + X12.t() * S.i() * X12;
      } else {
        n_already_in = idxexist.n_elem;
        arma::mat X(n_already_in + n_to_add,p_(idx),fill::zeros);
        idx_in_vec.fill(0);
        idx_in_vec(span(0,n_already_in - 1)) = idxexist;
        X.rows(span(0,n_already_in-1)) = X_all_list_(idx,0).rows(idxexist);
        for(arma::uword j = 0; j < n_to_add; j++){
          arma::rowvec z_j = Z_all_list_(idx,0).row(rowstoadd(j));
          arma::mat z_d = Z_all_list_(idx,0).rows(idx_in_vec(span(0,n_already_in - 1)));
          double sig_jj = arma::as_scalar(z_j * D_list_(idx,0) * z_j.t()); 
          sig_jj += W_all_diag_(rowstoadd(j));
          arma::vec f = z_d * D_list_(idx,0) * z_j.t();
          A = add_one_mat(A, 
                          sig_jj,
                          f);
          idx_in_vec(n_already_in) = rowstoadd(j);
          X.row(n_already_in) = X_all_list_(idx,0).row(rowstoadd(j));
          n_already_in++;
        }
        //check if positive definite
        M = X.t() * A * X;
      }
      issympd = M.is_sympd();
      if(issympd){
        if(keep){
          if(idx==0)r_in_design_ = A.n_rows;
          M_list_(idx,0) = M;
          A_list_.slice(idx).submat(0,0,r_in_design_-1,r_in_design_-1) = A;
        }
        vals(idx) = bayes_ ? c_obj_fun( M+V0_list_(idx,0), C_list_(idx,0)) : c_obj_fun( M, C_list_(idx,0));
      } else {
        vals.fill(10000);
        break;
      }
    }
    if(keep && issympd){
      if(userm){
        idx_in_ = join_idx(idx_in_rm_,inobs);
        curr_obs_(inobs-1)++;
        count_exp_cond_.subvec(0,idx_in_.n_elem-2) = count_exp_cond_rm_.subvec(0,idx_in_.n_elem-2);
        count_exp_cond_(idx_in_.n_elem-1) = n_to_add;
      } else {
        idx_in_ = join_idx(idx_in_,inobs);
        curr_obs_(inobs-1)++;
        count_exp_cond_(idx_in_.n_elem-1) = n_to_add;
      }
    }
    double rtn = rd_mode_ == 1 ? arma::dot(vals, weights_) : vals.max();
    if(rtn < 10000){
      return rtn;
    } else {
      return 10000;
    }
  }
  
  double add_obs_uncor(arma::uword inobs,
                       bool userm = true,
                       bool keep = false){
    arma::vec vals(nlist_);
    arma::uvec rowstoadd = find(exp_cond_ == inobs);
    bool issympd = true;
    for(arma::uword j=0; j<nlist_;j++){
      arma::mat X(rowstoadd.n_elem,p_(j),fill::zeros);
      arma::mat Z(rowstoadd.n_elem,q_(j),fill::zeros);
      arma::vec w_diag(rowstoadd.n_elem);
      
      for(arma::uword l=0;l<rowstoadd.n_elem;l++){
        X.row(l) = X_all_list_(j,0).row(rowstoadd(l));
        Z.row(l) = Z_all_list_(j,0).row(rowstoadd(l));
        w_diag(l) = W_all_diag_(rowstoadd(l),j);
      }
      
      arma::mat tmp = Z*D_list_(j,0)*Z.t();
      tmp.diag() += w_diag;
      arma::mat M = userm ? M_list_sub_(j,0) : M_list_(j,0);
      
      M += X.t() * tmp.i() * X;
      issympd = M.is_sympd();
      if(issympd){
        if(keep){
          M_list_(j,0) = M;
        }
        vals(j) = bayes_ ? c_obj_fun(M + V0_list_(j,0), C_list_(j,0)) : c_obj_fun(M, C_list_(j,0));
      } else {
        vals.fill(10000);
        break;
      }
    }
    if(keep && issympd){
      if(userm){
        idx_in_ = join_idx(idx_in_rm_,inobs);
        curr_obs_(inobs-1)++;
        count_exp_cond_.subvec(0,idx_in_.n_elem-2) = count_exp_cond_rm_.subvec(0,idx_in_.n_elem-2);
        count_exp_cond_(idx_in_.n_elem-1) = rowstoadd.n_elem;
      } else {
        idx_in_ = join_idx(idx_in_,inobs);
        curr_obs_(inobs-1)++;
        count_exp_cond_(idx_in_.n_elem-1) = rowstoadd.n_elem;
      }
    }
    double rtn = rd_mode_ == 1 ? arma::dot(vals, weights_) : vals.max();
    if(rtn < 10000){
      return rtn;
    } else {
      return 10000;
    }
  }
  
  arma::vec eval(bool userm = true, arma::uword obs = 0){
    arma::vec val_in_mat(k_,arma::fill::value(10000));
    if(userm){
      bool obsisin = any(idx_in_ == obs);
      if(obsisin){
        if(uncorr_){
          rm_obs_uncor(obs);
        } else {
          rm_obs(obs);
        }
#pragma omp parallel for
        for (arma::uword i = 1; i < k_+1; ++i) {
          if(obs != i && curr_obs_(i-1)<max_obs_(i-1)){
            if(uncorr_){
              val_in_mat(i-1) = add_obs_uncor(i,true,false);
              
            } else {
              val_in_mat(i-1) = add_obs(i,true,false);
            }
          } 
        }
        matops_ += k_*nlist_;
        fcalls_ += k_*nlist_;
      } 
    } else {
#pragma omp parallel for
      for (arma::uword i = 1; i < k_+1; ++i) {
        if(curr_obs_(i-1)<max_obs_(i-1)){
          if(uncorr_){
            val_in_mat(i-1) = add_obs_uncor(i,false,false);
          } else {
            val_in_mat(i-1) = add_obs(i,false,false);
          }
        }
      }
      matops_ += k_*nlist_;
      fcalls_ += k_*nlist_;
    }
    
    return val_in_mat;
  }
  
};

#endif