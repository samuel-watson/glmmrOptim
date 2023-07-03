#ifndef OPTIMCLASS_H
#define OPTIMCLASS_H

#include <cmath> 
#include <RcppEigen.h>
#include "optimlinalg.h"
#include "matrixfield.h"
#include "eigenext.h"
#include "optimmaths.h"
#include "optimdata.h"
#include <glmmr/openmpheader.h>

using namespace Eigen;

namespace glmmr {

// I plan to refactor this class as it is too big!

class OptimDesign {
private:
  glmmr::OptimData& data_;
  ArrayXi curr_obs_;
  const int nlist_;
  int n_;
  int k_;
  int nmax_;
  ArrayXi p_;
  ArrayXi q_;
  
public:
  ArrayXi idx_in_;
  ArrayXi idx_in_sub_;
  ArrayXi idx_in_rm_;
  int r_in_design_=0;
  int r_in_rm_=0;
  ArrayXi rows_in_design_;
  ArrayXi count_exp_cond_;
  ArrayXi count_exp_cond_rm_;
  double val_ = 0.0; // overall best value
  double new_val_=0.0; // new value
  double rm_val_=0.0;
  int fcalls_=0;
  int matops_=0;
  MatrixXd A_list_;// inverse sigma matrices
  MatrixXd rm1A_list_;// inverse sigma matrices with one removed - initialised to minus one but now needs to resize
  glmmr::MatrixField<MatrixXd> M_list_;
  glmmr::MatrixField<MatrixXd> M_list_sub_;
  const bool robust_log_; // robust designs mode: true = log sum, false = sum
  bool trace_;
  bool uncorr_;
  bool bayes_;
  
public:
  OptimDesign(const ArrayXi& idx_in, 
              int n,
              glmmr::OptimData& data,
              int nmax,
              bool robust_log = false, 
              bool trace=false,
              bool uncorr=false,
              bool bayes = false) :
  data_(data),
  curr_obs_(data_.max_obs_.size()),//set to zero
  nlist_(data_.weights_.size()),
  n_(n), 
  k_(data_.max_obs_.size()),
  nmax_(nmax),
  p_(nlist_), //set to zero
  q_(nlist_), //set to zero
  idx_in_(idx_in),
  idx_in_sub_(idx_in),
  idx_in_rm_(idx_in),
  rows_in_design_(nmax_),//set to zero
  count_exp_cond_(nmax_),//set to zero
  count_exp_cond_rm_(nmax_),//set to zero
  A_list_(nmax_*nlist_,nmax_), //set to zero
  rm1A_list_(nmax_*nlist_,nmax_),//set to zero
  robust_log_(robust_log), 
  trace_(trace),
  uncorr_(uncorr),
  bayes_(bayes){
    build_XZ();
  }
  
  ArrayXi join_idx(const ArrayXi &idx,int elem);
  void build_XZ();
  void local_search();
  void greedy_search();
  void reverse_greedy_search();
  
private:
  ArrayXi get_rows(int idx);
  ArrayXi get_all_rows(ArrayXi idx);
  ArrayXi idx_minus(int idx);
  double rm_obs(int outobs,bool keep = false,bool keep_mat = false,bool rtn_val = false);
  double rm_obs_uncor(int outobs,bool keep = false,bool keep_mat = false,bool rtn_val = false);
  double add_obs(int inobs,bool userm = true,bool keep = false);
  double add_obs_uncor(int inobs,bool userm = true,bool keep = false);
  ArrayXd eval(bool userm = true, int obs = 0);
};

}

inline ArrayXi glmmr::OptimDesign::join_idx(const ArrayXi &idx,
                        int elem){
  ArrayXi newidx(idx.size()+1);
  newidx.segment(0,idx.size()) = idx;
  newidx(idx.size()) = elem;
  return newidx;
}

inline void glmmr::OptimDesign::build_XZ(){
  curr_obs_ = ArrayXi::Zero(curr_obs_.size());
  p_ = ArrayXi::Zero(nlist_);
  q_ = ArrayXi::Zero(nlist_);
  rows_in_design_ = ArrayXi::Zero(nmax_);
  count_exp_cond_ = ArrayXi::Zero(nmax_);
  count_exp_cond_rm_ = ArrayXi::Zero(nmax_);
  A_list_ = MatrixXd::Zero(nmax_*nlist_,nmax_);
  rm1A_list_ = MatrixXd::Zero(nmax_*nlist_,nmax_);
  std::sort(idx_in_.data(),idx_in_.data()+idx_in_.size());
  for(int i=0; i<idx_in_.size(); i++){
    curr_obs_(idx_in_(i)-1)++;
  }
  idx_in_sub_ = idx_in_;
  rows_in_design_ = ArrayXi::LinSpaced(nmax_,0,nmax_-1);
  VectorXd vals(nlist_);
  int rowcount;
  for(int j=0; j<nlist_;j++){
    p_(j) = data_.X_all_list_.cols(j);
    q_(j) = data_.Z_all_list_.cols(j);
    MatrixXd X = MatrixXd::Zero(nmax_,p_(j));
    MatrixXd Z = MatrixXd::Zero(nmax_,q_(j));
    VectorXd w_diag(nmax_);
    rowcount = 0;
    for(int k=0; k<idx_in_.size();k++){
      ArrayXi rowstoincl = glmmr::Eigen_ext::find(data_.exp_cond_, idx_in_(k));
      for(int l=0;l<rowstoincl.size();l++){
        X.row(rowcount) = data_.X_all_list_.get_row(j,rowstoincl(l));
        Z.row(rowcount) = data_.Z_all_list_.get_row(j,rowstoincl(l));
        w_diag(rowcount) = data_.W_all_diag_(rowstoincl(l),j);
        if(j==0)count_exp_cond_(k)++;
        rowcount++;
      }
    }
    if(j==0)r_in_design_ = rowcount;
    MatrixXd Dt = data_.D_list_(j);
    MatrixXd tmp = Z.topRows(rowcount)*Dt*Z.topRows(rowcount).transpose();
    MatrixXd I = MatrixXd::Identity(tmp.rows(),tmp.cols());
    tmp.diagonal() += w_diag.head(rowcount);
    MatrixXd M = X.topRows(rowcount).transpose() * tmp.llt().solve(I) * X.topRows(rowcount);
    M_list_.add(M);
    M_list_sub_.add(M);
    if(uncorr_){
      vals(j) = bayes_ ? glmmr::maths::c_obj_fun( M_list_(j) + data_.V0_list_(j), data_.C_list_(j)) : glmmr::maths::c_obj_fun( M_list_(j), data_.C_list_(j));
    } else {
      A_list_.block(j*nmax_,0,r_in_design_,r_in_design_) = tmp.llt().solve(I);
      vals(j) = bayes_ ? glmmr::maths::c_obj_fun( X.topRows(rowcount).transpose() * A_list_.block(j*nmax_,0,r_in_design_,r_in_design_) * X.topRows(rowcount) +
        data_.V0_list_(j), data_.C_list_(j)) : glmmr::maths::c_obj_fun( X.topRows(rowcount).transpose() * A_list_.block(j*nmax_,0,r_in_design_,r_in_design_) * X.topRows(rowcount), data_.C_list_(j));
    }
  }
  new_val_ = robust_log_ ? vals.log().transpose()*data_.weights_ : vals.transpose()*data_.weights_;
  if(trace_)Rcpp::Rcout << "\nStarting val: " << new_val_;
}

// LOCAL SEARCH ALGORITHM
inline void glmmr::OptimDesign::local_search(){
  if (trace_) Rcpp::Rcout << "\nLOCAL SEARCH";
  int i = 0;
  double diff = -1.0;
  double tmp;
  
  while(diff < 0){
    i++;
    val_ = new_val_;
    if (trace_) Rcpp::Rcout << "\nIter " << i << ": Current value: " << val_;
    // evaluate the swaps
    ArrayXXd val_swap = ArrayXXd::Constant(k_,k_,10000);
    for(int j=1; j < k_+1; j++){
      if((idx_in_ == j).any()){
        ArrayXd val_in_vec = eval(true,j); //eval is a member function of this class
        val_swap.row(j-1) = val_in_vec.transpose();
      }
    }
    Index minrow, mincol;
    double newval = val_swap.minCoeff(&minrow,&mincol);
    diff = newval - val_;
    if (trace_) Rcpp::Rcout << " || Best Difference: " << diff << " Best New value: " << newval;
    
    if(diff < 0){
      if(uncorr_){
        tmp = rm_obs_uncor(1+(int)minrow,true);
        new_val_ = add_obs_uncor(1+(int)mincol,true,true);
      } else {
        tmp = rm_obs(1+(int)minrow,true);
        new_val_ = add_obs(1+(int)mincol,true,true);
      }
    } else {
      if (trace_) Rcpp::Rcout << "\nFINISHED LOCAL SEARCH";
    }
    
  }
}

inline void glmmr::OptimDesign::greedy_search(){
  // step 1: find optimal smallest design
  int i = 0;
  if (trace_) Rcpp::Rcout << "\nStarting conditions: " << idx_in_.transpose();
  if (trace_) Rcpp::Rcout << "\nGREEDY SEARCH for design of size " << n_;
  int idxcount = idx_in_.size();
  while(idxcount < n_){
    i++;
    idxcount++;
    val_ = new_val_;
    if (trace_) Rcpp::Rcout << "\nIter " << i << " size: " << idxcount << " Current value: " << val_ ;
    ArrayXd val_swap = eval(false);
    Index swap_sort;
    double min = val_swap.minCoeff(&swap_sort);
    if (trace_) Rcpp::Rcout << " adding " << swap_sort+1;
    if(uncorr_){
      new_val_ = add_obs_uncor((int)swap_sort+1,false,true);
    } else {
      new_val_ = add_obs((int)swap_sort+1,false,true); 
    }
  }
  if (trace_) Rcpp::Rcout << "\nFINISHED GREEDY SEARCH";
}

inline void glmmr::OptimDesign::reverse_greedy_search(){
  // start from the whole design space and then successively remove observations
  if (trace_) Rcpp::Rcout << "\nREVERSE GREEDY SEARCH for design of size " << n_;
  int i = 0;
  int idxcount = idx_in_.size();
  ArrayXd val_rm(k_);
  
  while(idxcount > n_){
    i++;
    val_ = new_val_;
    if (trace_) Rcpp::Rcout << "\nIter " << i << " size: " << idxcount << " Current value: " << val_ ;
    for(int j = 1; j< k_+1; j++){
      if((idx_in_ == j).any()){
        if(uncorr_){
          val_rm(j-1) = rm_obs_uncor(j,false,false,true);
        } else {
          val_rm(j-1) = rm_obs(j,false,false,true);
        }
      } else {
        val_rm(j-1) = 10000;
      }
    }
    Index swap_sort;
    double min = val_rm.minCoeff(&swap_sort);
    if (trace_) Rcpp::Rcout << " removing " << swap_sort+1;
    if(uncorr_){
      new_val_ = rm_obs_uncor((int)swap_sort+1,true,true,true);
    } else {
      new_val_ = rm_obs((int)swap_sort+1,true,true,true);
    }
    idxcount--;
  }
  val_ = new_val_;
  if (trace_) Rcpp::Rcout << "\nFINISHED REVERSE GREEDY SEARCH";
}

// get rows corresponding to an experimental condition
inline ArrayXi glmmr::OptimDesign::get_rows(int idx){
  int start = (idx == 0) ? 0 : count_exp_cond_.segment(0,idx).sum();
  return rows_in_design_.segment(start,count_exp_cond_(idx));
}

// get rows corresponding to a set of experimental condition
inline ArrayXi glmmr::OptimDesign::get_all_rows(ArrayXi idx){
  ArrayXi rowidx(nmax_);
  int count = 0;
  for(int i = 0; i < idx.size(); i++){
    ArrayXi addidx = glmmr::Eigen_ext::find(data_.exp_cond_,idx(i));
    rowidx.segment(count,addidx.size()) = addidx;
    count += addidx.size();
  }
  return rowidx.segment(0,count);
}

//remove rows corresponding to experimental condition
inline ArrayXi glmmr::OptimDesign::idx_minus(int idx){
  int start = (idx == 0) ? 0 : count_exp_cond_.segment(0,idx).sum();
  int end = start + count_exp_cond_(idx) - 1;
  ArrayXi idxrtn(r_in_design_-(end-start+1));
  idxrtn.head(start) = rows_in_design_.head(start);
  if(end < r_in_design_-1)idxrtn.tail(r_in_design_-end) = rows_in_design_.segment(end+1,r_in_design_-end-1);
  return rows_in_design_.segment(start,end-start+1);
}

// remove observation
// 
inline double glmmr::OptimDesign::rm_obs(int outobs,
              bool keep,
              bool keep_mat,
              bool rtn_val){
  ArrayXi rm_cond = glmmr::Eigen_ext::find(idx_in_,outobs);
  ArrayXi rowstorm = get_rows(rm_cond(0));
  idx_in_rm_ = glmmr::algo::uvec_minus(idx_in_,rm_cond(0));
  ArrayXi idxexist = get_all_rows(idx_in_rm_);
  VectorXd vals = VectorXd::Constant(nlist_,10000.0);
  for (int idx = 0; idx < nlist_; ++idx) {
    matops_++;
    MatrixXd A1 = A_list_.block(idx*nmax_,0,r_in_design_,r_in_design_);
    MatrixXd rm1A = glmmr::algo::remove_one_many_mat(A1, rowstorm);
    if(idx==0)r_in_rm_ = rm1A.rows();
    rm1A_list_.block(idx*nmax_,0,r_in_rm_,r_in_rm_) = rm1A;
    int p = data_.X_all_list_.cols(idx);
    MatrixXd X = glmmr::Eigen_ext::mat_indexing(data_.X_all_list_(idx),idxexist,ArrayXi::LinSpaced(p,0,p-1));
    MatrixXd M = X.transpose()*rm1A*X;
    M_list_sub_.replace(idx,M);
    if(rtn_val)vals(idx) = bayes_ ? glmmr::maths::c_obj_fun( M+data_.V0_list_(idx), data_.C_list_(idx)) : glmmr::maths::c_obj_fun( M, data_.C_list_(idx));
    if(keep_mat){
      M_list_.replace(idx,M);
      A_list_.block(idx*nmax_,0,r_in_rm_,r_in_rm_) = rm1A;
      if(idx==(nlist_-1))r_in_design_ = rm1A.rows();
    } 
  }
  count_exp_cond_rm_.head(rm_cond(0)) = count_exp_cond_.head(rm_cond(0));
  if(rm_cond(0)>=(idx_in_.size() - 1)){
    count_exp_cond_rm_(rm_cond(0)) = count_exp_cond_(rm_cond(0)+1);
  } else {
    count_exp_cond_rm_.segment(rm_cond(0),idx_in_.size()-rm_cond(0)-1) = count_exp_cond_.segment(rm_cond(0)+1,idx_in_.size()-rm_cond(0)-1);
  }
  if(keep){
    curr_obs_(outobs-1)--;
  }
  if(keep_mat){
    idx_in_ = idx_in_rm_;
    count_exp_cond_.segment(0,idx_in_.size()-1) = count_exp_cond_rm_.segment(0,idx_in_.size()-1);
  }
  double rtn = robust_log_ ? vals.log().transpose()*data_.weights_ : vals.transpose()*data_.weights_;
  return rtn;
}

inline double glmmr::OptimDesign::rm_obs_uncor(int outobs,
                    bool keep,
                    bool keep_mat,
                    bool rtn_val){
  ArrayXi rm_cond = glmmr::Eigen_ext::find(idx_in_,outobs);
  ArrayXi rowstorm = get_rows(rm_cond(0));
  VectorXd vals = VectorXd::Constant(nlist_,10000.0); 
  
  for(int j=0; j<nlist_;j++){
    MatrixXd X = MatrixXd::Zero(rowstorm.size(),p_(j));
    MatrixXd Z = MatrixXd::Zero(rowstorm.size(),q_(j));
    VectorXd w_diag(rowstorm.size());
    
    for(int l=0;l<rowstorm.size();l++){
      X.row(l) = data_.X_all_list_.get_row(j,rowstorm(l));
      Z.row(l) = data_.Z_all_list_.get_row(j,rowstorm(l));
      w_diag(l) = data_.W_all_diag_(rowstorm(l),j);
    }
    MatrixXd Dt = data_.D_list_(j);
    MatrixXd tmp = Z*Dt*Z.transpose();
    tmp.diagonal() += w_diag;
    MatrixXd M = M_list_(j);
    MatrixXd I = MatrixXd::Identity(tmp.rows(),tmp.cols());
    tmp = tmp.llt().solve(I);
    M.noalias() -= X.transpose()*tmp*X;
    M_list_sub_.replace(j,M);
    if(rtn_val)vals(j) = bayes_ ? glmmr::maths::c_obj_fun( M+data_.V0_list_(j), data_.C_list_(j)) : glmmr::maths::c_obj_fun( M, data_.C_list_(j));
    
    
    if(keep_mat){
      M_list_.replace(j,M);
    }
  }
  idx_in_rm_ = glmmr::algo::uvec_minus(idx_in_,rm_cond(0));
  count_exp_cond_rm_.head(rm_cond(0)) = count_exp_cond_.head(rm_cond(0));
  if(rm_cond(0)>=(idx_in_.size() - 1)){
    count_exp_cond_rm_(rm_cond(0)) = count_exp_cond_(rm_cond(0)+1);
  } else {
    count_exp_cond_rm_.segment(rm_cond(0),idx_in_.size()-rm_cond(0)-1) = count_exp_cond_.segment(rm_cond(0)+1,idx_in_.size()-rm_cond(0)-1);
  }
  
  if(keep){
    curr_obs_(outobs-1)--;
  }
  
  if(keep_mat){
    idx_in_ = idx_in_rm_;
    count_exp_cond_.segment(0,idx_in_.size()-1) = count_exp_cond_rm_.segment(0,idx_in_.size()-1);
  }
  double rtn = robust_log_ ? vals.log().transpose()*data_.weights_ : vals.transpose()*data_.weights_;
  return rtn;
}

// add a new observation
inline double glmmr::OptimDesign::add_obs(int inobs,
               bool userm,
               bool keep ){
  ArrayXi rowstoadd = glmmr::Eigen_ext::find(data_.exp_cond_,inobs);
  ArrayXi idxvec = userm ? idx_in_rm_ : idx_in_;
  ArrayXi idxexist = get_all_rows(idxvec);
  int n_to_add = rowstoadd.size();
  int n_already_in = idxexist.size();
  ArrayXi idx_in_vec = ArrayXi::Zero(n_already_in + n_to_add);
  VectorXd vals(nlist_);
  bool issympd;
  int r_in_design_tmp_ = r_in_design_;
  for (int idx = 0; idx < nlist_; ++idx) {
    MatrixXd A = userm ? rm1A_list_.block(idx*nmax_,0,r_in_rm_,r_in_rm_) : 
    A_list_.block(idx*nmax_,0,r_in_design_tmp_,r_in_design_tmp_);
    n_already_in = idxexist.size();
    MatrixXd X = MatrixXd::Zero(n_already_in + n_to_add,p_(idx));
    idx_in_vec = ArrayXi::Zero(n_already_in + n_to_add);
    idx_in_vec.segment(0,n_already_in) = idxexist;
    MatrixXd X0 = data_.X_all_list_(idx);
    X.block(0,0,n_already_in,X.cols()) = glmmr::Eigen_ext::mat_indexing(X0,idxexist,ArrayXi::LinSpaced(X0.cols(),0,X0.cols()-1));
    MatrixXd D = data_.D_list_(idx);
    for(int j = 0; j < n_to_add; j++){
      RowVectorXd z_j = data_.Z_all_list_.get_row(idx,rowstoadd(j));
      int zcols = data_.Z_all_list_.cols(idx);
      MatrixXd z_d = glmmr::Eigen_ext::mat_indexing(data_.Z_all_list_(idx),idx_in_vec.segment(0,n_already_in),ArrayXi::LinSpaced(zcols,0,zcols-1));
      double sig_jj = z_j * D * z_j.transpose(); 
      sig_jj += data_.W_all_diag_(rowstoadd(j),idx);
      VectorXd f = z_d * D * z_j.transpose();
      A = glmmr::algo::add_one_mat(A, sig_jj,f);
      idx_in_vec(n_already_in) = rowstoadd(j);
      X.row(n_already_in) = data_.X_all_list_.get_row(idx,rowstoadd(j));  
      n_already_in++;
    }
    MatrixXd M = X.transpose() * A * X;
    issympd = glmmr::Eigen_ext::issympd(M);
    if(!issympd){
      if(keep){
        if(idx==0)r_in_design_ = A.rows();
        M_list_.replace(idx,M);
        A_list_.block(idx*nmax_,0,r_in_design_,r_in_design_) = A;
      }
      vals(idx) = bayes_ ? glmmr::maths::c_obj_fun( M+data_.V0_list_(idx), data_.C_list_(idx)) : glmmr::maths::c_obj_fun( M, data_.C_list_(idx));
    } else {
      for(int k = 0; k<vals.size(); k++)vals(k) = 10000;
      break;
    }
  }
  if(keep && !issympd){
    if(userm){
      idx_in_ = join_idx(idx_in_rm_,inobs);
      curr_obs_(inobs-1)++;
      count_exp_cond_.segment(0,idx_in_.size()-1) = count_exp_cond_rm_.segment(0,idx_in_.size()-1);
      count_exp_cond_(idx_in_.size()-1) = n_to_add;
    } else {
      idx_in_ = join_idx(idx_in_,inobs);
      curr_obs_(inobs-1)++;
      count_exp_cond_(idx_in_.size()-1) = n_to_add;
    }
  }
  double rtn = robust_log_ ? vals.log().transpose()*data_.weights_ : vals.transpose()*data_.weights_;
  if(rtn < 10000){
    return rtn;
  } else {
    return 10000;
  }
}

inline double glmmr::OptimDesign::add_obs_uncor(int inobs,
                     bool userm,
                     bool keep){
  VectorXd vals(nlist_);
  ArrayXi rowstoadd = glmmr::Eigen_ext::find(data_.exp_cond_,inobs);
  bool issympd = true;
  for(int j=0; j<nlist_;j++){
    MatrixXd X = MatrixXd::Zero(rowstoadd.size(),p_(j));
    MatrixXd Z = MatrixXd::Zero(rowstoadd.size(),q_(j));
    VectorXd w_diag(rowstoadd.size());
    for(int l=0;l<rowstoadd.size();l++){
      X.row(l) = data_.X_all_list_.get_row(j,rowstoadd(l));
      Z.row(l) = data_.Z_all_list_.get_row(j,rowstoadd(l));
      w_diag(l) = data_.W_all_diag_(rowstoadd(l),j);
    }
    MatrixXd Dt = data_.D_list_(j);
    MatrixXd tmp = Z*Dt*Z.transpose();
    tmp.diagonal() += w_diag;
    MatrixXd M = userm ? M_list_sub_(j) : M_list_(j);
    MatrixXd iden = MatrixXd::Identity(tmp.rows(),tmp.cols());
    M += X.transpose() * tmp.llt().solve(iden) * X;
    issympd = glmmr::Eigen_ext::issympd(M);
    if(!issympd){
      if(keep){
        M_list_.replace(j,M);
      }
      vals(j) = bayes_ ? glmmr::maths::c_obj_fun( M+data_.V0_list_(j), data_.C_list_(j)) : glmmr::maths::c_obj_fun( M, data_.C_list_(j));
    } else {
      for(int k = 0; k<vals.size(); k++)vals(k) = 10000;
      break;
    }
  }
  if(keep && !issympd){
    if(userm){
      idx_in_ = join_idx(idx_in_rm_,inobs);
      curr_obs_(inobs-1)++;
      count_exp_cond_.segment(0,idx_in_.size()-1) = count_exp_cond_rm_.segment(0,idx_in_.size()-1);
      count_exp_cond_(idx_in_.size()-1) = rowstoadd.size();
    } else {
      idx_in_ = join_idx(idx_in_,inobs);
      curr_obs_(inobs-1)++;
      count_exp_cond_(idx_in_.size()-1) = rowstoadd.size();
    }
  }
  double rtn = robust_log_ ? vals.log().transpose()*data_.weights_ : vals.transpose()*data_.weights_;
  if(rtn < 10000){
    return rtn;
  } else {
    return 10000;
  }
}

inline ArrayXd glmmr::OptimDesign::eval(bool userm, int obs){
  ArrayXd val_in_mat = ArrayXd::Constant(k_,10000);
  double tmp;
  if(userm){
    if(uncorr_){
      tmp = rm_obs_uncor(obs);
    } else {
      tmp = rm_obs(obs);
    }
#pragma omp parallel for
    for (int i = 1; i < k_+1; ++i) {
      if(obs != i && curr_obs_(i-1)<data_.max_obs_(i-1)){
        if(uncorr_){
          val_in_mat(i-1) = add_obs_uncor(i,true,false);
        } else {
          val_in_mat(i-1) = add_obs(i,true,false);
        }
      } 
    }
    matops_ += k_*nlist_;
    fcalls_ += k_*nlist_;
  } else {
#pragma omp parallel for
    for (int i = 1; i < k_+1; ++i) {
      if(curr_obs_(i-1)<data_.max_obs_(i-1)){
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

#endif