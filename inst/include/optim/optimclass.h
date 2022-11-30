#ifndef OPTIMCLASS_H
#define OPTIMCLASS_H

#include <cmath> 
#include <RcppEigen.h>
#include <glmmr.h>
#include "optimlinalg.h"
#include "matrixfield.h"
#include "eigenext.h"
#include "optimmaths.h"

#ifdef _OPENMP
#include <omp.h>
#endif


namespace glmmr {

class OptimDesign {
private:
  glmmr::MatrixField<Eigen::VectorXd> C_list_;
  glmmr::MatrixField<Eigen::MatrixXd> D_list_;
  glmmr::MatrixField<Eigen::MatrixXd> X_all_list_;
  glmmr::MatrixField<Eigen::MatrixXd> Z_all_list_;
  const Eigen::MatrixXd W_all_diag_;
  Eigen::VectorXd weights_;
  Eigen::ArrayXi max_obs_;
  Eigen::ArrayXi curr_obs_;
  const int nlist_;
  int any_fix_;
  int n_;
  int k_;
  int nmax_;
  Eigen::ArrayXi p_;
  Eigen::ArrayXi q_;
  
public:
  Eigen::ArrayXi idx_in_;
  Eigen::ArrayXi idx_in_sub_;
  Eigen::ArrayXi idx_in_rm_;
  const Eigen::ArrayXi exp_cond_; //vector linking row to experimental condition number
  int r_in_design_;
  int r_in_rm_;
  Eigen::ArrayXi rows_in_design_;
  Eigen::ArrayXi count_exp_cond_;
  Eigen::ArrayXi count_exp_cond_rm_;
  double val_; // overall best value
  double new_val_; // new value
  double rm_val_;
  int fcalls_;
  int matops_;
  
  Eigen::MatrixXd A_list_;// inverse sigma matrices
  Eigen::MatrixXd rm1A_list_;// inverse sigma matrices with one removed - initialised to minus one but now needs to resize
  glmmr::MatrixField<Eigen::MatrixXd> M_list_;
  glmmr::MatrixField<Eigen::MatrixXd> M_list_sub_;
  glmmr::MatrixField<Eigen::MatrixXd> V0_list_;
  const Eigen::ArrayXi nfix_; //the indexes of the experimental conditions to keep
  const int rd_mode_; // robust designs mode: 1 == weighted, 2 == minimax.
  
  bool trace_;
  bool uncorr_;
  bool bayes_;
  
public:
  OptimDesign(Eigen::ArrayXi idx_in, 
              int n,
              const glmmr::MatrixField<Eigen::VectorXd> &C_list, 
              const glmmr::MatrixField<Eigen::MatrixXd> &X_list, 
              const glmmr::MatrixField<Eigen::MatrixXd> &Z_list, 
              const glmmr::MatrixField<Eigen::MatrixXd> &D_list,
              Eigen::MatrixXd w_diag,
              Eigen::ArrayXi max_obs,
              Eigen::VectorXd weights,
              Eigen::ArrayXi exp_cond,
              int any_fix,
              Eigen::ArrayXi nfix,
              const glmmr::MatrixField<Eigen::MatrixXd> &V0_list,
              int rd_mode = 0, 
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
  curr_obs_(max_obs.size()),//set to zero
  nlist_(weights_.size()),
  any_fix_(any_fix),
  n_(n), 
  k_(max_obs.size()),
  nmax_(2*ceil(X_all_list_.rows(0)/k_)*n_),
  p_(nlist_), //set to zero
  q_(nlist_), //set to zero
  idx_in_(idx_in),
  idx_in_sub_(idx_in),
  idx_in_rm_(idx_in),
  exp_cond_(exp_cond),
  r_in_design_(0),
  r_in_rm_(0),
  rows_in_design_(nmax_),//set to zero
  count_exp_cond_(nmax_),//set to zero
  count_exp_cond_rm_(nmax_),//set to zero
  val_(0.0), 
  new_val_(0.0),
  rm_val_(0.0),
  fcalls_(0),
  matops_(0),
  A_list_(nmax_*nlist_,nmax_), //set to zero
  rm1A_list_(nmax_*nlist_,nmax_),//set to zero
  M_list_(),
  M_list_sub_(),
  V0_list_(V0_list),
  nfix_(nfix), 
  rd_mode_(rd_mode), 
  trace_(trace),
  uncorr_(uncorr),
  bayes_(bayes){
    build_XZ();
  }
  
  Eigen::ArrayXi join_idx(const Eigen::ArrayXi &idx,
                          int elem){
    Eigen::ArrayXi newidx(idx.size()+1);
    newidx.segment(0,idx.size()) = idx;
    newidx(idx.size()) = elem;
    return newidx;
  }
  
  void build_XZ(){
    curr_obs_ = Eigen::ArrayXi::Zero(curr_obs_.size());
    p_ = Eigen::ArrayXi::Zero(nlist_);
    q_ = Eigen::ArrayXi::Zero(nlist_);
    rows_in_design_ = Eigen::ArrayXi::Zero(nmax_);
    count_exp_cond_ = Eigen::ArrayXi::Zero(nmax_);
    count_exp_cond_rm_ = Eigen::ArrayXi::Zero(nmax_);
    A_list_ = Eigen::MatrixXd::Zero(nmax_*nlist_,nmax_);
    rm1A_list_ = Eigen::MatrixXd::Zero(nmax_*nlist_,nmax_);

    
    std::sort(idx_in_.data(),idx_in_.data()+idx_in_.size());
    for(int i=0; i<idx_in_.size(); i++){
      curr_obs_(idx_in_(i)-1)++;
    }

    idx_in_sub_ = idx_in_;
    rows_in_design_ = Eigen::ArrayXi::LinSpaced(nmax_,0,nmax_-1);

    Eigen::VectorXd vals(nlist_);
    int rowcount;
    for(int j=0; j<nlist_;j++){
      p_(j) = X_all_list_.cols(j);
      q_(j) = Z_all_list_.cols(j);
      Eigen::MatrixXd X = Eigen::MatrixXd::Zero(nmax_,p_(j));
      Eigen::MatrixXd Z = Eigen::MatrixXd::Zero(nmax_,q_(j));
      Eigen::VectorXd w_diag(nmax_);
      rowcount = 0;
      for(int k=0; k<idx_in_.size();k++){
        Eigen::ArrayXi rowstoincl = glmmr::Eigen_ext::find(exp_cond_, idx_in_(k));
        for(int l=0;l<rowstoincl.size();l++){
          X.row(rowcount) = X_all_list_.get_row(j,rowstoincl(l));
          Z.row(rowcount) = Z_all_list_.get_row(j,rowstoincl(l));
          w_diag(rowcount) = W_all_diag_(rowstoincl(l),j);
          if(j==0)count_exp_cond_(k)++;
          rowcount++;
        }
      }
       if(j==0)r_in_design_ = rowcount;
       Eigen::MatrixXd Dt = D_list_(j);
       Eigen::MatrixXd tmp = Z.topRows(rowcount)*Dt*Z.topRows(rowcount).transpose();
       Eigen::MatrixXd I = Eigen::MatrixXd::Identity(tmp.rows(),tmp.cols());
       tmp.diagonal() += w_diag.head(rowcount);
       Eigen::MatrixXd M = X.topRows(rowcount).transpose() * tmp.llt().solve(I) * X.topRows(rowcount);
   
       
       M_list_.add(M);
       M_list_sub_.add(M);
      if(uncorr_){
        vals(j) = bayes_ ? glmmr::maths::c_obj_fun( M_list_(j) + V0_list_(j), C_list_(j)) : glmmr::maths::c_obj_fun( M_list_(j), C_list_(j));
      } else {
        A_list_.block(j*nmax_,0,r_in_design_,r_in_design_) = tmp.llt().solve(I);
        vals(j) = bayes_ ? glmmr::maths::c_obj_fun( X.topRows(rowcount).transpose() * A_list_.block(j*nmax_,0,r_in_design_,r_in_design_) * X.topRows(rowcount) +
          V0_list_(j), C_list_(j)) : glmmr::maths::c_obj_fun( X.topRows(rowcount).transpose() * A_list_.block(j*nmax_,0,r_in_design_,r_in_design_) * X.topRows(rowcount), C_list_(j));
      }

    }
    new_val_ = rd_mode_ == 1 ? vals.transpose()*weights_ : vals.maxCoeff();
    if(trace_)Rcpp::Rcout << "\nval: " << new_val_;
  }
  
  
  // LOCAL SEARCH ALGORITHM
  void local_search(){
    if (trace_) Rcpp::Rcout << "\nLocal search";
    int i = 0;
    double diff = -1.0;
    while(diff < 0){
      i++;
      val_ = new_val_;
      if (trace_) Rcpp::Rcout << "\nIter " << i << ": Var: " << val_;
      // evaluate the swaps
      Eigen::ArrayXXd val_swap = Eigen::ArrayXXd::Constant(k_,k_,10000);
      for(int j=1; j < k_+1; j++){
        if((idx_in_ == j).any()){
          Eigen::ArrayXd val_in_vec = eval(true,j); //eval is a member function of this class
          val_swap.row(j-1) = val_in_vec.transpose();
        }
      }
      
      Eigen::Index minrow, mincol;
      double newval = val_swap.minCoeff(&minrow,&mincol);
      diff = newval - val_;
      if (trace_) Rcpp::Rcout << " diff: " << diff << " newval " << newval << " val " << val_ ;
      
      if(diff < 0){
        int target = (int)mincol;//floor((int)minval/k_); 
        int rm_target = (int)minrow;//((int)minval) - target*k_;
        if(uncorr_){
          rm_obs_uncor(rm_target+1);
          new_val_ = add_obs_uncor(target+1,true,true);
        } else {
          rm_obs(rm_target+1);
          new_val_ = add_obs(target+1,true,true);
          Rcpp::Rcout << "\ncoord: " << target << " " << rm_target <<   " newval: " << new_val_; //<< "\nval swap: " << val_swap;
        }
      }
    }
  }
  
  void greedy_search(){
    // step 1: find optimal smallest design
    int i = 0;
    if (trace_) Rcpp::Rcout << "\nidx: " << idx_in_.transpose();
    if (trace_) Rcpp::Rcout << "\nGreedy search: " << n_;
    int idxcount = idx_in_.size();
    while(idxcount < n_){
      i++;
      idxcount++;
      val_ = new_val_;
      if (trace_) Rcpp::Rcout << "\nIter " << i << " size: " << idxcount << " Var: " << val_ ;
      Eigen::ArrayXd val_swap = eval(false);
      Eigen::Index swap_sort;
      double min = val_swap.minCoeff(&swap_sort);
      if (trace_) Rcpp::Rcout << " adding " << swap_sort+1;
      if(uncorr_){
        new_val_ = add_obs_uncor((int)swap_sort+1,false,true);
      } else {
        new_val_ = add_obs((int)swap_sort+1,false,true); 
      }
    }
  }
  
private:
  // get rows corresponding to an experimental condition
  Eigen::ArrayXi get_rows(int idx){
    int start = (idx == 0) ? 0 : count_exp_cond_.segment(0,idx).sum();
    return rows_in_design_.segment(start,count_exp_cond_(idx));
  }
  
  // get rows corresponding to a set of experimental condition
  Eigen::ArrayXi get_all_rows(Eigen::ArrayXi idx){
    Eigen::ArrayXi rowidx(nmax_);
    int count = 0;
    for(int i = 0; i < idx.size(); i++){
      Eigen::ArrayXi addidx = glmmr::Eigen_ext::find(exp_cond_,idx(i));
      rowidx.segment(count,addidx.size()) = addidx;
      count += addidx.size();
    }
    return rowidx.segment(0,count);
  }
  
  //remove rows corresponding to experimental condition
  Eigen::ArrayXi idx_minus(int idx){
    int start = (idx == 0) ? 0 : count_exp_cond_.segment(0,idx).sum();
    int end = start + count_exp_cond_(idx) - 1;
    Eigen::ArrayXi idxrtn(r_in_design_-(end-start+1));
    idxrtn.head(start) = rows_in_design_.head(start);
    if(end < r_in_design_-1)idxrtn.tail(r_in_design_-end) = rows_in_design_.segment(end+1,r_in_design_-end-1);
    return rows_in_design_.segment(start,end-start+1);
  }
  
  // remove observation
  void rm_obs(int outobs){
    Eigen::ArrayXi rm_cond = glmmr::Eigen_ext::find(idx_in_,outobs);
    Eigen::ArrayXi rowstorm = get_rows(rm_cond(0));
    idx_in_rm_ = glmmr::algo::uvec_minus(idx_in_,rm_cond(0));
    Eigen::ArrayXi idxexist = get_all_rows(idx_in_rm_);
    
    for (int idx = 0; idx < nlist_; ++idx) {
      matops_++;
      Eigen::MatrixXd A1 = A_list_.block(idx*nmax_,0,r_in_design_,r_in_design_);
      const Eigen::MatrixXd rm1A = glmmr::algo::remove_one_many_mat(A1, rowstorm);
      if(idx==0)r_in_rm_ = rm1A.rows();
      rm1A_list_.block(idx*nmax_,0,r_in_rm_,r_in_rm_) = rm1A;
      int p = X_all_list_.cols(idx);
      Eigen::MatrixXd X = glmmr::Eigen_ext::mat_indexing(X_all_list_(idx),idxexist,Eigen::ArrayXi::LinSpaced(p,0,p-1));
      M_list_sub_.replace(idx,X.transpose()*rm1A*X);
    }
    
    count_exp_cond_rm_.head(rm_cond(0)) = count_exp_cond_.head(rm_cond(0));
    if(rm_cond(0)>=(idx_in_.size() - 1)){
      count_exp_cond_rm_(rm_cond(0)) = count_exp_cond_(rm_cond(0)+1);
    } else {
      count_exp_cond_rm_.segment(rm_cond(0),idx_in_.size()-rm_cond(0)-1) = count_exp_cond_.segment(rm_cond(0)+1,idx_in_.size()-rm_cond(0)-1);
    }
  }
  
  void rm_obs_uncor(int outobs){
    Eigen::ArrayXi rm_cond = glmmr::Eigen_ext::find(idx_in_,outobs);
    Eigen::ArrayXi rowstorm = get_rows(rm_cond(0));
    
    for(int j=0; j<nlist_;j++){
      Eigen::MatrixXd X = Eigen::MatrixXd::Zero(rowstorm.size(),p_(j));
      Eigen::MatrixXd Z = Eigen::MatrixXd::Zero(rowstorm.size(),q_(j));
      Eigen::VectorXd w_diag(rowstorm.size());
      
      for(int l=0;l<rowstorm.size();l++){
        X.row(l) = X_all_list_.get_row(j,rowstorm(l));
        Z.row(l) = Z_all_list_.get_row(j,rowstorm(l));
        w_diag(l) = W_all_diag_(rowstorm(l),j);
      }
      
      Eigen::MatrixXd tmp = Z*D_list_(j)*Z.transpose();
      tmp.diagonal() += w_diag;
      Eigen::MatrixXd M = M_list_(j);
      M.noalias() -= X.transpose()*tmp.inverse()*X;
      M_list_sub_.replace(j,M);
    }
    idx_in_rm_ = glmmr::algo::uvec_minus(idx_in_,rm_cond(0));
    count_exp_cond_rm_.head(rm_cond(0)) = count_exp_cond_.head(rm_cond(0));
    if(rm_cond(0)>=(idx_in_.size() - 1)){
      count_exp_cond_rm_(rm_cond(0)) = count_exp_cond_(rm_cond(0)+1);
    } else {
      count_exp_cond_rm_.segment(rm_cond(0),idx_in_.size()-rm_cond(0)-1) = count_exp_cond_.segment(rm_cond(0)+1,idx_in_.size()-rm_cond(0)-1);
    }
  }
  
  
  
  // add a new observation
  double add_obs(int inobs,
                 bool userm = true,
                 bool keep = false){
    Eigen::ArrayXi rowstoadd = glmmr::Eigen_ext::find(exp_cond_,inobs);
    Eigen::ArrayXi idxvec = userm ? idx_in_rm_ : idx_in_;
    Eigen::ArrayXi idxexist = get_all_rows(idxvec);
    int n_to_add = rowstoadd.size();
    int n_already_in = idxexist.size();
    Eigen::ArrayXi idx_in_vec = Eigen::ArrayXi::Zero(n_already_in + n_to_add);
    Eigen::VectorXd vals(nlist_);
    bool issympd = true;
    int r_in_design_tmp_ = r_in_design_;
    
    for (int idx = 0; idx < nlist_; ++idx) {
      Eigen::MatrixXd M;
      Eigen::MatrixXd A = userm ? rm1A_list_.block(idx*nmax_,0,r_in_rm_,r_in_rm_) : 
        A_list_.block(idx*nmax_,0,r_in_design_tmp_,r_in_design_tmp_);
      
      if(!keep){
        Eigen::MatrixXd Z1 = Z_all_list_(idx);
        Eigen::MatrixXd Z2 = glmmr::Eigen_ext::mat_indexing(Z1,rowstoadd,Eigen::ArrayXi::LinSpaced(Z1.cols(),0,Z1.cols()-1));
        Z1 = glmmr::Eigen_ext::mat_indexing(Z1,idxexist,Eigen::ArrayXi::LinSpaced(Z1.cols(),0,Z1.cols()-1));
        Eigen::MatrixXd sig112 = Z2 * D_list_(idx) * Z1.transpose();
        Eigen::MatrixXd sig112A = sig112 * A;
        Eigen::MatrixXd sig2 = Z2 * D_list_(idx) * Z2.transpose();
        for(int i = 0; i < sig2.rows(); i++){
          sig2(i,i) += W_all_diag_(rowstoadd(i),idx);
        }
        sig2.noalias() -= sig112A*sig112.transpose();
        Eigen::MatrixXd X12 = X_all_list_(idx);
        Eigen::MatrixXd X1ex = glmmr::Eigen_ext::mat_indexing(X12,idxexist,Eigen::ArrayXi::LinSpaced(X12.cols(),0,X12.cols()-1));
        X12 = glmmr::Eigen_ext::mat_indexing(X12,rowstoadd,Eigen::ArrayXi::LinSpaced(X12.cols(),0,X12.cols()-1));
        X12 -= sig112A * X1ex;
        Eigen::MatrixXd iden = Eigen::MatrixXd::Identity(sig2.rows(),sig2.cols());
        M = userm ? M_list_sub_(idx) + X12.transpose() * sig2.llt().solve(iden) * X12 : M_list_(idx) + X12.transpose() * sig2.llt().solve(iden) * X12;
        
      } else {
        n_already_in = idxexist.size();
        Eigen::MatrixXd X = Eigen::MatrixXd::Zero(n_already_in + n_to_add,p_(idx));
        idx_in_vec = Eigen::ArrayXi::Zero(n_already_in + n_to_add);
        idx_in_vec.segment(0,n_already_in) = idxexist;
        Eigen::MatrixXd X0 = X_all_list_(idx);
        X.block(0,0,n_already_in,X.cols()) = glmmr::Eigen_ext::mat_indexing(X0,idxexist,Eigen::ArrayXi::LinSpaced(X0.cols(),0,X0.cols()-1));
        
        for(int j = 0; j < n_to_add; j++){
          Eigen::RowVectorXd z_j = Z_all_list_.get_row(idx,rowstoadd(j));
          Eigen::MatrixXd z_d = Z_all_list_(idx);
          z_d = glmmr::Eigen_ext::mat_indexing(z_d,idx_in_vec.segment(0,n_already_in),Eigen::ArrayXi::LinSpaced(z_d.cols(),0,z_d.cols()-1));
          double sig_jj = z_j * D_list_(idx) * z_j.transpose(); 
          sig_jj += W_all_diag_(rowstoadd(j),idx);
          Eigen::VectorXd f = z_d * D_list_(idx) * z_j.transpose();
          A = glmmr::algo::add_one_mat(A, sig_jj,f);
          idx_in_vec(n_already_in) = rowstoadd(j);
          X.row(n_already_in) = X_all_list_.get_row(idx,rowstoadd(j));  
          n_already_in++;
        }
        //check if positive definite
        M = X.transpose() * A * X;
      }
      issympd = glmmr::Eigen_ext::issympd(M);
      if(!issympd){
        if(keep){
          if(idx==0)r_in_design_ = A.rows();
          M_list_.replace(idx,M);
          A_list_.block(idx*nmax_,0,r_in_design_,r_in_design_) = A;
        }
        vals(idx) = bayes_ ? glmmr::maths::c_obj_fun( M+V0_list_(idx), C_list_(idx)) : glmmr::maths::c_obj_fun( M, C_list_(idx));
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
    
    double rtn = rd_mode_ == 1 ? vals.transpose()*weights_ : vals.maxCoeff();
    if(rtn < 10000){
      return rtn;
    } else {
      return 10000;
    }
  }
  
  double add_obs_uncor(int inobs,
                       bool userm = true,
                       bool keep = false){
    Eigen::VectorXd vals(nlist_);
    Eigen::ArrayXi rowstoadd = glmmr::Eigen_ext::find(exp_cond_,inobs);
    bool issympd = true;
    for(int j=0; j<nlist_;j++){
      Eigen::MatrixXd X = Eigen::MatrixXd::Zero(rowstoadd.size(),p_(j));
      Eigen::MatrixXd Z = Eigen::MatrixXd::Zero(rowstoadd.size(),q_(j));
      Eigen::VectorXd w_diag(rowstoadd.size());
      
      for(int l=0;l<rowstoadd.size();l++){
        X.row(l) = X_all_list_.get_row(j,rowstoadd(l));
        Z.row(l) = Z_all_list_.get_row(j,rowstoadd(l));
        w_diag(l) = W_all_diag_(rowstoadd(l),j);
      }
      
      Eigen::MatrixXd tmp = Z*D_list_(j)*Z.transpose();
      tmp.diagonal() += w_diag;
      Eigen::MatrixXd M = userm ? M_list_sub_(j) : M_list_(j);
      Eigen::MatrixXd iden = Eigen::MatrixXd::Identity(tmp.rows(),tmp.cols());
      
      M += X.transpose() * tmp.llt().solve(iden) * X;
      issympd = glmmr::Eigen_ext::issympd(M);
      if(issympd){
        if(keep){
          M_list_.replace(j,M);
        }
        vals(j) = bayes_ ? glmmr::maths::c_obj_fun( M+V0_list_(j), C_list_(j)) : glmmr::maths::c_obj_fun( M, C_list_(j));
      } else {
        for(int k = 0; k<vals.size(); k++)vals(k) = 10000;
        break;
      }
    }
    if(keep && issympd){
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
    double rtn = rd_mode_ == 1 ? vals.transpose()*weights_ : vals.maxCoeff();
    if(rtn < 10000){
      return rtn;
    } else {
      return 10000;
    }
  }
  
  Eigen::ArrayXd eval(bool userm = true, int obs = 0){
    Eigen::ArrayXd val_in_mat = Eigen::ArrayXd::Constant(k_,10000);
    if(userm){
      bool obsisin = (idx_in_ == obs).any();
      if(obsisin){
        if(uncorr_){
          rm_obs_uncor(obs);
        } else {
          rm_obs(obs);
        }
//#pragma omp parallel for
        for (int i = 1; i < k_+1; ++i) {
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
      for (int i = 1; i < k_+1; ++i) {
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

}

#endif