#include <glmmr/covariance.hpp>
#include <glmmr/openmpheader.h>
#include <glmmr/matrixfield.h>
#include <cmath> 
#include <RcppEigen.h>
#include <progress.hpp>
#include <progress_bar.hpp>

// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::depends(glmmrBase)]]

using namespace Rcpp;

// code helpfully taken from https://stackoverflow.com/questions/50027494/eigen-indices-of-dense-matrix-meeting-condition
// index code from http://eigen.tuxfamily.org/dox/TopicCustomizing_NullaryExpr.html#title1

namespace glmmr {
namespace OptimEigen {

template<typename Func>
struct lambda_as_visitor_wrapper : Func {
  lambda_as_visitor_wrapper(const Func& f) : Func(f) {}
  template<typename S,typename I>
  void init(const S& v, I i, I j) { return Func::operator()(v,i,j); }
};

template<typename Mat, typename Func>
inline void visit_lambda(const Mat& m, const Func& f)
{
  lambda_as_visitor_wrapper<Func> visitor(f);
  m.visit(visitor);
}

inline Eigen::ArrayXi find(Eigen::ArrayXi vec, int n){
  std::vector<int> indices;
  glmmr::OptimEigen::visit_lambda(vec,
                                  [&indices,n](int v, int i, int j) {
                                    if(v == n)
                                      indices.push_back(i);
                                  });
  return Eigen::Map<Eigen::ArrayXi>(indices.data(),indices.size());
}

template<class ArgType, class RowIndexType, class ColIndexType>
class indexing_functor {
  const ArgType &m_arg;
  const RowIndexType &m_rowIndices;
  const ColIndexType &m_colIndices;
public:
  typedef Eigen::Matrix<typename ArgType::Scalar,
                        RowIndexType::SizeAtCompileTime,
                        ColIndexType::SizeAtCompileTime,
                        ArgType::Flags&Eigen::RowMajorBit?Eigen::RowMajor:Eigen::ColMajor,
                        RowIndexType::MaxSizeAtCompileTime,
                        ColIndexType::MaxSizeAtCompileTime> MatrixType;
  
  indexing_functor(const ArgType& arg, const RowIndexType& row_indices, const ColIndexType& col_indices)
    : m_arg(arg), m_rowIndices(row_indices), m_colIndices(col_indices)
  {}
  
  const typename ArgType::Scalar& operator() (Eigen::Index row, Eigen::Index col) const {
    return m_arg(m_rowIndices[row], m_colIndices[col]);
  }
};

template <class ArgType, class RowIndexType, class ColIndexType>
Eigen::CwiseNullaryOp<indexing_functor<ArgType,RowIndexType,ColIndexType>, typename indexing_functor<ArgType,RowIndexType,ColIndexType>::MatrixType>
mat_indexing(const Eigen::MatrixBase<ArgType>& arg, const RowIndexType& row_indices, const ColIndexType& col_indices)
{
  typedef indexing_functor<ArgType,RowIndexType,ColIndexType> Func;
  typedef typename Func::MatrixType MatrixType;
  return MatrixType::NullaryExpr(row_indices.size(), col_indices.size(), Func(arg.derived(), row_indices, col_indices));
}

inline bool isnotsympd(Eigen::MatrixXd mat){
  Eigen::LLT<Eigen::MatrixXd> lltOfA(mat);
  return lltOfA.info() == Eigen::NumericalIssue;
}


}
}

namespace glmmr {
namespace algo {

//removes index from square matrix
inline Eigen::MatrixXd remove_one_many_mat(const Eigen::MatrixXd &A, 
                                           const Eigen::ArrayXi &i) {
  
  Eigen::ArrayXi isort = i;
  std::sort(isort.data(),isort.data()+isort.size(), std::greater<int>());
  Eigen::MatrixXd A2 = A;
  
  for(int j=0; j<i.size(); j++){
    Eigen::MatrixXd A3(A2.rows()-1,A2.cols()-1);
    double d = A2(isort(j), isort(j));
    Eigen::VectorXd b(A2.rows()-1);
    if(isort(j) == A2.rows()){
      b = A2.block(0,isort(j),A2.rows()-1,1);
      A3 = A2.block(0,0,A2.rows()-1,A2.rows()-1);
    } else if(isort(j) == 0){
      b = A2.block(1,isort(j),A2.rows()-1,1);
      A3 = A2.block(1,1,A2.rows()-1,A2.rows()-1);
    } else {
      b.segment(0,isort(j)) = A2.block(0,isort(j),isort(j),1);
      b.segment(isort(j),A2.rows()-1-isort(j)) = A2.block(isort(j)+1,isort(j),A2.rows()-1-isort(j),1);
      A3.block(0,0,isort(j),isort(j)) = A2.block(0,0,isort(j),isort(j));
      A3.block(0,isort(j),isort(j),A2.rows()-1-isort(j)) = A2.block(0,isort(j)+1,isort(j),A2.rows()-1-isort(j));
      A3.block(isort(j),0,A2.rows()-1-isort(j),isort(j)) = A2.block(isort(j)+1,0,A2.rows()-1-isort(j),isort(j));
      A3.block(isort(j),isort(j),A2.rows()-1-isort(j),A2.rows()-1-isort(j)) = A2.block(isort(j)+1,isort(j)+1,A2.rows()-1-isort(j),A2.rows()-1-isort(j));
    }
    A3 -= (1.0/d)*b*b.transpose();
    A2 = A3;
  }
  
  return A2;
}

// removes observations and calculates value of objective function
inline double remove_one_many(const Eigen::MatrixXd &A, 
                              const Eigen::ArrayXi &i,
                              const Eigen::VectorXd &u) {
  
  Eigen::MatrixXd A2 = glmmr::algo::remove_one_many_mat(A,i);
  int n = u.size()-i.size();
  Eigen::VectorXd u2(n);
  bool isin;
  int iter = 0;
  
  for(int j=0; j< i.size(); j++){
    isin = (i == j).any();
    if(!isin){
      u2(iter) = u(j);
      iter++;
    }
  }
  return u2.transpose()*A2 * u2;
}


inline Eigen::MatrixXd add_one_mat(const Eigen::MatrixXd &A, 
                                   double sigma_jj, 
                                   const Eigen::VectorXd &f) {
  
  Eigen::MatrixXd A2 = Eigen::MatrixXd::Zero(A.rows() + 1, A.rows() + 1);
  A2.block(0,0,A.rows(),A.cols()) = A;
  A2(A2.rows() - 1, A2.rows() - 1) = 1 / sigma_jj;
  
  Eigen::VectorXd u1 = Eigen::VectorXd::Zero(f.size()+1);
  u1.segment(0,f.size()) = f;
  Eigen::VectorXd v1 = Eigen::VectorXd::Zero(u1.size());
  v1(v1.size()-1) = 1.0;
  
  A2 -= ((A2 * u1) * (v1.transpose() * A2)) * (1.0/(1.0 + (v1.transpose() * A2) * u1));
  A2 -= ((A2 * v1) * (u1.transpose() * A2)) * (1.0/(1.0 + (u1.transpose() * A2) * v1));
  
  return A2;
}


inline double add_one(const Eigen::MatrixXd &A, 
                      double sigma_jj, 
                      const Eigen::VectorXd &f,
                      const Eigen::VectorXd &u) {
  
  Eigen::MatrixXd A2 = glmmr::algo::add_one_mat(A,sigma_jj,f);
  return u.transpose()*A2*u;
}

inline Eigen::ArrayXi uvec_minus(const Eigen::ArrayXi &v, 
                                 int rm_idx) {
  int n = v.size();
  if (rm_idx == 0) return v.tail(n-1);
  if (rm_idx == n-1) return v.head(n-1);
  Eigen::ArrayXi res(v.size()-1);
  res.head(rm_idx) = v.head(rm_idx);
  res.tail(n-1-rm_idx) = v.tail(n-1-rm_idx);
  return res;
}

}

}

namespace glmmr {
namespace maths {

using namespace Eigen;

inline double obj_fun(const MatrixXd &A, 
                      const VectorXd &U2) {
  return U2.transpose() * A * U2;
}

inline double c_obj_fun(const MatrixXd &M, 
                        const VectorXd &C) {
  // this is the objective function c-optimal
  //MatrixXd M_inv = M.inverse();
  return C.transpose() * M * C;
}

inline double c_d_deriv(const MatrixXd &M1, 
                        const MatrixXd &M2, 
                        const VectorXd &C) {
  // this is the directional derivative from M1 to M2 c-optimal
  MatrixXd M_inv = M1.inverse();
  return C.transpose() * M_inv * (M1 - M2) * M_inv * C;
}

inline double c_deriv(const MatrixXd &M, 
                      const VectorXd &C) {
  // this is the directional derivative from M1 to M2 c-optimal
  MatrixXd M_inv = M.inverse();
  return (-1 * M_inv * C * C.transpose() * M_inv).norm();
}

inline MatrixXd gen_m(const MatrixXd &X, 
                      const MatrixXd &A) {
  //generate information matrix
  return X.transpose() * A * X;
}


}

}

namespace glmmr{

using namespace Eigen;

class OptimDerivatives {
public: 
  std::vector<int> gaussian;
  std::vector<glmmr::MatrixField<MatrixXd> > FirstOrderDerivatives;
  std::vector<glmmr::MatrixField<MatrixXd> > SecondOrderDerivatives;
  OptimDerivatives(){};
  
  void addDesign(glmmr::Covariance& cov, bool is_gaussian){
    glmmr::MatrixField<MatrixXd> first;
    glmmr::MatrixField<MatrixXd> second;
    int curr_size = FirstOrderDerivatives.size();
    std::vector<MatrixXd> derivs;
    cov.derivatives(derivs,2);
    int R = cov.npar();
    gaussian.push_back((int)(is_gaussian));
    for(int i = 0; i < R; i++){
      first.add(derivs[1+i]);
    }
    FirstOrderDerivatives.push_back(first);
    for(int i = 0; i < R; i++){
      for(int j = i; j < R; j++){
        int scnd_idx = i + j*(R-1) - j*(j-1)/2;
        second.add(derivs[R+1+scnd_idx]);
      }
    }
    SecondOrderDerivatives.push_back(second);
  };
};

}

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

namespace glmmr {
using namespace Eigen;
// I plan to refactor this class as it is too big!

class OptimDesign {
private:
  glmmr::OptimData& data_;
  glmmr::OptimDerivatives& derivs_;
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
  bool kr_;
  bool bayes_;
  
  
public:
  
  OptimDesign(const ArrayXi& idx_in, 
              int n,
              glmmr::OptimData& data,
              glmmr::OptimDerivatives& deriv,
              int nmax,
              bool robust_log = false, 
              bool trace=false,
              bool uncorr=false,
              bool kr = false,
              bool bayes = false) :
  data_(data),
  derivs_(deriv),
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
  uncorr_(kr ? false : uncorr),
  kr_(kr),
  bayes_(bayes){
    if(bayes_ & kr_)Rcpp::stop("Can't have KR and Bayes");
    build_XZ();
  }
  
  ArrayXi join_idx(const ArrayXi &idx,int elem);
  void build_XZ();
  void local_search();
  void greedy_search();
  void reverse_greedy_search();
  MatrixXd KR_correction(const MatrixXd& M,const int& idx,const MatrixXd& X,const MatrixXd& Z,const MatrixXd& Sinv,const ArrayXi& indexes);
  double c_obj_fun(const MatrixXd &M,const VectorXd &C);
  
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

ArrayXi glmmr::OptimDesign::join_idx(const ArrayXi &idx,
                                            int elem){
  ArrayXi newidx(idx.size()+1);
  newidx.segment(0,idx.size()) = idx;
  newidx(idx.size()) = elem;
  return newidx;
}

MatrixXd glmmr::OptimDesign::KR_correction(const MatrixXd& M,
                                                  const int& idx,
                                                  const MatrixXd& X,
                                                  const MatrixXd& Z,
                                                  const MatrixXd& Sinv,
                                                  const ArrayXi& indexes){
  if(idx>=derivs_.FirstOrderDerivatives.size() )Rcpp::stop("idx out of range");
  int n = X.rows();
  if(n != indexes.size())Rcpp::stop("indexes not equal to X array extent");
  int R = derivs_.FirstOrderDerivatives[idx].size();
  int Rmod = derivs_.gaussian[idx] == 1 ? R+1 : R;
  MatrixXd M_new(M);
  MatrixXd SigX = Sinv*X;
  //MatrixXd PG = Sinv - SigX*M*SigX.transpose();
  MatrixXd M_theta = MatrixXd::Zero(Rmod,Rmod);
  MatrixXd partial1(n,n);
  MatrixXd partial2(n,n);
  MatrixXd meat = MatrixXd::Zero(X.cols(),X.cols());
  glmmr::MatrixField<MatrixXd> P;
  glmmr::MatrixField<MatrixXd> Q;
  glmmr::MatrixField<MatrixXd> RR;
  glmmr::MatrixField<MatrixXd> S;
  int counter = 0;
  for(int i = 0; i < Rmod; i++){
    if(i < R){
      partial1 = Z*derivs_.FirstOrderDerivatives[idx](i)*Z.transpose();
    } else {
      partial1 = MatrixXd::Identity(n,n);
    }
    P.add(-1*SigX.transpose()*partial1*SigX);
    for(int j = i; j < Rmod; j++){
      if(j < R){
        partial2 = Z*derivs_.FirstOrderDerivatives[idx](j)*Z.transpose();
      } else {
        partial2 = MatrixXd::Identity(n,n);
      }
      S.add(Sinv*partial1*Sinv*partial2);
      Q.add(X.transpose()*Sinv*partial1*Sinv*partial2*SigX);
      // counter++;
      if(i < R && j < R){
        int scnd_idx = i + j*(R-1) - j*(j-1)/2;
        if(scnd_idx >= derivs_.SecondOrderDerivatives[idx].size())Rcpp::stop("scnd > SoD s");
        RR.add(SigX.transpose()*Z*derivs_.SecondOrderDerivatives[idx](scnd_idx)*Z.transpose()*SigX);
      }
    }
  }
  counter = 0;
  for(int i = 0; i < Rmod; i++){
    for(int j = i; j < Rmod; j++){
      // using the expected REML information matrix here, but can use alternative commented out
      M_theta(i,j) = 0.5*(S(counter).trace())- (M*Q(counter)).trace() + 0.5*((M*P(i)*M*P(j)).trace());
      if(i!=j)M_theta(j,i)=M_theta(i,j);
      counter++;
    }
  }
  M_theta = M_theta.llt().solve(MatrixXd::Identity(Rmod,Rmod));
  for(int i = 0; i < (Rmod-1); i++){
    if(i < R){
      partial1 = Z*derivs_.FirstOrderDerivatives[idx](i)*Z.transpose();
    } else {
      partial1 = MatrixXd::Identity(n,n);
    }
    for(int j = (i+1); j < Rmod; j++){
      if(j < R){
        partial2 = Z*derivs_.FirstOrderDerivatives[idx](j)*Z.transpose();
      } else {
        partial2 = MatrixXd::Identity(n,n);
      }
      int scnd_idx = i + j*(Rmod-1) - j*(j-1)/2;
      meat += M_theta(i,j)*(Q(scnd_idx) + Q(scnd_idx).transpose() - P(i)*M*P(j) -P(j)*M*P(i));
      //meat += M_theta(i,j)*(SigX.transpose()*partial2*PG*partial1*SigX);
      if(i < R && j < R){
        scnd_idx = i + j*(R-1) - j*(j-1)/2;
        meat -= 0.5*M_theta(i,j)*(RR(scnd_idx));
      }
    }
  }
  for(int i = 0; i < Rmod; i++){
    if(i < R){
      partial1 = Z*derivs_.FirstOrderDerivatives[idx](i)*Z.transpose();
    } else {
      partial1 = MatrixXd::Identity(n,n);
    }
    int scnd_idx = i + i*(Rmod-1) - i*(i-1)/2;
    meat += M_theta(i,i)*(Q(scnd_idx) - P(i)*M*P(i));
    if(i < R){
      scnd_idx = i + i*(R-1) - i*(i-1)/2;
      meat -= 0.25*M_theta(i,i)*RR(scnd_idx);
    }
  }
  M_new = M + 2*M*meat*M;
  return M_new;
}

void glmmr::OptimDesign::build_XZ(){
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
    std::vector<int> starting_rows;
    for(int k=0; k<idx_in_.size();k++){
      ArrayXi rowstoincl = glmmr::OptimEigen::find(data_.exp_cond_, idx_in_(k));
      for(int l=0;l<rowstoincl.size();l++){
        X.row(rowcount) = data_.X_all_list_.get_row(j,rowstoincl(l));
        Z.row(rowcount) = data_.Z_all_list_.get_row(j,rowstoincl(l));
        w_diag(rowcount) = data_.W_all_diag_(rowstoincl(l),j);
        if(j==0)count_exp_cond_(k)++;
        starting_rows.push_back(rowstoincl(l));
        rowcount++;
      }
    }
    if(j==0)r_in_design_ = rowcount;
    MatrixXd Dt = data_.D_list_(j);
    MatrixXd tmp = Z.topRows(rowcount)*Dt*Z.topRows(rowcount).transpose();
    MatrixXd I = MatrixXd::Identity(tmp.rows(),tmp.cols());
    tmp.diagonal() += w_diag.head(rowcount);
    tmp = tmp.llt().solve(I);
    MatrixXd M = X.topRows(rowcount).transpose() * tmp * X.topRows(rowcount);
    if(bayes_){
      M += data_.V0_list_(j);
    }
    M = M.llt().solve(MatrixXd::Identity(M.rows(),M.cols()));
    if(kr_){
      M = KR_correction(M,j,X.topRows(rowcount),Z.topRows(rowcount),tmp,Map<ArrayXi,Unaligned>(starting_rows.data(),starting_rows.size()));
    }
    M_list_.add(M);
    M_list_sub_.add(M);
    vals(j) = c_obj_fun( M, data_.C_list_(j));
    if(!uncorr_){
      A_list_.block(j*nmax_,0,r_in_design_,r_in_design_) = tmp;
    } 
  }
  new_val_ = robust_log_ ? vals.log().transpose()*data_.weights_ : vals.transpose()*data_.weights_;
  if(trace_)Rcpp::Rcout << "\nStarting val: " << new_val_;
}

// LOCAL SEARCH ALGORITHM
void glmmr::OptimDesign::local_search(){
  if (trace_) Rcpp::Rcout << "\nLOCAL SEARCH";
  int i = 0;
  double diff = -1.0;
  while(diff < 0){
    i++;
    val_ = new_val_;
    if (trace_) Rcpp::Rcout << "\n|Iteration " << i << "| Current value: " << val_;
    // evaluate the swaps
    ArrayXXd val_swap = ArrayXXd::Constant(k_,k_,10000);
    if( trace_) Rcpp::Rcout << "\nCalculating swaps: \n";
    Progress bar(k_+1,trace_);
    for(int j=1; j < k_+1; j++){
      if((idx_in_ == j).any()){
        ArrayXd val_in_vec = eval(true,j); //eval is a member function of this class
        val_swap.row(j-1) = val_in_vec.transpose();
      }
      bar.increment();
    }
    Index minrow, mincol;
    double newval = val_swap.minCoeff(&minrow,&mincol);
    diff = newval - val_;
    if (trace_) Rcpp::Rcout << "\nBest Difference: " << diff << " | Best New value: " << newval;
    if(diff < 0){
      double tmp;
      if(uncorr_){
        tmp = rm_obs_uncor(1+(int)minrow,true);
        new_val_ = add_obs_uncor(1+(int)mincol,true,true);
      } else {
        tmp = rm_obs(1+(int)minrow,true);
        new_val_ = add_obs(1+(int)mincol,true,true);
      }
      (void)tmp;
    } else {
      if (trace_) Rcpp::Rcout << "\nFINISHED LOCAL SEARCH";
    }
    
  }
}

void glmmr::OptimDesign::greedy_search(){
  // step 1: find optimal smallest design
  int i = 0;
  if (trace_) Rcpp::Rcout << "\nStarting conditions: " << idx_in_.transpose();
  if (trace_) Rcpp::Rcout << "\nGREEDY SEARCH for design of size " << n_;
  int idxcount = idx_in_.size();
  while(idxcount < n_){
    i++;
    idxcount++;
    val_ = new_val_;
    if (trace_) Rcpp::Rcout << "\n|Iteration " << i << "| Size: " << idxcount << " Current value: " << val_ ;
    ArrayXd val_swap = eval(false);
    Index swap_sort;
    double min = val_swap.minCoeff(&swap_sort);
    (void)min;
    if (trace_) Rcpp::Rcout << " adding " << swap_sort+1;
    if(uncorr_){
      new_val_ = add_obs_uncor((int)swap_sort+1,false,true);
    } else {
      new_val_ = add_obs((int)swap_sort+1,false,true); 
    }
  }
  if (trace_) Rcpp::Rcout << "\nFINISHED GREEDY SEARCH";
}

void glmmr::OptimDesign::reverse_greedy_search(){
  // start from the whole design space and then successively remove observations
  if (trace_) Rcpp::Rcout << "\nREVERSE GREEDY SEARCH for design of size " << n_;
  int i = 0;
  int idxcount = idx_in_.size();
  ArrayXd val_rm(k_);
  
  while(idxcount > n_){
    i++;
    val_ = new_val_;
    if (trace_) Rcpp::Rcout << "\n|Iteration " << i << "| Size: " << idxcount << " Current value: " << val_ ;
    if( trace_) Rcpp::Rcout << "\nCalculating removals: \n";
    Progress bar(k_+1,trace_);
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
      bar.increment();
    }
    Index swap_sort;
    double min = val_rm.minCoeff(&swap_sort);
    (void)min;
    if (trace_) Rcpp::Rcout << "\nRemoving: " << swap_sort+1;
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
ArrayXi glmmr::OptimDesign::get_rows(int idx){
  int start = (idx == 0) ? 0 : count_exp_cond_.segment(0,idx).sum();
  return rows_in_design_.segment(start,count_exp_cond_(idx));
}

// get rows corresponding to a set of experimental condition
ArrayXi glmmr::OptimDesign::get_all_rows(ArrayXi idx){
  ArrayXi rowidx(nmax_);
  int count = 0;
  for(int i = 0; i < idx.size(); i++){
    ArrayXi addidx = glmmr::OptimEigen::find(data_.exp_cond_,idx(i));
    rowidx.segment(count,addidx.size()) = addidx;
    count += addidx.size();
  }
  return rowidx.segment(0,count);
}

//remove rows corresponding to experimental condition
ArrayXi glmmr::OptimDesign::idx_minus(int idx){
  int start = (idx == 0) ? 0 : count_exp_cond_.segment(0,idx).sum();
  int end = start + count_exp_cond_(idx) - 1;
  ArrayXi idxrtn(r_in_design_-(end-start+1));
  idxrtn.head(start) = rows_in_design_.head(start);
  if(end < r_in_design_-1)idxrtn.tail(r_in_design_-end) = rows_in_design_.segment(end+1,r_in_design_-end-1);
  return rows_in_design_.segment(start,end-start+1);
}

// remove observation
// 
double glmmr::OptimDesign::rm_obs(int outobs,
                                         bool keep,
                                         bool keep_mat,
                                         bool rtn_val){
  ArrayXi rm_cond = glmmr::OptimEigen::find(idx_in_,outobs);
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
    MatrixXd X = glmmr::OptimEigen::mat_indexing(data_.X_all_list_(idx),idxexist,ArrayXi::LinSpaced(p,0,p-1));
    MatrixXd M = X.transpose()*rm1A*X;
    if(bayes_){
      M += data_.V0_list_(idx);
    }
    bool issympd = glmmr::OptimEigen::isnotsympd(M);
    if(!issympd){
      M = M.llt().solve(MatrixXd::Identity(M.rows(),M.cols()));
      if(kr_){
        int q = data_.Z_all_list_.cols(idx);
        MatrixXd Z = glmmr::OptimEigen::mat_indexing(data_.Z_all_list_(idx),idxexist,ArrayXi::LinSpaced(q,0,q-1));
        MatrixXd Mkr(M);
        M = KR_correction(Mkr,idx,X,Z,rm1A,idxexist);
      } 
      M_list_sub_.replace(idx,M);
      if(rtn_val)vals(idx) = c_obj_fun( M, data_.C_list_(idx));
    }
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

double glmmr::OptimDesign::rm_obs_uncor(int outobs,
                                               bool keep,
                                               bool keep_mat,
                                               bool rtn_val){
  ArrayXi rm_cond = glmmr::OptimEigen::find(idx_in_,outobs);
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
    if(rtn_val)vals(j) = bayes_ ? c_obj_fun( M+data_.V0_list_(j), data_.C_list_(j)) : c_obj_fun( M, data_.C_list_(j));
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
double glmmr::OptimDesign::add_obs(int inobs,
                                          bool userm,
                                          bool keep ){
  ArrayXi rowstoadd = glmmr::OptimEigen::find(data_.exp_cond_,inobs);
  ArrayXi idxvec = userm ? idx_in_rm_ : idx_in_;
  ArrayXi idxexist = get_all_rows(idxvec);
  int n_to_add = rowstoadd.size();
  int n_already_in = idxexist.size();
  ArrayXi idx_in_vec = ArrayXi::Zero(n_already_in + n_to_add);
  VectorXd vals(nlist_);
  bool issympd;
  int r_in_design_tmp_ = r_in_design_;
  bool exit_loop = false;
  for (int idx = 0; idx < nlist_; ++idx) {
    MatrixXd A = userm ? rm1A_list_.block(idx*nmax_,0,r_in_rm_,r_in_rm_) : 
    A_list_.block(idx*nmax_,0,r_in_design_tmp_,r_in_design_tmp_);
    n_already_in = idxexist.size();
    MatrixXd X = MatrixXd::Zero(n_already_in + n_to_add,p_(idx));
    MatrixXd Z = MatrixXd::Zero(n_already_in + n_to_add,q_(idx));
    idx_in_vec = ArrayXi::Zero(n_already_in + n_to_add);
    idx_in_vec.segment(0,n_already_in) = idxexist;
    MatrixXd X0 = data_.X_all_list_(idx);
    MatrixXd Z0 = data_.Z_all_list_(idx);
    X.block(0,0,n_already_in,X.cols()) = glmmr::OptimEigen::mat_indexing(X0,idxexist,ArrayXi::LinSpaced(X0.cols(),0,X0.cols()-1));
    Z.block(0,0,n_already_in,Z.cols()) = glmmr::OptimEigen::mat_indexing(Z0,idxexist,ArrayXi::LinSpaced(Z0.cols(),0,Z0.cols()-1));
    MatrixXd D = data_.D_list_(idx);
    for(int j = 0; j < n_to_add; j++){
      RowVectorXd z_j = data_.Z_all_list_.get_row(idx,rowstoadd(j));
      int zcols = data_.Z_all_list_.cols(idx);
      MatrixXd z_d = glmmr::OptimEigen::mat_indexing(data_.Z_all_list_(idx),idx_in_vec.segment(0,n_already_in),ArrayXi::LinSpaced(zcols,0,zcols-1));
      double sig_jj = z_j * D * z_j.transpose(); 
      sig_jj += data_.W_all_diag_(rowstoadd(j),idx);
      VectorXd f = z_d * D * z_j.transpose();
      A = glmmr::algo::add_one_mat(A, sig_jj,f);
      idx_in_vec(n_already_in) = rowstoadd(j);
      X.row(n_already_in) = data_.X_all_list_.get_row(idx,rowstoadd(j));  
      Z.row(n_already_in) = data_.Z_all_list_.get_row(idx,rowstoadd(j));  
      n_already_in++;
    }
    MatrixXd M = X.transpose() * A * X;
    if(bayes_){
      M += data_.V0_list_(idx);
    }
    issympd = glmmr::OptimEigen::isnotsympd(M);
    if(!issympd){
      M = M.llt().solve(MatrixXd::Identity(M.rows(),M.cols()));
      if(kr_){
        MatrixXd Mkr(M);
        M = KR_correction(Mkr,idx,X,Z,A,idx_in_vec);
      }
      if(keep){
        if(idx==0)r_in_design_ = A.rows();
        M_list_.replace(idx,M);
        A_list_.block(idx*nmax_,0,r_in_design_,r_in_design_) = A;
      }
      vals(idx) = c_obj_fun( M, data_.C_list_(idx));
    } else {
      for(int k = 0; k<vals.size(); k++)vals(k) = 10000;
      exit_loop = true;
    }
    if(exit_loop)break;
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

double glmmr::OptimDesign::add_obs_uncor(int inobs,
                                                bool userm,
                                                bool keep){
  VectorXd vals(nlist_);
  ArrayXi rowstoadd = glmmr::OptimEigen::find(data_.exp_cond_,inobs);
  bool issympd = true;
  bool exit_loop = false;
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
    issympd = glmmr::OptimEigen::isnotsympd(M);
    if(!issympd){
      if(keep){
        M_list_.replace(j,M);
      }
      vals(j) = bayes_ ? c_obj_fun( M+data_.V0_list_(j), data_.C_list_(j)) : c_obj_fun( M, data_.C_list_(j));
    } else {
      for(int k = 0; k<vals.size(); k++)vals(k) = 10000;
      exit_loop = true;
    }
    if(exit_loop)break;
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

ArrayXd glmmr::OptimDesign::eval(bool userm, int obs){
  ArrayXd val_in_mat = ArrayXd::Constant(k_,10000);
  if(userm){
    double tmp;
    if(uncorr_){
      tmp = rm_obs_uncor(obs);
    } else {
      tmp = rm_obs(obs);
    }
    (void)tmp;
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

double glmmr::OptimDesign::c_obj_fun(const MatrixXd &M, 
                                            const VectorXd &C) {
  return C.transpose() * M * C;
}


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