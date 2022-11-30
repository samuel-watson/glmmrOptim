#ifndef MATRIXFIELD_H
#define MATRIXFIELD_H

#include <cmath> 
#include <RcppEigen.h>
#include <memory>

namespace glmmr {

// create a class to store different sized matrices, not perfect but it'll do!
template<typename T>
class MatrixField{
  public: 
    std::vector<std::shared_ptr<T> > data;
    
    MatrixField(int n){
      data.reserve(n);
    };
    
    MatrixField(){};
    
    MatrixField(const glmmr::MatrixField<T> &field){
      data.reserve(field.data.size());
      data = field.data;
    }
    
    void add(T matrix){
      //T* mat_ptr = new T(matrix);
      //data.push_back(mat_ptr);
      data.push_back(std::make_shared<T>(matrix));
    }
    
    
    // template<typename Mat>
    // void add(const Mat& matrix){ //Rcpp::NumericMatrix matrix
    //   std::shared_ptr<T> mat_ptr = std::make_shared<T>(Rcpp::as<Eigen::Map<T> >(matrix));
    //   //T* mat_ptr = new T(Rcpp::as<Eigen::Map<T> >(matrix));
    //   data.push_back(mat_ptr);
    // }
    
    T operator()(int i)
    {
      //return *(reinterpret_cast<T *>(data[i]));
      return *(data[i]);
    }
    
    Eigen::RowVectorXd get_row(int n, int i){
      return data[n]->row(i);
    }
    
    // T* get_ptr(int n){
    //   return data[n];
    // }
    std::shared_ptr<T> get_ptr(int n){
      return data[n];
    }
    
    void replace(int i, T matrix){
      *(data[i]) = matrix;
    }
    
    int mat_size(int i){
      return data[i]->size();
    }
    
    int size(){
      return data.size();
    }
    
    int rows(int i){
      return data[i]->rows();
    }
    
    int cols(int i){
      return data[i]->cols();
    }
    
    //~MatrixField();
    
    // ~MatrixField(){
    //   for(int i = data.size()-1; i >= 0; i--){
    //     delete data[i];
    //   }
    // }
    
};

}

#endif