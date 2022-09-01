// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include "../inst/include/glmmrOptim.h"
#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// GradRobustStep
Rcpp::List GradRobustStep(arma::uvec idx_in, arma::uword n, Rcpp::List C_list, Rcpp::List X_list, Rcpp::List Z_list, Rcpp::List D_list, arma::mat w_diag, arma::uvec max_obs, arma::vec weights, arma::uvec exp_cond, arma::uvec nfix, Rcpp::List V0_list, arma::uword any_fix, arma::uword type, arma::uword rd_mode, bool trace, bool uncorr, bool bayes);
RcppExport SEXP _glmmrOptim_GradRobustStep(SEXP idx_inSEXP, SEXP nSEXP, SEXP C_listSEXP, SEXP X_listSEXP, SEXP Z_listSEXP, SEXP D_listSEXP, SEXP w_diagSEXP, SEXP max_obsSEXP, SEXP weightsSEXP, SEXP exp_condSEXP, SEXP nfixSEXP, SEXP V0_listSEXP, SEXP any_fixSEXP, SEXP typeSEXP, SEXP rd_modeSEXP, SEXP traceSEXP, SEXP uncorrSEXP, SEXP bayesSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::uvec >::type idx_in(idx_inSEXP);
    Rcpp::traits::input_parameter< arma::uword >::type n(nSEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type C_list(C_listSEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type X_list(X_listSEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type Z_list(Z_listSEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type D_list(D_listSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type w_diag(w_diagSEXP);
    Rcpp::traits::input_parameter< arma::uvec >::type max_obs(max_obsSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type weights(weightsSEXP);
    Rcpp::traits::input_parameter< arma::uvec >::type exp_cond(exp_condSEXP);
    Rcpp::traits::input_parameter< arma::uvec >::type nfix(nfixSEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type V0_list(V0_listSEXP);
    Rcpp::traits::input_parameter< arma::uword >::type any_fix(any_fixSEXP);
    Rcpp::traits::input_parameter< arma::uword >::type type(typeSEXP);
    Rcpp::traits::input_parameter< arma::uword >::type rd_mode(rd_modeSEXP);
    Rcpp::traits::input_parameter< bool >::type trace(traceSEXP);
    Rcpp::traits::input_parameter< bool >::type uncorr(uncorrSEXP);
    Rcpp::traits::input_parameter< bool >::type bayes(bayesSEXP);
    rcpp_result_gen = Rcpp::wrap(GradRobustStep(idx_in, n, C_list, X_list, Z_list, D_list, w_diag, max_obs, weights, exp_cond, nfix, V0_list, any_fix, type, rd_mode, trace, uncorr, bayes));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_glmmrOptim_GradRobustStep", (DL_FUNC) &_glmmrOptim_GradRobustStep, 18},
    {NULL, NULL, 0}
};

RcppExport void R_init_glmmrOptim(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
