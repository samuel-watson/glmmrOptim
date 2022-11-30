#' A GLMM Design Space
#' 
#' A class-based representation of a "design space" that contains one or more \link[glmmrBase]{Model} objects.
#' @details 
#' An experimental study is comprised of a collection of experimental conditions, which are one or more observations made a pre-specified locations/values
#' of covariates. A design space represents the collection of all possible experimental conditions for the study design and plausible models describing
#' the data generating process. The main purpose of this class is to identify optimal study designs, that is the set of `n` experimental conditions 
#' from all possible experimental conditions that minimise the variance of a parameter of interest across the specified GLMMs.
#' 
#' A `DesignSpace` object is intialised using one or more \link[glmmrBase]{Model} objects. Design objects can be added or removed from the collection. 
#' All designs must have the same number of rows in their design matrices (X and Z) and the same number of experimental conditions. 
#' The DesignSpace functions can modify the linked design objects.
DesignSpace <- R6::R6Class("DesignSpace",
                 public = list(
                   #' @field weights A vector denoting the prior weighting of each Design in the design space. Required if robust optimisation is used based on a 
                   #' weighted average variance over the linked designs. If it is not specified in the call to `new()` then designs are assumed
                   #' to have equal weighting.
                   weights = NULL,
                   #' @field experimental_condition A vector indicating the unique identifier of the experimental condition for each observation/row in the matrices X and Z.
                   experimental_condition = NULL,
                   #' @description 
                   #' Create a new Design Space
                   #' 
                   #' Creates a new design space from one or more glmmr designs.
                   #' @details 
                   #' **Initialisation**
                   #' The experimental condition refers to the smallest "unit" of the study design that could be included in the design. For example, in a
                   #' cluster randomised trial, the experimental condition may be single individuals such that we can observed any number of individuals 
                   #' in any cluster period (including none at all). In this case the experimental condition would be equivalent to row number. Alternatively,
                   #' we may have to observe whole cluster periods, and we need to choose which cluster periods to observe, in which case the each observation 
                   #' in a different cluster-period would have the same experimental condition identifier. Finally, we may determine that the whole cluster in 
                   #' all periods (a "sequence") is either observed or not.
                   #' @param ... One or more glmmrBase \link[glmmrBase]{Model} objects. The designs must have an equal number of observations.
                   #' @param weights Optional. A numeric vector of values between 0 and 1 indicating the prior weights to assign to each of the designs. The weights
                   #' are required for optimisation, if a weighted average variance is used across the designs. If not specified then designs are assumed 
                   #' to have equal weighting.
                   #' @param experimental_condition Optional. A vector of the same length as the number of observations in each design indicating the unique
                   #' identifier of the experimental condition that observation belongs to, see Details. If not provided, then it is assumed that all observations
                   #' are separate experimental conditions.
                   #' @return A `DesignSpace` object
                   #' @examples
                   #' df <- nelder(~ ((int(2)*t(3)) > cl(3)) > ind(5))
                   #' df$int <- df$int - 1
                   #' mf1 <- MeanFunction$new(formula = ~ int + factor(t) - 1,
                   #'                         data=df,
                   #'                         parameters = rep(0,4),
                   #'                         family = gaussian())
                   #' cov1 <- Covariance$new(data = df,
                   #'                        formula = ~ (1|gr(cl)) + (1|gr(cl*t)),
                   #'                        parameters = c(0.25,0.1))
                   #' des <- Model$new(covariance = cov1,
                   #'                   mean.function = mf1,
                   #'                   var_par = 1)
                   #' ds <- DesignSpace$new(des)
                   #' #add another design
                   #' cov2 <- Covariance$new(data = df,
                   #'                        formula = ~ (1|gr(cl)*ar1(t)),
                   #'                        parameters = c(0.25,0.8))
                   #' des2 <- Model$new(covariance = cov2,
                   #'                   mean.function = mf1,
                   #'                   var_par = 1)
                   #' ds$add(des2)
                   #' #report the size of the design
                   #' ds$n()
                   #' #we can access specific designs
                   #' ds$show(2)$n()
                   #' #and then remove it
                   #' ds$remove(2)
                   #' #or we could add them when we construct object
                   #' ds <- DesignSpace$new(des,des2)
                   #' #we can specify weights
                   #' ds <- DesignSpace$new(des,des2,weights=c(0.1,0.9))
                   #' #and add experimental conditions
                   #' ds <- DesignSpace$new(des,des2,experimental_condition = df$cl)                   
                   initialize = function(...,
                                         weights=NULL,
                                         experimental_condition = NULL) {
                     samp.size <- c()
                     i <- 0
                     for (item in list(...)) {
                       i <- i + 1
                       if(!(is(item,"Model")|is(item,"ModelMCML")))stop("Not all Model objects")
                       samp.size[i] <- item$n()
                     }
                     #print(samp.size)
                     if(length(samp.size) > 1 && !all(samp.size==samp.size[1]))stop("designs are of different sizes")
                     samp.size <- unique(samp.size)
                     
                     #if the weights are null assign equal weighting
                     if(!is.null(weights)){
                       if(length(weights)!=length(list(...)))stop("weights not same length as designs")
                       self$weights <- weights
                     } else {
                       message("weights not provided, assigning equal weighting. weights can be changed manually in self$weights")
                       self$weights <- rep(1/length(list(...)),length(list(...)))
                     }
                     
                     #if the experimental condition is null assign all separate
                     if(!is.null(experimental_condition)){
                       if(length(experimental_condition)!=samp.size)stop("experimental condition not same size as designs")
                       self$experimental_condition <- experimental_condition
                     } else {
                       message("experimental condition not provided, assuming each observation is a separate experimental condition. experimental condition can be changed manually in self$experimental_condition")
                       self$experimental_condition <- 1:samp.size
                     }
                     
                     for (item in list(...)) {
                       self$add(item)
                     }

                   },
                   #' @description 
                   #' Add a design to the design space
                   #' @param x A `Design` to add to the design space
                   #' @return Nothing
                   #' @examples 
                   #' #See examples for constructing the class
                   add = function(x) {
                     if(length(private$designs)>0 && x$n()!=private$designs[[1]]$n())stop("New design is not same size as designs in this design space.")
                     private$designs <- append(private$designs, list(x))
                     self$weights <- rep(1/length(private$designs),length(private$designs))
                     invisible(self)
                   },
                   #' @description 
                   #' Removes a design from the design space
                   #' @param index Index of the design to remove
                   #' @return Nothing
                   #' @examples 
                   #' #See examples for constructing the class
                   remove = function(index) {
                     if (length(private$designs) == 0) return(NULL)
                     private$designs <- private$designs[-index]
                   },
                   #' @description 
                   #' Print method for the Design Space
                   #' @param ... ignored
                   #' @return Prints to the console all the designs in the design space
                   #' @examples 
                   #' #See examples for constructing the class
                   print = function(){
                     cat(paste0("Design space with ",self$n()[[1]]," design(s): \n"))
                     for(i in 1:length(private$designs)){
                       cat(paste0("=========================================================\nDESIGN ",i,"(weight ",self$weights[i],"):\n"))
                       print(private$designs[[i]])
                     }
                   },
                   #' @description 
                   #' Returns the size of the design space and number of observations
                   #' @examples 
                   #' #See examples for constructing the class
                   n = function(){
                     c("n.designs"=length(private$designs),"n" = private$designs[[1]]$n())
                   },
                   #' @description 
                   #' Approximate c-optimal design of size m
                   #' 
                   #' Algorithms to identify an approximate c-optimal design of size m within the design space.
                   #' @details 
                   #' **Approximate c-Optimal designs**
                   #' The algorithm identifies a c-optimal design of size m from the design space with N designs each with n observations. The objective
                   #' function is
                   #' 
                   #' \deqn{C^TM^{-1}C}
                   #' 
                   #' where M is the information matrix and C is a vector. Typically C will be a vector of zeros with a single 1 in the position of the
                   #' parameter of interest. For example, if the columns of X in the design are an interept, the treatment indicator, and then time 
                   #' period indicators, the vector C may be `c(0,1,0,0,...)`, such that the objective function is the variance of that parameter. 
                   #' If there are multiple designs in the design space, the C vectors do 
                   #' not have to be the same as the columns of X in each design might differ, in which case a list of vectors can be provided.
                   #' 
                   #' If the experimental conditions are correlated with one another, then a hill climbing algorithm is used to find the optimal 
                   #' design by using the convexity of the objective function to "climb the hill" towards the optimal design. 
                   #' If the experimental conditional are uncorrelated (but there is correlation between observations within the same
                   #' experimental condition) then optionally a fast algorithm can be used to approximate the optimal design using a second-order 
                   #' cone program (see Sangol 2015 and van Dette). The approximate algorithm will return weights for each unique experimental condition representing
                   #' the "proportion of effort" to spend on each design condition. There are different ways to translate these weights into integer
                   #' values. Use of the approximate optimal design algorithm can be disabled used `force_hill=TRUE`
                   #' 
                   #' In some cases the optimal design will not be full rank with respect to the design matrix X of the design space. This will result
                   #' in a non-positive definite information matrix, and an error. The program will indicate which columns of X are likely "empty" in the optimal
                   #' design. The user can then optionally remove these columns in the algorithm using the `rm_cols` argument, which will delete the
                   #' specified columns and linked observations before starting the algorithm. 
                   #' 
                   #' The algorithm will also identify robust optimal designs if there are multiple designs in the design space. 
                   #' There are two options for robust optimisation. First, a weighted average of objective functions, where the weights are specified 
                   #' by the `weights` field in the design space (`robust_function = "weighted"`). The weights may represent the prior probability or plausibility of each design, 
                   #' for example. Second, a minimax approach can be used, where the function identifies the design that minimises the maximum objective
                   #' function across all designs (`robust_function = "minimax"`).
                   #' @param m A positive integer specifying the number of experimental conditions to include.
                   #' @param C Either a vector or a list of vectors of the same length as the number of designs, see Details.
                   #' @param V0 Optional. If a Bayesian c-optimality problem then this should be a list of prior covariance matrices for the model parameters
                   #' the same length as the number of designs.
                   #' @param rm_cols Optional. A list of vectors indicating columns of X to remove from each design, see Details.
                   #' @param keep Logical indicating whether to "keep" the optimal design in the linked design objects and remove any experimental
                   #' conditions and columns that are not part of the optimal design. Irreversible, so that these observations will be lost from the 
                   #' linked design objects. Defaults to FALSE.
                   #' @param verbose Logical indicating whether to reported detailed output on the progress of the algorithm. Default is TRUE.
                   #' @param algo character string, either "local" for local search algorithm, or "greedy" for greedy search
                   #' @param force_hill Logical. If the experimental conditions are uncorrelated, if this option is TRUE then the hill climbing 
                   #' algorithm will be used, otherwise if it is FALSE, then a fast approximate alternative will be used. See Details
                   #' @param p Positive integer specifying the size of the starting design for the greedy algorithm
                   #' @return A vector indicating the identifiers of the experimental conditions in the optimal design, or a vector indicating the
                   #' weights if the approximate algorithm is used. Optionally the linked designs are also modified (see option `keep`).
                   #' @examples
                   #' df <- nelder(~(cl(6)*t(5)) > ind(5))
                   #' df$int <- 0
                   #' df[df$t >= df$cl, 'int'] <- 1
                   #' mf1 <- MeanFunction$new(
                   #'   formula = ~ factor(t) + int - 1,
                   #'   data=df,
                   #'   parameters = c(rep(0,5),0.6),
                   #'   family =gaussian()
                   #' )
                   #' cov1 <- Covariance$new(
                   #'   data = df,
                   #'   formula = ~ (1|gr(cl)),
                   #'   parameters = c(0.25)
                   #' )
                   #' des <- Model$new(
                   #'   covariance = cov1,
                   #'   mean.function = mf1,
                   #'   var_par = 1
                   #' )
                   #' ds <- DesignSpace$new(des)
                   #' 
                   #' #find the optimal design of size 30 individuals
                   #' opt <- ds$optimal(30,C=list(c(rep(0,5),1)))
                   #' 
                   #' #let the experimental condition be the cluster
                   #' # these experimental conditions are independent of one another
                   #' ds <- DesignSpace$new(des,experimental_condition = df$cl)
                   #' #now find the optimal 4 clusters to include
                   #' # approximately, finding the weights for each condition
                   #' # note it will ignore m and just return the weights
                   #' opt <- ds$optimal(4,C=list(c(rep(0,5),1)))
                   #' # or use the exact algorithm
                   #' opt <- ds$optimal(4,C=list(c(rep(0,5),1)),force_hill = TRUE)
                   #' 
                   #' #robust optimisation using two designs
                   #'   cov2 <- Covariance$new(
                   #'   data = df,
                   #'   formula = ~ (1|gr(cl)*ar1(t)),
                   #'   parameters = c(0.25,0.8)
                   #' )
                   #' des2 <- Model$new(
                   #'   covariance = cov1,
                   #'   mean.function = mf1,
                   #'   var_par = 1
                   #' )
                   #' ds <- DesignSpace$new(des,des2)
                   #' #weighted average
                   #' opt <- ds$optimal(30,C=list(c(rep(0,5),1),c(rep(0,5),1)))
                   optimal = function(m,
                                      C,
                                      V0=NULL,
                                      rm_cols=NULL,
                                      keep=FALSE,
                                      verbose=TRUE,
                                      algo = 1,
                                      force_hill=FALSE,
                                      p){
                     if(keep&verbose)message("linked design objects will be overwritten with the new design")
                     if(length(C)!=self$n()[[1]])stop("C not equal to number of designs")
                     if(!is.null(V0) & length(V0)!=self$n()[[1]])stop("V0 not equal to number of designs")
                     ## add checks
                     
                     # dispatch to correct algorithm
                     # check if the experimental conditions are correlated or not
                     #loop through each sigma
                     if(verbose)message("Checking experimental condition correlations...")
                     if(length(self$experimental_condition)!=private$designs[[1]]$n())stop("experimental condition not the same length as design")
                     uncorr <- TRUE
                     unique_exp_cond <- unique(self$experimental_condition)
                     for(i in 1:self$n()[[1]]){
                       for(j in unique_exp_cond){
                         uncorr <- all(private$designs[[i]]$Sigma[which(self$experimental_condition==j),which(self$experimental_condition!=j)]==0)
                         if(!uncorr)break
                       }
                       if(!uncorr)break
                     }
                     ## need to detect if the experimental conditions are duplicated
                     ## can update this but currently only provides a warning to the user
                     if(uncorr&!force_hill){
                       datahashes <- c()
                       for(j in unique_exp_cond){
                         datalist <- list()
                         for(k in 1:self$n()[[1]]){
                           datalist[[k]] <- list(private$designs[[i]]$mean_function$X[self$experimental_condition==j,],
                                                 private$designs[[i]]$Sigma[self$experimental_condition==j,self$experimental_condition==j])
                         }
                         datahashes <- c(datahashes, digest::digest(datalist))
                       }
                       
                       if(any(duplicated(datahashes))){
                         unique_hash <- unique(datahashes)
                         n_unique_hash <- length(unique_hash)
                         datahashes <- match(datahashes,unique_hash)
                         message(paste0("Duplicated experimental conditions in the design space, ",n_unique_hash," unique 
experimental conditions, which are uncorrelated. 
force_hill=FALSE so weights will be calculated for each experimental condition separately. Sum of weights for
each condition will be reported below."))
                       }
                     }
                     
                     if(!is(C,"list")){
                       C_list <- list()
                       for(i in 1:self$n()[[1]]){
                         C_list[[i]] <- matrix(C,ncol=1)
                       }
                     } else {
                       C_list <- C
                     }
                     
                     if(verbose&uncorr&!force_hill)message("Experimental conditions uncorrelated, using second-order cone program")
                     if(verbose&uncorr&force_hill)message("Experimental conditions uncorrelated, but using hill climbing algorithm")
                     if(verbose&!uncorr)message("Experimental conditions correlated, using combinatorial search algorithms")
                     
                     if(uncorr&!force_hill){
                       # this returns the experimental designs to keep
                       idx_out <- private$socp_roptimal(C_list,m)
                       idx_out <- drop(idx_out)
                       if(verbose)cat("Weights for each experimental condition in the optimal design: ", idx_out)
                       if(any(duplicated(datahashes))){
                         agg_weights <- aggregate(idx_out,list(datahashes),sum)
                         cat("\nSum of weights for unique experimental conditions: ",agg_weights$x)
                         idx_out <- list(weights = idx_out, unique_weights = agg_weights$x)
                       }
                       return(invisible(idx_out))
                     } else {
                       #initialise from random starting index
                       N <- private$designs[[1]]$mean_function$n()
                       X_list <- private$genXlist()
                       Z_list <- private$genZlist()
                       D_list <- private$genDlist()
                       
                       #sig_list <- private$genSlist()
                       weights <- self$weights
                       #rdmode <- 1#ifelse(robust_function=="weighted",1,0)
                       if(!is.null(rm_cols))
                       {
                         if(!is(rm_cols,"list"))stop("rm_cols should be a list")
                         zero_idx <- c()
                         idx_original <- 1:nrow(X_list[[1]])
                         
                         # find all the entries with non-zero values of the given columns in each design
                         for(i in 1:length(rm_cols))
                         {
                           if(!is.null(rm_cols[[i]])){
                             for(j in 1:length(rm_cols[[i]]))
                             {
                               zero_idx <- c(zero_idx,which(X_list[[i]][,rm_cols[[i]][j]]!=0))
                             }
                           }
                         }
                         zero_idx <- sort(unique(zero_idx))
                         
                         # exp_cond <- as.numeric(factor(self$experimental_condition[-zero_idx],
                         #                                  levels=unique(self$experimental_condition[-zero_idx])))
                         expcond <- self$experimental_condition[-zero_idx]
                         uexpcond <- unique(self$experimental_condition[-zero_idx])
                         ncond <- length(uexpcond)
                         
                         # idx_in <- sort(sample(1:(N-length(zero_idx)),m,replace=FALSE))
                         idx_original <- idx_original[-zero_idx]
                         # idx_in <- match(idx_in,idx_original)
                         
                         if(verbose)message(paste0("removing ",length(zero_idx)," observations"))
                         
                         #update the matrices
                         for(i in 1:length(rm_cols))
                         {
                           X_list[[i]] <- X_list[[i]][-zero_idx,-rm_cols[[i]]]
                           C_list[[i]] <- C_list[[i]][-rm_cols[[i]]]
                           sig_list[[i]] <- sig_list[[i]][-zero_idx,-zero_idx]
                         }
                         
                         N <- nrow(X_list[[1]])
                         
                       } else {
                         expcond <- self$experimental_condition
                         uexpcond <- unique(self$experimental_condition)
                       }
                       
                       ncond <- length(uexpcond)
                       XZ <- cbind(X_list[[1]],Z_list[[1]])
                       XZ.hash <- c()
                       for(i in unique(expcond)){
                         XZ.hash <- c(XZ.hash,digest::digest(XZ[expcond==i,]))
                       }
                       #XZ.hash <- apply(XZ,1,digest::digest)
                       row.hash <- as.numeric(factor(XZ.hash,levels=unique(XZ.hash)))
                       #idx.bool <- c(!duplicated(row.hash)) + c(duplicated(expcond)) 
                       ridx.nodup <- which(!duplicated(row.hash))
                       idx.nodup <- which(expcond %in% unique(expcond)[ridx.nodup])
                       n.uni.obs <- length(idx.nodup)
                       w_diag <- matrix(0,ncol=length(X_list),nrow=n.uni.obs)
                       for(i in 1:length(X_list)){
                         X_list[[i]] <- X_list[[i]][idx.nodup,]
                         Z_list[[i]] <- Z_list[[i]][idx.nodup,]
                         if(is.null(rm_cols)){
                           w_diag[,i] <- Matrix::diag(private$designs[[i]]$.__enclos_env__$private$W)[idx.nodup]
                         } else {
                           w_diag[,i] <- Matrix::diag(private$designs[[i]]$.__enclos_env__$private$W)[-zero_idx][idx.nodup]
                         }
                       }
                       
                       max_obs <- unname(table(row.hash))
                       expcond.id <- as.numeric(factor(expcond[idx.nodup],levels = unique(expcond[idx.nodup])))
                       
                       # row.hash <<- row.hash
                       if(algo == 1){
                         idx_in <- sort(sample(row.hash,m,replace=FALSE))
                       } else if(algo %in% 2:4){
                         # find a size p design
                         ispd <- FALSE
                         #n <- nrow(X_list[[1]])
                         while(!ispd){
                           idx_in <- sort(sample(unique(row.hash),p,replace=FALSE))
                           M <- crossprod(X_list[[1]][expcond.id%in%idx_in,])
                           cM <- suppressWarnings(tryCatch(chol(M),error=function(e)NULL))
                           if(!is.null(cM))ispd <- TRUE
                         }
                       }
                       
                       bayes <- FALSE
                       if(!is.null(V0)){
                         bayes <- TRUE
                         for(i in 1:length(X_list)){
                           if(dim(V0[[i]])[1] != ncol(X_list[[i]]))stop(paste0("V0 wrong dimension for design ",i))
                         }
                       } else {
                         V0 <- list()
                         for(i in 1:length(X_list)){
                           V0[[i]] <- matrix(1)
                         }
                       }
                       
                       # #for debugging
                       args1 <<- list(idx_in = idx_in, 
                                     n=m,
                                     C_list = C_list, 
                                     X_list = X_list,
                                     Z_list = Z_list,
                                     D_list = D_list,
                                     w_diag = w_diag,
                                     max_obs = max_obs,
                                     any_fix = 0,
                                     nfix = N+10,
                                     V0_list = V0,
                                     weights = weights, 
                                     exp_cond = expcond.id,
                                     type = algo-1,
                                     rd_mode=1,
                                     trace=verbose,
                                     uncorr=uncorr,
                                     bayes=bayes)
                       stop("testing")
                       out_list <- GradRobustStep(idx_in = idx_in, 
                                                  n=m,
                                                  C_list = C_list, 
                                                  X_list = X_list,
                                                  Z_list = Z_list,
                                                  D_list = D_list,
                                                  w_diag = w_diag,
                                                  max_obs = max_obs,
                                                  any_fix = 0,
                                                  nfix = N+10,
                                                  V0_list = V0,
                                                  weights = weights, 
                                                  exp_cond = expcond.id,
                                                  type = algo-1,
                                                  rd_mode=1,
                                                  trace=verbose,
                                                  uncorr=uncorr,
                                                  bayes=bayes)
                       idx_out <- drop(out_list[["idx_in"]] )
                       idx_out_exp <- sort(idx_out)
                       rows_in <- c()
                       for(i in 1:length(idx_out_exp)){
                         uni.hash <- which(row.hash == idx_out_exp[i])
                         if(length(uni.hash)==1){
                           idx_out_exp[i] <- uni.hash
                         } else {
                           idx_out_exp[i] <- uni.hash[!uni.hash %in% idx_out_exp][1]
                         }
                         rows_in <- c(rows_in, which(expcond == idx_out_exp[i]))
                       }
                       
                       if(!is.null(rm_cols)){
                         rows_in <- idx_original[rows_in]
                       } 
                       if(keep){
                         for(i in 1:self$n()[[1]]){
                           private$designs[[i]]$subset_rows(rows_in)
                           ncol <- 1:ncol(private$designs[[i]]$mean_function$X)
                           if(!is.null(rm_cols))private$designs[[i]]$subset_cols(ncol[-rm_cols[[i]]])
                           private$designs[[i]]$check(verbose=FALSE)
                         }
                       }
                       
                       #if(verbose)cat("Experimental conditions in the optimal design: ", idx_out_exp$rows)
                       return(invisible(list(rows = rows_in, exp.cond = idx_out_exp, val = out_list$best_val_vec,
                                             func_calls = out_list$func_calls, mat_ops = out_list$mat_ops)))
                     }
                   },
                   #' @description 
                   #' Returns a linked design
                   #' @param i Index of the design to return
                   #' @examples 
                   #' #See examples for constructing the class
                   show = function(i){
                     return(private$designs[[i]])
                   }
                 ),
                 private = list(
                   designs = list(),
                   genXlist = function(){
                     X_list <- list()
                     for(i in 1:self$n()[[1]]){
                       X_list[[i]] <- as.matrix(private$designs[[i]]$mean_function$X)
                     }
                     return(X_list)
                   },
                   genSlist = function(){
                     S_list <- list()
                     for(i in 1:self$n()[[1]]){
                       S_list[[i]] <- as.matrix(private$designs[[i]]$Sigma)
                     }
                     return(S_list)
                   },
                   genZlist = function(){
                     Z_list <- list()
                     for(i in 1:self$n()[[1]]){
                       Z_list[[i]] <- as.matrix(private$designs[[i]]$covariance$Z)
                     }
                     return(Z_list)
                   },
                   genDlist = function(){
                     D_list <- list()
                     for(i in 1:self$n()[[1]]){
                       D_list[[i]] <- as.matrix(private$designs[[i]]$covariance$D)
                     }
                     return(D_list)
                   },
                   socp_roptimal = function(C,
                                            m){
                     X <- private$genXlist()
                     sig <- private$genSlist()
                     weights <- self$weights
                     exp_cond <- self$experimental_condition
                     
                     n_r <- length(sig)
                     constr <- list()
                     n <- nrow(X[[1]])
                     n_ex <- unique(exp_cond)
                     mu <- CVXR::Variable(length(n_ex))
                     z <- CVXR::Variable(n*n_r)
                     
                     for(i in 1:n_r){
                       constr[[i]] <- t(X[[i]])%*%Matrix::t(Matrix::chol(Matrix::solve(sig[[i]])))%*%z[c(1:n + n*(i-1))] == C[[i]]
                     }
                     
                     for(i in n_ex){
                       #build expression
                       cons_str <- "weights[1]*CVXR::p_norm(z[which(exp_cond==i)])"
                       if(n_r > 1){
                         for(j in 1:(n_r-1)){
                           cons_str <- paste0(cons_str," + weights[",j+1,"]*p_norm(z[(which(exp_cond==i)+",j,"*n)])")
                         }
                       }
                       cons_str <- paste0(cons_str, " <= mu[i]")
                       
                       constr[[(length(constr)+1)]] <- eval(str2lang(cons_str))
                     }
                     obj <- sum(mu)
                     prob <- CVXR::Problem(CVXR::Minimize(obj),constr)
                     stopifnot(CVXR::is_dcp(prob))
                     res <- CVXR::solve(prob)
                     weights <- res$getValue(mu)
                     # choose the m biggest to keep
                     weights/sum(weights)
                     #order(weights,decreasing = TRUE)[1:m]
                   }
                 )
)








