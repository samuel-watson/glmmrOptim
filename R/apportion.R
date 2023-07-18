#' Generate exact designs from approximate weights
#' 
#' Given a set of optimal weights for experimental conditions generate 
#' exact designs using several rounding methods.
#' 
#' @details 
#' Allocating `n` items to `k` groups proportionally to set of weights `w` is
#' known as the apportionment problem. The problem most famously arose when 
#' determining how many members each state should have in the U.S. House of
#' Representatives based on their proportion of the population. The solutions are
#' named after their proposers in the early U.S. Hamilton's method initially 
#' allocates `floor(n*w)` observations to each experimental condition and then
#' allocates the remaining observations based on the largest remainders `n*w - floor(n*w)`.
#' The other methods (Adams, Jefferson, and Webster) are divisor methods. 
#' The vector of counts is `m`, which is either all zeros for Jefferson and 
#' Webster and all ones for Adams, and we define `pi <- n*w` and then 
#' iteratively add observations based on the largest values of `pi/alpha` 
#' where `alpha` is either:
#' * m + 0.5 (Webster)
#' * m + 1 (Jefferson)
#' * m (Adams)
#' Pukelsheim and Rieder, 1996 <doi:10.2307/2337232> discuss efficient rounding
#' of experimental condition weights and determine that a variant of Adam's 
#' method is the most efficient. Results using this method are labelled "Pukelsheim"
#' in the output; there may be multiple designs using this procedure. Pukelsheim 
#' and Rieder's method assumes there is a minimum of one experimental condition 
#' of each type, whereas the other methods do not have this restriction.
#' 
#' @param w A vector of weights.
#' @param n The size of the exact designs to return.
#' @return A named list. The names correspond to the method of rounding (see Details),
#' and the entries are vectors of integers indicating the count of each type of 
#' experimental condition.
#' @examples 
#' w <- c(0.45,0.03,0.02,0.02,0.03,0.45)
#' apportion(w,10)
#' @importFrom digest digest
#' @export
apportion <- function(w,n){
  if(sum(w)!=1)w <- w/sum(w)
  j <- length(w)
  
  ## Pukelsheim and Rieder approach
  v <- n - j/2
  wv <- w*v
  designs <- list()
  # first check for integers
  if(any(wv%%1 == 0)){
    nint <- sum(wv%%1 == 0)
    if(nint > 10){
      designs[[1]] <- ceiling(wv)
      message("Puckelsheim and Rieder designs skipped due to very large number of possible designs.")
      break;
    }
    idxint <- which(wv%%1 == 0)
    wint <- list()
    for(i in 1:nint){
      wint[[i]] <- c(wv[idxint[i]],wv[idxint[i]]+1)
    }
    
    dfint <- do.call(expand.grid,wint)
    
    for(i in 1:nrow(dfint)){
      wvtmp <- ceiling(wv)
      wvtmp[idxint] <- unname(unlist(dfint[i,]))
      designs[[i]] <- wvtmp
    }
    
  } else {
    designs[[1]] <- ceiling(wv)
  }
  
  for(i in 1:length(designs)){
    if(sum(designs[[i]])<n){
      while(sum(designs[[i]])<n){
        wvw <- designs[[i]]/w
        designs[[i]][which.min(wvw)] <- designs[[i]][which.min(wvw)]+1 
      }
    } else if(sum(designs[[i]])>n){
      while(sum(designs[[i]])>n){
        wvw <- (designs[[i]]-1)/w
        designs[[i]][which.max(wvw)] <- designs[[i]][which.max(wvw)]-1 
      }
    }
  }
  
  # remove duplicates
  if(length(designs)>1){
    hashd <- c()
    for(i in 1:length(designs)){
      hashd[i] <- digest::digest(designs[[i]])
    }
    und <- which(!duplicated(hashd))
    d2 <- list()
    for(i in 1:length(und)){
      d2[[i]] <- designs[[und[i]]]
    }
    designs <- d2
    rm(d2)
  }
  
  names(designs) <- paste0("Pukelsheim_",1:length(designs))
  
  ## Hamilton
  designs[[length(designs)+1]] <- floor(n*w)
  if(sum(designs[[length(designs)]])<n){
    while(sum(designs[[length(designs)]])<n){
      rem <- n*w - designs[[length(designs)]]
      designs[[length(designs)]][which.max(rem)] <- designs[[length(designs)]][which.max(rem)]+1
    }
  }
  
  names(designs)[length(designs)] <- "Hamilton"
  
  # jefferson
  designs[[length(designs)+1]] <- rep(0,j)
  pi <- n*w
  while(sum(designs[[length(designs)]])<n){
    alpha <- designs[[length(designs)]] + 1
    pia <- pi/alpha
    designs[[length(designs)]][which.max(pia)] <- designs[[length(designs)]][which.max(pia)]+1
  }
  
  names(designs)[length(designs)] <- "Jefferson"
  
  # webster
  designs[[length(designs)+1]] <- rep(0,j)
  pi <- n*w
  while(sum(designs[[length(designs)]])<n){
    alpha <- designs[[length(designs)]] + 0.5
    pia <- pi/alpha
    designs[[length(designs)]][which.max(pia)] <- designs[[length(designs)]][which.max(pia)]+1
  }
  
  names(designs)[length(designs)] <- "Webster"
  
  return(designs)
}