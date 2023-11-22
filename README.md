[![cran version](http://www.r-pkg.org/badges/version/glmmrOptim)](https://cran.r-project.org/web/packages/glmmrOptim)

# glmmrOptim
(This text relates to version 0.3.3)
R package for approximate optimal experimental designs using generalised linear mixed models (GLMM) and combinatorial optimisation methods,
built on the [glmmrBase](https://github.com/samuel-watson/glmmrBase) package. A discussion of the methods in this package can be found in [Watson et al (2023)]([https://arxiv.org/abs/2207.09183](https://journals.sagepub.com/doi/10.1177/09622802231202379)).

## Model specification
For model specification see readme of [glmmrBase](https://github.com/samuel-watson/glmmrBase). The `glmmrOptim` package adds the `DesignSpace` class. An instance 
of a `DesignSpace` object takes one or more `Model` objects, where each object specifies a particular design. Each object should be a complete enumeration of all
possible observations. Observations can be nested within "experimental conditions", which are immutable collections of observations, 
for example, we may wish to choose $m$ of $n$ possible clusters of observations. Often the experimental conditions will comprise only a single observation.
The aim is to identify an approximately c-optimal design with $m < n$ experimental conditions, or in the case of multiple designs a 
robust c-optimal design weighting over all designs. 

## Optimal designs
The algorithm searches for a c-optimal design of size m from the design space using either local search, greedy search, or some combination of the two. 
The objective function is

$$
c^TM^{-1}c
$$ 

where $M$ is the information matrix and $c$ is a vector. Typically $c$ will be a vector of zeros with a single 1 in the position of the parameter of interest. 
For example, if the columns of $X$ in the design are an intercept, the treatment indicator, and then time period indicators, the vector $c$ may be `c(0,1,0,0,...)`, 
such that the objective function is the variance of that parameter. If there are multiple designs in the design space, the $c$ vectors do not have to be the same 
as the columns of X in each design might differ, in which case a list of vectors can be provided.

There are a variety of algorithms available:
- For design spaces with correlated experimental units, one can use either combinatorial algorithms: `algo=1` local search, `algo=2` greedy search, or `algo=3`; or an optimal mixed model weights algorithm, the "Girling" algorithm with `algo="girling"`.
- For design spaces with uncorrelated experimental units by default the optimal experimental unit weights will be calculated using a second-order cone program. To instead use a combinatorial algorithm set `use_combin=TRUE`.

In some cases the optimal design will not be full rank with respect to the design matrix $X$ of the design space. This will result in a non-positive definite 
information matrix, and an error. The program will indicate which columns of $X$ are likely "empty" in the optimal design. The user can then optionally remove 
these columns in the algorithm using the `rm_cols` argument, which will delete the specified columns and linked observations before starting the algorithm. 
                   
The algorithm will also identify robust optimal designs if there are multiple designs in the design space. A weighted average of objective functions is used, 
where the weights are specified by the `weights` field in the design space with default $1/N$. The weights may represent the prior probability or plausibility of each design, for example.  The objective function can be either a linear combination of variances, or a linear combination of log variances (`robust_log=TRUE`).

An example of model specification and optimal design search is below.
```
df <- nelder(formula(~ (cl(7) * t(6)) > ind(1)))
df$int <- 0
df[df$t >= df$cl,'int'] <- 1
des <- Model$new(formula = ~factor(t)  + int - 1 + (1|gr(cl)) + (1|gr(cl,t)),
                 covariance = c(0.04,0.01),
                 mean = rep(0,7),
                 var_par = sqrt(0.95),
                 data = df,
                 family=gaussian())
ds <- DesignSpace$new(des)
w1 <- ds$optimal(100,C = list(c(rep(0,6),1)),verbose = TRUE,algo="girling")
```
The design space supports any model specified in the `glmmrBase` package. Where there are non-linear functions of covariates in the fixed effects, a first-order approximation is used.
