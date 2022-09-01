# glmmrMCML
R package for approximate optimal experimental designs using generalised linear mixed models (GLMM) and combinatorial optimisation methods,
built on the [glmmrBase](https://github.com/samuel-watson/glmmrBase) package.

## Model specification
For model specification see readme of [glmmrBase](https://github.com/samuel-watson/glmmrBase). The `glmmrMCML` package adds the `DesignSpace` class. An instance 
of a `DesignSpace` object takes one or more `Model` objects, where each object specifies a particular design. Each object should be a complete enumeration of all
possible observations. Observations can be nested within "experimental conditions", which are immutable collections of observations, 
for example, we may wish to choose $m$ of $n$ possible clusters of observations. Often the experimental conditions will comprise only a single observation.
The aim is to identify an approximately c-optimal design with $m < n$ experimental conditions, or in the case of multiple designs a 
robust c-optimal design weighting over all designs. 

## Optimal designs
The algorithm searches for a c-optimal design of size m from the design space using either local search, greedy search, or some combination of the two. 
The objective function is

$$
C^TM^{-1}C
$$ 

where $M$ is the information matrix and $C$ is a vector. Typically $C$ will be a vector of zeros with a single 1 in the position of the parameter of interest. 
For example, if the columns of $X$ in the design are an intercept, the treatment indicator, and then time period indicators, the vector $C$ may be `c(0,1,0,0,...)`, 
such that the objective function is the variance of that parameter. If there are multiple designs in the design space, the $C$ vectors do not have to be the same 
as the columns of X in each design might differ, in which case a list of vectors can be provided.

If the experimental conditions are correlated with one another, then one of the combinatorial algorithms is used to find an optimal design. 
If the experimental conditional are uncorrelated (but there is correlation between observations 
within the same experimental condition) then optionally a fast algorithm can be used to approximate the optimal design using a second-order cone program. 
The approximate algorithm will return weights for each unique experimental condition representing the 
"proportion of effort" to spend on each design condition. There are different ways to translate these weights into integer values. 
Use of the approximate optimal design algorithm can be disabled with the option `force_hill=TRUE`.

In some cases the optimal design will not be full rank with respect to the design matrix $X$ of the design space. This will result in a non-positive definite 
information matrix, and an error. The program will indicate which columns of $X$ are likely "empty" in the optimal design. The user can then optionally remove 
these columns in the algorithm using the `rm_cols` argument, which will delete the specified columns and linked observations before starting the algorithm. 
                   
The algorithm will also identify robust optimal designs if there are multiple designs in the design space. A weighted average of objective functions is used, 
where the weights are specified by the `weights` field in the design space with default $1/N$. 
The weights may represent the prior probability or plausibility of each design, for example. 

An example of model specification and optimal design search is below.
```
R> df <- nelder(~(cl(6)*t(5)) > ind(5))
R> df$int <- 0
R> df[df$t >= df$cl, 'int'] <- 1
R> mf1 <- MeanFunction$new(
R>   formula = ~ factor(t) + int - 1,
R>   data=df,
R>   parameters = c(rep(0,5),0.6),
R>   family =gaussian()
R> )
R> cov1 <- Covariance$new(
R>   data = df,
R>   formula = ~ (1|gr(cl)),
R>   parameters = c(0.25)
R> )
R> des <- Model$new(
R>   covariance = cov1,
R>   mean.function = mf1,
R>   var_par = 1
R> )
R> ds <- DesignSpace$new(des)
R> #find the optimal design of size 30 individuals
R> opt <- ds$optimal(30,C=c(rep(0,5),1))
```
