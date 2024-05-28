\name{glmmrOptim-package}
\alias{glmmrOptim-package}
\alias{glmmrOptim}
\docType{package}
\title{
\packageTitle{glmmrOptim}
}
\description{
\packageDescription{glmmrOptim}
\code{glmmrOptim} provides algorithms for identifying (approximately) c-optimal experimental designs for experiments described by a generalised linear mixed model. 
}
\section{Algorithms}{
\pkg{glmmrOptim} provides algorithms for identifying (approximately) c-optimal experimental designs for experiments described by a generalised linear mixed model. Each data row constitutes an observation, which 
can be grouped into experimental units. The aim is to then find either the discrete subset of experimental units, or the optimal weights on each unit, to minimise the GLS variance criterion. There are four main algorithms:
\itemize{
  \item Reverse greedy search. This combinatorial algorithm starts from the complete set of experimental units and successively removes the unit that minimises the variance until the target sample size is reached.
  \item Local search. This combinatorial algorithm starts from a design of the target sample size and successively swaps units that minimise the variance until no improving swap can be made.
  \item Optimal weights for uncorrelated experimental units. A second-order cone program is used to determine the optimal experimental weights for each unit, where the units are uncorrelated with one another.
  \item Optimal mixed model weights. An algorithm base on the mixed model weights is used to identify the optimal experimental weights, units may be correlated.
}
The package also provides support for finding the optimal rounding of weights to integers. Robust optimal experimental designs can be identified by including multiple plausible models in the algorithms.
}
\section{Model specification}{
The \pkg{glmmrOptim} package uses the \pkg{glmmrBase} package for model specification and calculation.
}
\section{Package development}{
The package is still in development and there may still be bugs and errors. While we do not expect the general user interface to change there may be changes to the underlying library as well as new additions and functionality.
}
\author{
\packageAuthor{glmmrOptim}

Maintainer: \packageMaintainer{glmmrOptim}
}
\keyword{ package }


