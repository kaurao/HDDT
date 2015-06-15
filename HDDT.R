# HDDT
# functions to create and use Hellinger distance decision tree (HDDT)
# written by: Kaustubh Patil - MIT Neuroecon lab (C) 2015

# DISCLAIMER
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
# OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
# ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.

# LICENSE
# CREATIVE COMMONS Attribution-NonCommercial 2.5 Generic (CC BY-NC 2.5)
# https://creativecommons.org/licenses/by-nc/2.5/

# References:
# Hellinger distance decision trees are robust and skew-insensitive, 
# ... David A. Cieslak, T. Ryan Hoens, Nitesh V. Chawla and W. Philip Kegelmeyer, Data Min Knowl Disc 2011
# https://www3.nd.edu/~dial/papers/DMKD11.pdf
# Learning Decision Trees for Unbalanced Data, David A. Cieslak and Nitesh V. Chawla, ECML 2008
# https://www3.nd.edu/~dial/papers/ECML08.pdf

# build a Hellinger distance decision tree
# it is a recursive function that calls itself with subsets
# of training data that matches the decision criterion
# using a list to create the tree structure
#
# Input
# X (matrix/data frame): training data, features/independent variables
#                       The columns of X must be either numeric or factor
# y (vector)           : training data, labels/dependent variable
# C (integer)          : minimum size of the training set at a node to attempt a split
# labels (vector)      : allowed labels [optional]
#
# Value
# node (list)          : the root node of the deicison tree
HDDT <- function(X, y, C, labels=unique(y)) {
  
  if(is.null(labels) || length(labels)==0) labels <- unique(y)  
  
  node <- list() # when called for first time, this will be the root
  node$C <- C
  node$labels <- labels
  
  if(length(unique(y))==1 || length(y) < C) {
    # calculate counts and frequencies
    # use Laplace smoothing, by adding 1 to count of each label
    y <- c(y, labels)
    node$count <- sort(table(y), decreasing=TRUE)
    node$freq  <- node$count/sum(node$count)
    # get the label of this leaf node
    node$label <- as.integer(names(node$count)[1])
    return(node)
  }
  else { # recursion
    # get Hellinger distance and their max
    # use for loop insread of apply as it will convert data.frame to a matrix and mess up column classes
    # e.g. factor will get coerced into character
    HD <- list()
    for(i in 1:ncol(X)) HD[[i]] <- HDDT_dist(X[,i],y=y,labels=labels)    
    hd <- sapply(HD, function(x) {return(x$d)})
    i  <- which(hd==max(hd))[1] # just taking the first 
    
    # save node attributes
    node$i    <- i
    node$v    <- HD[[i]]$v
    node$type <- HD[[i]]$type
    node$d    <- HD[[i]]$d
    
    if(node$type=="factor") {
      j <- X[,i]==node$v
      node$childLeft  <- HDDT(X[j,], y[j], C, labels)
      node$childRight <- HDDT(X[!j,], y[!j], C, labels)
    }
    else if(node$type=="numeric") {
      j <- X[,i]<=node$v
      node$childLeft  <- HDDT(X[j,], y[j], C, labels)
      node$childRight <- HDDT(X[!j,], y[!j], C, labels)      
    }
  }
  
  return(node) # returns root node
}

# given the root node as returned by the HDDT function and
# new data X return predictions
#
# Input
# root (list)           : root node as returned by the function HDDT
# X (matrix/data frame) : new data, features/independent variables
#
# Value
# y (integer vector)    : predicted labels for X
HDDT_predict <- function(root, X) {
  y <- rep(NA, nrow(X))
  for(i in 1:nrow(X)) {
    # traverse the tree until we find a leaf node
    node <- root
    while(!is.null(node$v)) {
      if(node$type=="factor") {
        if(X[i,node$i]==node$v) node <- node$childLeft
        else node <- node$childRight
      }
      else if(node$type=="numeric") {
        if(X[i,node$i]<=node$v) node <- node$childLeft
        else node <- node$childRight
      }
      else stop("unknown node type: ", node$type)
    }
    stopifnot(!is.null(node$label))
    y[i] <- node$label
  }
  
  return(y)
}


# given a feature vector calculate Hellinger distance
# it takes care of both discrete and continuous attributes
# also returns the "value" of the feature that is used as decision criterion
# and the "type" pf the feature which is either factor as numeric
# ONLY WORKS WITH BINARY LABELS
HDDT_dist <- function(f, y, labels=unique(y)) {  
  i1 <- y==labels[1]
  i0 <- y==labels[2]
  T1 <- sum(i1)
  T0 <- sum(i0)
  val <- NA
  hellinger <- -1
  
  cl <- class(f)  
  if(cl=="factor") {    
    for(v in levels(f)) {
      Tfv1 <- sum(i1 & f==v)
      Tfv0 <- sum(i0 & f==v)
      
      Tfw1 <- T1 - Tfv1
      Tfw0 <- T0 - Tfv0
      cur_value <- ( sqrt(Tfv1 / T1) - sqrt(Tfv0 / T0) )^2 + ( sqrt(Tfw1 / T1) - sqrt(Tfw0 / T0) )^2
      
      if(cur_value > hellinger) {
        hellinger <- cur_value
        val <- v
      }
    }
  }
  else if(cl=="numeric") {
    fs <- sort(unique(f))
    for(v in fs) {
      Tfv1 <- sum(i1 & f<=v)
      Tfv0 <- sum(i0 & f<=v)
      
      Tfw1 <- T1 - Tfv1
      Tfw0 <- T0 - Tfv0
      cur_value <- ( sqrt(Tfv1 / T1) - sqrt(Tfv0 / T0) )^2 + ( sqrt(Tfw1 / T1) - sqrt(Tfw0 / T0) )^2
      
      if(cur_value > hellinger) {
        hellinger <- cur_value
        val <- v
      }
    }
  }
  else stop("unknown class: ", cl)
  
  return(list(d=sqrt(hellinger), v=val, type=cl))
}
