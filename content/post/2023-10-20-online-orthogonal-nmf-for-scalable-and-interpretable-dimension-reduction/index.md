---
title: "Online Orthogonal NMF for Scalable and Interpretable Dimension Reduction"
author: "Zach DeBruine"
date: '2023-10-20'
output:
  pdf_document: default
  word_document: default
categories:
- NMF
- methods
tags:
- NMF
- HPC
subtitle: Overcoming issues with scalability for NMF with a new method yielding comparable results
summary: Non-negative matrix factorization is a useful method for additive decomposition
  of signal within a dataset. However, NMF does not scale well to large datasets, particularly
  during cross-validation. Here we show that a single-layer tied-weights autoencoder where a non-negativity
  constraint is applied to the weights matrix can yield results similar to NMF, and promises far better scalability.
lastmod: '2023-10-21T16:11:14-04:00'
featured: no
image:
  caption: ''
  focal_point: ''
  preview_only: yes
slug: online-orthogonal-nmf
---



## NMF has a scaling issue

Non-negative Matrix Factorization (NMF) is a popular machine learning method that is often used as a interpretable dimension reduction. While NMF is highly effective in many applications, it scales poorly, distributed algorithms are highly inefficient, and cross-validation is an almost insurmountable bottleneck at large ranks.

In this vignette, I explore a new method for achieving NMF-like results with much better scalability and potential for extension.

## NMF as an autoencoder

NMF has been observed to behave similarly to specific non-deep autoencoders. NMF seeks to minimize the following problem:

`$$min_{\{W,H\}\geq 0} \lVert A - WH \rVert^2_2$$`

where `\(W\)` and `\(H\)` are low-rank matrices the matrix multiplication product of which approximates `\(A\)`.

In contrast, autoencoders with a squared error loss function are minimizing the following:

`$$min \lVert A - A' \rVert^2_2$$`

where `\(A'\)` is the reconstruction of `\(A\)` in the output layer.

Suppose now that we have a single-layer autoencoder with tied weights (meaning the encoder and decoder weights matrices are transpose-identical), and a ReLu activation function in the hidden layer and the output layer:

`$$A' = max(0, max(0, AW)W^T)$$`

Now suppose we enforce that `\(W \geq 0\)` and assume `\(A \geq 0\)`, then it is impossible for `\(H < 0\)` and ReLu is not necessary to enforce. This reduces to the following optimization problem:

$$min_{\{W \geq 0\}} \lVert A - AWW^T \rVert^2_2 $$
which is the loss function for orthogonal NMF.

Thus, *online orthogonal NMF (ooNMF)* is a single-layer tied-weights autoencoder (with a ReLu activation function) where the weights are enforced to be non-negative.

*Hidden layer*. Note that `\(H = max(0, AW)\)` and is the "hidden layer" in this autoencoder. `\(H\)` may be linearly decoded to give a lower-dimensional representation of the input data, since the number of neurons ($k$) in the hidden layer is always much less than the rank of the matrix `\(A\)`.

*No bias terms*. Similarly to NMF, there are no bias terms added to the inputs of the hidden layer that could challenge interpretability. By omitting bias terms, the model is forced to learn a maximally informative and interpretable weights matrix. Indeed, non-negative bias terms cannot benefit the model when there are no negative weights.

*Choice of activation function*. This model is exceptionally well suited to ReLu as an activation function, and while other activation functions might theoretically be explored, they are unlikely to respond effectively to non-negative weights or compare well to existing dimensions, and would result in significant additional computational cost.

*Approximate orthogonality.* The expectation of orthogonal NMF is that `\(WW^T \approx I\)`, thus in online orthogonal NMF, we may think of orthogonalizing `\(W\)`, however due to non-negativity constraints this orthogonalization will never be accurate, and thus we do not explicitly enforce orthogonality in any way.

*Orthogonal random initialization*. He initialization is popularly used for autoencoders. We propose to orthogonalize a He initialized weights matrix to ensure that the distribution of models are consistent across random restarts, and as close to a truly orthogonal solution as can be expected without explicit enforcement of orthogonality. A simple way to impose orthogonality is with Gram Schmidt. 

## Properties of Online Orthogonal NMF (ooNMF)

ooNMF has several computational advantages:
* It scales in linear time with respect to the number of samples
* It replaces computationally intensive NNLS solvers with simple ReLu activations
* It can take advantage of moment-based optimizers to accelerate convergence (e.g. adam)

*Convergence properties.* In practice, NMF by alternating least squares (ALS-NMF) converges in fewer epoch than adam-optimized ooNMF, but ooNMF has the potential to be far faster per epoch.

*Dense inputs.* ooNMF for dense `\(A\)` will also benefit significantly from GPU acceleration, whereas the NNLS solvers required for oNMF or ALS-NMF reduce the performance of GPU-accelerated NMF.

*Sparse inputs.* ooNMF for sparse `\(A\)` will perform well on GPU or CPU, but the gains from GPU acceleration will be significantly reduced or negligible at small scale.

*Parallelization.* Forward- and back-propagation in ooNMF is embarrassingly parallel within batches or mini-batches, and can be readily parallelized with OpenMP with the exception of a critical reduction prior to weights updates.

## Implementation of ooNMF

Below is an implementation of ooNMF with an adam optimizer option:


```r
#' Online Orthogonal Non-negative Matrix Factorization
#' 
#' @param A input matrix
#' @param k rank of hidden layer
#' @param adam use adam optimizer
#' @param lr learning rate
#' @param beta1 adam hyperparameter for first moment
#' @param beta2 adam hyperparameter for second moment
#' @param epochs maximum number of training epochs
#' @param tol relative change in error at which to stop
#' @param verbose output tolerance and loss for each iteration
ooNMF <- function(A, k, lr = 0.01, epochs = 100, tol = 1e-5, verbose = F){
  # absolute-value He initialization of W
  w <- matrix(abs(rnorm(nrow(A) * k, sd = sqrt(2 / nrow(A)))), nrow(A), k)
  
  # orthogonalize our implementation
  w <- pracma::gramSchmidt(w)$Q

  err <- rep(0, epochs)
  for (epoch in 1:epochs) {
    for (i in 1:ncol(A)) {
      # forward pass with relu 
      #  (no need to impose pmax because w and A are non-negative)
      a1 <- A[, i] %*% w
      a2 <- a1 %*% t(w)

      # backpropagate error gradient with relu derivative
      error <- A[, i] - a2
      a2_delta <- error * (a2 > 0)
      a1_delta <- (a2_delta %*% w) * (a1 > 0)

      # because "w" is the same in the encoder and decoder (tied weights)
      #  the gradient is the sum of gradients from both sides of the network
      dW <- A[, i] %*% a1_delta + t(t(a1) %*% a2_delta)
      
      # update weight matrix and apply non-negativity constraints
      w <- pmax(w + lr * dW, 0)
      
      err[epoch] <- err[epoch] + mean(error^2)
    }
    err[epoch] <- err[epoch] / ncol(A)
    tol_ <- 1
    if (epoch > 1) tol_ <- abs(err[epoch - 1] - err[epoch]) / (err[epoch] + err[epoch - 1])
    if(verbose) cat("epoch: ", epoch, ", error: ", err[epoch], ", tol: ", tol_, "\n")
    if(tol_ < tol) break
  }
  return(list(w = as.matrix(w), h = as.matrix(t(w) %*% A), error = err))
}
```

## Comparison to Matrix Factorizations

Train several dimension reduction models on hawaiibirds dataset:


```r
set.seed(123)
library(RcppML)
library(Matrix)
library(ggplot2)
A <- as.matrix(hawaiibirds$counts)

oonmf_model <- ooNMF(A, 15)
nmf_model <- RcppML::nmf(A, 15)
svd_model <- irlba::irlba(A, 15)
pca_model <- irlba::prcomp_irlba(A, 15, scale = TRUE)
```

### Similarity of Reconstructions

Generate reconstructions for most models:


```r
recon_nmf <- as.matrix(nmf_model@w %*% Diagonal(x = nmf_model@d) %*% nmf_model@h)
recon_oonmf <- t(t(A) %*% oonmf_model$w %*% t(oonmf_model$w))
recon_svd <- as.matrix(svd_model$u %*% Diagonal(x = svd_model$d) %*% t(svd_model$v))
```

Compare the similarity of reconstructions:

<img src="{{< blogdown/postref >}}index_files/figure-html/unnamed-chunk-4-1.png" width="624" style="display: block; margin: auto auto auto 0;" />

All methods approximate the input data well.  

### Reconstruction Accuracy 

We can also measure the mean squared error of reconstruction for these methods:

<img src="{{< blogdown/postref >}}index_files/figure-html/unnamed-chunk-5-1.png" width="672" style="display: block; margin: auto auto auto 0;" />

ooNMF compares favorably with NMF and SVD when realizing that it is both approximately orthogonal and updated online.

### UMAP embeddings

Compare the UMAP plots from the low-rank matrices:

<img src="{{< blogdown/postref >}}index_files/figure-html/unnamed-chunk-6-1.png" width="480" style="display: block; margin: auto auto auto 0;" />

Visually, all plots appear informative and convey similar information.

### Overlap of Nearest Neighborhoods

Compare the overlap between nearest neighborhoods in the lower-dimensional space. We will do this using a jaccard measure of overlap between neighborhoods for each sample.

<img src="{{< blogdown/postref >}}index_files/figure-html/unnamed-chunk-7-1.png" width="576" style="display: block; margin: auto auto auto 0;" />

All of the shared nearest neighborhoods are very similar between all methods, which supports the observation from UMAP that the visualizations are similarly informative. However, ooNMF is most similar to NMF and the raw data. Notably, less similar to orthogonal methods SVD and PCA.

### Sparsity

Sparsity of the model is slightly lower than NMF, but comparable:


```
##   method model   sparsity
## 1    NMF     W 0.75482696
## 2    NMF     H 0.49783037
## 3  ooNMF     W 0.71366120
## 4  ooNMF     H 0.01780783
```

## Approximate Orthogonality 

The expectation of orthogonal NMF is that `\(WW^T\)` will approximately yield the identity matrix. This is the case for SVD, but it is only approximately true for NMF due to non-negativity constraints and no explicit enforcement of orthogonality during fitting. 

If indeed we did enforce orthogonality of "W" (for instance, with Gram Schmidt) during fitting, we would encounter negative values in "W", and setting those back to 0 would just complicate stochastic gradient descent after non-negativity constraints on W.


```r
WWT <- crossprod(oonmf_model$w)
```

<img src="{{< blogdown/postref >}}index_files/figure-html/unnamed-chunk-10-1.png" width="288" style="display: block; margin: auto auto auto 0;" />

It both looks pretty close to orthogonal, and it numerically close:

<img src="{{< blogdown/postref >}}index_files/figure-html/unnamed-chunk-11-1.png" width="288" style="display: block; margin: auto auto auto 0;" />

If this were a true identity matrix, we would expect all 1's on the diagonal and all 0's on the off-diagonal. This is exactly the case for rank-truncated SVD (within machine precision tolerances), but not for ooNMF.

ooNMF orthogonality is thus approximate. The same holds true for previously published orthogonal NMF. Note that a truly orthogonal NMF is not practical as a meaningful lower-dimensional representation of data.

## Extending the ooNMF Architecture

We may consider a number of alternative architectures to ooNMF proposed here.

### Untied weights

First, consider a non-tied weights autoencoder:

`\(min_{\{W \geq 0\}} \lVert A - AW_1W_2 \rVert_2^2\)`

This architecture can achieve better reconstruction accuracy, but loses interpretability since `\(W_1\)` can only be understood in terms of `\(W_2\)`, and vice versa. Furthermore, it allows for non-linear representations to creep into `\(H\)`, confounding linear assumptions in standard exploratory data analysis.

### Deep extensions

Consider deep ooNMF:

`\(min_{\{W_1, W_2\} \geq 0} \lVert A - AW_1W_2W_2^TW_1^T\rVert_2^2\)`

And similar deeper extensions. Note that ReLu can be considered to be applied, but it is never necessary in practice with `\(A \geq 0\)`. These deeper extensions introduce non-linearity into the middle latent space, and it is no longer interpretable. While this deep extension may be interesting as a self-regularizing semi-orthonormal method for disentangling parts of complex feature sets, it is unlikely to serve for any interpretable dimension reduction.

For the purposes of interpretability, we therefore restrict algorithms to a single hidden layer with constrained and tied-weights. 

### Variational inference

NMF publications have at times flirted with the idea of an angular penalty, to disentangle or force apart NMF factors. This penalty works poorly in practice, but variational autoencoders have shown significant success at disentangling latent spaces with self-regularization through a KL-divergence loss in the encoder.

It is possible to construct a tied-weights variational autoencoder, where the weights matrix will now be subject to updates that are a sum of KL loss + reconstruction error (from the encoder) and only reconstruction error (from the decoder). A mixing parameter (similar to \beta-VAE) can control the contribution of KL vs. MSE loss to the encoder, for instance, through gradual annealing during fitting.

KL loss would increase the sparsity of the hidden layer due to neuron disentanglement, and also generate W matrices that are more colinear and less of a sparse clustering. For most applications, this is a huge need. 

Framing NMF as an autoencoder allows us to adopt many exciting ideas from the deep learning space for simple dimension reduction.

## Future directions

### Outlook

ooNMF promises to produce interpretable low-dimensional latent spaces that are highly reminiscent of NMF but with much better scaling and easier cross-validation. We look forward to testing ooNMF on very large real-world data and comparing to NMF models that were trained at the limit of available compute resources.

### Development Priorities
* Methods for L1 regularization of this neural architecture should be explored to help boost sparsity of both "W" and "H"
* Minibatch updates and Adam optimization for faster convergence
* Special properties of the tied-weights enable more efficient algorithms for updating weights
* High-performance implementation in C++ will demonstrate scalability of the algorithm
* Systematic benchmarking of this method vs. NMF for current applications to derive insights or for exploratory data analysis
