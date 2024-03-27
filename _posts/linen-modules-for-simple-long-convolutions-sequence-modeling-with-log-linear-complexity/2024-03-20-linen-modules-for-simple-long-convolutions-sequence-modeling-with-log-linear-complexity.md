---
layout: post
title:  "Linen Modules for Simple Long Convolutions Sequence Modeling with Log-Linear Time Complexity"
date:   2024-03-20 00:00:00 +0000
categories: sequence-modelling
usemathjax: true
---

Within this blog post, I’m releasing an simple compute API package called LongConv-Jax powered by Jax, Flax, and NumPy, which basically provides robust Flax Linen modules for the paper “Simple Long Convolutions for Sequence Modeling” from HazyResearch, which is also reviewed below in the rest of the this post. LongConv-Jax basically benefits, FFT convolution to compute a long convolution in O(N log N) time (instead of O(N^²)), and applies a simple regularization through a Squash operator to the kernel weights. This implementation is particularly effective for processing long convolution sequences as demonstated by the authors of the Safari.

The original implementation of the paper was written in PyTorch for compatibility with Safari. LongConv-Jax, implemented in the Jax library, utilizes JIT high-performance forward computation, scales beyond the standalone modules with PyTorch. Includes required modules for individual LongConv layers as well as a model that incorporates multiple LongConv layers for sequence processing tasks.

[Medium Blog Post](https://medium.com/@simudt/linen-modules-for-simple-long-convolutions-sequence-modeling-with-log-linear-time-complexity-0f7fb1c2b308)

[GitHub Repository](https://github.com/simudt/LongConv-Jax)