---
layout: post
title:  "Releasing Tri-RMSNorm: Yielding Speed-up for Training Stabilization + [Let’s Review] Root Mean Square Layer Normalization"
date:   2024-03-14 00:00:00 +0000
categories: rms-normalization, layernorm
usemathjax: true
---

I'm thrilled to release the Tri-RMSNorm in the second entry of my review series, featuring a new kernel. At below post, I'm introducing an accelerated Tri-RMSNorm v0.1.0 compute kernel packaged for easy integration, fully compatible with PyTorch tensors. Then, in further sections in my post, going into the details of LayerNorm and Root Mean Square normalization processes for large DNNs. In the very early benchmarks, Tri-RMSNorm kernel shows speed-up in the RMSNorm process, compared to vanilla PyTorch RMSNorm, PyTorch LayerNorm and Triton LayerNorm, indicating that it scales with larger values of N. When compared to the standalone PyTorch RMSNorm implementation, and the Tri-RMSNorm kernel, considering both the forward and backward passes computations, yields a mean speedup of approximately 10.18%.

[Release & Let's Review Post](https://medium.com/@simudt/releasing-tri-rmsnorm-yielding-speed-up-for-training-stabilization-lets-review-root-mean-8e699eab947c)

[GitHub Repository](https://github.com/simudt/Tri-RMSNorm)
