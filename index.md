---
layout: none
title: Accelerating CTC Beam Search Decoding on GPUs using CUDA
---

<link
  rel="stylesheet"
  href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css"
  integrity="sha384-n8MVd4RsNIU0KOVEMmg9rtabNEJFvmbFe7aiCKzADTLpiOKgDCqUlkggVCiduneo"
  crossorigin="anonymous"
/>
<script
  src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"
  integrity="sha384-XjKyOOlGwcjNTAIQHIpgOno0Hl1YQqzUOEleOLALmuqehneUG+vnGctmUb0ZY0l8"
  crossorigin="anonymous"
></script>
<script
  src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"
  integrity="sha384-+VBxd3r6XgURycqtZ117nYw44OOcIax56Z4dCRWbxyPt0Koah1uHoK0o4+/RRE05"
  crossorigin="anonymous"
></script>
<script>
  document.addEventListener("DOMContentLoaded", function () {
    renderMathInElement(document.body, {
      delimiters: [
        { left: "\\[", right: "\\]", display: true },
        { left: "\\(", right: "\\)", display: false }
      ]
    });
  });
</script>

<style>
  :root {
    --bg-color: #ffffff;
    --text-color: #333333;
    --heading-color: #1a1a1a;
    --link-color: #007bff;
    --link-hover-color: #0056b3;
    --border-color: #e0e0e0;
    --table-header-bg: #f8f9fa;
    --code-bg: #f1f1f1;
  }

  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
      "Helvetica Neue", Arial, sans-serif;
    line-height: 1.7;
    color: var(--text-color);
    background-color: var(--bg-color);
    max-width: 900px;
    margin: 0 auto;
    padding: 2rem;
  }

  .header-main {
    text-align: center;
    margin-bottom: 4rem;
    border-bottom: 1px solid var(--border-color);
    padding-bottom: 2rem;
  }

  h1 {
    font-size: 2.5rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
    color: var(--heading-color);
  }

  .subtitle {
    font-size: 1.5rem;
    font-weight: 300;
    color: var(--text-color);
    margin-bottom: 1rem;
  }

  .authors,
  .date {
    font-size: 1.1rem;
    color: var(--text-color);
  }

  .project-link {
    margin-top: 2rem;
  }

  .project-link a {
    font-size: 1.1rem;
    display: inline-block;
    padding: 0.6rem 1.2rem;
    border: 1px solid var(--border-color);
    border-radius: 6px;
    background-color: var(--table-header-bg);
    transition: background-color 0.2s ease;
  }

  .project-link a:hover {
    background-color: #e9ecef;
    text-decoration: none;
  }

  h2 {
    font-size: 2rem;
    font-weight: 600;
    color: var(--heading-color);
    border-bottom: 1px solid var(--border-color);
    padding-bottom: 0.5rem;
    margin-top: 3rem;
    margin-bottom: 1.5rem;
  }

  a {
    color: var(--link-color);
    text-decoration: none;
  }

  a:hover {
    color: var(--link-hover-color);
    text-decoration: underline;
  }

  table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 1.5rem;
    margin-bottom: 1.5rem;
  }

  th,
  td {
    border: 1px solid var(--border-color);
    padding: 12px;
    text-align: left;
  }

  th {
    background-color: var(--table-header-bg);
    font-weight: 600;
  }

  code {
    background-color: var(--code-bg);
    padding: 0.2em 0.4em;
    border-radius: 3px;
    font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo,
      monospace;
  }

  .math-block {
    margin: 1.5rem 0;
  }
</style>

<div class="header-main">
  <h1>Accelerating CTC Beam Search Decoding on GPUs using CUDA</h1>
  <p class="subtitle">15-418 Project Proposal</p>
  <p class="authors">Julius Arolovitch, Ben Kleyner, Maxim Yagnyatinskiy</p>
  <p class="date">Fall 2025</p>
</div>

## Summary

We will implement a high-performance batched CTC Beam Search decoder in CUDA to accelerate autoregressive sequence decoding tasks on NVIDIA GPUs. The project will focus on optimizing the iterative decoding loop by developing custom device-side kernels for massively parallel hypothesis expansion and an efficient parallel top-k selection algorithm for beam pruning.

## Background

Connectionist Temporal Classification (CTC) is an output layer and loss function used in training recurrent neural networks (RNNs) for sequence-to-sequence tasks, such as speech recognition or handwriting recognition. Its key advantage is that it does not require a frame-by-frame alignment between the input sequence and the output sequence. A network trained with CTC outputs a probability distribution for each time step over the set of possible output labels, plus a special blank token. The decoding task is to find the most probable output sequence from this \\(T \\times N\\) matrix of probabilities, whereby \\(T\\) is the number of time steps and \\(N\\) is the number of labels.

The simplest method is greedy decoding, which takes the most probable label at each time step and then collapses repeated labels and removes blanks. This is computationally trivial but often suboptimal.

CTC Beam Search Decoding is a more effective algorithm that provides more accurate transcriptions. Instead of just tracking the single best path, it maintains a "beam" of the \\(k\\) most probable candidate sequences at each time step. At each step \\(t\\), it explores extending these \\(k\\) hypotheses with all possible labels, calculates their new probabilities, and merges paths that result in the same output sequence. The beam is then pruned back to the \\(k\\) most likely hypotheses to carry forward to step \\(t+1\\).

The aspects of this problem that benefit from parallelism are two-fold. First, as our summary suggests, the entire decoding process for one sequence can be run in parallel with other sequences in a batch. This provides a high level of coarse-grained data parallelism. Second, within a single sequence's decoding at each time step \\(t\\), we can explore all \\(k \\times N\\) possible extensions in parallel. The challenge, as detailed below, lies in the synchronization, merging, and pruning steps that follow.

## Challenge

The core challenge is mapping this dynamic, irregular, and memory-bound algorithm onto the rigid, data-parallel SIMT architecture of a GPU. The workload has several particular characteristics that make it difficult to parallelize.

First, the probability of a new hypothesis at time \\(t\\) is not independent. It depends on the probabilities of its prefix at time \\(t-1\\), as well as the CTC merging logic. This requires synchronization and careful management of state.

Additionally, the "fan-out" step of extending \\(k\\) hypotheses to \\(k \\times N\\) new ones has poor spatial locality; threads in a warp may access scattered locations in the \\(T \\times N\\) probability matrix and in the beam data structures. The "merge" step requires a many-to-one reduction, which is also memory-intensive and requires synchronization. This leads to a low computation-to-communication ratio; the kernel will be dominated by memory operations and sorting, not arithmetic.

Finally, the number of active, non-pruned hypotheses can vary significantly at each time step. Furthermore, the merging logic is data-dependent. This leads to significant thread divergence within a warp, as some threads may be performing complex merge operations while others are idle. This load imbalance is a poor fit for the SIMT model.

In short, the GPU execution model thrives on uniform, arithmetic-heavy tasks with high data locality; our problem is opposite in almost every respect.

## Resources

We will be developing our code primarily on the GHC cluster machines which are equipped with CUDA capable GPUs. The results on these machines will be compared with experimentation on the PSC machines which have considerably more powerful GPUs.

Our initial implementation of the beam search algorithm will be based on the pseudocode in [this paper](https://arxiv.org/abs/2204.02929).

## Goals and Deliverables

| Completion | Goal Description |
| :--- | :--- |
| **MVP** | Implement a correct, batched CTC beam search decoder on GPU that outperforms a correct CPU implementation. This includes basic kernels for hypothesis expansion and sequential top-k pruning within each thread block. |
| **MLP** | Optimize the batched decoder to achieve significant speedup over a CPU baseline, focusing on efficient parallel top-k selection and improved memory access patterns. Implement shared memory optimizations and basic warp-level intrinsics. |
| **Stretch** | Develop advanced load-balancing techniques for handling thread divergence due to data-dependent merging, explore dynamic parallelism or kernel fusion, and integrate with a simple deep learning framework (e.g., PyTorch, TensorFlow) for real-world inference. |

## Platform

The deployment of beam search decoders has shifted from CPU-centric logic to hybrid CPU-GPU systems, driven by the developments in GPU-native neural network approaches. This hybrid model, however, is fundamentally limited by the data transfer between the CPU and GPU at every decoding step. This bottleneck becomes more relevant due to the asymmetric scaling of on-device GPU performance versus off-device I/O speed. Driven by specialized hardware like NVIDIA's Tensor Cores and massive increases in memory bandwidth, the effective computational speed of GPUs has increased much faster than the PCIe bus which has seen only modest, linear improvements. This creates a classic scenario where overall system speedup is constrained by its slowest component, as stipulated by Amdahl's Law.

Let \\(T_{\text{step}}\\) be the total time for a single decoding step. Let \\(S\\) represent the effective speedup of the on-device (GPU) parallelizable portion of the task, which accounts for all GPU architectural improvements (cores, specialized hardware, and memory bandwidth). The transition from an early GPU generation (\\(G_1\\)) to a modern one (\\(G_2\\)) can be modeled as follows:

<div class="math-block">
\[
\begin{align*}
    T_{\text{step}} &= T_{\text{compute}} + T_{\text{transfer}} \\\\
    \text{Given} \quad T_{\text{compute}}(G_2) &= \frac{T_{\text{compute}}(G_1)}{S} \quad \text{where } S \gg 1 \\\\
    \text{and} \quad T_{\text{transfer}}(G_2) &\approx T_{\text{transfer}}(G_1)
\end{align*}
\]
</div>

The fraction of time spent on data transfer, \\(f\\), consequently shifts from \\(f_1\\) to \\(f_2\\):

<div class="math-block">
\[
f_1 = \frac{T_{\text{transfer}}(G_1)}{T_{\text{compute}}(G_1) + T_{\text{transfer}}(G_1)}
\quad \longrightarrow \quad
f_2 \approx \frac{T_{\text{transfer}}(G_1)}{\frac{T_{\text{compute}}(G_1)}{S} + T_{\text{transfer}}(G_1)}
\]
</div>

As the effective speedup \\(S\\) grows, the effect of \\(\frac{T_{\text{compute}}(G_1)}{S}\\) is diminished and the transfer begins to dominate. Consequently, in this project we will attempt to maximize the fraction of computation that can be performed efficiently on-device to attempt to minimize the effect of the transfer term on overall performance.

## Schedule

| Expected Completion Date | Goal | Assignee(s) |
| :--- | :--- | :--- |
| Monday, November 24 | MVP Complete | TBD |
| Monday, December 1 | MLP Complete | TBD |
| Sunday, December 7 | Stretch, Presentation, & Report | TBD |

Exact work distribution is yet to be determined and will be filled in by the mid-project check in.

