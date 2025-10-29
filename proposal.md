![color%20logo_u](media/image1.jpeg)

**MACAU UNIVERSITY OF SCIENCE AND TECHNOLOGY**

**School of Computer Science and Engineering**

**Faculty of Innovation Engineering**

**Final Year Project Proposal**

Title: Global-Context–Driven Instance-Level Mixture-of-Experts for 4K Multi-Degradation Image Restoration (Rain, Haze, Snow)

Student Name : Liu Benhuang

Student No. : 1220004875

Supervisor : Li Nannan

November, 2025

# Abstract

We address 4K image restoration under adverse weather—rain, haze, and snow—where dense all-in-one models suffer from task interference and token-level sparse MoE incurs prohibitive communication at high resolutions. We propose an **instance-level Mixture-of-Experts (MoE-I)** that selects a **single expert per image** at near single-path cost. A shared backbone produces a **global bottleneck vector** that feeds a tiny router; **experts** are implemented as **parameter-efficient adapters** specializing to each degradation. To make routing reliable rather than assumptive, we train the router with **utility distillation** from per-expert restoration errors and apply **post-hoc calibration** with an **abstention** fallback to a dense unified head. We establish a **compute-normalized protocol** FLOPs/latency/memory at 4K with tiled inference, report PSNR/SSIM/LPIPS together with **ms/MP**, and analyze **coverage–accuracy**, calibration (ECE), and an **oracle upper bound** to attribute gains to conditional computation rather than parameter growth. Extensive ablations (router utilities, expert capacity, anti-collapse regularization) and per-image logs ensure reproducibility and deployment relevance.

**Table of Contents**

[Abstract [1](#abstract)](#abstract)

[1. Introduction [3](#introduction)](#introduction)

[2. Objectives [4](#objectives)](#objectives)

[3. Related work [5](#related-work)](#related-work)

[4. Methodology [6](#methodology)](#methodology)

[5. Required hardware and software [7](#required-hardware-and-software)](#required-hardware-and-software)

[6. Project planning [8](#project-planning)](#project-planning)

[References [9](#references)](#references)

# 1. Introduction

Restoring ultra–high-resolution images (3840×2160) captured under rain, haze, and snow is central to autonomous driving, surveillance, and aerial imagery. Single-degradation systems remain strong in-domain, and modern backbones substantially improve the capacity–efficiency trade-off , Restormer, NAFNet, and global-bottleneck designs suitable for extreme resolution \[8,9,10\]. However, such pipelines do not transfer across weather types and often rely on tiling or expensive global context to meet latency/memory budgets.

All-in-one image restoration (AiOIR) reduces deployment friction by sharing parameters across degradations, with recent surveys and methods covering prompt-conditioning, degradation-aware modeling, and regularization/efficiency strategies \[21,23,22\]. Yet shared-encoder AiOIR often sacrifices peak per-task quality and is vulnerable to catastrophic forgetting arising from gradient conflicts between heterogeneous degradations \[20\]. Sparse Mixture-of-Experts (MoE) promises conditional capacity, but token/patch-level routing becomes communication-heavy at UHD scales: a 3840×2160 frame with 16×16 patches already yields 32,400 tokens, so all-to-all dispatch and multi-expert activation undermine single-image latency even when FLOPs appear comparable \[1,2,3\].

To address these issues, we propose an instance-level Mixture-of-Experts (MoE-I) that makes one routing decision per image at near single-path cost. Concretely, a shared encoder–global bottleneck–decoder backbone produces a compact global scene descriptor ggg via cross-scale aggregation (a few pooling/MLP layers); a tiny router consumes ggg and selects one parameter-efficient expert (adapter-based, ≤1–3% params) specialized for rain/haze/snow. To ensure routing is auditable rather than black-box, we train with utility distillation from per-expert restoration errors (soft targets) and apply post-hoc calibration (temperature scaling) with abstention: if p \< τ, we fallback to a dense unified head. We evaluate under a compute-normalized protocol—equal FLOPs, equal latency, and equal max memory—using tiled UHD inference with Hann blending, and we report PSNR/SSIM/LPIPS together with ms/MP, coverage–accuracy, calibration (ECE), and an oracle upper bound to attribute gains to conditional computation instead of parameter growth.

# 2. Objectives

**O1. 4K benchmarking & compute normalization**

- **What:** Establish a unified UHD protocol (tile=512, overlap=64, Hann) and cost logging (Latency@4K, MaxMem, Throughput, ms/MP).

- **How:** Implement bench.py (≥30 warmups, ≥100 timed runs, median), cost.py (FLOPs/params), fixed I/O and AMP policy.

- **Measure:** Dense vs. ours under **equal FLOPs / equal latency / equal max memory** settings.

- **Pass–Fail:** Re-run stability on the same machine: all four cost metrics drift \<5%; export per-image JSON + aggregate CSV.

**O2. Dense baselines (single-task & all-in-one)**

- **What:** Train a Dense-AiO (shared backbone + generic head) and three Dense-Single baselines (dehaze/derain/desnow).

- **How:** Reuse the same backbone; task-specific losses; brief top-layer unfreeze.

- **Measure:** PSNR/SSIM/LPIPS and ms/MP on DehazeXL, UHDRain, UHD-Snow.

- **Pass–Fail:** Dense-AiO within −0.2 dB of published medians; each Dense-Single ≥ Dense-AiO on its task.

**O3. Lightweight experts (three specialists)**

- **What:** Adapter experts (1×1 → DW-3×3 → 1×1) with reduction$`\ r \in \{ 1/32,1/16\}`$; ≤1–3% extra params per expert.

- **How:** Freeze lower 1/2–2/3 of the backbone; fine-tune tops with small LR. Losses:

  - Dehaze:$`\ \mathcal{l}_{1} + 0.05(1 - SSIM)`$

  - Derain: $`\mathcal{l}_{1} + 0.05 \parallel \nabla\widehat{y} - \nabla y \parallel 1 + 0.02\, L_{dir}`$

  - Desnow: $`\mathcal{l}_{1} + 0.05\, LPIPS + \lambda_{m}\,\mathcal{l}_{1}( \cdot \mid mask)`$ (LoG/DoG mask, train-time only)

- **Measure:** Expert ≥ Dense-AiO on its own task.

- **Pass–Fail:** If not, increase rrr or upsample severe cases until ≥0 dB gap.

**O4. Instance-level router (Top-1) with utility distillation**

- **What:** MLP router (1024→512→K) on global vector ggg; soft targets $`{q}_{k} \propto exp( - \beta\epsilon_{k})`$.

- **How:** Offline build utility cache; train with $`KL(q \parallel p) + \alpha\, KL(pˉ \parallel u)`$,$`\beta \in \{ 20,50,80\}`$, $`\alpha \in \lbrack 0,0.1\rbrack`$.

- **Measure:** Top-1 agreement with oracle choice; per-expert utilization histogram.

- **Pass–Fail:** Agreement ≥85%; no collapse (max expert share \<70%).

**O5. Calibration & abstention (reliability)**

- **What:** Temperature scaling + thresholding; low-confidence fallback to Dense-Unified.

- **How:** Calibrate TTT on a disjoint split (min NLL); grid search τ to cap abstention ≤10%.

- **Measure:** ECE (15 bins), coverage–accuracy curves.

- **Pass–Fail:** ECE ≤0.03; abstention ≤10%; Macro metrics non-decreasing after fallback.

**O6. Quality gains at equal budgets (primary endpoint)**

- **What:** Compare MoE-I (Actual) vs. Dense-AiO under **equal latency** and **equal max memory**.

- **How:** Report Macro-PSNR/SSIM/LPIPS + H-Mean; 95% CIs (paired bootstrap 10k) and Cohen’s ddd.

- **Measure:** Macro-PSNR **+0.2–0.5 dB**, Macro-LPIPS **−0.01–0.02**.

- **Pass–Fail:** At least one budget setting achieves CI not containing 0 and $`p < 0.05`$, $`d \geq 0.2d`$.

**O7. Cost control (deployability)**

- **What:** Maintain single-path runtime close to Dense-AiO.

- **How:** Log Latency@UHD, ms/MP, MaxMem under identical tile/I-O/AMP.

- **Measure:** Relative overheads vs. Dense-AiO.

- **Pass–Fail:** Latency and ms/MP overhead **≤5%**; MaxMem **≤+5%**.

**O8. Oracle attribution & routing quality**

- **What:** Report **Oracle** (per-image best expert), **Actual** (MoE-I), and **Dense**.

- **How:** Include **Oracle-Gap** (PSNR), routing agreement, and abstention breakdown.

- **Measure:** Actual–Oracle ≤ **0.3 dB**; agreement ≥ **85%**.

- **Pass–Fail:** Satisfy at least one; otherwise revisit O3/O4 ($`r,\beta,\alpha`$) and sampling.

**O9. Ablation suite (evidence, not anecdotes)**

- **What:** w/o distillation; w/o calibration; w/o abstention; $`\alpha\  = \ 0`$; $`r`$ grid; diagnostic Top-2; token-MoE proxy (downsampled); MoCE-IR-style complexity experts.

- **How:** Keep equal-budget settings fixed; toggle one factor at a time.

- **Measure:** Impact on Macro metrics, ECE, Oracle-Gap, costs.

- **Pass–Fail:** Main conclusions (O6–O8) remain under small hyper-changes; if not, document scope limits.

**O10. Mixed degradations & out-of-distribution**

- **What:** 200 mixed-weather images (UAV/drive/ CCTV) for generalization and stability checks.

- **How:** Report coverage–accuracy, abstention, Dense fallback macro metrics.

- **Measure:** Rank consistency vs. in-distribution (Kendall’s $`\tau`$).

- **Pass–Fail:**$`\tau\  \geq \ 0.6`$; if lower, mark as shift-sensitive and add recalibration experiment.

**O11. Reproducibility package**

- **What:** Release artifact.zip with code, configs, weights, env snapshot, per-image logs, seeds/hashes.

- **How:** scripts/reproduce\_\*.sh reproduces main table/figures.

- **Measure:** Third-party reruns differ by \<±0.05 dB (median).

- **Pass–Fail:** If not met, lock Docker/conda env and publish fixed seeds/weights.

# 3. Related work

**Restoration under adverse weather.**  
Single-degradation models remain strong within domain: AOD-Net unifies atmospheric light and transmission for end-to-end dehazing \[12\]; RESCAN uses recurrent squeeze-and-excitation for progressive deraining \[13\]; DesnowNet exploits multi-scale context for structured snow \[14\]. Modern backbones improve the capacity–efficiency trade-off: Restormer with efficient attention \[8\], NAFNet with activation-free blocks \[9\], and DehazeXL with a global bottleneck for extreme-resolution inference \[10\]. Yet such pipelines do not transfer across weather types and, at 4K, either rely on tiling (risking seams) or pay high global-context costs under tight latency/memory budgets. In parallel, VLM/agent orchestration (e.g., JarvisIR \[11\]) can dispatch experts by high-level cues but adds language-model latency and complicates compute-normalized comparisons, motivating lighter image-level routing.

**All-in-One image restoration (AiOIR).**  
Unified models lower deployment friction via shared parameters, with steady progress documented by recent surveys/benchmarks \[15,16,17,19,21\]. Representative directions—prompt-conditioning for flexibility (PromptIR) \[23\], degradation-aware adaptation via state-space/frequency cues (DPMambaIR, AdaIR) \[22,25\], and regularization/efficiency through task-aware penalties and progressive designs (TUR, PerNet) \[26,27\]—mitigate some conflicts. However, shared-encoder AiOIR often trades peak per-task quality for convenience, remains sensitive to distribution shift (e.g., rain density/structure), and is vulnerable to catastrophic forgetting due to gradient conflicts in shared encoders \[20\]. These limits motivate conditional specialization instead of further enlarging a single dense model.

**Mixture-of-Experts for vision and routing reliability.**  
Sparsely-gated MoE scales capacity with sublinear compute by activating a subset of experts \[1\]; Switch adopts Top-1 routing for stability/throughput \[2\]; V-MoE brings sparse gating to vision with budget–accuracy control \[3\], while GShard provides the distributed substrate \[4\]. Vision studies quantify expert-count/performance trade-offs and parameter-efficient expertization for recognition/restoration \[28,29,31,30,33\]. Crucially, token/patch-level MoE is communication-heavy at 4K: a 3840×2160 frame with 16×16 patches already yields 32,400 tokens, so all-to-all dispatch and multi-expert activation degrade latency even when FLOPs look similar. We instead use instance-level (image-level) Top-1 routing: a tiny router selects one expert per image, preserving single-path inference and enabling compute-normalized comparisons (equal FLOPs/latency; unit ms/MP). Unlike token-level V-MoE \[3\] or complexity-expert MoCE-IR \[30\], our design minimizes 4K costs while ensuring calibration: temperature scaling + selective prediction serve as utilities (not contributions) so low-confidence cases abstain to a dense fallback \[5,6\]. We further address expert collapse/load imbalance by utility distillation (soft targets from per-expert errors) and monitoring per-expert utilization, which promotes stable specialization without heavy load-balancing machinery. Overall, we decouple specialization and selection (light degradation-specific experts + utility-distilled instance-level router), report results under equal budgets, and quantify an oracle gap to attribute gains to conditional computation rather than parameter growth on 4K rain/haze/snow.

# 4. Methodology

**3.1 Overview**

We address ultra–high-resolution restoration of rain, haze, and snow with a single-path, instance-level Mixture-of-Experts (MoE-I). A shared encoder–global-bottleneck–decoder backbone produces a compact global scene descriptor and a reconstruction stream; a tiny router reads the descriptor and activates one parameter-efficient expert (Top-1). If the router’s confidence is low, the system abstains and falls back to a dense unified head. By making one decision per image, MoE-I avoids token/patch all-to-all that is prohibitive at UHD (3840×2160 with 16×16 patches yields 32,400 tokens), while preserving the specialization benefits of sparse experts seen in MoE literature \[1,2,3\]. The global bottleneck follows DehazeXL-style designs that encode long-range context without quadratic token interactions, which is crucial at 4K \[10\].

**3.2 Architecture: Backbone and Experts**

Given a degraded input $`x`$ with ground truth $`y`$, the backbone $`{F}_{\theta}`$ outputs a preliminary reconstruction and a global vector

``` math
(\widetilde{y},\, g) = F_{\theta}(x;\, return\_ global = 1),g \in \mathbb{R}^{d}\ (d = 1024)
```

Multi-scale encoder features are cross-scale pooled and passed through a small MLP (GELU, LayerNorm) to form ggg; a very light auxiliary head on the bottleneck weight $`10^{- 3}`$ stabilizes its information content. We keep the encoder/decoder depths and widths identical to the public DehazeXL configuration to ensure comparability; exact values are listed in our config file.  
Each expert augments selected backbone blocks with a residual adapter

1×1 reduce → depthwise 3×3 → 1×1 expand;

reduction r∈{1/32,1/16} ), adding ≤1–3% parameters per expert.

Task losses follow standard practice and are used **both f**or expert training and later for routing supervision:

dehaze :$`\mathcal{l}_{1}`$+0.05(1−SSIM);

derain :$`\mathcal{l}_{1} + 0.05 \parallel \nabla y\hat{} - \nabla y \parallel 1 + 0.02\, L_{dir}`$ ​;

desnow :$`\mathcal{l}_{1}`$+$`0.05\, LPIPS + \lambda_{m}\,\mathcal{l}_{1}( \cdot \mid mask)`$

with LoG/DoG snow masks at train time only. Experts are pretrained per task, freezing the lower half to two-thirds of the backbone and briefly unfreezing the top blocks for a small-LR fine-tune.

# 5. Required hardware and software

Hardware: 2x NVIDIA A100 80GB, NVMe \>= 2TB; CPU \>= 16C/32T; RAM \>= 128GB.

Software: Ubuntu 22.04; CUDA 12.x; PyTorch 2.x; timm; lpips; fvcore/ptflops; Python 3.10; W&B/TensorBoard; WebDataset/LMDB; Git.

Dataset: DehazeXL(8K), UHDRain, UHDSnow

# 6. Project planning

| Week | Tasks | Milestones/Deliverables |
|----|----|----|
| W1 | Scene-level splits; pHash de-dup; alignment; meta.jsonl; severity; evaluation scaffold | Per-image & summary JSON on val |
| W2 | Adapter plug-in; Dense-Unified training & v0; start experts (dehaze/derain/desnow) | Each expert \>= dense on its task |
| W3 | Small-LR unfreeze; unified inference; log ms/MP | Severity-stratified curves |
| W4 | Soft labels (beta grid); router training (KL); temperature scaling | Router ECE \< 0.05 |
| W5 | Integrated inference with tau (abstention \<=10%); Actual/Oracle/Dense v1 | Oracle gap \<= 0.3 dB; latency overhead \<= 5% |
| W6 | A/B/E ablations; compute-normalized Pareto | ECE \<= 0.03; Pareto done |
| W7 | External bridging (50-200 samples); cost re-measure (x3 median) | Rank consistency & stable costs |
| W8 | Figures/tables; reproducibility pack; thesis draft | artifact.zip + draft |
| W9-11 | Buffer: fill gaps; extended ablations/visualizations; polishing | Final version |

# References

\[1\] Shazeer et al., "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer," 2017.

\[2\] Fedus et al., "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity," 2021.

\[3\] Riquelme et al., "Vision Mixture of Experts (V-MoE)," 2021.

\[4\] Lepikhin et al., "GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding," 2020.

\[5\] Guo et al., "On Calibration of Modern Neural Networks," 2017.

\[6\] Geifman & El-Yaniv, "SelectiveNet: A Deep Neural Network with an Integrated Reject Option," 2019.

\[7\] Yang et al., "CondConv: Conditionally Parameterized Convolutions for Efficient Inference," 2019.

\[8\] Zamir et al., "Restormer," 2022.

\[9\] Chen et al., "NAFNet," 2022.

\[10\] "DehazeXL: Extreme-resolution dehazing with global bottleneck modeling," arXiv preprint.

\[11\] "JarvisIR: VLM-driven multi-expert restoration for adverse weather," CVPR.

\[12\] Li et al., "AOD-Net: All-in-One Dehazing Network," ICCV 2017.

\[13\] Yang et al., "RESCAN: Recurrent Squeeze-and-Excitation Context Aggregation Net for Single Image Rain Removal," ECCV 2018.

\[14\] Liu et al., "DesnowNet: Context-Aware Deep Network for Snow Removal," TIP 2018.

\[15\] Jiang et al., "Data-Driven single image deraining: A Comprehensive review and New Perspectives," Pattern Recognition, 2023.

\[16\] Chen et al., "Towards Unified Deep Image Deraining: A Survey and A New Benchmark," arXiv:2310.03535, 2023.

\[17\] Chen et al., "Towards Unified Deep Image Deraining: A Survey and A New Benchmark," arXiv update, 2025.

\[18\] Wang et al., "Real‐World Image Deraining Using Model‐Free Unsupervised Learning," Computational Intelligence and Neuroscience, 2024.

\[19\] Su et al., "A Survey of Single Image Rain Removal Based on Deep Learning," ACM Computing Surveys, 2023.

\[20\] Sun et al., "Re-examine all-in-one image restoration: A catastrophic forgetting view," Information Processing & Management, 2025.

\[21\] Li et al., "A Survey on All-in-One Image Restoration: Taxonomy, Evaluation and Future Trends," arXiv:2410.15067, 2024.

\[22\] Zhou et al., "DPMambaIR: All-in-One Image Restoration via Degradation-Aware State Space Model," arXiv:2504.17732, 2025.

\[23\] Yu et al., "PromptIR: Prompting for All-in-One Image Restoration," NeurIPS, 2023.

\[24\] Jiang et al., "Degradation-Aware Feature Perturbation for All-in-One Image Restoration," CVPR, 2025.

\[25\] Li et al., "AdaIR: Adaptive All-in-One Image Restoration via Frequency Mining and Modulation," ICLR, 2024.

\[26\] Wang et al., "Debiased All-in-one Image Restoration with Task-aware Regularization," AAAI, 2025.

\[27\] Liu et al., "PerNet: Progressive and Efficient All-in-One Image-Restoration Network," Electronics, 2024.

\[28\] Grootendorst, "Mixture of Experts for Image Classification: What's the Sweet Spot?," arXiv:2411.18322, 2024.

\[29\] Bouaouni, "Mixture of experts (MoE): A big data perspective," Information Fusion, 2025.

\[30\] Zhang et al., "Complexity Experts are Task-Discriminative Learners for All-in-One Image Restoration," CVPR, 2025.

\[31\] Grootendorst, "A Visual Guide to Mixture of Experts (MoE)," Newsletter, 2024.

\[32\] Smith et al., "A mixture of experts (MoE) model to improve AI-based computational pathology prediction performance under variable levels of image blur," ResearchGate, 2025.

\[33\] Kim et al., "Parameter Efficient Adaptation for Image Restoration with Mixture-of-Experts," NeurIPS, 2024.
