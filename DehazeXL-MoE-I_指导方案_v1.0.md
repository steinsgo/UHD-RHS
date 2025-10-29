
# DehazeXL-MoE-I 指导方案（v1.0）
> 基于 DehazeXL 骨干的三专家（去雾/去雨/去雪）+ **样本级 MoE-I 路由**方案。目标是在 **4K 分辨率**、**同等推理预算**下，取得优于多任务稠密基线（Dense‑Unified）的系统表现，并给出可复现、可审计的证据链。

---

## 0. 总体目标与成功判据
**目标**：在不破坏 DehazeXL 核心优势（瓶颈全局信息、4K 友好、异步小批次）的前提下，实现三专家 + MoE‑I 路由。  
**判据（建议门槛）**：
- 质量：**MoE‑I‑Actual ≥ Dense‑Unified** 的 Macro/H‑Mean（PSNR/SSIM/LPIPS 汇总）；
- 上界：**Oracle‑Gap ≤ 0.2–0.3 dB** 或逼近度 ≥ 90–95%；
- 稳定：**ECE ≤ 0.03**，弃权 ≤ 10%，误路由 10% 时 Macro 跌幅 < 0.2 dB；
- 成本：Latency@4K **增幅 ≤ 5%**；单位像素延迟（ms/MP）透明可复核。

---

## 1. 系统结构（不改主干逻辑，仅加可插拔模块）
### 1.1 统一骨干：DehazeXL（Encoder → Global Bottleneck → Decoder）
- **Encoder**：局部块抽特征，支持**异步/分批**降低 4K 显存。
- **Bottleneck**：全局注意 / 高效 Transformer，使**所有 token 互可见**并注入全局色调/空气光/对比度信息。
- **Decoder**：逐级重建，回写至全分辨率。

> 仅新增两个接口：  
> `backbone(x, return_global=True, adapter_task=None) → (y, global_vec)`  
> `register_adapter(task)`: 在每个 Block 输出后挂载 Adapter（或 LoRA/FiLM），以 `task∈{dehaze, derain, desnow}` 选择权重。

### 1.2 三专家的“轻外壳”
**首选 Adapter**：`1×1 → DW3×3 → 1×1` + 通道门 `tanh(gate)`，缩放比 **r≈1/16**（做 r∈{1/32, 1/16} 消融）。  
**备选**：Conv/Attention‑LoRA（rank 4–8）；**最轻**：FiLM（逐通道仿射）。  
> 三专家**共享骨干参数**，仅外壳不同，**参数增量 ≈ 1–3%**，推理 FLOPs 基本不变。

### 1.3 MoE‑I 路由（样本级 Top‑1 + 弃权回退）
- **路由输入**：直接使用**瓶颈导出的全局向量 `global_vec`**（不降采样图像）。
- **路由器**：Tiny‑ViT / ConvNeXt‑T（< 5M）。输出 K=3 专家概率 `p`。
- **效用蒸馏（而非硬分类）**：在验证集离线得到每个样本、每专家的 `MSE_k`/`PSNR_k`，构造软标签  
  `q_k ∝ exp(−β·MSE_k)`（β∈{20,50,80}）；最小化 **KL(q‖p)**。
- **校准与托底**：温度缩放搜索 `T`；若 `max p < τ`（阈值搜索），**弃权→Dense‑Unified**。

---

## 2. 数据协议（JVB4K）与采样
**切分**：按 **scene/sequence** 划分 train/val/test；pHash 去重；ECC/相位相关检查对齐异常。  
**强度分层 `severity∈[0,1]`**：雾（暗通道+对比度）、雨（斜向高频能量+方向一致性）、雪（LoG/DoG 斑点密度与尺度）。  
**统一评测口径**：sRGB；**tile=512，overlap=64，blend=Hann**；主榜禁 TTA（x8 放附表）。  
**采样**：训练 patch 256→384/512 的课程学习；后期提高高‑severity 与高边缘能量样本的采样权重。  
**混合退化小集**：200–500 张（雨+雾 / 雪+雾），**仅做边界分析**。

`meta.jsonl` 关键字段：`id,dataset,scene_id,split,path_raw,path_gt,w,h,degradation,severity,capture_id,hash,tile_cfg,crop_border`。

---

## 3. 训练细则
### 3.1 专家训练（每任务）
- **初始化**：从去雾预训骨干加载；冻结前 1/2～2/3 层，仅训外壳 + 顶层（5–10 万 iter）。
- **微调**：解冻顶部少量层，LR×0.1 再训 2–4 万 iter。  
- **损失**：  
  - Dehaze：Charbonnier/L1 + SSIM(0.05)  
  - Derain：Charbonnier/L1 + Sobel 边缘(0.05) + 方向一致性正则(≈0.02)  
  - Desnow：Charbonnier/L1 + LPIPS(0.05)；对 LoG/DoG “雪斑”区域做掩码加权（可选）  
- **优化**：AdamW(lr=2e‑4, betas=0.9/0.99, wd=1e‑4)；Cosine + Warmup(1.5k)；bf16；GradClip=1.0；EMA=0.999。  
- **批与显存**：每卡 bs=2–4；梯度累积使有效 batch ≥ 32；启用激活检查点与异步/分批。

### 3.2 路由训练与校准
- **软标签生成**：在 `JVB4K/val` 上跑三专家得到 `MSE_k`，构造 `q`；记录 Oracle `k*`。  
- **训练**：AdamW(lr=1e‑4, wd=1e‑4)，30–50 epoch；最小化 KL(q‖p)。  
- **校准**：独立“路由验证子集”上搜索温度 `T` 与阈值 `τ`（控制弃权 ≤ 10%）。  
- **并发（仅消融）**：Top‑2 激活评估路由“尖锐度”；主线保持 Top‑1。

---

## 4. 推理编排（统一出口、可审计）
**流程**：
1. `y, global_vec = backbone(x4k, return_global=True, adapter_task=None)`  
2. `p = softmax(router(global_vec)/T)`  
3. 若 `p.max()<τ` → `y = DenseUnified(x4k, tile=512, overlap=64, blend='hann')`  
   否则选 `argmax p` 专家 → `y = Expert_k(x4k, same tile/overlap)`  
4. 记录：专家ID、`pmax`、弃权标记、Latency/maxMem、ms/MP 与吞吐。

**优化要点**：缓存上次专家权重；路由 8‑bit 量化（可选）；减少不必要的 postproc；I/O 管线 WebDataset/内存映射。

---

## 5. 消融与分析（围绕“全局信息 × 路由质量”；避免下采样套路）
- **A. 全局信息可见性**：*Local‑only 路由*（仅局部统计） vs *Global‑aware 路由*（瓶颈向量）；比 Macro/H‑Mean、Oracle‑Gap、ECE。  
- **B. 校准曲线**：不同 `τ` 的 **Coverage‑Accuracy**（弃权率—性能）曲线；报告 ECE。  
- **C. 专家分歧度**：Adapter r：1/32 → 1/16；观察 MoE‑I 收益随“分工明确”是否上升。  
- **D. 选择策略**：Top‑1 vs Top‑2；若提升微小 → 路由已足够“尖锐”。  
- **E. 代价对齐**：固定 Latency/FLOPs，画 **Pareto 前沿**（性能‑成本）。  
- **F. 强度分层**：雾/雨/雪高强度 bin 的边际收益是否更显著。  
- **G. 滑窗策略**：overlap=32/64 与融合（Hann/均匀）对缝合伪影与稳定性影响。  
- **H. 误路由敏感性**：强制错派 5/10/20%，统计性能跌幅与弃权托底效果。  
- **I. 归因（可选）**：积分梯度/扰动归因生成路由 RAM，与专家误差热图对齐。

---

## 6. 指标与报告产物
**质量**：PSNR/SSIM（RGB，固定裁边） + LPIPS（Alex v0.1）。  
**汇总**：**Macro 平均**与 **H‑Mean（PSNR 与 −LPIPS 的调和）**。  
**统计**：样本级 **bootstrap 10k → 95% CI**；配对 t‑test 与 Cohen’s d。  
**成本**：FLOPs、Latency@4K（预热 ≥ 30，测量 ≥ 100 取中位）、maxMem、Throughput；单位像素延迟（ms/MP）。  
**MoE 专属**：路由准确率、ECE、弃权率、**Oracle‑Gap**。  
**JSON Schema（建议）**：
```json
// per-image
{"id":"UHDRAIN_000123","dataset":"UHDRain",
 "metrics":{"psnr":32.41,"ssim":0.925,"lpips":0.118},
 "cost":{"latency_ms":54.3,"mem_gb":7.8},
 "moe":{"type":"MoE-I","route":"derain","pmax":0.87,"abstained":false}}
// per-run summary
{"model":"MoE-I-Actual","backbone":"DehazeXL-DB","track":"System",
 "settings":{"tile":512,"overlap":64,"blend":"hann","tta":false},
 "macro":{"psnr":31.85,"ssim":0.918,"lpips":0.124,"hmean_psnr_nlpips":0.873},
 "efficiency":{"flops_g":1650.2,"latency_ms":55.1,"mem_gb":8.0,"throughput_fps":18.1},
 "moe":{"router_ece":0.021,"abstain_rate":0.07,"oracle_gap_psnr":0.18}}
```

---

## 7. 风险与对策（怀疑清单）
- **专家差异不足 → MoE‑I 收益被抵消**：提高 r（1/32→1/16）；加入任务特定损失；先看 Oracle 上界再决定是否上 MoE。  
- **路由不稳 / ECE 偏高**：更细 β/T/τ 网格；难例重采样；必要时启用 Top‑2 小并发。  
- **缝合伪影/边界偏色**：overlap=64 + Hann；统一归一化与裁边。  
- **时延超预算**：路由 8‑bit 量化；减小 overlap；AMP + 激活检查点；缓存专家权重。  
- **混合退化样本**：主榜不计入；单独“混合退化小集”展示 MoE‑T 的细粒度优势，**诚实界定边界**。

---

## 8. 代码组织与关键签名
```
repo/
 ├─ backbone/
 │   ├─ dehaze_xl.py            # 骨干：return_global, adapter_task
 │   └─ blocks.py               # Block + Adapter/LoRA/FiLM 挂载
 ├─ modules/
 │   ├─ adapter.py              # BottleneckAdapter(r)
 │   └─ lora_film.py            # 备选外壳
 ├─ experts/
 │   ├─ dehaze.py / derain.py / desnow.py
 ├─ router/
 │   ├─ make_softlabels.py      # 生成 q 与 Oracle 标签
 │   ├─ router_net.py           # Tiny‑ViT / ConvNeXt‑T
 │   └─ calibrate.py            # 温度缩放 + τ 搜索
 ├─ infer/
 │   └─ infer_moei.py           # 瓶颈向量→路由→专家→滑窗
 ├─ eval/
 │   ├─ metrics.py              # PSNR/SSIM/LPIPS + CI + H‑Mean
 │   └─ costs.py                # FLOPs/Latency/Mem/Throughput
 └─ reports/                    # per-image.jsonl & summary.json
```
**函数签名**：  
- `y, g = backbone(x, return_global=True, adapter_task=None)`  
- `p = router(g)`  
- `y = run_expert(task, x, tile=512, overlap=64, blend='hann')`

**配置片段**：
```yaml
experts:
  common: {tile:512, overlap:64, blend:hann, amp:bf16, ema:0.999}
  dehaze: {adapter_r:0.0625, ssim_w:0.05}
  derain: {adapter_r:0.0625, edge_w:0.05, dir_consist_w:0.02}
  desnow: {adapter_r:0.0625, lpips_w:0.05, snow_mask_w:0.1}
router:
  backbone: convnext_t
  in_dim: C_global
  loss: kl_q_p
  beta: [20,50,80]
  calibrate: {search_T:true, search_tau:true, abstain_max:0.10}
```

---

## 9. 里程碑（两周强化版，可并行）
- **W1‑D1~D2**：数据 `meta.jsonl`、splits、去重/配准、severity 计算；混合退化小集。  
- **W1‑D3**：评测脚手架完成（PSNR/SSIM/LPIPS + CI；FLOPs/Latency/Mem/Throughput）。  
- **W1‑D4~D5**：Dense‑Unified 训练与主表 v0；Dehaze/Derain Adapter 接入自检。  
- **W1‑D6~W2‑D2**：三专家微调（256→384/512）；统一推理入口；导出三专家 val 表。  
- **W2‑D3**：生成软标签 q 与 Oracle 标签；路由器训练（KL(q‖p)）。  
- **W2‑D4**：温度缩放与阈值 τ 搜索（弃权 ≤ 10%）。  
- **W2‑D5**：MoE‑I 集成推理（Actual/Oracle/Dense），导出主表 v1。  
- **W2‑D6~D7**：消融（A~E 必做；F~I 视时长），图表与成本四件套定版。

---

## 10. 交付清单（验收点）
- `reports/*.json`：样本级与汇总（含设置/版本/commit/权重哈希）；
- 主表（系统 3+1 对照） + 三子表（每任务 6–8 基线或桥接）；
- 图：分层曲线、Coverage‑Accuracy、误路由敏感性、Pareto、（可选）RAM×误差热图；
- 代码：`backbone/`（return_global + adapter_task）、`modules/adapter.py`、`router/*`、`infer/infer_moei.py`、`eval/*`；
- 复现说明：环境、随机种子、I/O、tile/overlap/融合写明。

> 审阅通过后，我会将本指导方案转换为“**实施版 Runbook**”（列出可打勾任务、owner、占卡计划与验收标准），并补齐示例配置与最小脚手架。
