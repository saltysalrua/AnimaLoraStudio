# AnimaLoraStudio 改进计划

来源：参考 `D:\Code\Claude\new\another` 目录下的训练脚本实现。

每步骤执行前先核查现有代码，确认是否已有或有更优实现，再决定是否修改。

---

## 状态说明

- ✅ 已完成
- 🔍 检查后跳过（已有 / 不适用）
- ⏳ 待处理
- 🚧 进行中

---

## 步骤列表

### 第一组：优化器增强（改动小、收益稳）

#### Step 1 — ProdigyPlus 版本兼容参数过滤 ⏳

**文件**：`utils/optimizer_utils.py`

**目标**：用 `inspect.signature` 自动过滤当前包版本不支持的参数，避免用户装了旧版
`prodigy-plus-schedule-free` 时直接 TypeError 崩溃。

**检查点**：
- [ ] 确认 `create_prodigy_plus()` 当前是否有任何兼容性保护
- [ ] 确认参数列表哪些是新版才有的

**预期改动**：在 `create_prodigy_plus()` 里用 `_filter_kwargs_by_signature()` 过滤 kwargs。

---

#### Step 2 — ProdigyPlus eps=None + use_stableadamw ⏳

**文件**：`utils/optimizer_utils.py`，`studio/schema.py`

**目标**：
- `eps=None` 启用 Adam-atan2，消除 epsilon 依赖，对 bf16 梯度尖峰更稳定
- `use_stableadamw=True` 抗梯度尖峰

**检查点**：
- [ ] 确认 prodigy-plus-schedule-free 当前版本是否支持这两个参数（Step 1 完成后可知）
- [ ] schema 里是否已有相关字段

**预期改动**：schema 加 `prodigy_use_stableadamw: bool`，optimizer_utils 透传。

---

### 第二组：训练稳定性

#### Step 3 — Forward / 梯度 NaN 检测与跳过 ⏳

**文件**：`runtime/anima_train.py`

**目标**：
- forward 输出 NaN → 跳过该 micro-batch，记录日志，不反向传播
- 梯度 NaN → 跳过本次 optimizer.step()，清零梯度继续

**检查点**：
- [ ] 确认当前训练循环有无任何 NaN 检测逻辑（sampling 里有 nan_to_num，训练循环里无）
- [ ] 确认 grad_clip 后是否有梯度检查

**预期改动**：在 `loss.backward()` 后、`optimizer.step()` 前各加 NaN 检测分支。

---

#### Step 4 — Loss 加权方案 + weight_cap_ratio ⏳

**文件**：`runtime/anima_train.py`，`studio/schema.py`

**目标**：
- 实现多种 loss 权重方案：`none`（当前默认）、`min_snr`、`detail_inv_t`、`cosmap`
- `weight_cap_ratio`：batch 内 max/min 权重比上限，防单样本主导，保护 Prodigy d 估计

**检查点**：
- [ ] 当前 loss 计算是否已有任何加权（除正则集 loss_weight 外）
- [ ] `sample_t` 返回的 t 是否已有 shift（已有 shift=3.0）

**预期改动**：
- 新增 `compute_loss_weight(t, mode, weight_cap_ratio)` 函数
- schema 加 `loss_weighting: Literal[...]`，`loss_weight_cap_ratio: float`
- 训练循环里在 `F.mse_loss` 后应用权重

---

### 第三组：训练效果

#### Step 5 — Timestep 采样模式扩展 ⏳

**文件**：`runtime/anima_train.py`，`studio/schema.py`

**目标**：扩展 `sample_t()` 支持多种模式。

**检查点**：
- [ ] 当前 `sample_t()` 实现：logit-normal + shift=3.0，已经是较先进的默认值
- [ ] 确认是否有必要加 uniform / mode 等其他模式，还是只加 `schedule_shift` 可配置

**预期改动**：
- schema 加 `timestep_sampling: Literal["logit_normal", "uniform", "mode", "logit_normal_low"]`
- schema 加 `timestep_shift: float`（当前硬编码 3.0）
- `sample_t()` 按模式分支

---

#### Step 6 — 噪声增强：noise_offset + pyramid_noise ⏳

**文件**：`runtime/anima_train.py`，`studio/schema.py`

**目标**：
- `noise_offset`：向噪声加低频偏移，缓解亮度偏差（来自 SDXL 论文）
- `pyramid_noise`：多尺度低频叠加，构图更自然（bilinear 下采样再叠加）

**检查点**：
- [ ] 确认当前噪声生成逻辑（`torch.randn_like(latents)`，无增强）
- [ ] 确认 VAE latent 维度（16 通道），pyramid 下采样是否适配

**预期改动**：
- 新增 `make_noise(latents, noise_offset, pyramid_iterations, pyramid_discount)` 函数
- schema 加对应参数（默认 0/关闭，不改变默认行为）

---

### 第四组：模块级控制

#### Step 7 — Per-block rank（lora_reg_dims / lora_reg_lrs） ⏳

**文件**：`utils/lycoris_adapter.py`，`utils/tlora_adapter.py`，`studio/schema.py`

**目标**：YAML 里用正则匹配模块名，覆盖该模块的 rank 和 lr，未匹配的用全局值。

**检查点**：
- [ ] 确认 lycoris_adapter 是否已暴露模块级配置接口
- [ ] 确认 LycorisNetwork（lycoris-lora 库）是否本身支持 per-layer rank（可能不需要手动实现）
- [ ] tlora_adapter 目前注入逻辑是否支持不同 rank

**预期改动**：
- schema 加 `lora_reg_dims: dict | None`，`lora_reg_lrs: dict | None`
- lycoris_adapter 注入时按正则匹配覆盖 rank
- tlora_adapter 同理

---

### 第五组：复杂度较高

#### Step 8 — 手动 OrthoGrad ⏳

**文件**：新增 `utils/orthograd.py`，`runtime/anima_train.py`，`studio/schema.py`

**目标**：解决 ProdigyPlus 内置 OrthoGrad 对 LoKr 的幅度锁定副作用。精确排除
`lokr_w1`、`lokr_w2_b`、`lora_B`，只对 `lora_A` / `lokr_w2_a` 做梯度投影。

**检查点**：
- [ ] 确认当前是否对用户暴露 OrthoGrad 相关配置
- [ ] 确认 prodigy-plus-schedule-free 内置 use_orthograd 参数，用户现在能否开启

**预期改动**：
- 新增 `utils/orthograd.py`，实现 `apply_partial_orthograd_()`
- schema 加 `orthograd_mode: Literal["off", "manual"]`，`orthograd_enable_after_step: int`
- 训练循环在 `optimizer.step()` 前调用（manual 模式下关闭 ProdigyPlus 内置）

---

#### Step 9 — CachedLatentDataset（npz latent 缓存） ⏳

**文件**：`utils/dataset.py`，`runtime/anima_train.py`，`studio/schema.py`

**目标**：第 1 个 epoch 把 VAE encoding 结果缓存到 npz，第 2 个 epoch 起直接读缓存，
跳过 VAE 推理，大幅加速。

**检查点**：
- [ ] 确认当前 dataset 是否已有 `cache_latents` 字段（schema 里有，实现是否完整）
- [ ] 确认 `flip_augment` 与缓存的冲突处理

**预期改动**：视 Step 7 完成情况决定是否补全缓存逻辑。

---

## 已确认跳过

| 功能 | 原因 |
|------|------|
| LLMAdapter 自动禁用保护 | 已有（`anima_train.py:556-563`）|
| logit-normal + shift 的 timestep 采样 | 已有（`sample_t()` 硬编码 shift=3.0）；Step 5 改为可配置 |

---

## Commit 节点规划

| Commit | 包含步骤 | 说明 |
|--------|----------|------|
| C1 | Step 1 + Step 2 | 优化器增强（独立、改动集中） |
| C2 | Step 3 | NaN 保护（独立功能） |
| C3 | Step 4 + Step 5 | 训练动态控制（loss 权重 + timestep） |
| C4 | Step 6 | 噪声增强（独立） |
| C5 | Step 7 | Per-block rank（跨多文件，单独 commit） |
| C6 | Step 8 | OrthoGrad（复杂，单独 commit） |
| C7 | Step 9 | Latent 缓存（视情况） |
