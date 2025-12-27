# 方法

## 1. 方法概述
本文提出一种基于 **in-context** 结构的可见光到红外图像转换方法。在训练与推理中，将可见光图像与红外图像拼接为双联图（diptych），并通过掩码仅预测右半部分。模型主干采用 Flux 填充式扩散模型，主干参数保持冻结；通过 LoRA 进行轻量适配，并引入可训练的 Adapter 与 Thermal Query，以文本语义作为调制信号，提升可见光到热成像的语义对齐能力。

整体流程为：
1) 构建 [可见光 | 红外] 双联图与右半掩码；
2) 指令文本进入主干文本编码器，作为任务定义；
3) 图像描述（caption）仅用于 Adapter/thermal query；
4) Adapter 生成额外文本 token 并注入 cross-attention；
5) 模型在掩码区域进行预测，使用 masked loss 优化。

## 2. 数据组织与 in-context 构造
**2.1 双联图构造**
给定可见光图像 $I_{vis}$ 与对应红外图像 $I_{ir}$，将二者水平拼接形成 $I_{pair}=[I_{vis}, I_{ir}]$。对应掩码 $M$ 仅覆盖右半区域（红外侧），训练目标为恢复右半部分。

**2.2 CSV 数据格式**
默认数据列：
- `kontext_images`：可见光图像路径；
- `image`：红外图像路径；
- `prompt`：可见光图像描述（caption，可选）。

所有图像统一 resize 到固定尺寸（默认 512）。

## 3. 指令与描述分离策略
为了保证 in-context 任务定义稳定，本文将文本输入划分为两部分：

**指令（instruction）**
固定任务描述，定义 “可见光 → 红外” 的目标形式：
> A diptych with two side-by-side images of the same scene. The left image is visible light. The right image is the corresponding infrared thermal image. Instruction: Transform the visible image into its infrared thermal counterpart while preserving scene structure and objects.

该指令进入主干文本编码器，控制整体编辑方向。

**图像描述（caption）**
由 CSV 提供的可见光图像语义描述，仅用于 Adapter 与 Thermal Query 调制，不直接影响主干指令流。这样可避免 caption 干扰 in-context 指令，同时为热成像生成提供样本级语义信号。

在实现中，通过配置项 `adapter.affect_prompt: false` 保证 caption 只影响 adapter，不改写主干 pooled embedding。

## 4. Adapter 与 Thermal Query 设计
### 4.1 Pooled Prompt Adapter
将 pooled text embedding 经过轻量 MLP 做残差变换：
\[
\tilde{e} = e + \alpha f(e)
\]
其中 $e$ 为 pooled embedding，$f(\cdot)$ 为两层 MLP。此模块可选启用。

### 4.2 Text Token Adapter
从 pooled embedding 生成 $N$ 个额外 token：
- 先经 MLP 生成 $N \times d$ 的向量；
- reshape 为 $N$ 个 token 并投影到 Transformer 文本维度；
- 与原始文本 token 拼接后注入 cross-attention。

该设计类似 prefix-tuning / IP-Adapter，使模型能在 cross-attention 中关注可学习的语义提示。

### 4.3 Thermal Query Tokens
Thermal Query 由少量可学习向量组成（默认 4 个），其作用是提供“热先验”查询：
- 初始为随机可学习参数；
- 训练中通过红外监督信号对齐热成像分布；
- 拼接到文本 token 序列后，参与 cross-attention。

该机制类似 DETR 的 query：在训练过程中逐步学习到与“热模式”相关的语义槽位，例如人体、车辆、道路、植被等高/低温区域的统计特征。

### 4.4 Gate 调制机制（可选）
为平衡 thermal query 与 caption token 的贡献，加入可训练 gate：
- **Global gate**：每个样本一个标量；
- **Headwise gate**：每个注意力头一个标量。

gate 对 thermal token 进行缩放，且可选择反向调制文本 token：
\[
T_{thermal}' = g \cdot T_{thermal}, \quad T_{text}' = (1-g) \cdot T_{text}
\]

该机制通过配置控制，支持 global/headwise 两种模式。

## 5. 训练流程
训练过程如下：
1. **文本编码**：
   - 指令进入主干文本编码器（冻结）；
   - caption 单独编码，仅供 adapter 使用。
2. **图像编码**：
   - diptych 图像与掩码送入 VAE（冻结）获取 latent。
3. **噪声注入**：
   - 使用 Flux/FlowMatch 风格的 sigma schedule。
4. **Adapter 注入**：
   - thermal query 与 adapter token 拼接进文本 token 序列。
5. **扩散预测**：
   - 预测噪声残差，计算 masked loss。

**可训练模块**
- LoRA（主干 Transformer 低秩适配）；
- Adapter 与 Thermal Gate（当 `adapter.trainable: true` 时）。

主干文本编码器与 VAE 全程冻结。

## 6. 推理流程
推理阶段：
- 使用固定指令作为主干提示；
- 若提供 caption，则作为 adapter 输入；
- 若 caption 为空，可选择继续为空（与“空描述训练”一致），或回退到指令作为 adapter 输入。

推理时自动加载：
- `prompt_adapter.pt`、`text_token_adapter.pt`、`thermal_query.pt`；
- `thermal_gate.pt`（若存在则启用，否则自动关闭）。

## 7. 备注与实现细节
- CLIP 文本编码器最大长度为 77 token，每个文本输入独立截断；
- 建议 caption 控制在 30–40 个英文词或 40–60 个中文字符；
- 若 caption 为空，则 adapter 退化为固定先验，不具备样本级调制能力。
