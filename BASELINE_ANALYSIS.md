# ICEdit Baseline 代码解析

> 快速参考文档 - 用于理解和修改ICEdit项目

---

## 📁 项目结构

```
ICEdit_raw/
├── scripts/                    # 推理和演示脚本
│   ├── inference.py           # ⭐ 基础推理脚本
│   ├── inference_moe.py       # MoE-LoRA版本推理
│   ├── gradio_demo.py         # Gradio Web界面
│   └── config.json
│
├── train/                      # ⭐ 训练代码库
│   ├── src/
│   │   ├── train/             # 训练核心模块
│   │   │   ├── model.py      # ⭐⭐ OminiModel定义
│   │   │   ├── data.py       # ⭐⭐ 数据集加载器
│   │   │   ├── train.py      # ⭐⭐ 主训练脚本
│   │   │   └── callbacks.py  # 训练回调(保存/日志)
│   │   │
│   │   └── flux/              # FLUX模型组件
│   │       ├── transformer.py
│   │       ├── condition.py
│   │       ├── pipeline_tools.py
│   │       └── lora_controller.py
│   │
│   ├── train/
│   │   ├── config/
│   │   │   ├── normal_lora.yaml  # ⭐ 训练配置
│   │   │   └── moe_lora.yaml
│   │   └── script/
│   │       └── train.sh          # ⭐ 训练启动脚本
│   │
│   └── parquet/               # 数据集存储
│       └── prepare.sh         # 数据下载脚本
│
├── assets/                    # 示例图像
└── README.md
```

---

## 🔑 核心组件详解

### 1. **模型架构** (`train/src/train/model.py`)

**类**: `OminiModel(L.LightningModule)`

#### 关键属性:
```python
- flux_fill_pipe: FluxFillPipeline  # 基础FLUX模型
- transformer: FluxTransformer       # 主干网络
- text_encoder: CLIP                 # 文本编码器
- text_encoder_2: T5                 # T5文本编码器
- lora_layers: List[Parameter]       # LoRA可训练参数
```

#### 关键方法:

| 方法 | 功能 | 代码位置 |
|------|------|----------|
| `__init__()` | 初始化FLUX模型 + 添加LoRA | 14-51行 |
| `init_lora()` | 配置LoRA适配器 | 53-64行 |
| `save_lora()` | 保存LoRA权重 | 66-74行 |
| `configure_optimizers()` | 设置Prodigy优化器 | 76-101行 |
| `training_step()` | 单步训练 | 103-110行 |
| `step()` | ⭐⭐ **核心训练逻辑** | 112-160行 |

#### **训练流程** (第112-160行):
```python
1. 获取batch数据
   - imgs: 目标图像 (编辑后)
   - mask_imgs: mask图像
   - prompts: 编辑指令

2. 编码 (with torch.no_grad)
   - prompt_embeds ← prepare_text_input(prompts)
   - x_0, x_cond, img_ids ← encode_images_fill(imgs, mask_imgs)

3. 流匹配采样
   - t ~ Sigmoid(N(0,1))          # 时间步
   - x_1 ~ N(0,1)                 # 纯噪声
   - x_t = (1-t)*x_0 + t*x_1     # 插值

4. 前向传播
   - input = concat(x_t, x_cond)  # 拼接条件
   - pred = transformer(input, t, prompt_embeds)

5. 计算损失
   - loss = MSE(pred, x_1 - x_0)  # 预测速度场
```

---

### 2. **数据处理** (`train/src/train/data.py`)

#### 三个数据集类:

| 类名 | 数据源 | 用途 |
|------|--------|------|
| `EditDataset` | MagicBrush (train+dev) | 仅MagicBrush |
| `OminiDataset` | OmniEdit (parquet) | 仅OmniEdit |
| `EditDataset_with_Omini` | MagicBrush + OmniEdit | ⭐ **混合数据集** |

#### **数据格式处理** (关键代码: 第49-83行):

```python
# 1. 读取数据
source_img = dataset["source_img"]        # 原始图像
target_img = dataset["target_img"]        # 编辑后图像
instruction = dataset["instruction"]      # 编辑指令

# 2. 调整尺寸
source_img = source_img.resize((512, 512)).convert("RGB")
target_img = target_img.resize((512, 512)).convert("RGB")

# 3. 创建Diptych (左右拼接)
combined_image = Image.new('RGB', (1024, 512))
combined_image.paste(source_img, (0, 0))    # 左半部分
combined_image.paste(target_img, (512, 0))  # 右半部分

# 4. 创建Mask (标记编辑区域)
mask = Image.new('L', (1024, 512), 0)
draw.rectangle([512, 0, 1024, 512], fill=255)  # 右半部分=255

# 5. 构造Prompt
prompt = "A diptych with two side-by-side images of the same scene. " \
         "On the right, the scene is exactly the same as on the left but " + instruction

# 6. 返回
return {
    "image": to_tensor(combined_image),    # [3, 512, 1024]
    "condition": to_tensor(mask),          # [1, 512, 1024]
    "description": prompt,
}
```

#### 重要参数:
- `condition_size = 512`: 固定宽度
- `drop_text_prob = 0.1`: 10%概率丢弃文本(用于CFG训练)
- `crop_the_noise = True`: 裁剪MagicBrush底部噪声

---

### 3. **训练配置** (`train/train/config/normal_lora.yaml`)

```yaml
# 模型路径
flux_path: "black-forest-labs/flux.1-fill-dev"
dtype: "bfloat16"

# 训练参数
train:
  batch_size: 2                    # 每GPU批次大小
  accumulate_grad_batches: 1       # 梯度累积
  dataloader_workers: 5
  save_interval: 1000              # 每1000步保存
  sample_interval: 1000            # 每1000步采样
  gradient_checkpointing: true     # 梯度检查点(省显存)

  # 数据集
  dataset:
    type: "edit_with_omini"        # ⭐ 使用混合数据集
    path: "parquet/*.parquet"      # OmniEdit数据路径
    condition_size: 512
    target_size: 512
    drop_text_prob: 0.1            # CFG训练

  # LoRA配置
  lora_config:
    r: 32                          # LoRA rank
    lora_alpha: 32                 # LoRA缩放因子
    init_lora_weights: "gaussian"
    target_modules: "(.*x_embedder|...|.*single_transformer_blocks\\.[0-9]+\\.attn.to_out)"
    # ⭐ 正则匹配目标模块:
    # - x_embedder
    # - transformer_blocks: norm1, attn (q/k/v/out), ff
    # - single_transformer_blocks: norm, proj_mlp/out, attn

  # 优化器
  optimizer:
    type: "Prodigy"                # 自适应优化器
    params:
      lr: 1                        # Prodigy推荐lr=1
      weight_decay: 0.01
```

---

### 4. **推理流程** (`scripts/inference.py`)

```python
# 1. 加载模型
pipe = FluxFillPipeline.from_pretrained("black-forest-labs/flux.1-fill-dev")
pipe.load_lora_weights("RiverZ/normal-lora")  # ⭐ 加载训练的LoRA
pipe.to("cuda")

# 2. 准备输入图像
image = Image.open(args.image).convert("RGB")
if image.width != 512:
    image = image.resize((512, new_height))  # ⭐ 强制宽度=512

# 3. 构造Diptych
combined_image = Image.new("RGB", (1024, height))
combined_image.paste(image, (0, 0))     # 左边: 原图
combined_image.paste(image, (512, 0))   # 右边: 原图(待编辑)

# 4. 创建Mask
mask = np.zeros((height, 1024), dtype=np.uint8)
mask[:, 512:] = 255  # 右半部分

# 5. 构造Prompt
instruction = f"A diptych with two side-by-side images of the same scene. " \
              f"On the right, the scene is exactly the same as on the left but {args.instruction}"

# 6. 推理
result = pipe(
    prompt=instruction,
    image=combined_image,
    mask_image=mask,
    height=height,
    width=1024,
    guidance_scale=50,       # ⭐ 高CFG=更强指令跟随
    num_inference_steps=28,
    generator=torch.Generator("cpu").manual_seed(args.seed)
).images[0]

# 7. 裁剪右半部分
result = result.crop((512, 0, 1024, height))  # 只保留编辑后的图像
```

---

## 🎯 核心设计理念

### 1. **Diptych (双联画) 设计**
- **动机**: 让模型同时看到原图和编辑后的图，学习"保持一致性"
- **格式**: `[原图 | 编辑后]` 水平拼接
- **优势**:
  - 隐式学习图像对应关系
  - 更好的ID/风格保持
  - 简化训练(不需要显式对齐损失)

### 2. **Flow Matching 训练**
```python
# 传统扩散模型: 预测噪声 ε
loss = MSE(model(x_t, t), ε)

# Flow Matching: 预测速度场 v
v = x_1 - x_0  # 从干净图像到噪声的"流动方向"
loss = MSE(model(x_t, t), v)
```

### 3. **Prompt Engineering**
固定前缀: `"A diptych with two side-by-side images of the same scene. On the right, the scene is exactly the same as on the left but {instruction}"`

- "diptych" → 明确双图格式
- "exactly the same" → 强调一致性
- "but {instruction}" → 指定编辑内容

---

## 🔧 训练启动

### 准备数据
```bash
cd train/parquet
bash prepare.sh  # 下载OmniEdit数据集
```

### 启动训练
```bash
cd train
export XFL_CONFIG=train/config/normal_lora.yaml
bash train/script/train.sh
```

**train.sh 内容** (推测):
```bash
XFL_CONFIG=train/config/normal_lora.yaml \
python -m torch.distributed.run \
    --nproc_per_node=4 \
    src/train/train.py
```

---

## 📊 关键超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| Image Size | 512×512 | 固定宽度512 |
| LoRA Rank | 32 | 较大=更强表达能力 |
| Batch Size | 2×4 GPU = 8 | 论文中总batch=16 (2×2×4) |
| Optimizer | Prodigy (lr=1) | 自适应学习率 |
| CFG Scale | 50 (推理) | 非常高=强指令跟随 |
| Steps | 28 (推理) | FLUX标准步数 |
| Drop Text Prob | 0.1 | CFG训练 |

---

## 🚀 快速定位代码

### 需要修改训练逻辑?
→ `train/src/train/model.py` 第112-160行 (`step()` 方法)

### 需要修改数据处理?
→ `train/src/train/data.py` 第49-84行 (`__getitem__()` 方法)

### 需要改变LoRA目标层?
→ `train/train/config/normal_lora.yaml` 第38行 (`target_modules`)

### 需要调整训练参数?
→ `train/train/config/normal_lora.yaml`

### 需要修改推理流程?
→ `scripts/inference.py` 第62-71行 (pipe调用)

---

## 💡 重要注意事项

1. **图像宽度必须是512**
   - 模型在512宽度上训练
   - 推理时自动resize到512

2. **Diptych格式固定**
   - 训练: `[原图 | 编辑图]`
   - 推理: `[原图 | 原图]` → 输出 `[原图 | 编辑图]`

3. **Mask固定右半部分**
   ```python
   mask[:, 512:] = 255  # 右半部分
   ```

4. **高CFG Scale (50)**
   - 远高于常规扩散模型(通常7-10)
   - 用于增强指令跟随能力

5. **数据集混合**
   - MagicBrush: 高质量标注
   - OmniEdit: 大规模多样性

---

## 🔍 调试技巧

### 查看训练进度
```bash
# WanDB (需要配置WANDB_API_KEY)
# 或查看本地日志
ls train/runs/20250513-*/
```

### 测试单个样本
```python
from train.src.train.data import EditDataset_with_Omini
dataset = EditDataset_with_Omini(...)
sample = dataset[0]
print(sample.keys())  # image, condition, description
```

### 验证LoRA加载
```python
pipe.load_lora_weights("path/to/lora")
# 检查是否成功
print(pipe.transformer.get_adapter_state_dict())
```

---

## 📚 相关资源

- **论文**: https://arxiv.org/abs/2504.20690
- **HuggingFace模型**: https://huggingface.co/RiverZ/normal-lora
- **基础代码**: OminiControl (https://github.com/Yuanshi9815/OminiControl)
- **FLUX模型**: https://huggingface.co/black-forest-labs/flux.1-fill-dev

---

**最后更新**: 2025-10-04
