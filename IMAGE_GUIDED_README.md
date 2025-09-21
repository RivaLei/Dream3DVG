# Dream3DVG 图像引导训练配置说明

## 概述
Dream3DVG支持使用图像作为约束条件进行训练，而不仅仅是文本提示。这通过`from_dataset: True`参数启用。

## 配置文件说明

### 1. dream3dvg_image.yaml
位置: `conf/x/dream3dvg_image.yaml`

关键参数:
- `from_dataset: True` - 启用图像引导模式
- `image_lambda: 1.0` - 图像损失权重
- `text_lambda: 0.3` - 文本损失权重（仍保留以获得更好的语义理解）
- 优化的内存使用参数

### 2. 使用方法

#### 基本命令结构:
```bash
python svg_render.py \
    x=dream3dvg_image \
    target="path/to/target/image.jpg" \
    prompt="descriptive text" \
    result_path="./output/folder"
```

#### 重要参数说明:

1. **target**: 目标图像路径
   - 支持格式: .jpg, .png, .jpeg
   - 建议分辨率: 512x512 或以下

2. **prompt**: 文本描述
   - 即使在图像模式下也建议提供
   - 帮助模型理解图像内容
   - 例如: "A sketch of a cat", "An icon of a house"

3. **x.style**: 渲染风格
   - sketch: 素描风格
   - iconography: 图标风格
   - painting: 绘画风格

4. **x.num_paths**: SVG路径数量
   - 建议值: 16-64
   - 更多路径 = 更精细细节，但更多显存

## 内存优化建议

### GPU显存不足时的解决方案:

1. **降低分辨率**:
   ```yaml
   x.sd_guidance.eval_height: 256
   x.sd_guidance.eval_width: 256
   ```

2. **减少批次大小**:
   ```yaml
   x.batch_size: 1
   x.sd_guidance.batch_size: 1
   ```

3. **降低路径数量**:
   ```yaml
   x.num_paths: 16
   ```

4. **启用梯度检查点**:
   ```yaml
   x.sd_guidance.grad_clip: True
   ```

## 训练流程说明

### 双重优化策略:
1. **阶段1**: 粗糙初始化 (iter 0-500)
   - 主要使用几何约束
   - 建立基本形状结构

2. **阶段2**: 精细优化 (iter 500+)
   - 结合图像和文本损失
   - 细化细节和纹理

### 损失函数组成:
- **Image Loss**: 与目标图像的相似度
- **Text Loss**: 与文本描述的语义一致性
- **View Consistency Loss**: 多视角一致性
- **Depth Regularization**: 深度合理性

## 示例配置

### 低显存配置 (8GB GPU):
```yaml
x.sd_guidance.eval_height: 256
x.sd_guidance.eval_width: 256
x.num_paths: 16
x.batch_size: 1
x.sd_guidance.batch_size: 1
```

### 高质量配置 (16GB+ GPU):
```yaml
x.sd_guidance.eval_height: 512
x.sd_guidance.eval_width: 512
x.num_paths: 64
x.batch_size: 2
x.sd_guidance.batch_size: 2
```

## 故障排除

### 1. CUDA内存不足:
```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

### 2. 进程清理:
```bash
# 查看GPU使用情况
nvidia-smi

# 杀死Python进程
pkill -f python
```

### 3. 结果质量差:
- 检查目标图像质量和分辨率
- 调整损失权重 (image_lambda, text_lambda)
- 增加训练迭代次数
- 尝试不同的随机种子

## 结果分析

训练完成后，输出目录包含:
- `svg_*.svg`: 不同训练阶段的SVG文件
- `*.png`: 渲染结果图像
- `log/`: 训练日志和配置文件

建议查看不同迭代次数的结果，选择最佳效果的版本。
