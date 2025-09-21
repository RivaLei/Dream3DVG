# Dream3DVG Launch Configuration 使用指南

## VS Code 调试配置说明

已将 `train_image.sh` 中的sketch模式命令转换为VS Code的launch.json配置，方便调试和开发。

## 可用的调试配置

### 1. Dream3DVG - Image Guided (Sketch)
- **用途**: 使用图像引导的sketch风格训练
- **配置**: 固定参数，使用 `./input_images/example.jpg` 作为输入
- **适用场景**: 快速测试sketch风格的图像引导功能

### 2. Dream3DVG - Image Guided (Iconography)  
- **用途**: 使用图像引导的iconography风格训练
- **配置**: 固定参数，路径数减少到64以节省显存
- **适用场景**: 测试iconography风格的图像引导功能

### 3. Dream3DVG - Image Guided (Custom)
- **用途**: 自定义图像引导训练
- **配置**: 运行时动态输入参数
- **输入参数**:
  - `targetImage`: 目标图像路径
  - `promptText`: 文本提示
  - `outputName`: 输出文件夹名称

## 使用步骤

### 准备工作
1. 创建输入图像目录:
   ```bash
   mkdir -p ./input_images
   ```

2. 将目标图像放入目录:
   ```bash
   cp your_image.jpg ./input_images/example.jpg
   ```

### VS Code 调试步骤
1. 打开 VS Code
2. 按 `F5` 或点击调试按钮
3. 从下拉菜单选择相应的配置:
   - `Dream3DVG - Image Guided (Sketch)` - 固定sketch配置
   - `Dream3DVG - Image Guided (Iconography)` - 固定iconography配置  
   - `Dream3DVG - Image Guided (Custom)` - 自定义配置
4. 如果选择Custom配置，会提示输入:
   - 图像路径 (默认: `./input_images/example.jpg`)
   - 文本提示 (默认: `A sketch drawing`)
   - 输出名称 (默认: `custom_output`)

## 配置详解

### 环境变量
```json
"env": {
    "CUDA_VISIBLE_DEVICES": "0",
    "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128"
}
```
- `CUDA_VISIBLE_DEVICES`: 指定使用GPU 0
- `PYTORCH_CUDA_ALLOC_CONF`: 内存分配优化，避免显存碎片

### 参数说明
- `x=dream3dvg_image`: 使用图像引导配置
- `x.style=sketch/iconography`: 风格选择
- `x.num_paths=32/64`: 路径数量（影响细节和显存使用）
- `target=./input_images/example.jpg`: 目标图像路径
- `prompt=A sketch drawing`: 文本提示
- `result_path=./workspace/3dvg/output`: 结果保存路径

## 调试技巧

1. **设置断点**: 在关键代码位置设置断点进行调试
2. **查看变量**: 在调试过程中查看张量数值和形状
3. **修改参数**: 直接在launch.json中修改参数进行快速测试
4. **显存监控**: 
   ```bash
   watch -n 1 nvidia-smi
   ```

## 故障排除

### 显存不足
- 减少 `x.num_paths` 参数
- 降低 `x.image_size` (在配置文件中)
- 关闭不必要的应用程序

### 图像路径错误
- 确保图像文件存在
- 检查路径是否正确
- 支持格式: .jpg, .png, .jpeg

### 配置文件缺失
- 确保 `dream3dvg_image.yaml` 配置文件存在
- 检查配置文件中的参数设置

## 输出结果

训练完成后，结果保存在:
- SVG文件: `./workspace/3dvg/{output_name}/svg_logs/`
- 3D高斯结果: `./workspace/3dvg/{output_name}/gs_logs/`
- 评估结果: `./workspace/3dvg/{output_name}/eval/`

## 原始命令对比

**原始bash命令**:
```bash
python svg_render.py \
    x=dream3dvg_image \
    seed=1 \
    x.style=sketch \
    x.num_paths=32 \
    target="./input_images/example.jpg" \
    prompt="A sketch drawing" \
    result_path="./workspace/3dvg/sketch_example" \
    "x.camera_param.init_prompt=A sketch of example"
```

**VS Code launch.json配置**:
```json
{
    "name": "Dream3DVG - Image Guided (Sketch)",
    "args": [
        "x=dream3dvg_image",
        "seed=1", 
        "x.style=sketch",
        "x.num_paths=32",
        "target=./input_images/example.jpg",
        "prompt=A sketch drawing",
        "result_path=./workspace/3dvg/sketch_example",
        "x.camera_param.init_prompt=A sketch of example"
    ]
}
```
