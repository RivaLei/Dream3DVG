#!/bin/bash

# Image-guided Dream3DVG Training Script
# 图像引导的Dream3DVG训练脚本

export CUDA_VISIBLE_DEVICES='0'
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# 创建示例图片目录
mkdir -p ./input_images

echo "=== Dream3DVG Image-guided Training ==="
echo "请将目标图片放在 ./input_images/ 目录下"
echo "支持的格式: .jpg, .png, .jpeg"
echo ""

# 检查是否有输入图片
if [ ! "$(ls -A ./input_images/*.{jpg,png,jpeg} 2>/dev/null)" ]; then
    echo "警告: ./input_images/ 目录下没有找到图片文件"
    echo "请先添加目标图片文件到该目录"
    echo ""
    echo "示例用法:"
    echo "1. 将图片 car.jpg 放到 ./input_images/ 目录"
    echo "2. 运行此脚本"
    exit 1
fi

# 列出可用的图片
echo "找到以下图片文件:"
ls -la ./input_images/*.{jpg,png,jpeg} 2>/dev/null
echo ""

# Sketch风格 + 图像引导
style='sketch'
num_paths=32

echo "=== 开始 Sketch 风格的图像引导训练 ==="

# 遍历输入图片进行训练
for img_file in ./input_images/*.{jpg,png,jpeg}; do
    [ -f "$img_file" ] || continue  # 跳过不存在的文件
    
    filename=$(basename "$img_file")
    name="${filename%.*}"  # 去掉扩展名
    
    echo "处理图片: $filename"
    
    python svg_render.py \
        x=dream3dvg_image \
        seed=1 \
        x.style=$style \
        x.num_paths=$num_paths \
        target="$img_file" \
        prompt="A sketch drawing" \
        result_path="./workspace/3dvg/sketch_${name}" \
        "x.camera_param.init_prompt=A sketch of ${name}"
    
    echo "完成处理: $filename"
    echo "结果保存在: ./workspace/3dvg/sketch_${name}"
    echo ""
done

# Iconography风格 + 图像引导
style='iconography'
num_paths=64  # 减少路径数以节省显存

echo "=== 开始 Iconography 风格的图像引导训练 ==="

for img_file in ./input_images/*.{jpg,png,jpeg}; do
    [ -f "$img_file" ] || continue
    
    filename=$(basename "$img_file")
    name="${filename%.*}"
    
    echo "处理图片: $filename"
    
    python svg_render.py \
        x=dream3dvg_image \
        seed=2 \
        x.style=$style \
        x.num_paths=$num_paths \
        target="$img_file" \
        prompt="An icon illustration" \
        result_path="./workspace/3dvg/icon_${name}" \
        "x.camera_param.init_prompt=An icon of ${name}"
    
    echo "完成处理: $filename"
    echo "结果保存在: ./workspace/3dvg/icon_${name}"
    echo ""
done

echo "=== 所有图像引导训练完成 ==="
echo "查看结果:"
echo "ls -la ./workspace/3dvg/"
