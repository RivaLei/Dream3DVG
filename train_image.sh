#!/bin/bash

# Image-guided Dream3DVG Training Script
# 图像引导的Dream3DVG训练脚本
# 基于VS Code launch.json配置

export CUDA_VISIBLE_DEVICES='0'
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# 创建必要的目录
mkdir -p ./input_images/dtu_scan24_max
mkdir -p ./workspace/3dvg

echo "=== Dream3DVG Image-guided Training ==="
echo "基于VS Code launch.json配置"
echo ""

# 检查目标图片文件
TARGET_IMAGE="./input_images/dtu_scan24_max/rect_014_max.png"
if [ ! -f "$TARGET_IMAGE" ]; then
    echo "❌ 错误: 目标图片不存在!"
    echo "请将图片文件放置到: $TARGET_IMAGE"
    echo ""
    echo "或者将你的图片重命名为 rect_014_max.png 并放到 ./input_images/dtu_scan24_max/ 目录"
    exit 1
fi

echo "✅ 找到目标图片: $TARGET_IMAGE"
echo ""





# 执行与launch.json完全匹配的配置
echo "=== 执行 Dream3DVG - Image Guided (Sketch) 配置-sfm ==="
echo "参数配置:"
echo "  - 目标图片: $TARGET_IMAGE"
echo "  - 风格: sketch"
echo "  - 路径数: 32"
echo "  - 种子: 1"
echo "  - 提示词: A sketch of building"
echo "  - 结果路径: ./workspace/3dvg/dtu_scan24_max_sfm"
echo ""




# 执行与launch.json完全匹配的配置
echo "=== 执行 Dream3DVG - Image Guided (Sketch) 配置 ==="
echo "参数配置:"
echo "  - 目标图片: $TARGET_IMAGE"
echo "  - 风格: sketch"
echo "  - 路径数: 32"
echo "  - 种子: 1"
echo "  - 提示词: A sketch of building"
echo "  - 结果路径: ./workspace/3dvg/dtu_scan24_max"
echo ""

python svg_render.py \
    x=dream3dvg_image \
    seed=1 \
    x.style=sketch \
    x.num_paths=32 \
    target=./input_images/dtu_scan24_max/rect_014_max.png \
    prompt="A sketch of building" \
    result_path=./workspace/3dvg/dtu_scan24_max \
    x.camera_param.init_prompt="A sketch of building" 

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 训练完成!"
    echo "结果保存在: ./workspace/3dvg/dtu_scan24_max"
    echo ""
    echo "查看生成的文件:"
    ls -la ./workspace/3dvg/dtu_scan24_max/
else
    echo ""
    echo "❌ 训练过程中出现错误"
    echo "请检查错误信息并重试"
fi

echo ""
echo "=== 训练完成 ==="



python svg_render.py \
    x=dream3dvg_image \
    seed=1 \
    x.style=sketch \
    x.num_paths=32 \
    target=./input_images/dtu_scan24_max/rect_014_max.png \
    prompt="A sketch of building" \
    result_path=./workspace/3dvg/dtu_scan24_max_sfm \
    x.camera_param.init_prompt="A sketch of building"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 训练完成!"
    echo "结果保存在: ./workspace/3dvg/dtu_scan24_max_sfm"
    echo "" 
    echo "查看生成的文件:"
    ls -la ./workspace/3dvg/dtu_scan24_max_sfm/
else
    echo ""
    echo "❌ 训练过程中出现错误"
    echo "请检查错误信息并重试"
fi

echo ""
echo "=== 训练完成 ==="
