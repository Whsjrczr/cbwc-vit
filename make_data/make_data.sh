#!/bin/bash

# 定义源目录
src_dir="imagenet100"

# 定义训练和验证目录
train_dir="$src_dir/train"
val_dir="$src_dir/val"

# 创建训练和验证目录
mkdir -p "$train_dir"
mkdir -p "$val_dir"

# 遍历每个类别文件夹
for class_dir in "$src_dir"/*; do
    if [ -d "$class_dir" ]; then
        class_name=$(basename "$class_dir")
        train_class_dir="$train_dir/$class_name"
        val_class_dir="$val_dir/$class_name"

        # 创建类别对应的训练和验证目录
        mkdir -p "$train_class_dir"
        mkdir -p "$val_class_dir"

        # 获取所有图片文件
        images=("$class_dir"/*)
        
        # 计算验证集图片数量
        val_count=$((${#images[@]} * 20 / 100))

        # 随机选择20%的图片作为验证集
        val_images=()
        while [ ${#val_images[@]} -lt $val_count ]; do
            image=${images[$RANDOM]}
            if [[ ! "${val_images[@]}" =~ "$image" ]]; then
                val_images+=("$image")
            fi
        done

        # 移动图片到train和val目录
        for image in "${images[@]}"; do
            if [[ "${val_images[@]}" =~ "$image" ]]; then
                mv "$image" "$val_class_dir"
            else
                mv "$image" "$train_class_dir"
            fi
        done
    fi
done

echo "图片分配完成。"