import os
import shutil
import random

# 定义源目录和目标目录
src_dir = 'imagenet100'
train_dir = os.path.join(src_dir, 'train')
val_dir = os.path.join(src_dir, 'val')

# 创建train和val目录
if not os.path.exists(train_dir):
    os.makedirs(train_dir)
if not os.path.exists(val_dir):
    os.makedirs(val_dir)

# 遍历每个类别文件夹
for class_dir in next(os.walk(src_dir))[1]:
    class_path = os.path.join(src_dir, class_dir)
    train_class_dir = os.path.join(train_dir, class_dir)
    val_class_dir = os.path.join(val_dir, class_dir)

    # 创建类别对应的train和val目录
    if not os.path.exists(train_class_dir):
        os.makedirs(train_class_dir)
    if not os.path.exists(val_class_dir):
        os.makedirs(val_class_dir)

    # 获取所有图片文件
    images = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
    
    # 随机选择20%的图片作为验证集
    val_images = random.sample(images, int(len(images) * 0.2))

    # 移动图片到train和val目录
    for image in images:
        if image in val_images:
            shutil.move(os.path.join(class_path, image), os.path.join(val_class_dir, image))
        else:
            shutil.move(os.path.join(class_path, image), os.path.join(train_class_dir, image))

print("图片分配完成。")