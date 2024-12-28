import os
import torch
from torch.utils.data import DataLoader, Dataset, random_split
import random
from torchvision import transforms, datasets
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

try:
    from torchvision.transforms import InterpolationMode

    def _pil_interp(method):
        if method == 'bicubic':
            return InterpolationMode.BICUBIC
        elif method == 'lanczos':
            return InterpolationMode.LANCZOS
        elif method == 'hamming':
            return InterpolationMode.HAMMING
        else:
            # default bilinear, do we want to allow nearest?
            return InterpolationMode.BILINEAR

    import timm.data.transforms as timm_transforms

    timm_transforms._pil_interp = _pil_interp
except:
    from timm.data.transforms import _pil_interp


class CustomDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.image_paths = []
        self.labels = []

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = datasets.folder.default_loader(image_path)
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label



def make_dataset(root_dir, splite_rate):
    train_transform = build_transform(True)
    test_transform = build_transform(False)
    train_dataset = CustomDataset(train_transform)
    test_dataset = CustomDataset(test_transform)
    index = 0
    for label_dir in os.listdir(root_dir):
        # get each label dir
        if label_dir[0] == '.':
            continue
        label_dir_path = os.path.join(root_dir, label_dir)
        if os.path.isdir(label_dir_path):
            # get all image under this label
            image_names = [f for f in os.listdir(label_dir_path) if os.path.isfile(os.path.join(label_dir_path, f))]
            val_images = random.sample(image_names, int(len(image_names) * 0.2))

            for image_name in image_names:
                image_path = os.path.join(label_dir_path, image_name)
                if image_name in val_images:
                    test_dataset.image_paths.append(image_path)
                    test_dataset.labels.append(index)
                else:
                    train_dataset.image_paths.append(image_path)
                    train_dataset.labels.append(index)
        index += 1
    
    return train_dataset, test_dataset



def build_transform(is_train):
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=224,
            is_training=True,
            color_jitter=0.4,
            auto_augment='rand-m9-mstd0.5-inc1',
            re_prob=0.25,
            re_mode='pixel',
            re_count=1,
            interpolation='bicubic',
        )
        return transform

    t = []
    size = int((256 / 224) * 224)
    t.append(
        transforms.Resize(size, interpolation=_pil_interp('bicubic')),
        # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(224))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)