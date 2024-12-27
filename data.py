import os
import torch
from torch.utils.data import DataLoader, Dataset
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
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self._load_data()

    def _load_data(self):
        index = 0
        for label_dir in os.listdir(self.root_dir):
            label_dir_path = os.path.join(self.root_dir, label_dir)
            if os.path.isdir(label_dir_path):
                for image_name in os.listdir(label_dir_path):
                    image_path = os.path.join(label_dir_path, image_name)
                    self.image_paths.append(image_path)
                    self.labels.append(index)  # 假设文件夹名即为标签
            index += 1

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
    train_path = os.path.join(root_dir,"train")
    test_path = os.path.join(root_dir,"val")
    train_dataset = CustomDataset(root_dir=train_path, transform=train_transform)
    test_dataset = CustomDataset(root_dir=test_path, transform=test_transform)
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