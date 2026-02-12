import os
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import pandas as pd
import random
import torchvision.datasets as datasets


class FoodSubset(datasets.ImageFolder):

    def __init__(
            self,
            root,
            split='train',
            cfg=None,
            transform=None,
            target_transform=None,
            is_valid_file=None,
    ):
        self.cfg = cfg
        self.food_subset_dir = getattr(self.cfg.DATA, "FOOD_SUBSET_DIR", "food-101")
        dataset_root = os.path.join(root, self.food_subset_dir)

        super().__init__(
            os.path.join(dataset_root, "train"),
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file
        )

        if self.cfg.DATA.CLASSES == None:
            self.classes = []
            with open(os.path.join(dataset_root, "meta", "labels.txt")) as f:
                for line in f:
                    self.classes.append(
                        line.replace('\n', '').replace(' ', '_').lower()
                    )
        else:
            self.classes = sorted(self.cfg.DATA.CLASSES)
        df = pd.read_csv(
            os.path.join(self.cfg.DATA.ROOT, self.food_subset_dir, "meta", "all_images.csv")
        )
        df = pd.concat(
            [df[df['label'] == c] for c in self.classes]
        )
        self.df = df[df[self.cfg.DATA.SPLIT] == split]
        self.filename_array = self.df["abs_file_path"].values
        self.imgs = [
            (os.path.join(dataset_root, f), self.classes.index(l))
            for f, l in zip(self.df["abs_file_path"], self.df['label'])
        ]
        self.data = np.array(
            [os.path.join(dataset_root, f) for f in df["abs_file_path"]]
        )
        self.samples = self.imgs
        self.groups = [0 for i in range(len(self.imgs))]

        self.return_attention = cfg.DATA.ATTENTION_DIR != "NONE"
        self.size = cfg.DATA.SIZE

        if self.return_attention:
            self.attention_data = np.array(
                [
                    os.path.join(
                        self.root.replace('/train', '').replace('/test', '').replace('/val', ''),
                        cfg.DATA.ATTENTION_DIR,
                        path.replace('.jpg', '.pth')
                    )
                    for path in self.filename_array
                ]
            )


    def __getitem__(self, index):
        path, target = self.imgs[index]
        group = self.groups[index]

        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.return_attention:
            att = torch.load(self.attention_data[index])
            if self.cfg.DATA.ATTENTION_DIR == 'deeplabv3_attention':
                att = att['mask']
            else:
                att = att['unnormalized_attentions']
            att = F.interpolate(att, size=(self.size, self.size), mode='bilinear',
                                align_corners=False)#[0]
        else:
            att = torch.Tensor([-1]) # So batches correctly

        NULL = torch.Tensor([-1]) # So batches correctly

        return {
            'image_path': path,
            'image': sample,
            'label': target,
            'seg':   NULL,
            'group': group,
            'bbox': NULL,
            'attention': att,
        }
