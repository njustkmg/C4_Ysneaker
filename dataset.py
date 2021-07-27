import paddle
from paddle.io import Dataset
from paddle.vision.transforms import Compose, ToTensor, Normalize, Resize

import os
import json
import numpy as np
from PIL import Image

import PIL
from PIL import UnidentifiedImageError

# 统一图片大小，并将图片转换为Tensor
process = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

class MyDataset(Dataset):
    def __init__(self, args, split=''):
        super(MyDataset, self).__init__()
        if split == 'train':
            with open(args.identify_path_train, "r") as f:
                data_img = json.load(f)

            with open(args.retrieval_path_train, "r") as f:
                re_data_img = json.load(f)

        elif split == 'val':
            with open(args.identify_path_val, "r") as f:
                data_img = json.load(f)

            with open(args.retrieval_path_val, "r") as f:
                re_data_img = json.load(f)

        elif split == 'test':
            with open(args.identify_path_test, "r") as f:
                data_img = json.load(f)

            with open(args.retrieval_path_test, "r") as f:
                re_data_img = json.load(f)

        else:
            assert ("Please input dataset path!")

        re_imgs = data_img['img_id']
        self.re_imgs = []
        for img in re_imgs:
            tmp_path = []
            for i in img:
                path = os.path.join(args.dataset_path, i)
                tmp_path.append(path)
            self.re_imgs.append(tmp_path)

        re_labels = list(data_img['label'])
        self.re_label = paddle.to_tensor(np.array(re_labels))

        imgs = re_data_img.keys()
        self.imgs = [os.path.join(args.dataset_path, k) for k in imgs]
        labels = list(re_data_img.values())
        self.label = paddle.to_tensor(np.array(labels))
        self.transformers = process

    def __getitem__(self, index):
        img_path = self.imgs[index]

        pil_img = Image.open(img_path)
        data = self.transformers(pil_img)
        label = self.label[index]

        re_img_path = self.re_imgs[index]
        re_data = paddle.empty((6, 3, 224, 224))

        try:

            for i in range(len(re_img_path)):
                re_pil_img = Image.open(re_img_path[i])
                re_data[i] = self.transformers(re_pil_img)

        except:

            print("cannot identify image file or image file is truncated : ", img_path)
            re_data = paddle.randn((6, 3, 224, 224))

        re_label = self.re_label[index]

        return data, label, re_data, re_label

    def __len__(self):
        return len(self.imgs)